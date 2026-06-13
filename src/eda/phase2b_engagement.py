import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from pathlib import Path
from datetime import datetime, UTC
from scipy.stats import mannwhitneyu, kruskal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLEANED_CORPUS = Path("output_data/parquet/cleaned_corpus.parquet")
OUTPUT_PARQUET = Path("output_data/parquet")
OUTPUT_IMG = Path("output_data/img")

LABEL_ORDER = ["positive", "neutral", "negative"]
LABEL_COLORS = {"positive": "#55A868", "neutral": "#4C72B0", "negative": "#C44E52"}

COVID_ERAS = {
    "pre_covid": ("1900-01-01", "2019-12-31"),
    "covid": ("2020-01-01", "2021-12-31"),
    "post_covid": ("2022-01-01", "2099-12-31"),
}


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)


def load_corpus() -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2B — Loading cleaned corpus")
    logger.info("=" * 70)

    if not CLEANED_CORPUS.exists():
        raise FileNotFoundError(f"Run phase0_cleaning.py first. Missing: {CLEANED_CORPUS}")

    t0 = time.time()
    df = pl.read_parquet(CLEANED_CORPUS)
    logger.info(f"  Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    return df


def build_engagement_features(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2B — Computing engagement derived features")
    logger.info("=" * 70)

    df = df.with_columns([
        ((pl.col("reply_count") + 1) / (pl.col("like_count") + 1)).alias("reply_to_like_ratio"),
        pl.col("like_count").log1p().alias("like_count_log"),
        pl.col("reply_count").log1p().alias("reply_count_log"),
    ])

    df = df.with_columns(
        pl.when(pl.col("like_count") < 10).then(pl.lit("Micro (<10)"))
        .when(pl.col("like_count") < 100).then(pl.lit("Small (10-100)"))
        .when(pl.col("like_count") < 1000).then(pl.lit("Medium (100-1K)"))
        .when(pl.col("like_count") < 10000).then(pl.lit("Large (1K-10K)"))
        .otherwise(pl.lit("Viral (>=10K)"))
        .alias("engagement_tier")
    )

    if "published_at" in df.columns and "crawled_at" in df.columns:
        crawled_expr = pl.col("crawled_at")
        if df["crawled_at"].dtype == pl.String:
            crawled_expr = crawled_expr.str.to_datetime(time_zone="UTC")
        elif df["crawled_at"].dtype.time_zone is None:
            crawled_expr = crawled_expr.dt.replace_time_zone("UTC")

        published_expr = pl.col("published_at")
        if df["published_at"].dtype == pl.String:
            published_expr = published_expr.str.to_datetime(time_zone="UTC")
        elif df["published_at"].dtype.time_zone is None:
            published_expr = published_expr.dt.replace_time_zone("UTC")

        df = df.with_columns([
            (
                (crawled_expr - published_expr)
                .dt.total_seconds() / 86400.0
            ).alias("days_since_publish")
        ])
        df = df.with_columns([
            (pl.col("like_count") / (pl.col("days_since_publish").clip(lower_bound=1))).alias("engagement_velocity"),
        ])
        logger.info("  days_since_publish and engagement_velocity computed")
    else:
        logger.warning("  published_at or crawled_at missing — skipping temporal velocity")
        df = df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("days_since_publish"),
            pl.lit(None).cast(pl.Float64).alias("engagement_velocity"),
        ])

    tiers = df["engagement_tier"].value_counts().sort("count", descending=True)
    logger.info("  Engagement tier distribution:")
    for row in tiers.iter_rows(named=True):
        logger.info(f"    {row['engagement_tier']:20s}: {row['count']:>8,}")

    return df


def assign_covid_era(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2B — Assigning COVID era labels")
    logger.info("=" * 70)

    published_expr = pl.col("published_at")
    if df["published_at"].dtype == pl.String:
        published_expr = published_expr.str.to_datetime(time_zone="UTC")
    elif df["published_at"].dtype.time_zone is None:
        published_expr = published_expr.dt.replace_time_zone("UTC")

    df = df.with_columns(
        pl.when(published_expr < pl.lit(datetime(2020, 1, 1, tzinfo=UTC)))
        .then(pl.lit("pre_covid"))
        .when(published_expr <= pl.lit(datetime(2021, 12, 31, tzinfo=UTC)))
        .then(pl.lit("covid"))
        .otherwise(pl.lit("post_covid"))
        .alias("covid_era")
    )

    era_dist = df.group_by("covid_era").agg(pl.len().alias("count")).sort("count", descending=True)
    for row in era_dist.iter_rows(named=True):
        logger.info(f"  {row['covid_era']:15s}: {row['count']:>8,} comments")

    return df


def compute_engagement_semantics_by_label(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2B — Table 4a: Engagement Semantics by Label")
    logger.info("=" * 70)

    tier_order = ["Micro (<10)", "Small (10-100)", "Medium (100-1K)", "Large (1K-10K)", "Viral (>=10K)"]

    rows = []
    for label in LABEL_ORDER:
        subset = df.filter(pl.col("label") == label)
        n = len(subset)
        if n == 0:
            continue

        likes = subset["like_count"].drop_nulls()
        replies = subset["reply_count"].drop_nulls()
        ratio = subset["reply_to_like_ratio"].drop_nulls()

        controversial = (
            subset.filter(
                (pl.col("reply_count") > pl.col("like_count")) & (pl.col("like_count") > 0)
            ).height / n * 100
        )
        endorsed = (
            subset.filter(
                (pl.col("like_count") > pl.col("reply_count")) & (pl.col("like_count") > 0)
            ).height / n * 100
        )

        tier_dist = subset.group_by("engagement_tier").agg(
            (pl.len() / n * 100).alias("pct")
        ).to_dicts()
        tier_pct_str = " | ".join(
            f"{t['engagement_tier']}={t['pct']:.1f}%" for t in tier_dist
        )

        row = {
            "label": label,
            "n": n,
            "mean_like": round(float(likes.mean()), 3),
            "median_like": round(float(likes.median()), 3),
            "mean_reply": round(float(replies.mean()), 3),
            "median_reply": round(float(replies.median()), 3),
            "mean_reply_to_like_ratio": round(float(ratio.mean()), 4),
            "pct_controversial": round(controversial, 3),
            "pct_endorsed": round(endorsed, 3),
        }
        rows.append(row)

        logger.info(
            f"  {label:10s}: n={n:>8,}  likes_med={row['median_like']:.1f}  "
            f"replies_med={row['median_reply']:.1f}  controversial={controversial:.1f}%"
        )

    sem_table = pl.DataFrame(rows)

    neg_ratios = df.filter(pl.col("label") == "negative")["reply_to_like_ratio"].drop_nulls().to_numpy()
    pos_ratios = df.filter(pl.col("label") == "positive")["reply_to_like_ratio"].drop_nulls().to_numpy()
    if len(neg_ratios) > 0 and len(pos_ratios) > 0:
        mwu_stat, mwu_p = mannwhitneyu(neg_ratios, pos_ratios, alternative="greater")
        logger.info(
            f"  Controversy hypothesis (Neg > Pos reply_to_like_ratio): "
            f"U={mwu_stat:.1f}  p={'<0.001' if mwu_p < 0.001 else f'{mwu_p:.4f}'}"
        )

    return sem_table


def compute_temporal_engagement(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2B — Table 4b: Temporal Engagement with COVID Context")
    logger.info("=" * 70)

    if "published_at" not in df.columns:
        logger.warning("  published_at missing — skipping temporal analysis")
        return pl.DataFrame()

    df_month = df.with_columns(
        pl.col("published_at").dt.truncate("1mo").alias("month")
    )

    monthly_baseline = (
        df_month.group_by("month")
        .agg(pl.col("like_count").median().alias("monthly_median_likes"))
        .sort("month")
    )

    df_month = df_month.join(monthly_baseline, on="month", how="left")
    df_month = df_month.with_columns(
        (pl.col("like_count") / (pl.col("monthly_median_likes") + 1)).alias("like_inflation_index")
    )

    temporal_table = (
        df_month.group_by(["month", "covid_era"])
        .agg([
            pl.len().alias("volume"),
            pl.col("monthly_median_likes").first(),
            (pl.col("label") == "positive").sum().alias("n_positive"),
            (pl.col("label") == "neutral").sum().alias("n_neutral"),
            (pl.col("label") == "negative").sum().alias("n_negative"),
            pl.col("days_since_publish").median().alias("median_lag_days"),
            pl.col("engagement_velocity").mean().alias("mean_velocity"),
            pl.col("like_inflation_index").mean().alias("mean_inflation_index"),
        ])
        .with_columns([
            (pl.col("n_positive") / pl.col("volume") * 100).alias("pct_positive"),
            (pl.col("n_neutral") / pl.col("volume") * 100).alias("pct_neutral"),
            (pl.col("n_negative") / pl.col("volume") * 100).alias("pct_negative"),
        ])
        .sort("month")
    )

    for era in ["pre_covid", "covid", "post_covid"]:
        era_sub = temporal_table.filter(pl.col("covid_era") == era)
        if len(era_sub) == 0:
            continue
        logger.info(
            f"  {era:15s}: {len(era_sub)} months  "
            f"avg_vol={era_sub['volume'].mean():.0f}  "
            f"avg_neg%={era_sub['pct_negative'].mean():.1f}%  "
            f"avg_vel={era_sub['mean_velocity'].mean():.3f}"
        )

    return temporal_table


def compute_kw_engagement(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2B — Kruskal-Wallis H across labels for engagement metrics")
    logger.info("=" * 70)

    metrics = ["like_count", "reply_count", "reply_to_like_ratio", "engagement_velocity"]
    rows = []

    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"  {metric} not found — skipping")
            continue

        groups = [
            df.filter(pl.col("label") == lbl)[metric].drop_nulls().to_numpy()
            for lbl in LABEL_ORDER
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue

        h_stat, p_val = kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        eta_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))

        rows.append({
            "metric": metric,
            "H_stat": round(h_stat, 4),
            "p_value": round(p_val, 8),
            "eta_squared": round(max(0, eta_sq), 6),
        })

        logger.info(
            f"  {metric:30s}: H={h_stat:.2f}  p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}  "
            f"η²={max(0, eta_sq):.4f}"
        )

    return pl.DataFrame(rows)


def plot_engagement_semantics_by_label(sem_table: pl.DataFrame, df: pl.DataFrame):
    logger.info("Generating engagement semantics charts")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = sem_table["label"].to_list()
    colors = [LABEL_COLORS.get(l, "#8172B3") for l in labels]

    for i, (metric, ylabel, ax) in enumerate([
        ("mean_like", "Mean Like Count", axes[0, 0]),
        ("mean_reply", "Mean Reply Count", axes[0, 1]),
        ("mean_reply_to_like_ratio", "Mean Reply/Like Ratio", axes[1, 0]),
        ("pct_controversial", "% Controversial (reply > like)", axes[1, 1]),
    ]):
        vals = sem_table[metric].to_list()
        bars = ax.bar(labels, vals, color=colors, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(ylabel, fontweight="bold")
        ax.set_xlabel("Sentiment Label")
        ax.set_ylabel(ylabel)

    plt.suptitle("Engagement Semantics: Like = Endorsement, Reply = Controversy Proxy", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t4_engagement_semantics_by_label.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't4_engagement_semantics_by_label.png'}")


def plot_temporal_engagement(temporal_table: pl.DataFrame):
    if len(temporal_table) == 0:
        return

    logger.info("Generating temporal engagement charts with COVID era shading")

    era_colors = {"pre_covid": "#4C72B0", "covid": "#C44E52", "post_covid": "#55A868"}

    df_pd = temporal_table.sort("month").to_pandas()
    df_pd["month"] = df_pd["month"].astype(str)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    for era, color in era_colors.items():
        mask = df_pd["covid_era"] == era
        for ax in axes:
            ax.fill_between(
                range(len(df_pd)), 0, 1,
                where=mask.values,
                transform=ax.get_xaxis_transform(),
                alpha=0.08, color=color, label=era if ax == axes[0] else "_nolegend_",
            )

    axes[0].plot(range(len(df_pd)), df_pd["volume"], color="#4C72B0", linewidth=2)
    axes[0].set_title("Monthly Comment Volume", fontweight="bold")
    axes[0].set_ylabel("Volume")
    axes[0].legend(title="COVID Era", fontsize=8)

    for label, color in LABEL_COLORS.items():
        pct_col = f"pct_{label}"
        if pct_col in df_pd.columns:
            axes[1].plot(range(len(df_pd)), df_pd[pct_col], color=color, linewidth=2, label=label.capitalize())
    axes[1].set_title("Label Distribution Over Time", fontweight="bold")
    axes[1].set_ylabel("% within month")
    axes[1].legend(fontsize=9)

    if "monthly_median_likes" in df_pd.columns:
        axes[2].plot(range(len(df_pd)), df_pd["monthly_median_likes"], color="#DD8452", linewidth=2)
        axes[2].set_title("Monthly Median Like Count (Engagement Baseline)", fontweight="bold")
        axes[2].set_ylabel("Median Likes")

    step = max(1, len(df_pd) // 20)
    axes[2].set_xticks(range(0, len(df_pd), step))
    axes[2].set_xticklabels(df_pd["month"].tolist()[::step], rotation=45, ha="right", fontsize=8)

    plt.suptitle("Temporal Engagement Dynamics (COVID Context)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t4_temporal_engagement_covid.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't4_temporal_engagement_covid.png'}")


def plot_engagement_tier_by_label(df: pl.DataFrame):
    logger.info("Generating engagement tier × label heatmap")

    tier_order = ["Micro (<10)", "Small (10-100)", "Medium (100-1K)", "Large (1K-10K)", "Viral (>=10K)"]

    cross = (
        df.group_by(["label", "engagement_tier"])
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over("label") * 100).alias("pct_within_label")
        )
    )

    pivot = cross.pivot(
        index="label", columns="engagement_tier", values="pct_within_label", aggregate_function="first"
    ).fill_null(0)

    pivot_pd = pivot.to_pandas().set_index("label")
    existing_tiers = [t for t in tier_order if t in pivot_pd.columns]
    pivot_pd = pivot_pd[existing_tiers]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_pd, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("Engagement Tier Distribution by Label (% within label)", fontweight="bold")
    ax.set_xlabel("Engagement Tier")
    ax.set_ylabel("Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t4_engagement_tier_by_label.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't4_engagement_tier_by_label.png'}")


def save_outputs(
    df: pl.DataFrame,
    sem_table: pl.DataFrame,
    temporal_table: pl.DataFrame,
    kw_table: pl.DataFrame,
):
    logger.info("=" * 70)
    logger.info("PHASE 2B — Saving outputs")
    logger.info("=" * 70)

    sem_table.write_parquet(OUTPUT_PARQUET / "t4_engagement_semantics_by_label.parquet")
    logger.info(f"  Engagement semantics → {OUTPUT_PARQUET / 't4_engagement_semantics_by_label.parquet'}")

    if len(temporal_table) > 0:
        temporal_table.write_parquet(OUTPUT_PARQUET / "t4_temporal_engagement_baseline.parquet")
        logger.info(f"  Temporal baseline    → {OUTPUT_PARQUET / 't4_temporal_engagement_baseline.parquet'}")

    kw_table.write_parquet(OUTPUT_PARQUET / "t4_engagement_kruskal_wallis.parquet")
    logger.info(f"  KW test results      → {OUTPUT_PARQUET / 't4_engagement_kruskal_wallis.parquet'}")

    df.write_parquet(OUTPUT_PARQUET / "corpus_with_engagement.parquet")
    logger.info(f"  Corpus+engagement    → {OUTPUT_PARQUET / 'corpus_with_engagement.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 2B ENGAGEMENT SEMANTICS — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()
    df = load_corpus()
    df = build_engagement_features(df)
    df = assign_covid_era(df)

    sem_table = compute_engagement_semantics_by_label(df)
    temporal_table = compute_temporal_engagement(df)
    kw_table = compute_kw_engagement(df)

    plot_engagement_semantics_by_label(sem_table, df)
    plot_temporal_engagement(temporal_table)
    plot_engagement_tier_by_label(df)

    save_outputs(df, sem_table, temporal_table, kw_table)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 2B COMPLETE — elapsed: {elapsed:.1f}s")
    logger.info("=" * 70)
