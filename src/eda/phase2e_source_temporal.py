import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from pathlib import Path
from datetime import datetime, UTC

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
BIAS_THRESHOLD_PCT = 80.0
ANOMALY_SIGMA = 2.0


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)


def load_corpus() -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2E — Loading cleaned corpus")
    logger.info("=" * 70)

    if not CLEANED_CORPUS.exists():
        raise FileNotFoundError(f"Run phase0_cleaning.py first. Missing: {CLEANED_CORPUS}")

    t0 = time.time()
    df = pl.read_parquet(CLEANED_CORPUS)
    logger.info(f"  Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    return df


def compute_source_query_profile(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2E — Table 7a: Source Query Bias & Intent Profiling")
    logger.info("=" * 70)

    if "source_query" not in df.columns:
        logger.warning("  source_query column not found — returning empty table")
        return pl.DataFrame()

    total = len(df)

    base = (
        df.group_by("source_query")
        .agg([
            pl.len().alias("n"),
            (pl.col("label") == "positive").sum().alias("n_positive"),
            (pl.col("label") == "neutral").sum().alias("n_neutral"),
            (pl.col("label") == "negative").sum().alias("n_negative"),
            pl.col("like_count").mean().alias("avg_likes"),
            pl.col("reply_count").mean().alias("avg_replies"),
            pl.col("char_count").mean().alias("avg_length"),
        ])
        .with_columns([
            (pl.col("n_positive") / pl.col("n") * 100).alias("pct_positive"),
            (pl.col("n_neutral") / pl.col("n") * 100).alias("pct_neutral"),
            (pl.col("n_negative") / pl.col("n") * 100).alias("pct_negative"),
        ])
        .sort("n", descending=True)
    )

    base = base.with_columns(
        pl.when(
            (pl.col("pct_positive") > BIAS_THRESHOLD_PCT) |
            (pl.col("pct_neutral") > BIAS_THRESHOLD_PCT) |
            (pl.col("pct_negative") > BIAS_THRESHOLD_PCT)
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("sampling_biased")
    )

    if "primary_language" in df.columns:
        dominant_lang = (
            df.group_by(["source_query", "primary_language"])
            .agg(pl.len().alias("cnt"))
            .sort("cnt", descending=True)
            .group_by("source_query")
            .agg(pl.col("primary_language").first().alias("dominant_lang"))
        )
        base = base.join(dominant_lang, on="source_query", how="left")
    else:
        base = base.with_columns(pl.lit("unknown").alias("dominant_lang"))

    logger.info(f"  Source queries found: {len(base)}")
    biased_count = base.filter(pl.col("sampling_biased")).height
    logger.info(f"  Biased queries (>80% single-label): {biased_count}")

    for row in base.iter_rows(named=True):
        logger.info(
            f"  {str(row['source_query'])[:40]:40s}: n={row['n']:>7,}  "
            f"pos={row['pct_positive']:.1f}%  neu={row['pct_neutral']:.1f}%  neg={row['pct_negative']:.1f}%  "
            f"bias={row['sampling_biased']}"
        )

    return base


def compute_temporal_baseline(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2E — Table 7b: Temporal Volume & Sentiment Dynamics")
    logger.info("=" * 70)

    if "published_at" not in df.columns:
        logger.warning("  published_at missing — returning empty temporal table")
        return pl.DataFrame()

    df_month = df.with_columns(
        pl.col("published_at").dt.truncate("1mo").alias("month")
    )

    monthly = (
        df_month.group_by("month")
        .agg([
            pl.len().alias("volume"),
            (pl.col("label") == "positive").sum().alias("n_positive"),
            (pl.col("label") == "neutral").sum().alias("n_neutral"),
            (pl.col("label") == "negative").sum().alias("n_negative"),
            pl.col("like_count").median().alias("median_likes"),
        ])
        .with_columns([
            (pl.col("n_positive") / pl.col("volume") * 100).alias("pct_positive"),
            (pl.col("n_neutral") / pl.col("volume") * 100).alias("pct_neutral"),
            (pl.col("n_negative") / pl.col("volume") * 100).alias("pct_negative"),
        ])
        .sort("month")
    )

    if "crawled_at" in df.columns:
        crawled_expr = pl.col("crawled_at")
        if df_month["crawled_at"].dtype == pl.String:
            crawled_expr = crawled_expr.str.to_datetime(time_zone="UTC")
        elif df_month["crawled_at"].dtype.time_zone is None:
            crawled_expr = crawled_expr.dt.replace_time_zone("UTC")

        published_expr = pl.col("published_at")
        if df_month["published_at"].dtype == pl.String:
            published_expr = published_expr.str.to_datetime(time_zone="UTC")
        elif df_month["published_at"].dtype.time_zone is None:
            published_expr = published_expr.dt.replace_time_zone("UTC")

        lag_monthly = (
            df_month.with_columns(
                (
                    (crawled_expr - published_expr)
                    .dt.total_seconds() / 86400.0
                ).alias("lag_days")
            )
            .group_by("month")
            .agg(pl.col("lag_days").median().alias("median_lag_days"))
        )
        monthly = monthly.join(lag_monthly, on="month", how="left")
    else:
        monthly = monthly.with_columns(pl.lit(None).cast(pl.Float64).alias("median_lag_days"))

    neg_pct_vals = monthly["pct_negative"].to_numpy()
    vol_vals = monthly["volume"].to_numpy()
    mean_neg = neg_pct_vals.mean()
    std_neg = neg_pct_vals.std()
    mean_vol = vol_vals.mean()
    std_vol = vol_vals.std()

    monthly = monthly.with_columns(
        pl.when(
            (pl.col("pct_negative") > mean_neg + ANOMALY_SIGMA * std_neg) |
            (pl.col("volume") > mean_vol + ANOMALY_SIGMA * std_vol)
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("anomaly_flag")
    )

    anomaly_count = monthly.filter(pl.col("anomaly_flag")).height
    logger.info(f"  Monthly records: {len(monthly)}")
    logger.info(f"  Temporal anomalies flagged: {anomaly_count}")
    logger.info(f"  Global avg neg%: {mean_neg:.2f}%  std: {std_neg:.2f}%")
    logger.info(f"  Global avg volume: {mean_vol:.0f}  std: {std_vol:.0f}")

    if anomaly_count > 0:
        anomalies = monthly.filter(pl.col("anomaly_flag"))
        for row in anomalies.iter_rows(named=True):
            logger.info(
                f"    Anomaly at {row['month']}: vol={row['volume']:,}  neg={row['pct_negative']:.1f}%"
            )

    return monthly


def plot_source_query_divergence(sq_profile: pl.DataFrame):
    if len(sq_profile) == 0:
        return

    logger.info("Generating source query bias divergence chart")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    queries = [str(q)[:30] for q in sq_profile["source_query"].to_list()]
    x = np.arange(len(queries))
    width = 0.28

    pct_pos = sq_profile["pct_positive"].to_list()
    pct_neu = sq_profile["pct_neutral"].to_list()
    pct_neg = sq_profile["pct_negative"].to_list()

    axes[0].barh(x + width, pct_pos, width, label="Positive", color=LABEL_COLORS["positive"], alpha=0.85)
    axes[0].barh(x, pct_neu, width, label="Neutral", color=LABEL_COLORS["neutral"], alpha=0.85)
    axes[0].barh(x - width, pct_neg, width, label="Negative", color=LABEL_COLORS["negative"], alpha=0.85)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(queries, fontsize=9)
    axes[0].set_xlabel("% within source query")
    axes[0].set_title("Label Distribution per Source Query", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].axvline(BIAS_THRESHOLD_PCT, color="red", linestyle="--", linewidth=1, alpha=0.5, label=f"Bias threshold {BIAS_THRESHOLD_PCT}%")

    ns = sq_profile["n"].to_list()
    avg_likes = sq_profile["avg_likes"].to_list()
    avg_replies = sq_profile["avg_replies"].to_list()

    axes[1].scatter(avg_likes, avg_replies, s=[n / 100 for n in ns],
                    c=[LABEL_COLORS["negative"] if b else LABEL_COLORS["positive"]
                       for b in sq_profile["sampling_biased"].to_list()],
                    alpha=0.8, edgecolor="white", linewidth=1)
    for i, q in enumerate(queries):
        axes[1].annotate(q[:15], (avg_likes[i], avg_replies[i]), fontsize=7, ha="left")
    axes[1].set_xlabel("Average Like Count")
    axes[1].set_ylabel("Average Reply Count")
    axes[1].set_title("Engagement Profile by Source Query\n(size=comment volume, red=biased)", fontweight="bold")
    axes[1].set_xscale("log")

    plt.suptitle("Source Query Intent & Bias Analysis", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t7_source_query_profile.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't7_source_query_profile.png'}")


def plot_temporal_dynamics(temporal: pl.DataFrame):
    if len(temporal) == 0:
        return

    logger.info("Generating temporal dynamics chart")

    df_pd = temporal.sort("month").to_pandas()
    df_pd["month_str"] = df_pd["month"].astype(str)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    axes[0].fill_between(range(len(df_pd)), df_pd["volume"], alpha=0.6, color="#4C72B0")
    axes[0].plot(range(len(df_pd)), df_pd["volume"], color="#4C72B0", linewidth=2)
    anomaly_mask = df_pd["anomaly_flag"]
    axes[0].scatter(
        [i for i, a in enumerate(anomaly_mask) if a],
        [df_pd["volume"].iloc[i] for i, a in enumerate(anomaly_mask) if a],
        color="red", zorder=5, s=60, label="Anomaly", marker="^"
    )
    axes[0].set_title("Monthly Comment Volume", fontweight="bold")
    axes[0].set_ylabel("Volume")
    axes[0].legend(fontsize=8)

    for label, color in LABEL_COLORS.items():
        pct_col = f"pct_{label}"
        axes[1].plot(range(len(df_pd)), df_pd[pct_col], color=color, linewidth=2, label=label.capitalize())
    axes[1].set_title("Label % Trends Over Time", fontweight="bold")
    axes[1].set_ylabel("% within month")
    axes[1].legend(fontsize=9)

    axes[2].plot(range(len(df_pd)), df_pd["median_likes"], color="#DD8452", linewidth=2)
    axes[2].fill_between(range(len(df_pd)), df_pd["median_likes"], alpha=0.3, color="#DD8452")
    axes[2].set_title("Median Likes per Month", fontweight="bold")
    axes[2].set_ylabel("Median Like Count")

    step = max(1, len(df_pd) // 18)
    axes[2].set_xticks(range(0, len(df_pd), step))
    axes[2].set_xticklabels(df_pd["month_str"].tolist()[::step], rotation=45, ha="right", fontsize=8)

    plt.suptitle("Temporal Volume & Sentiment Dynamics", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t7_temporal_dynamics.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't7_temporal_dynamics.png'}")


def plot_lag_distribution(df: pl.DataFrame):
    if "crawled_at" not in df.columns or "published_at" not in df.columns:
        return

    logger.info("Generating crawl lag distribution chart")

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

    df_lag = df.with_columns(
        (
            (crawled_expr - published_expr)
            .dt.total_seconds() / 86400.0
        ).alias("lag_days")
    )

    lag_vals = df_lag["lag_days"].drop_nulls().to_numpy()
    cap = float(np.percentile(lag_vals, 99))
    lag_clipped = np.clip(lag_vals, 0, cap)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(lag_clipped, bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(lag_clipped), color="red", linestyle="--", linewidth=2,
               label=f"Median: {np.median(lag_clipped):.1f} days")
    ax.set_xlabel("Days between publish and crawl (99th pct clipped)")
    ax.set_ylabel("Comment Count")
    ax.set_title("Crawl Lag Distribution (Publish → Crawl)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t7_crawl_lag_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't7_crawl_lag_distribution.png'}")


def save_outputs(sq_profile: pl.DataFrame, temporal: pl.DataFrame):
    logger.info("=" * 70)
    logger.info("PHASE 2E — Saving outputs")
    logger.info("=" * 70)

    if len(sq_profile) > 0:
        sq_profile.write_parquet(OUTPUT_PARQUET / "t7_source_query_profile.parquet")
        logger.info(f"  Source query profile → {OUTPUT_PARQUET / 't7_source_query_profile.parquet'}")

    if len(temporal) > 0:
        temporal.write_parquet(OUTPUT_PARQUET / "t7_temporal_baseline.parquet")
        logger.info(f"  Temporal baseline    → {OUTPUT_PARQUET / 't7_temporal_baseline.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 2E SOURCE & TEMPORAL — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()
    df = load_corpus()

    sq_profile = compute_source_query_profile(df)
    temporal = compute_temporal_baseline(df)

    plot_source_query_divergence(sq_profile)
    plot_temporal_dynamics(temporal)
    plot_lag_distribution(df)

    save_outputs(sq_profile, temporal)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 2E COMPLETE — elapsed: {elapsed:.1f}s")
    logger.info("=" * 70)
