import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import logging
import time
import os
from pathlib import Path
from datetime import datetime
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLEANED_CORPUS = Path("output_data/parquet/cleaned_corpus.parquet")
OUTPUT_PARQUET = Path("output_data/parquet")
OUTPUT_IMG = Path("output_data/img")

NUMERIC_FEATURES = [
    "char_count", "word_count", "like_count", "reply_count",
    "emoji_count", "uppercase_ratio", "exclamation_count", "question_count",
]
LABEL_ORDER = ["positive", "neutral", "negative"]
LABEL_COLORS = {"positive": "#55A868", "neutral": "#4C72B0", "negative": "#C44E52"}


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)


def load_corpus() -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 1 — Loading cleaned corpus")
    logger.info("=" * 70)

    if not CLEANED_CORPUS.exists():
        raise FileNotFoundError(
            f"Cleaned corpus not found at {CLEANED_CORPUS}. Run phase0_cleaning.py first."
        )

    t0 = time.time()
    df = pl.read_parquet(CLEANED_CORPUS)
    logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns in {time.time() - t0:.2f}s")
    logger.info(f"  Label distribution: {df['label'].value_counts().to_dicts()}")
    return df


def compute_gini(counts: np.ndarray) -> float:
    props = counts / counts.sum()
    return 1.0 - float(np.sum(props ** 2))


def compute_label_distribution(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 1 — Table 2a: Label Distribution & Imbalance Diagnostics")
    logger.info("=" * 70)

    total = len(df)
    counts = df.group_by("label").agg(pl.len().alias("count")).sort("count", descending=True)

    gini = compute_gini(counts["count"].to_numpy())
    logger.info(f"  Gini coefficient: {gini:.4f}")

    pos_count = counts.filter(pl.col("label") == "positive")["count"][0] if "positive" in counts["label"].to_list() else 1

    rows = []
    cumulative = 0
    for row in counts.iter_rows(named=True):
        cumulative += row["count"]
        rows.append({
            "label": row["label"],
            "count": row["count"],
            "pct": round(row["count"] / total * 100, 3),
            "cumulative_pct": round(cumulative / total * 100, 3),
            "imbalance_ratio_vs_Positive": round(row["count"] / pos_count, 3),
            "gini_coefficient": round(gini, 4),
        })

    dist_table = pl.DataFrame(rows)

    for row in dist_table.iter_rows(named=True):
        logger.info(
            f"  {row['label']:10s}: {row['count']:>8,} ({row['pct']:6.2f}%)  "
            f"imbalance_vs_pos={row['imbalance_ratio_vs_Positive']:.3f}"
        )

    leakage_check_log = check_label_leakage(df)
    logger.info(f"  Leakage check: {leakage_check_log}")

    return dist_table


def check_label_leakage(df: pl.DataFrame) -> str:
    if "source_file" not in df.columns or "source_row" not in df.columns:
        return "source_file/source_row columns not present — leakage check skipped"

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())

    source_file_encoded = LabelEncoder().fit_transform(df["source_file"].fill_null("unknown").to_numpy())
    source_row_vals = df["source_row"].fill_null(0).to_numpy().astype(float).reshape(-1, 1)

    mi_file = mutual_info_classif(source_file_encoded.reshape(-1, 1), y, discrete_features=True, random_state=42)[0]
    mi_row = mutual_info_classif(source_row_vals, y, discrete_features=False, random_state=42)[0]

    return f"MI(source_file→label)={mi_file:.4f}, MI(source_row→label)={mi_row:.4f}"


def compute_stratified_stats(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 1 — Table 2b: Stratified Descriptive Statistics per Label")
    logger.info("=" * 70)

    t0 = time.time()
    rows = []
    for label in LABEL_ORDER:
        subset = df.filter(pl.col("label") == label)
        n = len(subset)
        if n == 0:
            logger.warning(f"  Label '{label}' has 0 rows — skipping")
            continue

        for feat in NUMERIC_FEATURES:
            if feat not in subset.columns:
                logger.warning(f"  Feature '{feat}' not in dataframe — skipping")
                continue
            vals = subset[feat].drop_nulls()
            rows.append({
                "label": label,
                "feature": feat,
                "n": n,
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "std": float(vals.std()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })

        logger.info(
            f"  {label:10s}: n={n:>8,}  "
            f"chars={subset['char_count'].mean():.1f}  "
            f"likes={subset['like_count'].mean():.2f}  "
            f"emojis={subset['emoji_count'].mean():.3f}"
        )

    strat_table = pl.DataFrame(rows)
    logger.info(f"  Stratified stats computed in {time.time() - t0:.2f}s — {len(strat_table)} rows")
    return strat_table


def compute_label_by_source_query(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 1 — Table 2c: Label × Source Query Cross-Tab")
    logger.info("=" * 70)

    if "source_query" not in df.columns:
        logger.warning("  source_query column not found — skipping cross-tab")
        return pl.DataFrame()

    cross = (
        df.group_by(["source_query", "label"])
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over("source_query") * 100).alias("pct_within_query")
        )
        .sort(["source_query", "label"])
    )

    for query in cross["source_query"].unique().to_list():
        rows = cross.filter(pl.col("source_query") == query)
        dist_str = "  ".join(
            f"{r['label']}={r['pct_within_query']:.1f}%"
            for r in rows.iter_rows(named=True)
        )
        logger.info(f"  {str(query)[:40]:40s}: {dist_str}")

    return cross


def compute_label_by_quarter(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 1 — Table 2d: Label × Quarter Distribution")
    logger.info("=" * 70)

    if "published_at" not in df.columns:
        logger.warning("  published_at column not found — skipping quarter table")
        return pl.DataFrame()

    df_q = df.with_columns(
        pl.col("published_at").dt.truncate("1q").alias("quarter")
    )

    cross = (
        df_q.group_by(["quarter", "label"])
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over("quarter") * 100).alias("pct_within_quarter")
        )
        .sort(["quarter", "label"])
    )

    quarters = sorted(cross["quarter"].unique().to_list())
    logger.info(f"  {len(quarters)} quarters spanning {quarters[0]} → {quarters[-1]}")
    return cross


def plot_label_distribution(dist_table: pl.DataFrame):
    logger.info("Generating label distribution charts")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = dist_table["label"].to_list()
    counts = dist_table["count"].to_list()
    pcts = dist_table["pct"].to_list()
    colors = [LABEL_COLORS.get(l, "#8172B3") for l in labels]

    axes[0].bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    for i, (c, p) in enumerate(zip(counts, pcts)):
        axes[0].text(i, c * 1.01, f"{c:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_title("Label Counts", fontweight="bold")
    axes[0].set_ylabel("Count")

    wedges, texts, autotexts = axes[1].pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(11)
    axes[1].set_title("Label Distribution (%)", fontweight="bold")

    imbalance = dist_table["imbalance_ratio_vs_Positive"].to_list()
    gini = dist_table["gini_coefficient"][0]
    fig.suptitle(
        f"Label Distribution — Gini={gini:.4f}  |  Imbalance ratios: {', '.join(f'{l}={r:.2f}x' for l, r in zip(labels, imbalance))}",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t2_label_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't2_label_distribution.png'}")


def plot_stratified_violins(strat_table: pl.DataFrame, df: pl.DataFrame):
    logger.info("Generating stratified violin plots for all numeric features")

    n_feats = len(NUMERIC_FEATURES)
    ncols = 4
    nrows = (n_feats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(NUMERIC_FEATURES):
        if feat not in df.columns:
            axes[i].set_visible(False)
            continue
        data_by_label = {
            lbl: df.filter(pl.col("label") == lbl)[feat].drop_nulls().to_numpy()
            for lbl in LABEL_ORDER
        }
        parts = axes[i].violinplot(
            [data_by_label[l] for l in LABEL_ORDER if len(data_by_label[l]) > 0],
            positions=range(len(LABEL_ORDER)),
            showmedians=True,
        )
        for pc, lbl in zip(parts["bodies"], LABEL_ORDER):
            pc.set_facecolor(LABEL_COLORS[lbl])
            pc.set_alpha(0.7)
        axes[i].set_xticks(range(len(LABEL_ORDER)))
        axes[i].set_xticklabels(LABEL_ORDER, fontsize=9)
        axes[i].set_title(feat, fontweight="bold", fontsize=10)

        h_stat, p_val = sp_stats.kruskal(*[
            data_by_label[l] for l in LABEL_ORDER if len(data_by_label[l]) > 0
        ])
        axes[i].set_xlabel(f"KW H={h_stat:.1f}  p={'<0.001' if p_val < 0.001 else f'{p_val:.3f}'}", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Stratified Feature Distributions by Label (Kruskal-Wallis H)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t2_stratified_violins.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't2_stratified_violins.png'}")


def plot_source_query_label_heatmap(cross_sq: pl.DataFrame):
    if len(cross_sq) == 0:
        return

    logger.info("Generating source query × label heatmap")

    pivot = cross_sq.pivot(
        index="source_query", columns="label", values="pct_within_query", aggregate_function="first"
    ).fill_null(0)

    pivot_pd = pivot.to_pandas().set_index("source_query")

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot_pd) * 0.6)))
    sns.heatmap(
        pivot_pd, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "% within query"},
    )
    ax.set_title("Label Distribution by Source Query (%)", fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Source Query")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t2_source_query_label_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't2_source_query_label_heatmap.png'}")


def plot_temporal_label_trends(cross_q: pl.DataFrame):
    if len(cross_q) == 0:
        return

    logger.info("Generating temporal label trend chart")

    fig, ax = plt.subplots(figsize=(14, 5))
    for label in LABEL_ORDER:
        subset = cross_q.filter(pl.col("label") == label).sort("quarter")
        ax.plot(
            [str(q) for q in subset["quarter"].to_list()],
            subset["pct_within_quarter"].to_list(),
            marker="o", markersize=4, linewidth=2,
            label=label.capitalize(), color=LABEL_COLORS[label],
        )

    ax.set_title("Label Distribution Over Time (by Quarter)", fontweight="bold")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("% of comments in quarter")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t2_temporal_label_trends.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't2_temporal_label_trends.png'}")


def save_outputs(
    dist_table: pl.DataFrame,
    strat_table: pl.DataFrame,
    cross_sq: pl.DataFrame,
    cross_q: pl.DataFrame,
):
    logger.info("=" * 70)
    logger.info("PHASE 1 — Saving outputs")
    logger.info("=" * 70)

    dist_table.write_parquet(OUTPUT_PARQUET / "t2_label_distribution.parquet")
    logger.info(f"  Label distribution → {OUTPUT_PARQUET / 't2_label_distribution.parquet'}")

    strat_table.write_parquet(OUTPUT_PARQUET / "t2_stratified_baseline_stats.parquet")
    logger.info(f"  Stratified stats   → {OUTPUT_PARQUET / 't2_stratified_baseline_stats.parquet'}")

    if len(cross_sq) > 0:
        cross_sq.write_parquet(OUTPUT_PARQUET / "t2_label_by_source_query.parquet")
        logger.info(f"  Label×source_query → {OUTPUT_PARQUET / 't2_label_by_source_query.parquet'}")

    if len(cross_q) > 0:
        cross_q.write_parquet(OUTPUT_PARQUET / "t2_label_by_quarter.parquet")
        logger.info(f"  Label×quarter      → {OUTPUT_PARQUET / 't2_label_by_quarter.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 1 LABEL STATISTICS — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()
    df = load_corpus()

    dist_table = compute_label_distribution(df)
    strat_table = compute_stratified_stats(df)
    cross_sq = compute_label_by_source_query(df)
    cross_q = compute_label_by_quarter(df)

    plot_label_distribution(dist_table)
    plot_stratified_violins(strat_table, df)
    plot_source_query_label_heatmap(cross_sq)
    plot_temporal_label_trends(cross_q)

    save_outputs(dist_table, strat_table, cross_sq, cross_q)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 1 COMPLETE — elapsed: {elapsed:.1f}s")
    logger.info("=" * 70)
