import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from features.emoji_classifier import (
    EMOJI_TYPES, EMOJI_TYPE_MAP, EMOJI_REGEX,
    extract_emojis_from_text, classify_emoji,
    count_emojis_by_type, is_emoji_only,
)

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


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)


def load_corpus() -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2A — Loading cleaned corpus")
    logger.info("=" * 70)

    if not CLEANED_CORPUS.exists():
        raise FileNotFoundError(f"Run phase0_cleaning.py first. Missing: {CLEANED_CORPUS}")

    t0 = time.time()
    df = pl.read_parquet(CLEANED_CORPUS)
    logger.info(f"  Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    return df


def extract_emoji_lists(texts: list[str]) -> list[list[str]]:
    logger.info(f"  Extracting emoji lists from {len(texts):,} texts...")
    result = []
    for text in tqdm(texts, desc="Emoji extraction", unit="comment", ncols=80):
        result.append(extract_emojis_from_text(text))
    return result


def data_driven_emoji_survey(df: pl.DataFrame, emoji_lists: list[list[str]]) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2A — Data-driven emoji survey (correlation before classification)")
    logger.info("=" * 70)

    labels = df["label"].to_list()
    emoji_label_counts: dict[str, dict[str, int]] = Counter()

    logger.info("  Building emoji × label co-occurrence matrix...")
    for emojis, label in tqdm(
        zip(emoji_lists, labels), total=len(labels), desc="Emoji survey", unit="comment", ncols=80
    ):
        for emoji in set(emojis):
            key = f"{emoji}|||{label}"
            emoji_label_counts[key] = emoji_label_counts.get(key, 0) + 1

    emoji_set = set()
    label_set = set(LABEL_ORDER)
    for key in emoji_label_counts:
        e, l = key.split("|||", 1)
        emoji_set.add(e)

    top_emojis_global = Counter()
    for emojis in emoji_lists:
        top_emojis_global.update(emojis)

    top_50 = [e for e, _ in top_emojis_global.most_common(50)]
    logger.info(f"  Top 50 most frequent emojis: {top_50[:10]}... (showing first 10)")

    rows = []
    for emoji in top_50:
        row = {"emoji": emoji}
        total = 0
        for label in LABEL_ORDER:
            cnt = emoji_label_counts.get(f"{emoji}|||{label}", 0)
            row[f"count_{label}"] = cnt
            total += cnt
        row["total"] = total
        row["assigned_type"] = classify_emoji(emoji)
        for label in LABEL_ORDER:
            row[f"pct_{label}"] = round(row[f"count_{label}"] / total * 100, 2) if total > 0 else 0.0
        rows.append(row)

    survey_df = pl.DataFrame(rows).sort("total", descending=True)

    logger.info("  Top 20 emojis with label breakdown:")
    for row in survey_df.head(20).iter_rows(named=True):
        logger.info(
            f"    {row['emoji']}  ({row['assigned_type']:15s})  "
            f"total={row['total']:>6,}  "
            f"pos={row['pct_positive']:.1f}%  neu={row['pct_neutral']:.1f}%  neg={row['pct_negative']:.1f}%"
        )

    return survey_df


def add_emoji_type_columns(df: pl.DataFrame, emoji_lists: list[list[str]]) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2A — Adding 12 emoji type columns (6 presence + 6 count)")
    logger.info("=" * 70)

    t0 = time.time()
    type_counts_all: dict[str, list[int]] = {t: [] for t in EMOJI_TYPES}
    type_presence_all: dict[str, list[bool]] = {t: [] for t in EMOJI_TYPES}
    emoji_only_flags: list[bool] = []

    for emojis, text in tqdm(
        zip(emoji_lists, df["comment_text"].to_list()),
        total=len(emoji_lists), desc="Type assignment", unit="comment", ncols=80
    ):
        type_counts = {t: 0 for t in EMOJI_TYPES}
        for e in emojis:
            type_counts[classify_emoji(e)] += 1
        for t in EMOJI_TYPES:
            type_counts_all[t].append(type_counts[t])
            type_presence_all[t].append(type_counts[t] > 0)
        non_emoji = EMOJI_REGEX.sub("", text if isinstance(text, str) else "").strip()
        emoji_only_flags.append(len(non_emoji) == 0 and len(emojis) > 0)

    new_cols = {}
    for t in EMOJI_TYPES:
        new_cols[f"emoji_{t}_count"] = pl.Series(type_counts_all[t], dtype=pl.Int32)
        new_cols[f"emoji_{t}_present"] = pl.Series(type_presence_all[t], dtype=pl.Boolean)

    new_cols["is_emoji_only"] = pl.Series(emoji_only_flags, dtype=pl.Boolean)
    new_cols["emoji_density"] = pl.Series(
        [c / max(cc, 1) for c, cc in zip(df["emoji_count"].to_list(), df["char_count"].to_list())],
        dtype=pl.Float32,
    )

    df = df.with_columns([pl.lit(v).alias(k) for k, v in new_cols.items()])

    emoji_only_count = sum(emoji_only_flags)
    logger.info(f"  Emoji-only comments: {emoji_only_count:,} ({emoji_only_count / len(df) * 100:.2f}%)")
    logger.info(f"  Column addition completed in {time.time() - t0:.2f}s")
    return df


def compute_emoji_type_inventory(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2A — Table 3a: Emoji Type Inventory")
    logger.info("=" * 70)

    total_comments = len(df)
    rows = []

    for etype in EMOJI_TYPES:
        count_col = f"emoji_{etype}_count"
        present_col = f"emoji_{etype}_present"

        total_occ = df[count_col].sum()
        pct_corpus = df[present_col].sum() / total_comments * 100

        type_emojis = [e for e, t in EMOJI_TYPE_MAP.items() if t == etype]
        rows.append({
            "emoji_type": etype,
            "total_occurrences": total_occ,
            "pct_comments_with_type": round(pct_corpus, 3),
            "n_emojis_in_type": len(type_emojis),
            "example_emojis": " ".join(type_emojis[:5]),
        })
        logger.info(
            f"  {etype:20s}: {total_occ:>8,} occurrences  "
            f"{pct_corpus:.2f}% of comments have it"
        )

    emoji_only_count = df["is_emoji_only"].sum()
    logger.info(f"  Emoji-only comments: {emoji_only_count:,}")

    return pl.DataFrame(rows)


def compute_emoji_label_correlation(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    logger.info("=" * 70)
    logger.info("PHASE 2A — Table 3b: Emoji-Label Correlation (Chi², Cramér's V, MI)")
    logger.info("=" * 70)

    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    total = len(df)

    corr_rows = []
    for etype in EMOJI_TYPES:
        count_col = f"emoji_{etype}_count"
        present_col = f"emoji_{etype}_present"

        presence = df[present_col].to_numpy().astype(int)
        contingency = np.zeros((2, 3), dtype=int)

        for label_idx, label in enumerate(LABEL_ORDER):
            label_mask = df["label"].to_numpy() == label
            contingency[0, label_idx] = int(np.sum(presence[label_mask] == 0))
            contingency[1, label_idx] = int(np.sum(presence[label_mask] == 1))

        chi2_stat, p_val, _, _ = chi2_contingency(contingency)
        n = contingency.sum()
        k = min(contingency.shape) - 1
        cramers_v = float(np.sqrt(chi2_stat / (n * k))) if n > 0 and k > 0 else 0.0

        mi_score = mutual_info_classif(
            presence.reshape(-1, 1), y, discrete_features=True, random_state=42
        )[0]

        density_by_label = {}
        for label in LABEL_ORDER:
            mask = df["label"].to_numpy() == label
            if mask.sum() == 0:
                density_by_label[label] = 0.0
            else:
                density_vals = df.filter(pl.col("label") == label)["emoji_density"].drop_nulls().to_numpy()
                density_by_label[label] = float(density_vals.mean()) if len(density_vals) > 0 else 0.0

        corr_rows.append({
            "emoji_type": etype,
            "chi2_stat": round(chi2_stat, 4),
            "p_value": round(p_val, 6),
            "cramers_v": round(cramers_v, 4),
            "mi_score": round(mi_score, 6),
            "mean_density_positive": round(density_by_label.get("positive", 0), 5),
            "mean_density_neutral": round(density_by_label.get("neutral", 0), 5),
            "mean_density_negative": round(density_by_label.get("negative", 0), 5),
        })

        logger.info(
            f"  {etype:20s}: chi2={chi2_stat:.2f}  p={p_val:.4f}  "
            f"cramers_v={cramers_v:.4f}  MI={mi_score:.4f}"
        )

    logger.info("  Computing per-individual-emoji MI scores...")
    top_emoji_mi = compute_individual_emoji_mi(df, le, y)

    return pl.DataFrame(corr_rows), top_emoji_mi


def compute_individual_emoji_mi(
    df: pl.DataFrame, le: LabelEncoder, y: np.ndarray
) -> pl.DataFrame:
    all_emojis_in_data = set()
    for text in df["comment_text"].to_list():
        all_emojis_in_data.update(extract_emojis_from_text(text or ""))

    logger.info(f"  Unique emojis in corpus: {len(all_emojis_in_data):,}")

    global_counts = Counter()
    for text in df["comment_text"].to_list():
        global_counts.update(set(extract_emojis_from_text(text or "")))

    top_100 = [e for e, _ in global_counts.most_common(100)]
    logger.info(f"  Computing MI for top 100 emojis...")

    rows = []
    for emoji in tqdm(top_100, desc="Emoji MI", ncols=80):
        presence = np.array(
            [1 if emoji in extract_emojis_from_text(t or "") else 0 for t in df["comment_text"].to_list()],
            dtype=int,
        )
        if presence.sum() < 5:
            continue

        mi = mutual_info_classif(presence.reshape(-1, 1), y, discrete_features=True, random_state=42)[0]

        label_counts = {}
        for label in LABEL_ORDER:
            mask = df["label"].to_numpy() == label
            label_counts[label] = int(presence[mask].sum())
        most_common_label = max(label_counts, key=label_counts.get)

        rows.append({
            "emoji": emoji,
            "total_occurrences": global_counts[emoji],
            "mi_score": round(mi, 6),
            "most_common_label": most_common_label,
            "emoji_type": classify_emoji(emoji),
            "count_positive": label_counts.get("positive", 0),
            "count_neutral": label_counts.get("neutral", 0),
            "count_negative": label_counts.get("negative", 0),
        })

    top_mi_df = pl.DataFrame(rows).sort("mi_score", descending=True)
    logger.info(f"  Top discriminating emojis (by MI):")
    for row in top_mi_df.head(10).iter_rows(named=True):
        logger.info(
            f"    {row['emoji']}  MI={row['mi_score']:.4f}  "
            f"most_common={row['most_common_label']}  type={row['emoji_type']}"
        )
    return top_mi_df


def plot_emoji_survey_heatmap(survey_df: pl.DataFrame):
    logger.info("Generating emoji survey correlation heatmap")

    pct_cols = ["pct_positive", "pct_neutral", "pct_negative"]
    top20 = survey_df.head(20)

    heatmap_data = top20.select(["emoji"] + pct_cols).to_pandas().set_index("emoji")
    heatmap_data.columns = ["Positive %", "Neutral %", "Negative %"]

    fig, ax = plt.subplots(figsize=(9, 10))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
        linewidths=0.5, ax=ax, vmin=0, vmax=100,
        cbar_kws={"label": "% of emoji comments with label"},
    )
    ax.set_title("Emoji × Label Distribution (Top 20 most frequent emojis)", fontweight="bold")
    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Emoji")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t3_emoji_label_survey_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't3_emoji_label_survey_heatmap.png'}")


def plot_cramers_v_bar(corr_df: pl.DataFrame):
    logger.info("Generating Cramér's V bar chart by emoji type")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    types = corr_df["emoji_type"].to_list()
    cramers = corr_df["cramers_v"].to_list()
    mi_scores = corr_df["mi_score"].to_list()

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(types)))

    axes[0].barh(types, cramers, color=colors, edgecolor="white")
    axes[0].set_title("Cramér's V by Emoji Type", fontweight="bold")
    axes[0].set_xlabel("Cramér's V")
    axes[0].axvline(0.1, color="red", linestyle="--", linewidth=1, alpha=0.7, label="V=0.1 threshold")
    axes[0].legend(fontsize=8)

    axes[1].barh(types, mi_scores, color=colors, edgecolor="white")
    axes[1].set_title("Mutual Information by Emoji Type", fontweight="bold")
    axes[1].set_xlabel("MI Score")

    plt.suptitle("Emoji Type → Label Discriminative Power", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t3_emoji_type_discriminative_power.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't3_emoji_type_discriminative_power.png'}")


def plot_top_mi_emojis(top_mi_df: pl.DataFrame):
    logger.info("Generating top MI emojis bar chart")

    top20 = top_mi_df.head(20)
    emojis = top20["emoji"].to_list()
    mi = top20["mi_score"].to_list()
    labels = top20["most_common_label"].to_list()
    colors = [LABEL_COLORS.get(l, "#8172B3") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(emojis)), mi, color=colors, edgecolor="white")
    ax.set_yticks(range(len(emojis)))
    ax.set_yticklabels(emojis, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Top 20 Most Label-Discriminating Emojis", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=l.capitalize()) for l, c in LABEL_COLORS.items()]
    ax.legend(handles=legend_elems, title="Most common label", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t3_top_discriminating_emojis.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't3_top_discriminating_emojis.png'}")


def plot_emoji_density_by_label(df: pl.DataFrame):
    logger.info("Generating emoji density violin by label")

    fig, ax = plt.subplots(figsize=(9, 5))
    data_by_label = [
        df.filter(pl.col("label") == lbl)["emoji_density"].clip(upper_bound=0.5).drop_nulls().to_numpy()
        for lbl in LABEL_ORDER
    ]
    parts = ax.violinplot(data_by_label, positions=range(len(LABEL_ORDER)), showmedians=True)
    for pc, lbl in zip(parts["bodies"], LABEL_ORDER):
        pc.set_facecolor(LABEL_COLORS[lbl])
        pc.set_alpha(0.75)
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels([l.capitalize() for l in LABEL_ORDER])
    ax.set_title("Emoji Density Distribution by Label", fontweight="bold")
    ax.set_ylabel("emoji_density (emoji_count / char_count)")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t3_emoji_density_by_label.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't3_emoji_density_by_label.png'}")


def save_outputs(
    survey_df: pl.DataFrame,
    inventory_df: pl.DataFrame,
    corr_df: pl.DataFrame,
    top_mi_df: pl.DataFrame,
    df_with_types: pl.DataFrame,
):
    logger.info("=" * 70)
    logger.info("PHASE 2A — Saving outputs")
    logger.info("=" * 70)

    survey_df.write_parquet(OUTPUT_PARQUET / "t3_emoji_survey_top50.parquet")
    logger.info(f"  Emoji survey         → {OUTPUT_PARQUET / 't3_emoji_survey_top50.parquet'}")

    inventory_df.write_parquet(OUTPUT_PARQUET / "t3_emoji_type_inventory.parquet")
    logger.info(f"  Emoji type inventory → {OUTPUT_PARQUET / 't3_emoji_type_inventory.parquet'}")

    corr_df.write_parquet(OUTPUT_PARQUET / "t3_emoji_label_correlation.parquet")
    logger.info(f"  Emoji-label corr     → {OUTPUT_PARQUET / 't3_emoji_label_correlation.parquet'}")

    top_mi_df.write_parquet(OUTPUT_PARQUET / "t3_top_discriminating_emojis.parquet")
    logger.info(f"  Top MI emojis        → {OUTPUT_PARQUET / 't3_top_discriminating_emojis.parquet'}")

    save_cols = [c for c in df_with_types.columns]
    df_with_types.write_parquet(OUTPUT_PARQUET / "corpus_with_emoji_types.parquet")
    logger.info(f"  Corpus+emoji types   → {OUTPUT_PARQUET / 'corpus_with_emoji_types.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 2A EMOJI ANALYSIS — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()
    df = load_corpus()

    texts = df["comment_text"].to_list()
    emoji_lists = extract_emoji_lists(texts)

    survey_df = data_driven_emoji_survey(df, emoji_lists)
    df = add_emoji_type_columns(df, emoji_lists)

    inventory_df = compute_emoji_type_inventory(df)
    corr_df, top_mi_df = compute_emoji_label_correlation(df)

    plot_emoji_survey_heatmap(survey_df)
    plot_cramers_v_bar(corr_df)
    plot_top_mi_emojis(top_mi_df)
    plot_emoji_density_by_label(df)

    save_outputs(survey_df, inventory_df, corr_df, top_mi_df, df)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 2A COMPLETE — elapsed: {elapsed:.1f}s")
    logger.info("=" * 70)
