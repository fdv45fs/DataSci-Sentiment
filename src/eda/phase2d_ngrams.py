import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import re
import os
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, chi2 as sklearn_chi2
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from tqdm import tqdm

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

NGRAM_ORDERS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
NGRAM_NAMES = ["1-gram", "2-gram", "3-gram", "4-gram", "5-gram", "6-gram", "7-gram"]
MIN_DF = 5
MAX_DF_RATIO = 0.50
TOP_K_PER_LABEL = 20
TOP_K_DISCRIMINATIVE = 100
SAMPLE_SIZE_FOR_HIGH_NGRAMS = 50000

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F9FF"
    r"\U00002600-\U000027BF\U0001FA00-\U0001FAFF\U0001F1E0-\U0001F1FF]",
    flags=re.UNICODE,
)


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)


def load_corpus() -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2D — Loading cleaned corpus")
    logger.info("=" * 70)

    if not CLEANED_CORPUS.exists():
        raise FileNotFoundError(f"Run phase0_cleaning.py first. Missing: {CLEANED_CORPUS}")

    t0 = time.time()
    df = pl.read_parquet(CLEANED_CORPUS)
    logger.info(f"  Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    return df


def preprocess_text_for_ngrams(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(lambda m: f" EMOJI_{ord(m.group()):X} ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_corpus(df: pl.DataFrame) -> tuple[list[str], np.ndarray, LabelEncoder]:
    logger.info("=" * 70)
    logger.info("PHASE 2D — Preprocessing texts for n-gram analysis")
    logger.info("=" * 70)

    t0 = time.time()
    texts_raw = df["comment_text"].to_list()

    logger.info(f"  Preprocessing {len(texts_raw):,} texts...")
    texts_clean = [
        preprocess_text_for_ngrams(t)
        for t in tqdm(texts_raw, desc="Text preprocessing", unit="text", ncols=80)
    ]

    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())

    logger.info(f"  Preprocessing completed in {time.time() - t0:.2f}s")
    logger.info(f"  Label classes: {le.classes_.tolist()}")
    return texts_clean, y, le


def check_existing_ngram_data() -> dict[str, Path]:
    existing = {}

    gram_13_path = Path("data/13gram/13gram.parquet")
    gram_37_path = Path("data/37gram/37gram.parquet")

    for path, name in [(gram_13_path, "1-3gram"), (gram_37_path, "3-7gram")]:
        if path.exists():
            try:
                schema = pl.read_parquet(path, n_rows=1)
                logger.info(f"  Found existing n-gram data: {path}  columns={schema.columns[:5]}")
                existing[name] = path
            except Exception as e:
                logger.warning(f"  Cannot read {path}: {e}")

    return existing


def compute_ngram_stats_for_order(
    texts: list[str],
    y: np.ndarray,
    le: LabelEncoder,
    ngram_range: tuple[int, int],
    ngram_name: str,
    use_sample: bool = False,
) -> dict:
    logger.info(f"  Processing {ngram_name} (range={ngram_range})...")

    if use_sample and len(texts) > SAMPLE_SIZE_FOR_HIGH_NGRAMS:
        logger.info(f"    Sampling {SAMPLE_SIZE_FOR_HIGH_NGRAMS:,} texts for {ngram_name}")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), SAMPLE_SIZE_FOR_HIGH_NGRAMS, replace=False)
        texts_use = [texts[i] for i in idx]
        y_use = y[idx]
    else:
        texts_use = texts
        y_use = y

    try:
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=MIN_DF,
            max_df=MAX_DF_RATIO,
            max_features=500000,
            dtype=np.float32,
        )
        X = vectorizer.fit_transform(texts_use)
        vocab = vectorizer.get_feature_names_out()

        logger.info(f"    {ngram_name}: vocabulary size={X.shape[1]:,}  matrix shape={X.shape}")

    except Exception as e:
        logger.error(f"    Failed to vectorize {ngram_name}: {e}")
        return {}

    total_docs = X.shape[0]
    avg_grams_per_comment = float(np.array(X.sum(axis=1)).flatten().mean())

    logger.info(f"    Computing DF and relative frequency per label...")
    label_stats = {}
    for label_idx, label_name in enumerate(le.classes_):
        mask = y_use == label_idx
        X_label = X[mask]
        n_label = mask.sum()

        df_counts = np.array((X_label > 0).sum(axis=0)).flatten()
        rel_freq = np.array(X_label.sum(axis=0)).flatten() / max(n_label, 1) * 1000.0

        top_idx = np.argsort(rel_freq)[-TOP_K_PER_LABEL:][::-1]
        top_grams = [
            {"gram": vocab[i], "rel_freq_per_1k": round(float(rel_freq[i]), 4), "df": int(df_counts[i])}
            for i in top_idx
        ]
        label_stats[label_name] = {"top_grams": top_grams, "n": int(n_label)}

    logger.info(f"    Computing MI and Chi² for {ngram_name}...")

    sample_size = min(50000, X.shape[0])
    if X.shape[0] > sample_size:
        idx_mi = np.random.default_rng(42).choice(X.shape[0], sample_size, replace=False)
        X_mi = X[idx_mi]
        y_mi = y_use[idx_mi]
    else:
        X_mi = X
        y_mi = y_use

    try:
        mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=True, random_state=42, n_jobs=min(12, os.cpu_count()))
        chi2_scores, _ = sklearn_chi2(X_mi, y_mi)
    except Exception as e:
        logger.error(f"    MI/Chi² computation failed for {ngram_name}: {e}")
        mi_scores = np.zeros(len(vocab))
        chi2_scores = np.zeros(len(vocab))

    mean_mi = float(mi_scores.mean())
    top_mi_idx = np.argsort(mi_scores)[-3:][::-1]
    top_mi_grams = [vocab[i] for i in top_mi_idx]
    logger.info(f"    {ngram_name}: mean_MI={mean_mi:.6f}  top_3_by_MI={top_mi_grams}")

    logger.info(f"    Computing coverage (% comments covered by top-{TOP_K_DISCRIMINATIVE} n-grams)...")
    top_100_mi_idx = np.argsort(mi_scores)[-TOP_K_DISCRIMINATIVE:]
    X_top100 = X[:, top_100_mi_idx]

    coverage_by_label = {}
    for label_idx, label_name in enumerate(le.classes_):
        mask = y_use == label_idx
        X_label = X_top100[mask]
        covered = (np.array(X_label.sum(axis=1)).flatten() > 0).sum()
        coverage_by_label[label_name] = round(covered / max(mask.sum(), 1) * 100, 3)

    return {
        "ngram_name": ngram_name,
        "ngram_range": ngram_range,
        "total_unique_grams": int(X.shape[1]),
        "avg_grams_per_comment": round(avg_grams_per_comment, 3),
        "mean_mi": round(mean_mi, 6),
        "top_3_grams_by_mi": " | ".join(top_mi_grams),
        "coverage_pct_positive": coverage_by_label.get("positive", 0),
        "coverage_pct_neutral": coverage_by_label.get("neutral", 0),
        "coverage_pct_negative": coverage_by_label.get("negative", 0),
        "label_top_grams": label_stats,
        "mi_scores_sample": mi_scores[:200].tolist(),
    }


def compute_all_ngram_stats(
    texts: list[str],
    y: np.ndarray,
    le: LabelEncoder,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    logger.info("=" * 70)
    logger.info("PHASE 2D — Computing N-gram statistics for orders 1–7")
    logger.info("=" * 70)

    existing = check_existing_ngram_data()
    if existing:
        logger.info(f"  Found pre-computed n-gram data: {list(existing.keys())}")
        logger.info("  Will use existing data where schema matches, fallback to recompute")

    summary_rows = []
    freq_rows = []

    for i, (ngram_range, ngram_name) in enumerate(zip(NGRAM_ORDERS, NGRAM_NAMES)):
        logger.info(f"\n  [{i+1}/{len(NGRAM_ORDERS)}] Processing {ngram_name}...")
        use_sample = ngram_range[0] >= 4

        result = compute_ngram_stats_for_order(texts, y, le, ngram_range, ngram_name, use_sample)
        if not result:
            continue

        summary_rows.append({
            "ngram_order": ngram_name,
            "n_min": ngram_range[0],
            "n_max": ngram_range[1],
            "total_unique_grams": result["total_unique_grams"],
            "avg_grams_per_comment": result["avg_grams_per_comment"],
            "mean_mi": result["mean_mi"],
            "top_3_grams_by_mi": result["top_3_grams_by_mi"],
            "coverage_pct_positive": result["coverage_pct_positive"],
            "coverage_pct_neutral": result["coverage_pct_neutral"],
            "coverage_pct_negative": result["coverage_pct_negative"],
        })

        for label_name, stats in result["label_top_grams"].items():
            for rank, gram_info in enumerate(stats["top_grams"]):
                freq_rows.append({
                    "ngram_order": ngram_name,
                    "label": label_name,
                    "rank": rank + 1,
                    "gram": gram_info["gram"],
                    "rel_freq_per_1k": gram_info["rel_freq_per_1k"],
                    "df": gram_info["df"],
                })

    discriminative_table = pl.DataFrame(summary_rows)
    freq_table = pl.DataFrame(freq_rows)

    logger.info("\n  === N-gram Discriminative Power Summary ===")
    for row in discriminative_table.sort("mean_mi", descending=True).iter_rows(named=True):
        logger.info(
            f"  {row['ngram_order']:8s}: mean_MI={row['mean_mi']:.6f}  "
            f"unique_grams={row['total_unique_grams']:>8,}  "
            f"top3=[{row['top_3_grams_by_mi']}]"
        )

    best_order = discriminative_table.sort("mean_mi", descending=True)["ngram_order"][0]
    logger.info(f"\n  Recommendation: Best n-gram order by mean MI = {best_order}")

    return discriminative_table, freq_table


def plot_mi_comparison(discriminative_table: pl.DataFrame):
    logger.info("Generating MI comparison bar chart across n-gram orders")

    orders = discriminative_table["ngram_order"].to_list()
    mi_vals = discriminative_table["mean_mi"].to_list()

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(orders)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(orders, mi_vals, color=colors, edgecolor="white")
    for bar, val in zip(bars, mi_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f"{val:.5f}", ha="center", va="bottom", fontsize=8, rotation=20)
    axes[0].set_title("Mean Mutual Information by N-gram Order", fontweight="bold")
    axes[0].set_xlabel("N-gram Order")
    axes[0].set_ylabel("Mean MI Score")

    unique_grams = discriminative_table["total_unique_grams"].to_list()
    bars2 = axes[1].bar(orders, unique_grams, color=colors, edgecolor="white")
    axes[1].set_yscale("log")
    for bar, val in zip(bars2, unique_grams):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{val:,}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Vocabulary Size by N-gram Order", fontweight="bold")
    axes[1].set_xlabel("N-gram Order")
    axes[1].set_ylabel("Unique N-grams (log scale)")

    plt.suptitle("N-gram Feature Comparison (1 to 7+)", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t6_ngram_mi_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't6_ngram_mi_comparison.png'}")


def plot_top_grams_per_label(freq_table: pl.DataFrame):
    logger.info("Generating top unigram relative frequency heatmap per label")

    unigrams = freq_table.filter(pl.col("ngram_order") == "1-gram")
    if len(unigrams) == 0:
        return

    all_grams = unigrams["gram"].unique().to_list()[:30]

    rows_for_pivot = []
    for gram in all_grams:
        for label in LABEL_ORDER:
            match = unigrams.filter((pl.col("gram") == gram) & (pl.col("label") == label))
            rel_freq = float(match["rel_freq_per_1k"][0]) if len(match) > 0 else 0.0
            rows_for_pivot.append({"gram": gram, "label": label, "rel_freq_per_1k": rel_freq})

    pivot_df = pl.DataFrame(rows_for_pivot).pivot(
        index="gram", columns="label", values="rel_freq_per_1k", aggregate_function="first"
    ).fill_null(0).to_pandas().set_index("gram")

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.3, ax=ax)
    ax.set_title("Top Unigrams — Relative Frequency per 1K Comments per Label", fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Unigram")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t6_top_unigrams_per_label.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't6_top_unigrams_per_label.png'}")


def plot_coverage_comparison(discriminative_table: pl.DataFrame):
    logger.info("Generating coverage comparison chart")

    orders = discriminative_table["ngram_order"].to_list()
    x = np.arange(len(orders))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, color) in enumerate(LABEL_COLORS.items()):
        col = f"coverage_pct_{label}"
        if col in discriminative_table.columns:
            ax.bar(x + i * width, discriminative_table[col].to_list(), width, label=label.capitalize(), color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(orders)
    ax.set_xlabel("N-gram Order")
    ax.set_ylabel("Coverage % (comments covered by top-100 grams)")
    ax.set_title("N-gram Coverage by Label — Top 100 Discriminating Grams", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t6_ngram_coverage_by_label.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't6_ngram_coverage_by_label.png'}")


def save_outputs(discriminative_table: pl.DataFrame, freq_table: pl.DataFrame):
    logger.info("=" * 70)
    logger.info("PHASE 2D — Saving outputs")
    logger.info("=" * 70)

    discriminative_table.write_parquet(OUTPUT_PARQUET / "t6_ngram_discriminative_power.parquet")
    logger.info(f"  Discriminative power → {OUTPUT_PARQUET / 't6_ngram_discriminative_power.parquet'}")

    freq_table.write_parquet(OUTPUT_PARQUET / "t6_ngram_freq_profile.parquet")
    logger.info(f"  Frequency profile    → {OUTPUT_PARQUET / 't6_ngram_freq_profile.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 2D N-GRAM ANALYSIS — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()
    df = load_corpus()
    texts, y, le = preprocess_corpus(df)

    discriminative_table, freq_table = compute_all_ngram_stats(texts, y, le)

    plot_mi_comparison(discriminative_table)
    plot_top_grams_per_label(freq_table)
    plot_coverage_comparison(discriminative_table)

    save_outputs(discriminative_table, freq_table)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 2D COMPLETE — elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)
