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
from collections import Counter
from scipy.stats import chi2_contingency
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

TOP_N_LANGUAGES = 5
MIN_COMMENTS_FOR_LANG = 200
FASTTEXT_MODEL_SEARCH_PATHS = [
    "data/lid.176.bin",
    os.path.expanduser("~/.fasttext/lid.176.bin"),
    "/tmp/lid.176.bin",
]
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

SCRIPT_PATTERNS = {
    "Latin": re.compile(r"[\u0041-\u007A\u00C0-\u024F]"),
    "CJK": re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF\u3400-\u4DBF]"),
    "Cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "Arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F]"),
    "Devanagari": re.compile(r"[\u0900-\u097F]"),
    "Thai": re.compile(r"[\u0E00-\u0E7F]"),
}


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)


def load_corpus() -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2C — Loading cleaned corpus")
    logger.info("=" * 70)

    if not CLEANED_CORPUS.exists():
        raise FileNotFoundError(f"Run phase0_cleaning.py first. Missing: {CLEANED_CORPUS}")

    t0 = time.time()
    df = pl.read_parquet(CLEANED_CORPUS)
    logger.info(f"  Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    return df


def locate_fasttext_lid_model() -> str:
    for path in FASTTEXT_MODEL_SEARCH_PATHS:
        if os.path.exists(path):
            logger.info(f"  Found FastText LID model at: {path}")
            return path

    logger.warning("  FastText LID model not found in default paths.")
    logger.warning(f"  Downloading from: {FASTTEXT_MODEL_URL}")
    logger.warning("  This will download ~126MB to /tmp/lid.176.bin")

    import urllib.request
    dest = "/tmp/lid.176.bin"
    urllib.request.urlretrieve(FASTTEXT_MODEL_URL, dest)
    logger.info(f"  Downloaded to: {dest}")
    return dest


def detect_languages_fasttext_batch(
    texts: list[str], batch_size: int = 10000
) -> tuple[list[str], list[float]]:
    import fasttext
    fasttext.FastText.eprint = lambda x: None

    model_path = locate_fasttext_lid_model()
    logger.info(f"  Loading FastText LID model from: {model_path}")
    model = fasttext.load_model(model_path)

    n = len(texts)
    languages = ["unknown"] * n
    confidences = [0.0] * n

    logger.info(f"  Running FastText LID on {n:,} texts in batches of {batch_size:,}...")

    for start in tqdm(
        range(0, n, batch_size), desc="FastText LID", unit="batch", ncols=80
    ):
        batch = texts[start : start + batch_size]
        cleaned = [
            re.sub(r"\s+", " ", t.replace("\n", " ").strip()) if isinstance(t, str) else ""
            for t in batch
        ]
        cleaned = [t if t else "unknown" for t in cleaned]

        try:
            preds, probs = model.predict(cleaned, k=1)
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                lang = pred[0].replace("__label__", "") if pred else "unknown"
                conf = float(prob[0]) if prob else 0.0
                languages[start + i] = lang
                confidences[start + i] = conf
        except Exception as e:
            logger.warning(f"  Batch {start}-{start+batch_size} failed: {e}")

    return languages, confidences


def detect_languages_ensemble(
    texts: list[str], fasttext_langs: list[str], fasttext_confs: list[float], conf_threshold: float = 0.5
) -> list[str]:
    logger.info(f"  Applying langdetect fallback for low-confidence predictions (threshold={conf_threshold})")

    low_conf_indices = [i for i, c in enumerate(fasttext_confs) if c < conf_threshold]
    logger.info(f"  Low-confidence predictions: {len(low_conf_indices):,} ({len(low_conf_indices)/len(texts)*100:.2f}%)")

    if len(low_conf_indices) == 0:
        return fasttext_langs

    try:
        import langdetect
        langdetect.DetectorFactory.seed = 42

        final_langs = list(fasttext_langs)
        for i in tqdm(low_conf_indices, desc="langdetect fallback", unit="comment", ncols=80):
            try:
                final_langs[i] = langdetect.detect(texts[i])
            except Exception:
                pass

        logger.info("  Ensemble language detection completed")
        return final_langs

    except ImportError:
        logger.warning("  langdetect not available — using FastText predictions only")
        return fasttext_langs


def detect_script_family(text: str) -> str:
    if not isinstance(text, str):
        return "Other"
    for script, pattern in SCRIPT_PATTERNS.items():
        if pattern.search(text):
            return script
    return "Other"


def detect_code_switching_ratio(text: str) -> float:
    if not isinstance(text, str) or len(text.split()) < 5:
        return 0.0
    tokens = text.split()
    window_size = 3
    lang_windows = []
    for i in range(0, len(tokens) - window_size + 1, 2):
        window = " ".join(tokens[i : i + window_size])
        try:
            import langdetect
            lang_windows.append(langdetect.detect(window))
        except Exception:
            lang_windows.append("unknown")
    if len(lang_windows) < 2:
        return 0.0
    switches = sum(
        1 for i in range(len(lang_windows) - 1)
        if lang_windows[i] != lang_windows[i + 1]
        and lang_windows[i] != "unknown"
        and lang_windows[i + 1] != "unknown"
    )
    return switches / (len(lang_windows) - 1)


def add_language_columns(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2C — Running language detection pipeline")
    logger.info("=" * 70)

    texts = df["comment_text"].to_list()
    t0 = time.time()

    fasttext_langs, fasttext_confs = detect_languages_fasttext_batch(texts)
    final_langs = detect_languages_ensemble(texts, fasttext_langs, fasttext_confs)

    logger.info(f"  Language detection total time: {time.time() - t0:.1f}s")

    logger.info("  Detecting script families (vectorized regex)...")
    script_families = [detect_script_family(t) for t in tqdm(texts, desc="Script detection", ncols=80)]

    df = df.with_columns([
        pl.Series("primary_language", final_langs, dtype=pl.Utf8),
        pl.Series("lang_confidence", fasttext_confs, dtype=pl.Float32),
        pl.Series("script_family", script_families, dtype=pl.Utf8),
    ])

    lang_dist = (
        df.group_by("primary_language")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    logger.info(f"  Top 10 languages detected:")
    for row in lang_dist.head(10).iter_rows(named=True):
        logger.info(
            f"    {row['primary_language']:10s}: {row['count']:>8,} "
            f"({row['count'] / len(df) * 100:.2f}%)"
        )

    return df


def group_languages(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2C — Grouping languages (top 5 + other)")
    logger.info("=" * 70)

    top_langs = (
        df.group_by("primary_language")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(TOP_N_LANGUAGES)["primary_language"]
        .to_list()
    )
    logger.info(f"  Top {TOP_N_LANGUAGES} languages: {top_langs}")

    df = df.with_columns(
        pl.when(pl.col("primary_language").is_in(top_langs))
        .then(pl.col("primary_language"))
        .otherwise(pl.lit("other"))
        .alias("language_group")
    )

    return df


def compute_language_distribution(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2C — Table 5a: Language Distribution")
    logger.info("=" * 70)

    total = len(df)
    lang_table = (
        df.group_by("primary_language")
        .agg([
            pl.len().alias("comment_count"),
            pl.col("author_id").n_unique().alias("unique_authors"),
            pl.col("post_id").n_unique().alias("unique_videos"),
            pl.col("lang_confidence").mean().alias("avg_confidence"),
            pl.col("script_family").mode().first().alias("dominant_script"),
        ])
        .with_columns(
            (pl.col("comment_count") / total * 100).alias("pct_of_corpus")
        )
        .sort("comment_count", descending=True)
    )

    logger.info(f"  Total unique languages: {len(lang_table)}")
    for row in lang_table.head(10).iter_rows(named=True):
        logger.info(
            f"    {row['primary_language']:10s}: {row['comment_count']:>8,} "
            f"({row['pct_of_corpus']:.2f}%)  conf={row['avg_confidence']:.3f}  "
            f"script={row['dominant_script']}"
        )

    return lang_table


def compute_multilingual_stratification(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 2C — Table 5b: Cross-Lingual Label & Engagement Stratification")
    logger.info("=" * 70)

    top_langs = (
        df.group_by("primary_language")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(TOP_N_LANGUAGES)["primary_language"]
        .to_list()
    )

    rows = []
    for lang in top_langs:
        subset = df.filter(pl.col("primary_language") == lang)
        n = len(subset)
        if n < MIN_COMMENTS_FOR_LANG:
            logger.warning(f"  {lang}: only {n} comments — below threshold, skipping")
            continue

        label_counts = {
            lbl: subset.filter(pl.col("label") == lbl).height
            for lbl in LABEL_ORDER
        }

        row = {
            "language": lang,
            "n": n,
            "pct_positive": round(label_counts["positive"] / n * 100, 2),
            "pct_neutral": round(label_counts["neutral"] / n * 100, 2),
            "pct_negative": round(label_counts["negative"] / n * 100, 2),
            "mean_chars": round(float(subset["char_count"].mean()), 2),
            "mean_words": round(float(subset["word_count"].mean()), 2),
            "mean_likes": round(float(subset["like_count"].mean()), 3),
            "mean_replies": round(float(subset["reply_count"].mean()), 3),
            "avg_confidence": round(float(subset["lang_confidence"].mean()), 4),
        }

        if "emoji_density" in subset.columns:
            row["emoji_density"] = round(float(subset["emoji_density"].mean()), 5)
        else:
            row["emoji_density"] = round(
                float((subset["emoji_count"] / subset["char_count"].clip(lower_bound=1)).mean()), 5
            )

        rows.append(row)

        logger.info(
            f"  {lang:10s}: n={n:>7,}  "
            f"pos={row['pct_positive']:.1f}%  neu={row['pct_neutral']:.1f}%  neg={row['pct_negative']:.1f}%  "
            f"chars={row['mean_chars']:.0f}  likes={row['mean_likes']:.2f}"
        )

    strat_df = pl.DataFrame(rows)

    contingency_matrix = np.array([
        [row[f"pct_{lbl}"] * row["n"] / 100 for lbl in LABEL_ORDER]
        for row in rows
    ]).astype(int)

    if contingency_matrix.shape[0] > 1:
        chi2_stat, p_val, _, _ = chi2_contingency(contingency_matrix)
        logger.info(
            f"  Chi-square homogeneity test across languages: "
            f"chi2={chi2_stat:.2f}  p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}"
        )

        neg_rates = strat_df["pct_negative"].to_numpy()
        mean_neg = neg_rates.mean()
        std_neg = neg_rates.std()
        high_neg_langs = [
            row["language"] for row in rows
            if row["pct_negative"] > mean_neg + std_neg
        ]
        logger.info(f"  Languages with statistically higher Negative rates: {high_neg_langs}")

    return strat_df


def plot_language_distribution(lang_table: pl.DataFrame):
    logger.info("Generating language distribution chart")

    top20 = lang_table.head(20)
    langs = top20["primary_language"].to_list()
    counts = top20["comment_count"].to_list()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(langs)))
    bars = ax.barh(langs, counts, color=colors, edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Comment Count")
    ax.set_title("Language Distribution — Top 20 Languages (FastText LID)", fontweight="bold")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t5_language_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't5_language_distribution.png'}")


def plot_multilingual_label_heatmap(strat_df: pl.DataFrame):
    logger.info("Generating multilingual label distribution heatmap")

    heatmap_data = strat_df.select(["language", "pct_positive", "pct_neutral", "pct_negative"]).to_pandas()
    heatmap_data = heatmap_data.set_index("language")
    heatmap_data.columns = ["Positive %", "Neutral %", "Negative %"]

    fig, ax = plt.subplots(figsize=(9, max(5, len(strat_df) * 0.7)))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
        linewidths=0.5, ax=ax, vmin=0, vmax=100,
    )
    ax.set_title("Label Distribution by Language (%)", fontweight="bold")
    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Language")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t5_multilingual_label_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't5_multilingual_label_heatmap.png'}")


def plot_multilingual_engagement(strat_df: pl.DataFrame):
    logger.info("Generating multilingual engagement comparison")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    langs = strat_df["language"].to_list()
    colors = plt.cm.Set2(np.linspace(0, 1, len(langs)))

    for ax, (metric, label) in zip(axes, [
        ("mean_chars", "Mean Char Count"),
        ("emoji_density", "Mean Emoji Density"),
        ("mean_likes", "Mean Like Count"),
    ]):
        vals = strat_df[metric].to_list()
        ax.bar(langs, vals, color=colors, edgecolor="white")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Language")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Multilingual Engagement & Text Characteristics", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t5_multilingual_engagement.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't5_multilingual_engagement.png'}")


def save_outputs(
    df: pl.DataFrame,
    lang_table: pl.DataFrame,
    strat_df: pl.DataFrame,
):
    logger.info("=" * 70)
    logger.info("PHASE 2C — Saving outputs")
    logger.info("=" * 70)

    lang_table.write_parquet(OUTPUT_PARQUET / "t5_language_distribution.parquet")
    logger.info(f"  Language dist   → {OUTPUT_PARQUET / 't5_language_distribution.parquet'}")

    strat_df.write_parquet(OUTPUT_PARQUET / "t5_multilingual_stratification.parquet")
    logger.info(f"  Multilingual    → {OUTPUT_PARQUET / 't5_multilingual_stratification.parquet'}")

    df.write_parquet(OUTPUT_PARQUET / "corpus_with_language.parquet")
    logger.info(f"  Corpus+language → {OUTPUT_PARQUET / 'corpus_with_language.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 2C MULTILINGUAL — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()
    df = load_corpus()
    df = add_language_columns(df)
    df = group_languages(df)

    lang_table = compute_language_distribution(df)
    strat_df = compute_multilingual_stratification(df)

    plot_language_distribution(lang_table)
    plot_multilingual_label_heatmap(strat_df)
    plot_multilingual_engagement(strat_df)

    save_outputs(df, lang_table, strat_df)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 2C COMPLETE — elapsed: {elapsed:.1f}s")
    logger.info("=" * 70)
