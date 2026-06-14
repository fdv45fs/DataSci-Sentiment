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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
CLEANED_CORPUS = Path("output_data/parquet/cleaned_corpus.parquet")
OUTPUT_PARQUET = Path("output_data/parquet")
OUTPUT_IMG = Path("output_data/img")

# Aesthetics
LABEL_ORDER = ["positive", "neutral", "negative"]
LABEL_COLORS = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}
HEATMAP_CMAP = "RdBu_r"

MIN_DF = 5
MAX_DF_RATIO = 0.50
TOP_K_FREQUENT = 20
TOP_K_CORRELATED = 15

# Patterns for text preprocessing (matching phase2d_ngrams.py)
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
    logger.info("1-2-3 GRAM CORRELATION ANALYSIS — Loading cleaned corpus")
    logger.info("=" * 70)

    if not CLEANED_CORPUS.exists():
        raise FileNotFoundError(f"Missing corpus file: {CLEANED_CORPUS}")

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
    logger.info("1-2-3 GRAM CORRELATION ANALYSIS — Preprocessing texts")
    logger.info("=" * 70)

    t0 = time.time()
    texts_raw = df["comment_text"].to_list()

    logger.info(f"  Preprocessing {len(texts_raw):,} texts...")
    texts_clean = [
        preprocess_text_for_ngrams(t)
        for t in tqdm(texts_raw, desc="Preprocessing", unit="text", ncols=80)
    ]

    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())

    logger.info(f"  Preprocessing completed in {time.time() - t0:.2f}s")
    logger.info(f"  Classes found: {le.classes_.tolist()}")
    return texts_clean, y, le


def compute_correlations_fast(X, y_bin: np.ndarray) -> np.ndarray:
    """Computes Pearson correlation coefficient between sparse matrix X columns and binary y_bin."""
    N = X.shape[0]
    y_centered = y_bin - y_bin.mean()
    
    # Numerator: X^T * y_centered
    numerator = np.array(X.T.dot(y_centered)).flatten()
    
    # Standard deviation of y
    std_y = y_bin.std()
    if std_y < 1e-12:
        return np.zeros(X.shape[1], dtype=np.float32)
        
    # Mean of X columns
    mean_X = np.array(X.mean(axis=0)).flatten()
    
    # Variance of X columns: Var(X) = E[X^2] - (E[X])^2
    X_sq = X.copy()
    X_sq.data = X_sq.data ** 2
    sum_X_sq = np.array(X_sq.sum(axis=0)).flatten()
    
    var_X = (sum_X_sq / N) - (mean_X ** 2)
    std_X = np.sqrt(np.maximum(var_X, 1e-12))
    
    # Correlation: numerator / (N * std_X * std_y)
    r = numerator / (N * std_X * std_y)
    return r


def analyze_ngrams(texts: list[str], y: np.ndarray, le: LabelEncoder) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("1-2-3 GRAM CORRELATION ANALYSIS — Vectorizing and Computing Stats")
    logger.info("=" * 70)

    # Convert y to binary indicators for correlation
    y_pos = (y == le.transform(["positive"])[0]).astype(np.float32)
    y_neu = (y == le.transform(["neutral"])[0]).astype(np.float32)
    y_neg = (y == le.transform(["negative"])[0]).astype(np.float32)

    ngram_orders = [(1, 1), (2, 2), (3, 3)]
    ngram_names = ["1-gram", "2-gram", "3-gram"]
    
    all_results = []
    total_docs = len(texts)

    for (n_min, n_max), name in zip(ngram_orders, ngram_names):
        logger.info(f"Processing {name}...")
        t0 = time.time()
        
        vectorizer = CountVectorizer(
            ngram_range=(n_min, n_max),
            min_df=MIN_DF,
            max_df=MAX_DF_RATIO,
            max_features=300000,
            dtype=np.int32
        )
        
        X = vectorizer.fit_transform(texts)
        vocab = vectorizer.get_feature_names_out()
        
        logger.info(f"  Vocabulary size: {X.shape[1]:,}")
        
        # 1. Compute frequencies
        logger.info("  Computing term and document frequencies...")
        term_counts = np.array(X.sum(axis=0)).flatten()
        
        # Binary presence matrix for doc frequency & presence correlation
        X_bin = X.copy()
        X_bin.data = np.ones_like(X_bin.data)
        doc_counts = np.array(X_bin.sum(axis=0)).flatten()
        
        # 2. Compute correlations (using binary presence)
        logger.info("  Computing correlations with sentiment labels...")
        corr_pos = compute_correlations_fast(X_bin, y_pos)
        corr_neu = compute_correlations_fast(X_bin, y_neu)
        corr_neg = compute_correlations_fast(X_bin, y_neg)
        
        # Compile dataframe rows
        for i, gram in enumerate(vocab):
            all_results.append({
                "ngram_order": name,
                "gram": gram,
                "total_freq": int(term_counts[i]),
                "doc_freq": int(doc_counts[i]),
                "doc_freq_ratio": float(doc_counts[i] / total_docs),
                "corr_positive": float(corr_pos[i]),
                "corr_neutral": float(corr_neu[i]),
                "corr_negative": float(corr_neg[i]),
            })
            
        logger.info(f"  Completed {name} in {time.time() - t0:.2f}s")

    df_results = pl.DataFrame(all_results)
    return df_results


def plot_top_frequent_ngrams(df_analysis: pl.DataFrame):
    logger.info("Plotting top frequent n-grams...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    sns.set_theme(style="whitegrid")
    
    ngram_names = ["1-gram", "2-gram", "3-gram"]
    colors = ["#34495e", "#2c3e50", "#1a252f"]
    
    for idx, (name, color) in enumerate(zip(ngram_names, colors)):
        df_sub = (
            df_analysis.filter(pl.col("ngram_order") == name)
            .sort("doc_freq", descending=True)
            .head(TOP_K_FREQUENT)
        )
        
        grams = df_sub["gram"].to_list()
        freqs = df_sub["doc_freq"].to_list()
        
        bars = axes[idx].barh(grams, freqs, color=color, edgecolor="none", height=0.6)
        axes[idx].invert_yaxis()  # top-down
        axes[idx].set_title(f"Top {TOP_K_FREQUENT} Most Used {name}s\n(by Document Frequency)", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Document Frequency (Count)")
        
        # Add labels to the ends of the bars
        for bar in bars:
            width = bar.get_width()
            axes[idx].text(
                width + (max(freqs) * 0.01),
                bar.get_y() + bar.get_height()/2,
                f"{width:,}",
                ha="left",
                va="center",
                fontsize=8,
                color="#555"
            )
            
    plt.suptitle("N-gram Overall Usage Analysis (Unigrams, Bigrams, Trigrams)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plot_path = OUTPUT_IMG / "t6_123gram_top_frequent.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plot_path}")


def plot_correlation_heatmaps(df_analysis: pl.DataFrame):
    logger.info("Plotting correlation heatmaps...")
    
    ngram_names = ["1-gram", "2-gram", "3-gram"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    for idx, name in enumerate(ngram_names):
        df_sub = df_analysis.filter(pl.col("ngram_order") == name)
        
        # Find top correlated ngrams for this order
        # We look at the top positive correlation for positive, neutral, and negative labels
        # Also top negative correlation for each label
        top_pos_set = set()
        for label_col in ["corr_positive", "corr_neutral", "corr_negative"]:
            # Top positive
            top_pos = df_sub.sort(label_col, descending=True).head(5)["gram"].to_list()
            # Top negative (which means most correlated in the opposite direction)
            top_neg = df_sub.sort(label_col, descending=False).head(5)["gram"].to_list()
            top_pos_set.update(top_pos)
            top_pos_set.update(top_neg)
            
        list_grams = list(top_pos_set)
        
        # Filter dataframe for these selected grams
        df_heatmap_data = df_sub.filter(pl.col("gram").is_in(list_grams))
        
        # Calculate maximum absolute correlation for sorting rows
        df_heatmap_data = df_heatmap_data.with_columns(
            abs_max_corr=pl.max_horizontal(
                pl.col("corr_positive").abs(),
                pl.col("corr_neutral").abs(),
                pl.col("corr_negative").abs()
            )
        ).sort("abs_max_corr", descending=True)
        
        # Convert to pandas for seaborn
        pdf = df_heatmap_data.select([
            "gram", "corr_positive", "corr_neutral", "corr_negative"
        ]).to_pandas().set_index("gram")
        
        pdf.columns = ["Positive", "Neutral", "Negative"]
        
        sns.heatmap(
            pdf,
            annot=True,
            fmt=".3f",
            cmap=HEATMAP_CMAP,
            center=0.0,
            linewidths=0.5,
            ax=axes[idx],
            cbar_kws={"shrink": 0.8}
        )
        
        axes[idx].set_title(f"{name} Label Correlations\n(Top Associated Phrases)", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Sentiment Label")
        axes[idx].set_ylabel(name.capitalize())
        
    plt.suptitle("Pearson Correlation (Phi Coefficient) between N-grams and Sentiment Labels", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plot_path = OUTPUT_IMG / "t6_123gram_correlation_heatmap.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plot_path}")


def save_data(df_analysis: pl.DataFrame):
    logger.info("=" * 70)
    logger.info("1-2-3 GRAM CORRELATION ANALYSIS — Saving Data")
    logger.info("=" * 70)
    
    # Save the full analysis table
    full_path = OUTPUT_PARQUET / "t6_123gram_analysis.parquet"
    df_analysis.write_parquet(full_path)
    logger.info(f"  Saved full stats: {full_path}")
    
    # Save a top-correlated summary table for quick review
    # Keep top 100 most correlated grams (by maximum absolute correlation) per order
    df_top = (
        df_analysis.with_columns(
            abs_max_corr=pl.max_horizontal(
                pl.col("corr_positive").abs(),
                pl.col("corr_neutral").abs(),
                pl.col("corr_negative").abs()
            )
        )
        .sort(["ngram_order", "abs_max_corr"], descending=[False, True])
        .group_by("ngram_order")
        .head(100)
    )
    
    summary_path = OUTPUT_PARQUET / "t6_123gram_top_correlated.parquet"
    df_top.write_parquet(summary_path)
    logger.info(f"  Saved top-correlated summary: {summary_path}")


if __name__ == "__main__":
    start_time = time.time()
    setup_output_dirs()
    
    try:
        df_corpus = load_corpus()
        texts, y, le = preprocess_corpus(df_corpus)
        
        df_analysis = analyze_ngrams(texts, y, le)
        
        plot_top_frequent_ngrams(df_analysis)
        plot_correlation_heatmaps(df_analysis)
        
        save_data(df_analysis)
        
        duration = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"1-2-3 GRAM CORRELATION ANALYSIS COMPLETE — elapsed: {duration:.2f}s ({duration/60:.2f} min)")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
