import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import logging
import time
import unicodedata
from pathlib import Path
from datetime import UTC, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path("data/combined_embeddings.parquet")
LABELED_PATH = Path("data/combined_labeled.parquet")
OUTPUT_PARQUET = Path("output_data/parquet")
OUTPUT_IMG = Path("output_data/img")
CLEANED_CORPUS_PATH = OUTPUT_PARQUET / "cleaned_corpus.parquet"

NUMERIC_COLS = [
    "char_count", "word_count", "avg_word_length", "uppercase_ratio",
    "exclamation_count", "question_count", "hashtag_count",
    "mention_count", "emoji_count", "like_count_log",
]
EMBEDDING_COLS = ["embedding", "embedding_char", "embedding_word", "embedding_ft"]

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F9FF"
    "\U00002600-\U000027BF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    logger.info("=" * 70)
    logger.info("PHASE 0 — Loading raw parquet files")
    logger.info("=" * 70)

    t0 = time.time()
    logger.info(f"Reading embeddings from: {EMBEDDINGS_PATH}")
    df_emb = pl.read_parquet(EMBEDDINGS_PATH)
    logger.info(f"  embeddings shape  : {df_emb.shape}")
    logger.info(f"  embeddings columns: {df_emb.columns}")

    logger.info(f"Reading labels from: {LABELED_PATH}")
    df_lbl = pl.read_parquet(LABELED_PATH)
    logger.info(f"  labeled shape     : {df_lbl.shape}")
    logger.info(f"  labeled columns   : {df_lbl.columns}")
    logger.info(f"  unique labels     : {df_lbl['label'].unique().to_list()}")
    logger.info(f"Data load completed in {time.time() - t0:.2f}s")

    return df_emb, df_lbl


def inner_merge_and_validate(df_emb: pl.DataFrame, df_lbl: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 0 — Inner merge on comment_id + post_id")
    logger.info("=" * 70)

    t0 = time.time()
    raw_emb_rows = len(df_emb)
    raw_lbl_rows = len(df_lbl)

    df_lbl_deduped = df_lbl.unique(subset=["comment_id"], keep="first")
    logger.info(f"  Labels after dedup on comment_id: {len(df_lbl_deduped):,} (was {raw_lbl_rows:,})")

    label_counts_per_id = (
        df_lbl_deduped.group_by("comment_id")
        .agg(pl.len().alias("n_labels"))
        .filter(pl.col("n_labels") > 1)
    )
    if len(label_counts_per_id) > 0:
        logger.warning(f"  {len(label_counts_per_id):,} comment_ids have >1 label — these will be dropped")
        valid_ids = (
            df_lbl_deduped.group_by("comment_id")
            .agg(pl.len().alias("n_labels"))
            .filter(pl.col("n_labels") == 1)
            .select("comment_id")
        )
        df_lbl_deduped = df_lbl_deduped.join(valid_ids, on="comment_id", how="inner")
    else:
        logger.info("  1:1 label mapping validated — all comment_ids have exactly one label")

    df_merged = df_emb.join(
        df_lbl_deduped.select(["comment_id", "label", "source_file", "source_row", "post_id"]),
        on=["comment_id"],
        how="inner",
        suffix="_lbl",
    )
    logger.info(f"  After inner join: {len(df_merged):,} rows (from {raw_emb_rows:,} embeddings, {len(df_lbl_deduped):,} labels)")

    post_id_mismatch = df_merged.filter(
        pl.col("post_id") != pl.col("post_id_lbl")
    )
    logger.info(f"  post_id mismatches detected: {len(post_id_mismatch):,}")
    if len(post_id_mismatch) > 0:
        df_merged = df_merged.filter(
            pl.col("post_id") == pl.col("post_id_lbl")
        )
        logger.info(f"  Dropped mismatches. Remaining: {len(df_merged):,}")

    df_merged = df_merged.drop("post_id_lbl") if "post_id_lbl" in df_merged.columns else df_merged

    logger.info(f"Merge completed in {time.time() - t0:.2f}s")
    return df_merged


def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 0 — Deduplication")
    logger.info("=" * 70)

    before = len(df)

    df = df.unique(subset=["comment_id"], keep="first")
    after_id = len(df)
    logger.info(f"  Dropped {before - after_id:,} exact duplicate comment_ids — {after_id:,} remain")

    df = df.unique(subset=["comment_id", "comment_text"], keep="first")
    after_text = len(df)
    logger.info(f"  Dropped {after_id - after_text:,} duplicate comment_texts tied to the same comment_id — {after_text:,} remain")

    return df


def drop_invalid_rows(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 0 — Dropping invalid rows")
    logger.info("=" * 70)

    before = len(df)

    df = df.filter(pl.col("comment_id").is_not_null())
    logger.info(f"  After drop NULL comment_id  : {len(df):,} (-{before - len(df):,})")

    n = len(df)
    df = df.filter(pl.col("post_id").is_not_null())
    logger.info(f"  After drop NULL post_id      : {len(df):,} (-{n - len(df):,})")

    n = len(df)
    df = df.filter(
        pl.col("comment_text").is_not_null() &
        (pl.col("comment_text").str.strip_chars() != "")
    )
    logger.info(f"  After drop NULL/empty text   : {len(df):,} (-{n - len(df):,})")

    n = len(df)
    df = df.filter(pl.col("published_at").is_not_null())
    logger.info(f"  After drop NULL published_at : {len(df):,} (-{n - len(df):,})")

    n = len(df)
    now_utc = datetime.now(UTC)
    if df["published_at"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("published_at").str.to_datetime(strict=False, time_zone="UTC").alias("published_at")
        )
    elif df["published_at"].dtype == pl.Datetime and df["published_at"].dtype.time_zone is None:
        df = df.with_columns(
            pl.col("published_at").dt.replace_time_zone("UTC").alias("published_at")
        )
    df = df.filter(pl.col("published_at") <= pl.lit(now_utc))
    logger.info(f"  After drop future published_at: {len(df):,} (-{n - len(df):,})")

    logger.info(f"  Total dropped in this step: {before - len(df):,}")
    return df


def impute_engagement(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 0 — Imputing & clamping engagement columns")
    logger.info("=" * 70)

    for col in ["like_count", "reply_count"]:
        null_before = df[col].is_null().sum()
        neg_before = (df[col] < 0).sum() if df[col].dtype in [pl.Int64, pl.Float64] else 0
        df = df.with_columns(
            pl.col(col).fill_null(0).clip(lower_bound=0).alias(col)
        )
        logger.info(f"  {col}: imputed {null_before:,} NULLs → 0, clamped {neg_before:,} negatives → 0")

    null_author = df["author_id"].is_null().sum()
    df = df.with_columns(
        pl.col("author_id").fill_null("unknown").alias("author_id")
    )
    logger.info(f"  author_id: flagged {null_author:,} NULLs as 'unknown'")

    return df


def recompute_text_features(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 0 — Recomputing text-derived features from comment_text")
    logger.info("=" * 70)

    t0 = time.time()

    df = df.with_columns([
        pl.col("comment_text").str.len_chars().alias("char_count"),
        pl.col("comment_text").str.split(" ").list.len().alias("word_count"),
    ])

    df = df.with_columns([
        (
            pl.col("comment_text")
            .str.split(" ")
            .list.eval(pl.element().str.len_chars())
            .list.mean()
        ).alias("avg_word_length"),
        (
            pl.col("comment_text").str.count_matches(r"[A-Z]") /
            (pl.col("char_count").clip(lower_bound=1))
        ).alias("uppercase_ratio"),
        pl.col("comment_text").str.count_matches(r"!").alias("exclamation_count"),
        pl.col("comment_text").str.count_matches(r"\?").alias("question_count"),
        pl.col("comment_text").str.count_matches(r"#\w+").alias("hashtag_count"),
        pl.col("comment_text").str.count_matches(r"@\w+").alias("mention_count"),
    ])

    emoji_re = (
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF"
        "\U00002600-\U000027BF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U0000FE00-\U0000FE0F"
        "\U0001F1E0-\U0001F1FF]"
    )
    df = df.with_columns([
        pl.col("comment_text").str.count_matches(emoji_re).alias("emoji_count"),
    ])

    df = df.with_columns([
        (pl.col("like_count") + 1).log(base=float(np.e)).alias("like_count_log"),
    ])

    logger.info(f"  char_count   — mean: {df['char_count'].mean():.1f}, nulls: {df['char_count'].is_null().sum()}")
    logger.info(f"  word_count   — mean: {df['word_count'].mean():.1f}, nulls: {df['word_count'].is_null().sum()}")
    logger.info(f"  emoji_count  — mean: {df['emoji_count'].mean():.3f}")
    logger.info(f"  like_count_log — mean: {df['like_count_log'].mean():.3f}")
    logger.info(f"  Recomputation completed in {time.time() - t0:.2f}s")

    return df


def audit_embeddings(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 0 — Embedding null audit")
    logger.info("=" * 70)

    for col in EMBEDDING_COLS:
        if col not in df.columns:
            logger.warning(f"  Embedding column '{col}' not found in dataframe")
            continue
        null_count = df[col].is_null().sum()
        logger.info(f"  {col}: {null_count:,} NULL rows ({null_count / len(df) * 100:.3f}%)")
        if null_count > 0:
            df = df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias(f"{col}_null_flag")
            )

    return df


def build_null_pattern_matrix(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 0 — Building Null Pattern Matrix")
    logger.info("=" * 70)

    rows = []
    for col in df.columns:
        if col.endswith("_null_flag"):
            continue
        nc = df[col].is_null().sum()
        rows.append({
            "column": col,
            "null_count": nc,
            "null_pct": round(nc / len(df) * 100, 4),
            "post_cleaning_null_count": nc,
        })

    null_matrix = pl.DataFrame(rows).sort("null_pct", descending=True)
    logger.info(f"  Columns with any nulls: {null_matrix.filter(pl.col('null_count') > 0).shape[0]}")
    for row in null_matrix.filter(pl.col("null_count") > 0).iter_rows(named=True):
        logger.info(f"    {row['column']:35s}: {row['null_count']:>8,} nulls ({row['null_pct']:.3f}%)")

    return null_matrix


def build_corpus_cardinality(df_raw: pl.DataFrame, df_clean: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 0 — Corpus Cardinality Summary (Table 1)")
    logger.info("=" * 70)

    time_min = df_clean["published_at"].min()
    time_max = df_clean["published_at"].max()

    cardinality = pl.DataFrame([{
        "rows_retained": len(df_clean),
        "rows_raw": len(df_raw),
        "pct_of_raw": round(len(df_clean) / len(df_raw) * 100, 3),
        "unique_comments": df_clean["comment_id"].n_unique(),
        "unique_videos": df_clean["post_id"].n_unique(),
        "unique_authors": df_clean["author_id"].n_unique(),
        "time_span_start": str(time_min),
        "time_span_end": str(time_max),
        "unique_labels": df_clean["label"].n_unique() if "label" in df_clean.columns else 0,
    }])

    logger.info(f"  rows_retained   : {cardinality['rows_retained'][0]:,}")
    logger.info(f"  pct_of_raw      : {cardinality['pct_of_raw'][0]:.2f}%")
    logger.info(f"  unique_comments : {cardinality['unique_comments'][0]:,}")
    logger.info(f"  unique_videos   : {cardinality['unique_videos'][0]:,}")
    logger.info(f"  unique_authors  : {cardinality['unique_authors'][0]:,}")
    logger.info(f"  time_span       : {cardinality['time_span_start'][0]} → {cardinality['time_span_end'][0]}")

    return cardinality


def plot_null_matrix(null_matrix: pl.DataFrame):
    logger.info("Generating null pattern matrix chart")

    non_zero = null_matrix.filter(pl.col("null_count") > 0)
    if len(non_zero) == 0:
        logger.info("  No nulls remain — skipping null matrix chart")
        return

    fig, ax = plt.subplots(figsize=(12, max(5, len(non_zero) * 0.4)))
    cols = non_zero["column"].to_list()
    pcts = non_zero["null_pct"].to_list()

    bars = ax.barh(cols, pcts, color="#C44E52", edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{pct:.2f}%", va="center", fontsize=8)

    ax.set_xlabel("Null Percentage (%)")
    ax.set_title("Null Pattern Matrix — Post-Cleaning", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t0_null_pattern_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't0_null_pattern_matrix.png'}")


def plot_cardinality_summary(cardinality: pl.DataFrame):
    logger.info("Generating corpus cardinality summary chart")

    row = cardinality.row(0, named=True)
    metrics = {
        "Rows Retained": row["rows_retained"],
        "Unique Comments": row["unique_comments"],
        "Unique Videos": row["unique_videos"],
        "Unique Authors": row["unique_authors"],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor="white")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_yscale("log")
    ax.set_title(f"Cleaned Corpus Cardinality ({row['pct_of_raw']:.1f}% of raw data retained)",
                 fontweight="bold")
    ax.set_ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t1_corpus_cardinality.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't1_corpus_cardinality.png'}")


def save_outputs(
    df_clean: pl.DataFrame,
    null_matrix: pl.DataFrame,
    cardinality: pl.DataFrame,
):
    logger.info("=" * 70)
    logger.info("PHASE 0 — Saving outputs")
    logger.info("=" * 70)

    save_cols = [c for c in df_clean.columns if not c.endswith("_null_flag")]
    df_clean.select(save_cols).write_parquet(CLEANED_CORPUS_PATH)
    logger.info(f"  Cleaned corpus  → {CLEANED_CORPUS_PATH}  ({CLEANED_CORPUS_PATH.stat().st_size / 1e6:.1f} MB)")

    null_matrix.write_parquet(OUTPUT_PARQUET / "t0_null_pattern_matrix.parquet")
    logger.info(f"  Null matrix     → {OUTPUT_PARQUET / 't0_null_pattern_matrix.parquet'}")

    cardinality.write_parquet(OUTPUT_PARQUET / "t1_corpus_cardinality.parquet")
    logger.info(f"  Cardinality     → {OUTPUT_PARQUET / 't1_corpus_cardinality.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 0 CLEANING PIPELINE — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()

    df_emb, df_lbl = load_raw_data()
    raw_count = len(df_emb)

    df = inner_merge_and_validate(df_emb, df_lbl)
    df = deduplicate(df)
    df = drop_invalid_rows(df)
    df = impute_engagement(df)
    df = recompute_text_features(df)
    df = audit_embeddings(df)

    null_matrix = build_null_pattern_matrix(df)
    cardinality = build_corpus_cardinality(pl.DataFrame({"_": [None] * raw_count}), df)

    plot_null_matrix(null_matrix)
    plot_cardinality_summary(cardinality)

    save_outputs(df, null_matrix, cardinality)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 0 COMPLETE — {len(df):,} clean rows — elapsed: {elapsed:.1f}s")
    logger.info("=" * 70)
