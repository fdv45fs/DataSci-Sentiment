# %% [markdown]
# # YouTube Comments EDA — Comprehensive Exploratory Data Analysis
# 
# **Dataset**: YouTube Comments with Engagement Metrics & Sentiment Labels  
# **Rows**: ~395,000 | **Labeled subset**: ~362,000 comments  
# **Author**: Intro to Data Science Course  
# **Date**: 2026-05-07
# 
# ---
# 
# ## Table of Contents
# 
# 1. [Environment Setup & Configuration](#1-environment-setup--configuration)
# 2. [Data Loading & Memory Optimisation](#2-data-loading--memory-optimisation)
# 3. [Data Overview & Structure](#3-data-overview--structure)
# 4. [Data Quality Assessment](#4-data-quality-assessment)
# 5. [Univariate Statistical Analysis](#5-univariate-statistical-analysis)
# 6. [Distributions & Visualisations](#6-distributions--visualisations)
# 7. [Text Analysis](#7-text-analysis)
# 8. [Temporal Analysis](#8-temporal-analysis)
# 9. [Bivariate & Multivariate Analysis](#9-bivariate--multivariate-analysis)
# 10. [Labeled Data / Sentiment Analysis](#10-labeled-data--sentiment-analysis)
# 11. [Key Insights & Recommendations](#11-key-insights--recommendations)
# 
# ---

# %% [markdown]
# ## 1. Environment Setup & Configuration
# 
# Import all required libraries, configure global settings for reproducibility, and define the visual theme used throughout the notebook.

# %%
# Core libraries
import sys
import os
import gc
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as sp_stats

# Text & NLP
import nltk

# Visualisation settings
import src.plots as plots

# Statistical helpers
from src.stats import (
    describe_numeric, mode_value, frequency_table,
    detect_outliers_iqr, detect_outliers_zscore, outlier_summary,
    correlation_matrix, top_correlations, distribution_stats,
    grouped_stats, engagement_tier,
)

# Text analysis helpers
from src.text_analysis import (
    extract_emojis, emoji_count, top_emojis,
    clean_text_basic, simple_tokenize,
    extract_ngrams, plot_wordcloud,
    uppercase_ratio, engagement_by_length,
    sentiment_word_summary,
)

warnings.filterwarnings("ignore")

# %%
# ── Global plot configuration ────────────────────────────────────────────────
plots.configure_style(style="whitegrid")
COLOR_PRIMARY = "#4C72B0"
COLOR_SECONDARY = "#DD8452"
COLOR_TERTIARY = "#55A868"
COLOR_QUATERNARY = "#C44E52"

# Pandas display options
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 50)
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

# Paths
DATA_DIR = Path("data")
MAIN_CSV = DATA_DIR / "final_dataset.csv"
LABELED_DIR = DATA_DIR / "labeled"

print(f"Python: {sys.version.split()[0]}")
print(f"NumPy:  {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Seaborn: {sns.__version__}")

# %%
# ── Download NLTK resources (first run) ──────────────────────────────────────
for resource in ["punkt", "stopwords", "wordnet", "punkt_tab", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

print("NLTK resources ready.")

# %% [markdown]
# ## 2. Data Loading & Memory Optimisation
# 
# Load the main CSV using dtype-optimised reading, then load and concatenate all labeled JSONL files. Show memory footprints before and after optimisation.

# %%
from src.loaders import load_main_dataset, load_labeled_jsonl, health_report

print("Loading main dataset (final_dataset.csv)...")
df = load_main_dataset(MAIN_CSV, verbose=True)

# %%
print("\nLoading labeled JSONL files (this may take a moment)...")
# Load a sample of label files first to check structure
df_labels_sample = load_labeled_jsonl(LABELED_DIR, limit_files=5, verbose=True)
print(f"\nLabel schema — columns: {df_labels_sample.columns.tolist()}")
print(f"Labels distribution (sample):\n{df_labels_sample['label'].value_counts()}")

# %%
# Load ALL labeled data (full corpus)
df_labels = load_labeled_jsonl(LABELED_DIR, verbose=True)
print(f"\nLabeled records: {len(df_labels):,}")
print(f"Unique labels: {df_labels['label'].unique().tolist()}")
gc.collect()

# %%
# ── Merge labeled sentiment onto main dataset ──────────────────────────────────
df_full = df.merge(
    df_labels[["comment_id", "label", "model", "labeled_at"]],
    on="comment_id",
    how="left",
)
print(f"After merge — shape: {df_full.shape}")
print(f"Rows with sentiment label: {df_full['label'].notna().sum():,} ({df_full['label'].notna().mean():.1%})")
gc.collect()

# %% [markdown]
# ## 3. Data Overview & Structure
# 
# Inspect column dtypes, the health report, and categorise columns by type for targeted analysis downstream.

# %%
# ── Shape summary ─────────────────────────────────────────────────────────────
print(f"{'═'*60}")
print(f"{'Main DataFrame':<20}  {df.shape[0]:>10,} rows  × {df.shape[1]:>3} columns")
print(f"{'Label DataFrame':<20}  {df_labels.shape[0]:>10,} rows  × {df_labels.shape[1]:>3} columns")
print(f"{'Merged DataFrame':<20} {df_full.shape[0]:>10,} rows  × {df_full.shape[1]:>3} columns")
print(f"{'═'*60}")

# %%
# ── Column dtypes & nulls ─────────────────────────────────────────────────────
print("\n── Main DataFrame .info() ──\n")
df.info(memory_usage="deep")

# %%
# ── Health report ───────────────────────────────────────────────────────────────
health = health_report(df)
print("\n── Column Health Report ──\n")
health

# %%
# ── Column categories ──────────────────────────────────────────────────────────
from src.loaders import COLUMN_CATEGORIES

ID_COLS      = COLUMN_CATEGORIES["id_cols"]
TEXT_COLS    = COLUMN_CATEGORIES["text_cols"]
NUMERIC_COLS = COLUMN_CATEGORIES["numeric_cols"]
DT_COLS      = COLUMN_CATEGORIES["datetime_cols"]
CAT_COLS     = COLUMN_CATEGORIES["categorical_cols"]

print("── Column Categories ──")
for name, cols in COLUMN_CATEGORIES.items():
    print(f"  {name:<20}: {cols}")

# %%
# ── Sample rows ────────────────────────────────────────────────────────────────
print("── First 5 rows ──\n")
df.head()

# %%
# ── Unique value counts ────────────────────────────────────────────────────────
print("── Unique value counts (top 5 per category) ──\n")
for col in ID_COLS + CAT_COLS:
    print(f"  {col:<20} {df[col].nunique():>10,} unique")

# %%
# ── Date range ─────────────────────────────────────────────────────────────────
print("── Date Range ──")
print(f"  published_at : {df['published_at'].min()}  →  {df['published_at'].max()}")
print(f"  crawled_at    : {df['crawled_at'].min()}  →  {df['crawled_at'].max()}")
print(f"  labeled_at    : {df_labels['labeled_at'].min()}  →  {df_labels['labeled_at'].max()}")

# %% [markdown]
# ## 4. Data Quality Assessment
# 
# Examine missing values, duplicate rows, invalid entries, and outlier statistics across all numeric columns.

# %%
# ── Missing values ─────────────────────────────────────────────────────────────
missing_pct = df.isnull().mean() * 100
missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)

if len(missing_cols) > 0:
    print("── Columns with Missing Values ──\n")
    for col, pct in missing_cols.items():
        print(f"  {col:<25}  {pct:>6.2f}%  ({df[col].isnull().sum():,} / {len(df):,})")
else:
    print("No missing values found in the dataset.")

fig, ax = plt.subplots(figsize=(12, max(4, len(missing_cols) * 0.35)))
plots.plot_missing(df, ax=ax, title="Missing Values per Column (%)")
plt.tight_layout()
plt.show()

# %%
# ── Duplicate rows ─────────────────────────────────────────────────────────────
print("── Duplicate Analysis ──\n")

# Exact duplicates (all columns)
exact_dup = df.duplicated().sum()
print(f"  Exact duplicate rows  : {exact_dup:,}")

# Duplicate comment_ids
dup_ids = df["comment_id"].duplicated().sum()
print(f"  Duplicate comment_ids  : {dup_ids:,}")

# Near-duplicate comment_text (exact string matches)
dup_text = df["comment_text"].duplicated().sum()
print(f"  Duplicate comment_text : {dup_text:,}")

# Show a few duplicate comment_ids if any
if dup_ids > 0:
    print("\n── Sample duplicate comment_ids ──\n")
    dup_id_examples = df[df["comment_id"].duplicated(keep=False)]["comment_id"].unique()[:3]
    for cid in dup_id_examples:
        display(df[df["comment_id"] == cid][["comment_id", "comment_text", "like_count"]].head(3))

# %%
# ── Invalid values ────────────────────────────────────────────────────────────
print("── Invalid Value Checks ──\n")

# Negative counts
for col in ["like_count", "reply_count", "char_count", "word_count", "exclamation_count", "question_count"]:
    if col in df.columns:
        neg = (df[col] < 0).sum()
        print(f"  Negative {col:<25}: {neg:,}")

# Future dates
now = pd.Timestamp.now(tz="UTC")
future_dates = df["published_at"] > now
print(f"  Future published_at dates    : {future_dates.sum():,}")

# Zero-length comments
zero_len = (df["char_count"] == 0).sum()
print(f"  Zero-length comments          : {zero_len:,}")

# Missing comment_text
missing_text = df["comment_text"].isna().sum()
print(f"  Missing comment_text          : {missing_text:,}")

# %%
# ── Outlier detection (IQR & Z-score) ────────────────────────────────────────
outlier_results = outlier_summary(df, numeric_cols=NUMERIC_COLS, methods=("iqr", "zscore"))

print("── Outlier Summary (IQR method, multiplier=1.5) ──\n")
display(outlier_results["iqr"][["lower_bound", "upper_bound", "outlier_count", "outlier_pct"]])

print("\n── Outlier Summary (Z-score method, threshold=3.0) ──\n")
display(outlier_results["zscore"][["median", "mad", "threshold", "outlier_count", "outlier_pct"]])

# %%
# Visualise outlier distribution for engagement columns
engage_cols = ["like_count", "reply_count", "char_count", "word_count"]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
colors = plots.get_colors(4)

for i, col in enumerate(engage_cols):
    o = detect_outliers_iqr(df[col])
    labels = np.where(o["is_outlier"], "Outlier", "Normal")
    sns.violinplot(x=labels, y=df[col], ax=axes[i], palette=["#55A868", "#C44E52"], order=["Normal", "Outlier"])
    axes[i].set_title(f"{col} — {o['outlier_pct']:.1f}% outliers (IQR)")
    axes[i].set_xlabel("")

plt.suptitle("Outlier Distribution by Engagement Column (IQR Method)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Univariate Statistical Analysis
# 
# Comprehensive descriptive statistics, central tendency measures, dispersion metrics, and distribution shape diagnostics for all numeric columns.

# %%
# ── Descriptive statistics ────────────────────────────────────────────────────
desc = describe_numeric(df, cols=NUMERIC_COLS)
print("── Extended Descriptive Statistics ──\n")
display(desc)

# %%
# ── Central tendency & dispersion ────────────────────────────────────────────
engage_cols = ["like_count", "reply_count", "char_count", "word_count"]

print("── Central Tendency & Dispersion (Engagement Columns) ──\n")
ct_rows = []
for col in engage_cols:
    s = df[col].dropna()
    ct_rows.append({
        "column": col,
        "mean": s.mean(),
        "median": s.median(),
        "mode": s.mode().iloc[0] if not s.mode().empty else np.nan,
        "std": s.std(),
        "variance": s.var(),
        "cv (coef. of variation)": s.std() / s.mean() if s.mean() != 0 else np.nan,
        "skewness": s.skew(),
        "kurtosis": s.kurt(),
    })

ct_df = pd.DataFrame(ct_rows).set_index("column").round(4)
display(ct_df)

# %%
# ── Frequency analysis: source_query ──────────────────────────────────────────
print("── Frequency: source_query ──\n")
display(frequency_table(df["source_query"], n=15))

# %%
# ── Frequency analysis: top authors ──────────────────────────────────────────
print("── Top 15 Authors by Comment Count ──\n")
display(frequency_table(df["author_name"], n=15))

# %%
# ── Frequency analysis: top YouTube videos ─────────────────────────────────────
print("── Top 15 YouTube Videos by Comment Count ──\n")
display(frequency_table(df["title_youtube"], n=15))

# %% [markdown]
# ## 6. Distributions & Visualisations
# 
# Histograms with KDE, box plots, violin plots, and bar charts for all key columns.

# %%
# ── Histogram grid: all numeric columns ────────────────────────────────────────
fig = plots.plot_histogram_grid(df, cols=NUMERIC_COLS, ncols=4, kde=True, stat="count")
fig.suptitle("Distribution of All Numeric Features", fontsize=16, fontweight="bold", y=1.02)
plt.show()

# %%
# ── Engagement columns: log-scale histograms ───────────────────────────────────
log_cols = ["like_count", "reply_count"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, col in enumerate(log_cols):
    s = df[col].dropna()
    plots.plot_histogram(
        np.log10(s[s > 0] + 1),
        ax=axes[i],
        bins=50,
        kde=True,
        title=f"Log10 Distribution of {col}",
        xlabel=f"log10({col} + 1)",
        color=plots.PALETTE[["primary", "secondary"][i]],
    )

plt.suptitle("Engagement Metrics — Log-Scale Distributions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Box plots: engagement metrics ─────────────────────────────────────────────
engage_cols = ["like_count", "reply_count", "char_count", "word_count"]
fig = plots.plot_boxplot_grid(df, cols=engage_cols, ncols=2, figsize_scale=7)
fig.suptitle("Box Plots — Engagement & Text Metrics", fontsize=14, fontweight="bold", y=1.02)
plt.show()

# %%
# ── Box plot: like_count by source_query ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
order = df.groupby("source_query")["like_count"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="source_query", y="like_count", order=order, ax=ax,
            palette="Spectral", showfliers=False)
ax.set_yscale("symlog")
ax.set_title("Like Count Distribution by Source Query (log scale, outliers hidden)", fontsize=13)
ax.set_xlabel("Source Query")
ax.set_ylabel("like_count (symlog)")
plots.wrap_labels(ax, width=20)
plt.tight_layout()
plt.show()

# %%
# ── Violin plot: engagement by source_query ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (col, ylabel) in enumerate([("like_count", "like_count"), ("reply_count", "reply_count")]):
    sns.violinplot(data=df, x="source_query", y=col, ax=axes[i],
                   palette="Spectral", inner="box")
    axes[i].set_yscale("symlog")
    axes[i].set_title(f"{col} Distribution by Source Query")
    axes[i].set_xlabel("Source Query")
    axes[i].set_ylabel(f"{ylabel} (symlog)")
    plots.wrap_labels(axes[i], width=20)

plt.suptitle("Engagement Distribution — Violin Plots by Source Query", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Bar chart: source_query distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
plots.plot_bar(df["source_query"], ax=ax, n=15, title="Comment Count by Source Query",
               xlabel="Number of Comments", palette=[plots.PALETTE["primary"]])
plt.tight_layout()
plt.show()

# %%
# ── Scatter: word_count vs like_count ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sample = df.sample(min(50000, len(df)), random_state=42)
sns.scatterplot(data=sample, x="word_count", y="like_count",
                hue="source_query", alpha=0.4, s=10, ax=ax)
ax.set_yscale("symlog")
ax.set_title("Word Count vs. Like Count (50K sample)", fontsize=13)
ax.set_xlabel("word_count")
ax.set_ylabel("like_count (symlog)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Text Analysis
# 
# Comment length distributions, emoji usage patterns, hashtag/mention analysis, uppercase ratios, and n-gram extraction per sentiment class.

# %%
# ── Comment length distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Character count
plots.plot_histogram(df["char_count"], ax=axes[0], bins=80, kde=True,
                    title="Comment Length (Characters)", xlabel="char_count")

# Word count
plots.plot_histogram(df["word_count"], ax=axes[1], bins=80, kde=True,
                    title="Comment Length (Words)", xlabel="word_count")

plt.suptitle("Text Length Distributions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Engagement by comment length bin ─────────────────────────────────────────
eng_by_len = engagement_by_length(df)
print("── Engagement Metrics by Comment Length Bin ──\n")
display(eng_by_len)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=eng_by_len, x="length_bin", y="mean_likes", ax=ax,
            palette="Blues_d", errorbar=None)
ax.set_title("Average Like Count by Comment Length Bin")
ax.set_xlabel("Comment Length Bin (characters)")
ax.set_ylabel("Mean Like Count")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

# %%
# ── Emoji analysis ────────────────────────────────────────────────────────────
print("── Emoji Usage Statistics ──\n")
print(f"  Comments with at least 1 emoji : {(df['emoji_count'] > 0).sum():,} ({(df['emoji_count'] > 0).mean():.1%})")
print(f"  Mean emoji count                : {df['emoji_count'].mean():.3f}")
print(f"  Max emoji count                 : {df['emoji_count'].max():,}")

# Top emojis
top_emo = top_emojis(df["comment_text"], n=25)
print("\n── Top 25 Emojis ──\n")
display(top_emo)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_emo.head(15), x="count", y="emoji", ax=ax, palette="viridis")
ax.set_title("Top 15 Most Frequent Emojis")
ax.set_xlabel("Count")
ax.set_ylabel("Emoji")
plt.tight_layout()
plt.show()

# %%
# ── Hashtag & mention analysis ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plots.plot_histogram(df["hashtag_count"], ax=axes[0], bins=15, kde=False,
                    title="Hashtag Count Distribution", xlabel="hashtag_count",
                    color=plots.PALETTE["secondary"])

plots.plot_histogram(df["mention_count"], ax=axes[1], bins=15, kde=False,
                    title="Mention (@) Count Distribution", xlabel="mention_count",
                    color=plots.PALETTE["tertiary"])

plt.suptitle("Social Features: Hashtags & Mentions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print(f"Comments with hashtags : {(df['hashtag_count'] > 0).sum():,} ({(df['hashtag_count'] > 0).mean():.1%})")
print(f"Comments with mentions: {(df['mention_count'] > 0).sum():,} ({(df['mention_count'] > 0).mean():.1%})")

# %%
# ── Uppercase ratio (shouting detection) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
plots.plot_histogram(df["uppercase_ratio"], ax=ax, bins=50, kde=True,
                    title="Uppercase Ratio Distribution (Shouting Detection)",
                    xlabel="uppercase_ratio", color=plots.PALETTE["quaternary"])
ax.axvline(0.3, color=plots.PALETTE["quaternary"], linestyle="--", linewidth=1.5,
           label="Potential shouting (>30% uppercase)")
ax.legend()
plt.tight_layout()
plt.show()

shouting_pct = (df["uppercase_ratio"] > 0.3).mean()
print(f"Comments flagged as 'shouting' (>30% uppercase): {shouting_pct:.1%}")

# %%
# ── Exclamation & question mark patterns ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plots.plot_histogram(df["exclamation_count"], ax=axes[0], bins=15, kde=False,
                    title="Exclamation Mark Count Distribution",
                    xlabel="exclamation_count", color=plots.PALETTE["secondary"])

plots.plot_histogram(df["question_count"], ax=axes[1], bins=10, kde=False,
                    title="Question Mark Count Distribution",
                    xlabel="question_count", color=plots.PALETTE["tertiary"])

plt.suptitle("Punctuation Patterns", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print(f"Comments with '!' : {(df['exclamation_count'] > 0).sum():,} ({(df['exclamation_count'] > 0).mean():.1%})")
print(f"Comments with '?' : {(df['question_count'] > 0).sum():,} ({(df['question_count'] > 0).mean():.1%})")

# %%
# ── Top n-grams: all comments ────────────────────────────────────────────────
print("── Top 30 Unigrams (all comments) ──\n")
top_unigrams = extract_ngrams(df["comment_text"], n=1, top_k=30, min_freq=5, remove_stopwords=True)
display(top_unigrams)

print("\n── Top 30 Bigrams (all comments) ──\n")
top_bigrams = extract_ngrams(df["comment_text"], n=2, top_k=30, min_freq=5, remove_stopwords=True)
display(top_bigrams)

# %%
# ── Word clouds by source_query ───────────────────────────────────────────────
query_groups = df.groupby("source_query")["comment_text"].apply(lambda x: " ".join(x.dropna().astype(str)))

n_queries = len(query_groups)
ncols = min(3, n_queries)
nrows = int(np.ceil(n_queries / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten() if n_queries > 1 else [axes]

cmaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
for i, (query, text) in enumerate(query_groups.items()):
    try:
        plot_wordcloud(text, ax=axes[i], colormap=cmaps[i % len(cmaps)], max_words=100)
        axes[i].set_title(f"Source: {query}", fontsize=12, fontweight="bold")
    except Exception as e:
        axes[i].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Word Clouds by Source Query", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Temporal Analysis
# 
# Comment volume over time, engagement trends, and temporal patterns by day-of-week and hour.

# %%
# ── Comment volume over time ───────────────────────────────────────────────────
df_ts = df.copy()
df_ts["published_date"] = df_ts["published_at"].dt.date

daily_counts = df_ts.groupby("published_date").size()

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

# Raw daily counts
plots.plot_time_series(daily_counts, ax=axes[0], title="Daily Comment Volume",
                      ylabel="Number of Comments", rolling=7, color=COLOR_PRIMARY)

# Monthly aggregation
df_ts["published_month"] = df_ts["published_at"].dt.to_period("M")
monthly_counts = df_ts.groupby("published_month").size()
monthly_counts.index = monthly_counts.index.astype(str)

axes[1].bar(monthly_counts.index, monthly_counts.values, color=COLOR_PRIMARY, alpha=0.85)
axes[1].set_title("Monthly Comment Volume", fontsize=12)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Comment Count")
axes[1].tick_params(axis="x", rotation=45)
sns.despine(ax=axes[1])

plt.suptitle("Comment Volume Over Time", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Engagement trends over time ────────────────────────────────────────────────
daily_engagement = df_ts.groupby("published_date").agg(
    mean_likes=("like_count", "mean"),
    median_likes=("like_count", "median"),
    mean_replies=("reply_count", "mean"),
    total_likes=("like_count", "sum"),
).reset_index()
daily_engagement["published_date"] = pd.to_datetime(daily_engagement["published_date"])
daily_engagement = daily_engagement.set_index("published_date").sort_index()

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

axes[0].plot(daily_engagement.index, daily_engagement["mean_likes"].rolling(7).mean(),
             color=COLOR_PRIMARY, linewidth=2, label="7-day rolling mean")
axes[0].fill_between(daily_engagement.index, daily_engagement["mean_likes"].rolling(7).mean(),
                    alpha=0.3, color=COLOR_PRIMARY)
axes[0].set_title("Average Likes per Comment Over Time (7-day rolling)")
axes[0].set_ylabel("Mean Like Count")
sns.despine(ax=axes[0])

axes[1].plot(daily_engagement.index, daily_engagement["median_likes"].rolling(7).mean(),
             color=COLOR_SECONDARY, linewidth=2, label="7-day rolling median")
axes[1].set_title("Median Likes per Comment Over Time (7-day rolling)")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Median Like Count")
sns.despine(ax=axes[1])

plt.suptitle("Engagement Trends Over Time", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Day-of-week × Hour-of-day heatmap ─────────────────────────────────────────
fig = plots.plot_temporal_heatmap(df, dt_col="published_at", value_col="like_count",
                                  aggfunc="mean",
                                  title="Average Likes by Day of Week & Hour of Day")
plt.show()

# %%
# ── Day-of-week distribution ───────────────────────────────────────────────────
df_ts["dow"] = df_ts["published_at"].dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dow_counts = df_ts.groupby("dow").size().reindex(dow_order)
axes[0].bar(dow_counts.index, dow_counts.values, color=COLOR_PRIMARY, alpha=0.85)
axes[0].set_title("Comment Volume by Day of Week")
axes[0].set_xlabel("Day of Week")
axes[0].set_ylabel("Comment Count")
axes[0].tick_params(axis="x", rotation=45)
sns.despine(ax=axes[0])

dow_likes = df_ts.groupby("dow")["like_count"].mean().reindex(dow_order)
axes[1].bar(dow_likes.index, dow_likes.values, color=COLOR_SECONDARY, alpha=0.85)
axes[1].set_title("Average Likes by Day of Week")
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Mean Like Count")
axes[1].tick_params(axis="x", rotation=45)
sns.despine(ax=axes[1])

plt.suptitle("Day-of-Week Patterns", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Hour-of-day distribution ──────────────────────────────────────────────────
df_ts["hour"] = df_ts["published_at"].dt.hour

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

hour_counts = df_ts.groupby("hour").size()
axes[0].bar(hour_counts.index, hour_counts.values, color=COLOR_PRIMARY, alpha=0.85)
axes[0].set_title("Comment Volume by Hour of Day")
axes[0].set_xlabel("Hour (UTC)")
axes[0].set_ylabel("Comment Count")
axes[0].set_xticks(range(0, 24))
sns.despine(ax=axes[0])

hour_likes = df_ts.groupby("hour")["like_count"].mean()
axes[1].bar(hour_likes.index, hour_likes.values, color=COLOR_SECONDARY, alpha=0.85)
axes[1].set_title("Average Likes by Hour of Day")
axes[1].set_xlabel("Hour (UTC)")
axes[1].set_ylabel("Mean Like Count")
axes[1].set_xticks(range(0, 24))
sns.despine(ax=axes[1])

plt.suptitle("Hour-of-Day Patterns", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Time between publish and crawl ────────────────────────────────────────────
if df_ts["crawled_at"].dt.tz is None:
    df_ts["crawled_at"] = df_ts["crawled_at"].dt.tz_localize("UTC")
df_ts["lag_hours"] = (df_ts["crawled_at"] - df_ts["published_at"]).dt.total_seconds() / 3600
df_ts["lag_days"] = df_ts["lag_hours"] / 24

print("── Time Lag (Publish → Crawl) Statistics ──\n")
lag_stats = distribution_stats(df_ts["lag_days"])
display(lag_stats)

fig, ax = plt.subplots(figsize=(10, 5))
plots.plot_histogram(
    df_ts["lag_days"].clip(upper=df_ts["lag_days"].quantile(0.99)),
    ax=ax, bins=60, kde=True,
    title="Days Between Publish and Crawl (99th percentile clipped)",
    xlabel="Days",
    color=COLOR_TERTIARY,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Bivariate & Multivariate Analysis
# 
# Correlation analysis, scatter matrices, grouped statistics, cross-tabulations, and pair plots.

# %%
# ── Correlation matrices (Pearson & Spearman) ──────────────────────────────────
corrs = correlation_matrix(df, numeric_cols=NUMERIC_COLS)

for method, corr in corrs.items():
    fig = plots.plot_correlation_heatmap(
        corr, title=f"{method.capitalize()} Correlation Matrix",
        figsize=(14, 12)
    )
    plt.show()

# %%
# ── Top correlation pairs ───────────────────────────────────────────────────────
print("── Top 20 Correlation Pairs (Pearson) ──\n")
display(top_correlations(corrs["pearson"], n=20, absolute=True))

print("\n── Top 20 Correlation Pairs (Spearman) ──\n")
display(top_correlations(corrs["spearman"], n=20, absolute=True))

# %%
# ── Scatter matrix (sampled) ───────────────────────────────────────────────────
scatter_cols = ["char_count", "word_count", "avg_word_length", "uppercase_ratio", "like_count", "like_count_log"]
sample = df[scatter_cols].sample(min(20000, len(df)), random_state=42)

g = sns.pairplot(sample, vars=scatter_cols, hue="like_count_log",
                palette="viridis", diag_kind="kde",
                plot_kws={"alpha": 0.4, "s": 8}, diag_kws={"linewidth": 2})
g.fig.suptitle("Scatter Matrix — Key Numeric Features", fontsize=14, fontweight="bold", y=1.02)
plt.show()

# %%
# ── Grouped statistics by source_query ─────────────────────────────────────────
print("── Grouped Statistics by Source Query ──\n")
grouped = grouped_stats(df, group_col="source_query",
                        value_cols=["like_count", "reply_count", "char_count", "word_count"])
display(grouped.sort_values("like_count_mean", ascending=False))

# %%
# ── Engagement tiers ───────────────────────────────────────────────────────────
df["engagement_tier"] = engagement_tier(df["like_count"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

tier_counts = df["engagement_tier"].value_counts()
plots.plot_pie(df["engagement_tier"], ax=axes[0],
               title="Comment Distribution by Engagement Tier")

tier_avg = df.groupby("engagement_tier", observed=True)["like_count"].mean().reindex(
    ["Micro (<10)", "Small (10-100)", "Medium (100-1K)", "Large (1K-10K)", "Viral (>=10K)"]
)
axes[1].barh(tier_avg.index.astype(str), tier_avg.values, color=plots.get_colors(5))
axes[1].set_title("Average Likes by Engagement Tier")
axes[1].set_xlabel("Average Like Count")
for i, v in enumerate(tier_avg.values):
    axes[1].text(v + tier_avg.max() * 0.02, i, f"{v:,.0f}", va="center")
sns.despine(ax=axes[1])

plt.suptitle("Engagement Tier Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Cross-tabulation: source_query × engagement_tier ─────────────────────────
ct = pd.crosstab(df["source_query"], df["engagement_tier"], normalize="index") * 100
tier_order = ["Micro (<10)", "Small (10-100)", "Medium (100-1K)", "Large (1K-10K)", "Viral (>=10K)"]
ct = ct.reindex(columns=[c for c in tier_order if c in ct.columns])

print("── Cross-tabulation: Source Query × Engagement Tier (% per row) ──\n")
display(ct.round(2))

fig, ax = plt.subplots(figsize=(12, 6))
plots.plot_heatmap(ct, ax=ax, cmap="Blues", fmt=".1f",
                  title="Engagement Tier Distribution by Source Query (%)",
                  annot_kws={"size": 10})
ax.set_xlabel("Engagement Tier")
ax.set_ylabel("Source Query")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Labeled Data / Sentiment Analysis
# 
# Analyse the labeled subset: sentiment distribution, engagement by sentiment class, text characteristics, and n-grams per class.

# %%
# ── Labeled data overview ─────────────────────────────────────────────────────
print("── Labeled Dataset Overview ──\n")
print(f"  Total labeled records : {len(df_labels):,}")
print(f"  Unique comment_ids   : {df_labels['comment_id'].nunique():,}")
print(f"  Unique posts          : {df_labels['post_id'].nunique():,}")
print(f"  Labeler model         : {df_labels['model'].unique().tolist()}")

print("\n── Sentiment Distribution ──\n")
label_counts = df_labels["label"].value_counts()
display(label_counts)
display(label_counts / len(df_labels) * 100)

# %%
# ── Sentiment distribution plots ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plots.plot_pie(df_labels["label"], ax=axes[0],
               title="Sentiment Label Distribution", n=10)

plots.plot_bar(df_labels["label"], ax=axes[1],
               n=10, title="Sentiment Label Counts",
               xlabel="Count", horizontal=True,
               palette=[plots.PALETTE["neutral"],
                        plots.PALETTE["positive"],
                        plots.PALETTE["negative"]])

plt.suptitle("Sentiment Label Distribution", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Merge labels onto main df for analysis ───────────────────────────────────
df_labeled = df_full[df_full["label"].notna()].copy()
print(f"Labeled rows in merged df: {len(df_labeled):,}")

# Engagement by sentiment
print("\n── Engagement Stats by Sentiment ──\n")
sent_engage = df_labeled.groupby("label").agg(
    count=("like_count", "count"),
    mean_likes=("like_count", "mean"),
    median_likes=("like_count", "median"),
    mean_replies=("reply_count", "mean"),
    median_replies=("reply_count", "median"),
    mean_char_count=("char_count", "mean"),
    mean_word_count=("word_count", "mean"),
).round(3)
display(sent_engage)

# %%
# ── Box plot: engagement by sentiment ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, col in enumerate(["like_count", "reply_count"]):
    order = df_labeled.groupby("label")[col].median().sort_values(ascending=False).index
    sns.boxplot(data=df_labeled, x="label", y=col, ax=axes[i],
                order=order, palette={"positive": plots.PALETTE["tertiary"],
                                     "neutral": plots.PALETTE["neutral"],
                                     "negative": plots.PALETTE["negative"]},
                showfliers=False)
    axes[i].set_yscale("symlog")
    axes[i].set_title(f"{col} by Sentiment (log scale)")
    axes[i].set_xlabel("Sentiment")
    axes[i].set_ylabel(f"{col} (symlog)")

plt.suptitle("Engagement Metrics by Sentiment Class", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Violin plot: text length by sentiment ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, col in enumerate(["char_count", "word_count"]):
    sns.violinplot(data=df_labeled, x="label", y=col, ax=axes[i],
                   palette={"positive": plots.PALETTE["tertiary"],
                            "neutral": plots.PALETTE["neutral"],
                            "negative": plots.PALETTE["negative"]},
                   inner="box")
    axes[i].set_title(f"{col} Distribution by Sentiment")
    axes[i].set_xlabel("Sentiment")
    axes[i].set_ylabel(col)

plt.suptitle("Text Length by Sentiment Class", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── N-grams per sentiment class ──────────────────────────────────────────────
sentiment_words = sentiment_word_summary(df_labeled, text_col="comment_text",
                                         label_col="label", top_n=20)

for label, top_words in sentiment_words.items():
    if top_words.empty:
        continue
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_words.head(15), x="count", y="1-gram", ax=ax,
                palette="Spectral")
    ax.set_title(f"Top 15 Words — {label.capitalize()} Sentiment")
    ax.set_xlabel("Count")
    ax.set_ylabel("Word")
    plt.tight_layout()
    plt.show()

# %%
# ── Word clouds per sentiment class ─────────────────────────────────────────
label_colors = {
    "positive": "Greens",
    "neutral": "Blues",
    "negative": "Reds",
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (label, cmap) in enumerate(label_colors.items()):
    text = " ".join(df_labeled[df_labeled["label"] == label]["comment_text"].dropna().astype(str))
    if text.strip():
        try:
            plot_wordcloud(text, ax=axes[i], colormap=cmap, max_words=80)
            axes[i].set_title(f"{label.capitalize()} Comments", fontsize=13, fontweight="bold")
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

plt.suptitle("Word Clouds by Sentiment Class", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# ── Emoji usage by sentiment ─────────────────────────────────────────────────
emoji_by_sentiment = df_labeled.groupby("label")["emoji_count"].agg(["mean", "median", "std"]).round(3)
print("── Emoji Usage by Sentiment ──\n")
display(emoji_by_sentiment)

fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=df_labeled, x="label", y="emoji_count", ax=ax,
               palette={"positive": plots.PALETTE["tertiary"],
                        "neutral": plots.PALETTE["neutral"],
                        "negative": plots.PALETTE["negative"]},
               inner="box")
ax.set_title("Emoji Count Distribution by Sentiment")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Emoji Count")
plt.tight_layout()
plt.show()

# %%
# ── Labeler model analysis ────────────────────────────────────────────────────
print("── Labeling Model Distribution ──\n")
display(df_labels["model"].value_counts())

# Sentiment distribution per model
print("\n── Sentiment Distribution per Model ──\n")
display(pd.crosstab(df_labels["model"], df_labels["label"], normalize="index").round(3) * 100)

# %%
# ── Labeling timestamp analysis ─────────────────────────────────────────────
df_labels_ts = df_labels.copy()
df_labels_ts["labeled_hour"] = df_labels_ts["labeled_at"].dt.hour
df_labels_ts["labeled_date"] = df_labels_ts["labeled_at"].dt.date

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

hour_dist = df_labels_ts.groupby("labeled_hour").size()
axes[0].bar(hour_dist.index, hour_dist.values, color=COLOR_PRIMARY, alpha=0.85)
axes[0].set_title("Labels Created by Hour of Day")
axes[0].set_xlabel("Hour (UTC)")
axes[0].set_ylabel("Number of Labels Created")
axes[0].set_xticks(range(0, 24))
sns.despine(ax=axes[0])

daily_labels = df_labels_ts.groupby("labeled_date").size()
axes[1].plot(daily_labels.index, daily_labels.values,
             color=COLOR_SECONDARY, linewidth=2)
axes[1].fill_between(daily_labels.index, daily_labels.values, alpha=0.3, color=COLOR_SECONDARY)
axes[1].set_title("Labels Created Over Time")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Labels Created per Day")
axes[1].tick_params(axis="x", rotation=45)
sns.despine(ax=axes[1])

plt.suptitle("Labeling Activity Over Time", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Key Insights
# 
# Automated data quality profiling and top findings.

# %%
# ── Automated summary statistics per column ───────────────────────────────────
print("── Automated Distribution Summary (All Numeric Columns) ──\n")
summary_frames = []
for col in NUMERIC_COLS:
    if col in df.columns:
        summary_frames.append(distribution_stats(df[col]).rename(col))

full_summary = pd.concat(summary_frames, axis=1).T
full_summary.index.name = "column"
display(full_summary)

# %%
# ── Final memory summary ──────────────────────────────────────────────────────
print("\n── Final Memory Summary ──\n")
for name, df_obj in [("df (main)", df), ("df_labels", df_labels), ("df_full (merged)", df_full)]:
    mem = df_obj.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  {name:<25} {mem:>10.2f} MB")

print(f"\n{'='*60}")
print("EDA COMPLETE — YouTube Comments Dataset")
print(f"{'='*60}")


