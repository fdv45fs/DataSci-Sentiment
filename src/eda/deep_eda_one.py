import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import orthogonal_procrustes
import umap
import hdbscan
import networkx as nx
import ripser
import persim
import re
import langdetect
from collections import Counter
import warnings
import time
import logging
from datetime import datetime

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

COMBINED_DATA_PATH = "data/combined_embeddings.parquet"
NUMERIC_COLS = [
    "like_count", "reply_count", "char_count", "word_count",
    "avg_word_length", "uppercase_ratio", "exclamation_count",
    "question_count", "hashtag_count", "mention_count",
    "emoji_count", "like_count_log"
]
EMBEDDING_COLS = ["embedding", "embedding_char", "embedding_word", "embedding_ft"]
TEXT_COL = "comment_text"

def load_data():
    logger.info("=" * 60)
    logger.info("Starting data loading process")
    logger.info("=" * 60)
    
    start_time = time.time()
    logger.info(f"Reading parquet file from: {COMBINED_DATA_PATH}")
    logger.info(f"Target columns: {NUMERIC_COLS + EMBEDDING_COLS + [TEXT_COL, 'published_at', 'crawled_at', 'source_query']}")
    
    df = pl.read_parquet(
        COMBINED_DATA_PATH,
        columns=NUMERIC_COLS + EMBEDDING_COLS + [TEXT_COL, "published_at", "crawled_at", "source_query"]
    )
    
    logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Memory usage estimate: {df.estimated_size() / 1024**2:.2f} MB")
    
    logger.info("Extracting numeric features to numpy array")
    X_numeric = df.select(NUMERIC_COLS).to_numpy().astype(np.float32)
    logger.info(f"Numeric features shape: {X_numeric.shape}, dtype: {X_numeric.dtype}")
    
    logger.info("Extracting embedding columns")
    embeddings = {}
    for col in EMBEDDING_COLS:
        start_col = time.time()
        embeddings[col] = np.stack(df[col].to_numpy())
        logger.info(f"  Loaded {col}: shape {embeddings[col].shape}, took {time.time() - start_col:.2f}s")
    
    logger.info(f"Combining all features: {len(NUMERIC_COLS)} numeric + {sum(e.shape[1] for e in embeddings.values())} embedding dimensions")
    X_all = np.hstack([X_numeric] + list(embeddings.values()))
    logger.info(f"Combined feature matrix shape: {X_all.shape}")
    
    logger.info("Initializing StandardScaler for feature normalization")
    scaler = StandardScaler()
    logger.info("Fitting scaler and transforming features")
    X_scaled = scaler.fit_transform(X_all)
    logger.info(f"Scaled features completed. Mean ~0, Std ~1 verification: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")
    
    elapsed = time.time() - start_time
    logger.info(f"Data loading complete. Total time: {elapsed:.2f} seconds")
    logger.info("=" * 60)
    
    return df, X_scaled, embeddings, scaler

def detect_language(text):
    try:
        return langdetect.detect(text)
    except Exception:
        return "unknown"

def detect_script(text):
    scripts = {
        "Latin": r"[\u0041-\u005A\u0061-\u007A\u00C0-\u024F]",
        "Cyrillic": r"[\u0400-\u04FF]",
        "CJK": r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF]",
        "Arabic": r"[\u0600-\u06FF\u0750-\u077F]",
        "Devanagari": r"[\u0900-\u097F]"
    }
    for script, pattern in scripts.items():
        if re.search(pattern, text):
            return script
    return "Other"

def detect_code_switching(text):
    tokens = text.split()
    if len(tokens) < 5:
        return 0.0
    window_size = 3
    langs = []
    for i in range(0, len(tokens) - window_size + 1, 2):
        window = " ".join(tokens[i:i+window_size])
        try:
            langs.append(langdetect.detect(window))
        except Exception:
            langs.append("unknown")
    if len(langs) < 2:
        return 0.0
    switches = sum(1 for i in range(len(langs)-1) if langs[i] != langs[i+1] and langs[i] != "unknown" and langs[i+1] != "unknown")
    return switches / (len(langs) - 1)

def apply_language_architecture(df):
    logger.info("Applying language detection and script analysis")
    total_rows = len(df)
    logger.info(f"Processing {total_rows} comments for language features")
    
    start_time = time.time()
    
    logger.info("  Detecting primary language for each comment")
    df = df.with_columns([
        pl.col(TEXT_COL).map_elements(detect_language, return_dtype=pl.Utf8).alias("lang_primary"),
        pl.col(TEXT_COL).map_elements(detect_script, return_dtype=pl.Utf8).alias("script_type"),
        pl.col(TEXT_COL).map_elements(detect_code_switching, return_dtype=pl.Float64).alias("code_switch_ratio")
    ])
    
    logger.info("  Computing code-switching indicators")
    df = df.with_columns([
        pl.when(pl.col("code_switch_ratio") > 0.3).then(pl.lit(True)).otherwise(pl.lit(False)).alias("is_code_switched"),
        pl.when(pl.col("lang_primary") != pl.col("lang_primary").filter(pl.col("script_type") == "Latin").mode().first())
          .then(pl.lit(True)).otherwise(pl.lit(False)).alias("lang_script_mismatch")
    ])
    
    lang_dist = df.group_by("lang_primary").agg(pl.len().alias("count")).sort("count", descending=True)
    logger.info("  Language distribution (top 10):")
    for row in lang_dist.head(10).iter_rows():
        logger.info(f"    {row[0]}: {row[1]} comments ({row[1]/total_rows*100:.1f}%)")
    
    code_switch_count = df.filter(pl.col("is_code_switched")).height
    logger.info(f"  Code-switched comments: {code_switch_count} ({code_switch_count/total_rows*100:.2f}%)")
    
    mismatch_count = df.filter(pl.col("lang_script_mismatch")).height
    logger.info(f"  Language-script mismatches: {mismatch_count} ({mismatch_count/total_rows*100:.2f}%)")
    
    elapsed = time.time() - start_time
    logger.info(f"Language architecture completed in {elapsed:.2f} seconds")
    
    return df

def handle_nulls_stratified(df):
    logger.info("Handling null values with stratified imputation")
    total_before = len(df)
    
    start_time = time.time()
    
    logger.info(f"  Initial rows: {total_before}")
    logger.info("  Filtering out null comment texts")
    df = df.filter(pl.col(TEXT_COL).is_not_null())
    after_text_filter = len(df)
    logger.info(f"  Removed {total_before - after_text_filter} rows with null text ({ (total_before - after_text_filter)/total_before*100:.2f}%)")
    
    logger.info("  Imputing like_count by language median")
    df = df.with_columns([
        pl.col("like_count").fill_null(
            pl.col("like_count").over("lang_primary").median()
        ),
        pl.col("reply_count").fill_null(
            pl.col("reply_count").over("lang_primary").median()
        ),
        pl.col("published_at").fill_null(pl.col("crawled_at")).alias("published_at_proxy"),
        pl.when(pl.col("published_at").is_null()).then(pl.lit(1)).otherwise(pl.lit(0)).alias("time_uncertainty_flag")
    ])
    
    missing_likes = df.filter(pl.col("like_count").is_null()).height
    missing_replies = df.filter(pl.col("reply_count").is_null()).height
    logger.info(f"  Remaining nulls after imputation - like_count: {missing_likes}, reply_count: {missing_replies}")
    
    time_uncertain = df.filter(pl.col("time_uncertainty_flag") == 1).height
    logger.info(f"  Time uncertainty flags set for {time_uncertain} rows ({time_uncertain/len(df)*100:.2f}%)")
    
    elapsed = time.time() - start_time
    logger.info(f"Null handling completed in {elapsed:.2f} seconds. Final rows: {len(df)}")
    
    return df

def compute_multilingual_descriptive_stats(df):
    logger.info("Computing multilingual descriptive statistics")
    
    start_time = time.time()
    
    logger.info("  Filtering languages with >1000 comments for stable statistics")
    lang_counts = df.group_by("lang_primary").agg(pl.len().alias("count")).filter(pl.col("count") > 1000)
    valid_langs = lang_counts["lang_primary"].to_list()
    logger.info(f"  Languages meeting threshold: {len(valid_langs)}")
    for lang in valid_langs[:10]:
        count = lang_counts.filter(pl.col("lang_primary") == lang)["count"][0]
        logger.info(f"    {lang}: {count} comments")
    
    df_filtered = df.filter(pl.col("lang_primary").is_in(valid_langs))
    logger.info(f"  Filtered dataset size: {len(df_filtered)} rows")
    
    logger.info("  Aggregating statistics by language")
    stats = df_filtered.group_by("lang_primary").agg([
        pl.col("char_count").mean().alias("avg_chars"),
        pl.col("word_count").mean().alias("avg_words"),
        (pl.col("emoji_count") / pl.col("char_count").clip(1, None) * 100).mean().alias("emoji_density"),
        pl.col("uppercase_ratio").mean().alias("avg_uppercase"),
        pl.col("exclamation_count").mean().alias("avg_exclamation"),
        pl.col("question_count").mean().alias("avg_question"),
        pl.col("like_count").mean().alias("avg_likes"),
        pl.col("reply_count").mean().alias("avg_replies")
    ]).sort("avg_likes", descending=True)
    
    logger.info(f"  Generated statistics for {len(stats)} languages")
    logger.info("  Top 5 languages by average likes:")
    for i in range(min(5, len(stats))):
        row = stats.row(i)
        logger.info(f"    {row[0]}: avg_likes={row[7]:.2f}, avg_chars={row[1]:.1f}, emoji_density={row[3]:.3f}%")
    
    logger.info("  Generating visualization: multilingual descriptive statistics")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.barplot(data=stats.to_pandas(), x="lang_primary", y="avg_chars", ax=axes[0, 0], palette="mako")
    axes[0, 0].set_title("Average Character Count by Language")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=stats.to_pandas(), x="lang_primary", y="emoji_density", ax=axes[0, 1], palette="flare")
    axes[0, 1].set_title("Emoji Density (per 100 chars) by Language")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=stats.to_pandas(), x="lang_primary", y="avg_likes", ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title("Average Like Count by Language")
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    corr_matrix = df_filtered.select(NUMERIC_COLS + ["lang_primary"]).to_pandas().pivot_table(index="lang_primary", values=NUMERIC_COLS, aggfunc="mean").corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=axes[1, 1], fmt=".2f")
    axes[1, 1].set_title("Cross-Lingual Feature Correlation Matrix")
    
    plt.tight_layout()
    plt.savefig("output_data/img/p0_multilingual_descriptive_stats.png", dpi=300)
    plt.close()
    
    logger.info("  Saved visualization to output_data/img/p0_multilingual_descriptive_stats.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Multilingual descriptive statistics completed in {elapsed:.2f} seconds")
    
    return stats

def perform_dimensionality_reduction(embeddings, lang_labels, sample_size=10000):
    logger.info("Performing dimensionality reduction on embeddings")
    
    start_time = time.time()
    
    np.random.seed(42)
    n_samples = min(sample_size, len(lang_labels))
    logger.info(f"  Sampling {n_samples} points from {len(lang_labels)} total")
    
    idx = np.random.choice(len(lang_labels), n_samples, replace=False)
    lang_sample = np.array(lang_labels)[idx]
    
    logger.info("  Sample language distribution:")
    unique, counts = np.unique(lang_sample, return_counts=True)
    for lang, count in zip(unique[:5], counts[:5]):
        logger.info(f"    {lang}: {count} ({count/n_samples*100:.1f}%)")
    
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    for i, (name, emb) in enumerate(embeddings.items()):
        logger.info(f"  Processing embedding: {name} (shape: {emb.shape})")
        emb_sample = emb[idx]
        
        logger.info("    Applying PCA (95% variance explained)")
        pca = PCA(n_components=0.95)
        pca.fit(emb_sample)
        n_components = pca.n_components_
        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f"    PCA selected {n_components} components explaining {explained_var*100:.2f}% of variance")
        results[f"{name}_pca_var"] = pca.explained_variance_ratio_
        
        logger.info("    Applying UMAP (n_neighbors=30, min_dist=0.1, metric=cosine)")
        reducer_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=2, random_state=42)
        emb_umap = reducer_umap.fit_transform(emb_sample)
        logger.info(f"    UMAP projection completed, shape: {emb_umap.shape}")
        results[f"{name}_umap"] = emb_umap
        
        logger.info(f"    Generating scatter plot for {name}")
        sns.scatterplot(x=emb_umap[:, 0], y=emb_umap[:, 1], hue=lang_sample, palette="tab20", s=15, alpha=0.7, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f"UMAP Projection: {name}")
        axes[i//2, i%2].legend([], [], frameon=False)
        
        progress_pct = (i + 1) / len(embeddings) * 100
        logger.info(f"    Progress: {progress_pct:.1f}% complete for dimensionality reduction")
    
    plt.tight_layout()
    plt.savefig("output_data/img/p1_embedding_umap_projections.png", dpi=300)
    plt.close()
    logger.info("  Saved UMAP projections to output_data/img/p1_embedding_umap_projections.png")
    
    logger.info("  Computing PCA cumulative variance plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, emb in embeddings.items():
        emb_sample = emb[idx]
        pca = PCA().fit(emb_sample)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(cumsum, label=name)
        logger.info(f"    {name}: {cumsum[10]*100:.1f}% variance in first 10 components, {cumsum[50]*100:.1f}% in first 50")
    
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Cumulative Variance by Embedding Type")
    ax.legend()
    plt.savefig("output_data/img/p1_pca_cumulative_variance.png", dpi=300)
    plt.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Dimensionality reduction completed in {elapsed:.2f} seconds")
    
    return results

def verify_cross_lingual_alignment(embeddings, df, lang_labels):
    logger.info("Verifying cross-lingual alignment between embedding spaces")
    
    start_time = time.time()
    
    logger.info("  Identifying anchor points using emoji-containing comments")
    anchor_mask = df.filter(pl.col("emoji_count") > 0).select(pl.col("comment_text").str.contains(r"[😀-🙏]")).to_numpy().flatten()
    anchor_indices = np.where(anchor_mask)[0]
    logger.info(f"  Found {len(anchor_indices)} anchor points with emojis")
    
    if len(anchor_indices) < 100:
        logger.warning(f"  Insufficient anchor points ({len(anchor_indices)} < 100). Returning default scores.")
        return {"alignment_score": 0.0, "clsc": 0.0}
    
    np.random.seed(42)
    sample_size = min(500, len(anchor_indices))
    sample_idx = np.random.choice(anchor_indices, sample_size, replace=False)
    logger.info(f"  Sampling {sample_size} anchor points for alignment")
    
    emb_base = embeddings["embedding"][sample_idx]
    emb_ft = embeddings["embedding_ft"][sample_idx]
    logger.info(f"  Base embedding shape: {emb_base.shape}, FT embedding shape: {emb_ft.shape}")
    
    logger.info("  Applying orthogonal Procrustes alignment")
    R, s = orthogonal_procrustes(emb_base, emb_ft)
    aligned_ft = emb_ft @ R
    logger.info(f"  Procrustes rotation matrix shape: {R.shape}")
    
    logger.info("  Computing similarity metrics")
    sim_before = np.diag(cosine_similarity(emb_base, emb_ft)).mean()
    sim_after = np.diag(cosine_similarity(emb_base, aligned_ft)).mean()
    
    improvement = (sim_after - sim_before) / sim_before * 100
    logger.info(f"  Similarity before alignment: {sim_before:.4f}")
    logger.info(f"  Similarity after alignment: {sim_after:.4f}")
    logger.info(f"  Improvement: {improvement:.1f}%")
    
    logger.info("  Generating alignment visualization")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(np.diag(cosine_similarity(emb_base, emb_ft)), bins=30, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"Cross-Lingual Similarity Before Alignment (Mean: {sim_before:.3f})")
    axes[0].set_xlabel("Cosine Similarity")
    
    sns.histplot(np.diag(cosine_similarity(emb_base, aligned_ft)), bins=30, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title(f"Cross-Lingual Similarity After Procrustes (Mean: {sim_after:.3f})")
    axes[1].set_xlabel("Cosine Similarity")
    
    plt.tight_layout()
    plt.savefig("output_data/img/p1_cross_lingual_alignment.png", dpi=300)
    plt.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Cross-lingual alignment verification completed in {elapsed:.2f} seconds")
    
    return {"sim_before": sim_before, "sim_after": sim_after, "alignment_matrix": R}

def estimate_local_intrinsic_dimensionality(embeddings, lang_labels, k=20):
    logger.info(f"Estimating Local Intrinsic Dimensionality (LID) with k={k}")
    
    start_time = time.time()
    
    results = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, (name, emb) in enumerate(embeddings.items()):
        logger.info(f"  Processing {name} (shape: {emb.shape})")
        start_emb = time.time()
        
        nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(emb)
        distances, _ = nbrs.kneighbors(emb)
        distances = distances[:, 1:]
        
        lid_scores = -1.0 / np.mean(np.log(distances[:, -1:] / distances[:, :-1] + 1e-8), axis=1)
        lid_scores = np.clip(lid_scores, 0, 100)
        results[name] = lid_scores
        
        logger.info(f"    LID statistics - mean: {lid_scores.mean():.2f}, std: {lid_scores.std():.2f}, median: {np.median(lid_scores):.2f}")
        logger.info(f"    LID range: [{lid_scores.min():.2f}, {lid_scores.max():.2f}]")
        
        sns.kdeplot(lid_scores, label=name, ax=ax, fill=True, alpha=0.3)
        
        emb_elapsed = time.time() - start_emb
        logger.info(f"    Completed in {emb_elapsed:.2f} seconds")
        progress_pct = (idx + 1) / len(embeddings) * 100
        logger.info(f"    Overall LID progress: {progress_pct:.1f}%")
    
    ax.set_xlabel("Local Intrinsic Dimensionality (LID)")
    ax.set_ylabel("Density")
    ax.set_title("LID Distribution Across Embedding Spaces")
    ax.legend()
    plt.savefig("output_data/img/p1_lid_distributions.png", dpi=300)
    plt.close()
    
    elapsed = time.time() - start_time
    logger.info(f"LID estimation completed in {elapsed:.2f} seconds")
    
    return results

def compute_persistent_homology(embeddings, sample_size=2000):
    logger.info(f"Computing persistent homology with sample size {sample_size}")
    
    start_time = time.time()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (name, emb) in enumerate(embeddings.items()):
        logger.info(f"  Processing {name} for persistent homology")
        np.random.seed(42)
        
        actual_sample = min(sample_size, len(emb))
        idx = np.random.choice(len(emb), actual_sample, replace=False)
        emb_sample = emb[idx]
        
        logger.info(f"    Sampled {actual_sample} points from {len(emb)} total")
        logger.info("    Computing Rips filtration up to dimension 1")
        
        start_rips = time.time()
        diagrams = ripser.ripser(emb_sample, maxdim=1)["dgms"]
        rips_time = time.time() - start_rips
        
        logger.info(f"    Homology computation took {rips_time:.2f} seconds")
        logger.info(f"    H0 persistent intervals: {len(diagrams[0])}")
        logger.info(f"    H1 persistent intervals: {len(diagrams[1])}")
        
        persim.plot_diagrams(diagrams, show=False, ax=axes[i])
        axes[i].set_title(f"Persistent Homology: {name}")
        
        progress_pct = (i + 1) / len(embeddings) * 100
        logger.info(f"    Progress: {progress_pct:.1f}% complete")
    
    plt.tight_layout()
    plt.savefig("output_data/img/p1_persistent_homology.png", dpi=300)
    plt.close()
    logger.info("  Saved persistent homology diagrams to output_data/img/p1_persistent_homology.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Persistent homology computation completed in {elapsed:.2f} seconds")
    
    return {"diagrams_computed": True}

def perform_semantic_clustering(embeddings, df, lang_labels):
    logger.info("Performing semantic clustering using UMAP + HDBSCAN")
    
    start_time = total_start = time.time()
    
    emb_ft = embeddings["embedding_ft"]
    logger.info(f"  Using fasttext embeddings with shape {emb_ft.shape}")
    
    logger.info("  Step 1/4: Reducing dimensionality with UMAP (10 components)")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=10, random_state=42)
    
    start_umap = time.time()
    emb_reduced = reducer.fit_transform(emb_ft)
    umap_time = time.time() - start_umap
    logger.info(f"  UMAP reduction completed in {umap_time:.2f} seconds")
    logger.info(f"  Reduced embedding shape: {emb_reduced.shape}")
    
    logger.info("  Step 2/4: Applying HDBSCAN clustering")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10, metric="euclidean", cluster_selection_method="eom")
    
    start_hdbscan = time.time()
    labels = clusterer.fit_predict(emb_reduced)
    hdbscan_time = time.time() - start_hdbscan
    logger.info(f"  HDBSCAN completed in {hdbscan_time:.2f} seconds")
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    logger.info(f"  Found {n_clusters} clusters with {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")
    
    cluster_sizes = Counter(labels)
    logger.info("  Top 10 cluster sizes:")
    for label, size in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
        if label == -1:
            logger.info(f"    Noise (-1): {size} points")
        else:
            logger.info(f"    Cluster {label}: {size} points")
    
    logger.info("  Step 3/4: Adding cluster labels to dataframe")
    df_clustered = df.with_columns([
        pl.Series("cluster_label", labels),
        pl.Series("lang_primary", lang_labels)
    ])
    
    logger.info("  Step 4/4: Computing cluster statistics")
    cluster_stats = df_clustered.group_by("cluster_label").agg([
        pl.len().alias("size"),
        pl.col("like_count").mean().alias("avg_likes"),
        pl.col("reply_count").mean().alias("avg_replies"),
        pl.col("lang_primary").mode().first().alias("dominant_lang")
    ]).filter(pl.col("cluster_label") != -1).sort("size", descending=True)
    
    logger.info(f"  Generated statistics for {len(cluster_stats)} non-noise clusters")
    
    top_clusters = cluster_stats.head(10)
    logger.info("  Top 10 clusters by size:")
    for row in top_clusters.iter_rows():
        logger.info(f"    Cluster {row[0]}: size={row[1]}, avg_likes={row[2]:.2f}, dominant_lang={row[4]}")
    
    logger.info("  Generating clustering visualizations")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(x=emb_reduced[:, 0], y=emb_reduced[:, 1], hue=labels, palette="tab20", s=10, alpha=0.6, ax=axes[0])
    axes[0].set_title(f"UMAP + HDBSCAN Semantic Clusters ({n_clusters} clusters)")
    axes[0].legend([], [], frameon=False)
    
    top_clusters_df = top_clusters.to_pandas()
    sns.barplot(data=top_clusters_df, x="cluster_label", y="size", hue="dominant_lang", ax=axes[1], palette="Set2")
    axes[1].set_title("Top 10 Cluster Sizes & Dominant Language")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("output_data/img/p1_semantic_clustering.png", dpi=300)
    plt.close()
    
    total_time = time.time() - total_start
    logger.info(f"Semantic clustering completed in {total_time:.2f} seconds")
    
    return df_clustered, cluster_stats

def build_and_analyze_graphs(df, embeddings, lang_labels, sample_size=5000):
    logger.info(f"Building and analyzing comment graph with sample size {sample_size}")
    
    start_time = time.time()
    
    np.random.seed(42)
    actual_sample = min(sample_size, len(df))
    idx = np.random.choice(len(df), actual_sample, replace=False)
    
    logger.info(f"  Sampled {actual_sample} comments from {len(df)} total")
    
    texts = df.select(TEXT_COL).to_numpy().flatten()[idx]
    emb_sample = embeddings["embedding"][idx]
    lang_sample = np.array(lang_labels)[idx]
    
    logger.info(f"  Computing cosine similarity matrix for {actual_sample} points")
    start_sim = time.time()
    sim_matrix = cosine_similarity(emb_sample)
    sim_time = time.time() - start_sim
    logger.info(f"  Similarity matrix computation took {sim_time:.2f} seconds")
    logger.info(f"  Similarity matrix shape: {sim_matrix.shape}, memory: {sim_matrix.nbytes / 1024**2:.1f} MB")
    
    logger.info("  Building adjacency matrix (threshold > 0.75)")
    adj_matrix = (sim_matrix > 0.75).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    
    edge_density = adj_matrix.sum() / (actual_sample * (actual_sample - 1))
    logger.info(f"  Edge density: {edge_density:.6f} ({adj_matrix.sum()} edges)")
    
    logger.info("  Converting to NetworkX graph")
    G = nx.from_numpy_array(adj_matrix)
    
    logger.info("  Finding connected components")
    components = list(nx.connected_components(G))
    logger.info(f"  Found {len(components)} connected components")
    
    component_sizes = [len(c) for c in components]
    logger.info(f"  Component size statistics - min: {min(component_sizes)}, max: {max(component_sizes)}, mean: {np.mean(component_sizes):.1f}, median: {np.median(component_sizes):.1f}")
    
    giant_component = max(components, key=len)
    giant_size = len(giant_component)
    logger.info(f"  Giant component contains {giant_size} nodes ({giant_size/actual_sample*100:.1f}% of sample)")
    
    G_giant = G.subgraph(giant_component)
    
    logger.info("  Computing degree distribution")
    degrees = [d for n, d in G_giant.degree()]
    logger.info(f"  Degree statistics - mean: {np.mean(degrees):.2f}, std: {np.std(degrees):.2f}, max: {max(degrees)}")
    
    logger.info("  Computing clustering coefficient")
    avg_clustering = nx.average_clustering(G_giant)
    logger.info(f"  Average clustering coefficient: {avg_clustering:.4f}")
    
    logger.info("  Computing language assortativity")
    total_edges = G_giant.number_of_edges()
    lang_edges = 0
    for u, v in G_giant.edges():
        if lang_sample[u] == lang_sample[v]:
            lang_edges += 1
    
    assortativity = lang_edges / total_edges if total_edges > 0 else 0
    logger.info(f"  Language assortativity: {assortativity:.4f} ({lang_edges}/{total_edges} same-language edges)")
    
    metrics = {
        "nodes": G_giant.number_of_nodes(),
        "edges": total_edges,
        "avg_clustering": avg_clustering,
        "language_assortativity": assortativity,
        "components": len(components)
    }
    
    logger.info("  Generating graph analysis visualization")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(degrees, bins=50, kde=True, ax=axes[0], color="teal")
    axes[0].set_title(f"Comment-Comment Graph Degree Distribution (n={giant_size})")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale("log")
    
    axes[1].bar(metrics.keys(), metrics.values(), color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    axes[1].set_title("Graph Topology Metrics")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("output_data/img/p1_comment_graph_analysis.png", dpi=300)
    plt.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Graph analysis completed in {elapsed:.2f} seconds")
    
    return metrics

def run_p0_pipeline(df):
    logger.info("=" * 60)
    logger.info("STARTING P0 PIPELINE: Data Processing & Language Architecture")
    logger.info("=" * 60)
    
    step_start = time.time()
    
    df_lang = apply_language_architecture(df)
    logger.info("P0 - Step 1/3 completed: Language architecture applied")
    
    df_clean = handle_nulls_stratified(df_lang)
    logger.info("P0 - Step 2/3 completed: Null values handled")
    
    stats = compute_multilingual_descriptive_stats(df_clean)
    logger.info("P0 - Step 3/3 completed: Descriptive statistics computed")
    
    total_time = time.time() - step_start
    logger.info(f"P0 PIPELINE COMPLETED in {total_time:.2f} seconds")
    logger.info("=" * 60)
    
    return df_clean, stats

def run_p1_pipeline(df_clean, embeddings, lang_labels):
    logger.info("=" * 60)
    logger.info("STARTING P1 PIPELINE: Advanced Multilingual Analysis")
    logger.info("=" * 60)
    
    pipeline_start = time.time()
    
    logger.info("P1 - Task 1/6: Dimensionality Reduction")
    dim_results = perform_dimensionality_reduction(embeddings, lang_labels)
    logger.info("P1 - Task 1/6 completed")
    
    logger.info("P1 - Task 2/6: Cross-lingual Alignment")
    align_results = verify_cross_lingual_alignment(embeddings, df_clean, lang_labels)
    logger.info("P1 - Task 2/6 completed")
    
    logger.info("P1 - Task 3/6: Local Intrinsic Dimensionality")
    lid_results = estimate_local_intrinsic_dimensionality(embeddings, lang_labels)
    logger.info("P1 - Task 3/6 completed")
    
    logger.info("P1 - Task 4/6: Persistent Homology")
    homology_results = compute_persistent_homology(embeddings)
    logger.info("P1 - Task 4/6 completed")
    
    logger.info("P1 - Task 5/6: Semantic Clustering")
    df_clustered, cluster_stats = perform_semantic_clustering(embeddings, df_clean, lang_labels)
    logger.info("P1 - Task 5/6 completed")
    
    logger.info("P1 - Task 6/6: Graph Analysis")
    graph_metrics = build_and_analyze_graphs(df_clean, embeddings, lang_labels)
    logger.info("P1 - Task 6/6 completed")
    
    total_time = time.time() - pipeline_start
    logger.info(f"P1 PIPELINE COMPLETED in {total_time:.2f} seconds")
    logger.info("=" * 60)
    
    return {
        "dimensionality": dim_results,
        "alignment": align_results,
        "lid": lid_results,
        "homology": homology_results,
        "clustering": {"df": df_clustered, "stats": cluster_stats},
        "graph": graph_metrics
    }

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info(f"MULTILINGUAL COMMENT ANALYSIS PIPELINE STARTING at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    overall_start = time.time()
    
    df_raw, X_scaled, embeddings, scaler = load_data()
    
    logger.info("Creating temporary language labels (will be replaced by actual detection)")
    lang_labels_raw = ["en"] * len(df_raw)
    logger.info(f"Initial dummy labels created: {len(lang_labels_raw)} entries")
    
    logger.info("Executing P0 Pipeline")
    df_p0, p0_stats = run_p0_pipeline(df_raw)
    
    lang_labels = df_p0["lang_primary"].to_list()
    logger.info(f"Extracted language labels from P0 output: {len(lang_labels)} entries")
    
    unique_langs = set(lang_labels)
    logger.info(f"Unique languages detected: {len(unique_langs)}")
    for lang in sorted(unique_langs)[:10]:
        count = lang_labels.count(lang)
        logger.info(f"  {lang}: {count} ({count/len(lang_labels)*100:.1f}%)")
    
    logger.info("Executing P1 Pipeline")
    p1_results = run_p1_pipeline(df_p0, embeddings, lang_labels)
    
    overall_time = time.time() - overall_start
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE - FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    logger.info(f"Descriptive Stats Shape: {p0_stats.shape}")
    logger.info(f"Alignment Score Improvement: {p1_results['alignment']['sim_before']:.3f} -> {p1_results['alignment']['sim_after']:.3f}")
    logger.info(f"  Improvement: {(p1_results['alignment']['sim_after'] - p1_results['alignment']['sim_before']) / p1_results['alignment']['sim_before'] * 100:.1f}%")
    logger.info("Graph Metrics:")
    for key, value in p1_results['graph'].items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    logger.info("Visualizations saved to output_data/img/ directory")
    logger.info(f"Pipeline completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)