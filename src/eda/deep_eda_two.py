import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import ripser
import persim
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
    logger.info(f"Target columns: {NUMERIC_COLS + EMBEDDING_COLS + [TEXT_COL, 'published_at', 'crawled_at', 'source_query', 'post_id']}")
    
    df = pl.read_parquet(
        COMBINED_DATA_PATH,
        columns=NUMERIC_COLS + EMBEDDING_COLS + [TEXT_COL, "published_at", "crawled_at", "source_query", "post_id"]
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
        return langdetect.detect(str(text))
    except Exception:
        return "unknown"

def add_language_labels(df):
    logger.info("Adding language labels to dataframe")
    start_time = time.time()
    
    total_rows = len(df)
    logger.info(f"Processing {total_rows} comments for language detection")
    
    df = df.with_columns(
        pl.col(TEXT_COL).map_elements(detect_language, return_dtype=pl.Utf8).alias("lang_primary")
    )
    
    lang_dist = df.group_by("lang_primary").agg(pl.len().alias("count")).sort("count", descending=True)
    logger.info("Language distribution (top 10):")
    for row in lang_dist.head(10).iter_rows():
        logger.info(f"  {row[0]}: {row[1]} comments ({row[1]/total_rows*100:.1f}%)")
    
    unknown_count = df.filter(pl.col("lang_primary") == "unknown").height
    logger.info(f"Unknown language detected: {unknown_count} ({unknown_count/total_rows*100:.2f}%)")
    
    elapsed = time.time() - start_time
    logger.info(f"Language labeling completed in {elapsed:.2f} seconds")
    
    return df

def analyze_embedding_pca(embeddings):
    logger.info("Starting PCA analysis on embeddings")
    start_time = time.time()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    pca_results = {}
    
    for i, (name, emb) in enumerate(embeddings.items()):
        logger.info(f"  Computing PCA for {name} (shape: {emb.shape})")
        start_pca = time.time()
        
        pca = PCA()
        pca.fit(emb)
        pca_results[name] = pca.explained_variance_ratio_
        
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        n_95 = np.argmax(cumulative_var >= 0.95) + 1
        n_99 = np.argmax(cumulative_var >= 0.99) + 1
        
        logger.info(f"    {name}: {n_95} components explain 95% variance, {n_99} components explain 99% variance")
        logger.info(f"    First 10 components explain {cumulative_var[9]*100:.1f}% of variance")
        
        sns.lineplot(
            x=np.arange(1, len(pca.explained_variance_ratio_) + 1),
            y=cumulative_var,
            ax=axes[i],
            color=sns.color_palette()[i]
        )
        axes[i].set_title(f"Cumulative Explained Variance: {name}")
        axes[i].set_xlabel("Number of Components")
        axes[i].set_ylabel("Cumulative Variance Ratio")
        axes[i].axhline(0.95, color='red', linestyle='--', label='95% Threshold')
        axes[i].legend()
        
        elapsed_pca = time.time() - start_pca
        logger.info(f"    PCA completed in {elapsed_pca:.2f} seconds")
        progress_pct = (i + 1) / len(embeddings) * 100
        logger.info(f"    Progress: {progress_pct:.1f}%")
    
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_pca_variance.png", dpi=300)
    plt.close()
    logger.info("Saved PCA visualization to output_data/img/phase2_pca_variance.png")
    
    elapsed = time.time() - start_time
    logger.info(f"PCA analysis completed in {elapsed:.2f} seconds")
    
    return pca_results

def analyze_embedding_projections(embeddings, lang_labels, sample_size=10000):
    logger.info(f"Starting embedding projections analysis with sample size {sample_size}")
    start_time = time.time()
    
    np.random.seed(42)
    n_samples = min(sample_size, len(lang_labels))
    idx = np.random.choice(len(lang_labels), n_samples, replace=False)
    lang_sample = np.array(lang_labels)[idx]
    
    logger.info(f"Sampled {n_samples} points from {len(lang_labels)} total")
    
    unique_langs, lang_counts = np.unique(lang_sample, return_counts=True)
    logger.info("Sample language distribution:")
    for lang, count in zip(unique_langs[:5], lang_counts[:5]):
        logger.info(f"  {lang}: {count} ({count/n_samples*100:.1f}%)")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    projection_results = {}
    
    for i, (name, emb) in enumerate(embeddings.items()):
        logger.info(f"  Computing UMAP projection for {name}")
        start_umap = time.time()
        
        emb_sample = emb[idx]
        
        reducer_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=2, random_state=42)
        emb_umap = reducer_umap.fit_transform(emb_sample)
        projection_results[f"{name}_umap"] = emb_umap
        
        logger.info(f"    UMAP completed in {time.time() - start_umap:.2f} seconds")
        
        sns.scatterplot(
            x=emb_umap[:, 0], y=emb_umap[:, 1], hue=lang_sample, 
            palette="tab20", s=15, alpha=0.7, ax=axes[i], legend=False
        )
        axes[i].set_title(f"UMAP Projection (Cross-Lingual): {name}")
        
        progress_pct = (i + 1) / len(embeddings) * 100
        logger.info(f"    Overall UMAP progress: {progress_pct:.1f}%")
    
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_umap_projections.png", dpi=300)
    plt.close()
    logger.info("Saved UMAP projections to output_data/img/phase2_umap_projections.png")
    
    logger.info("  Computing t-SNE projections for selected embeddings")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, name in enumerate(["embedding", "embedding_ft"]):
        logger.info(f"  Computing t-SNE projection for {name}")
        start_tsne = time.time()
        
        emb_sample = embeddings[name][idx]
        reducer_tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42, init='pca')
        emb_tsne = reducer_tsne.fit_transform(emb_sample)
        projection_results[f"{name}_tsne"] = emb_tsne
        
        logger.info(f"    t-SNE completed in {time.time() - start_tsne:.2f} seconds")
        
        sns.scatterplot(
            x=emb_tsne[:, 0], y=emb_tsne[:, 1], hue=lang_sample, 
            palette="tab20", s=15, alpha=0.7, ax=axes[i], legend=False
        )
        axes[i].set_title(f"t-SNE Projection: {name}")
    
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_tsne_projections.png", dpi=300)
    plt.close()
    logger.info("Saved t-SNE projections to output_data/img/phase2_tsne_projections.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Embedding projections analysis completed in {elapsed:.2f} seconds")
    
    return projection_results

def cluster_embeddings_hdbscan(embeddings, lang_labels):
    logger.info("Starting HDBSCAN semantic clustering")
    start_time = time.time()
    
    emb_ft = embeddings["embedding_ft"]
    logger.info(f"  Using fasttext embeddings with shape {emb_ft.shape}")
    
    logger.info("  Step 1/3: Reducing dimensionality with UMAP (10 components)")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=10, random_state=42)
    
    start_umap = time.time()
    emb_reduced = reducer.fit_transform(emb_ft)
    umap_time = time.time() - start_umap
    logger.info(f"  UMAP reduction completed in {umap_time:.2f} seconds")
    logger.info(f"  Reduced embedding shape: {emb_reduced.shape}")
    
    logger.info("  Step 2/3: Applying HDBSCAN clustering")
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
    
    logger.info("  Step 3/3: Generating clustering visualization")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(
        x=emb_reduced[:, 0], y=emb_reduced[:, 1], hue=labels, 
        palette="tab20", s=10, alpha=0.6, ax=axes[0], legend=False
    )
    axes[0].set_title("UMAP + HDBSCAN Semantic Clusters")
    
    cluster_counts = Counter(labels)
    valid_clusters = [k for k, v in cluster_counts.items() if k != -1]
    sizes = [cluster_counts[k] for k in valid_clusters[:10]]
    sns.barplot(x=np.arange(len(sizes)), y=sizes, ax=axes[1], palette="Set2")
    axes[1].set_title("Top 10 Cluster Sizes (Excluding Noise)")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_hdbscan_clustering.png", dpi=300)
    plt.close()
    logger.info("Saved clustering visualization to output_data/img/phase2_hdbscan_clustering.png")
    
    elapsed = time.time() - start_time
    logger.info(f"HDBSCAN clustering completed in {elapsed:.2f} seconds")
    
    return labels, emb_reduced

def characterize_clusters(df, cluster_labels, lang_labels):
    logger.info("Characterizing clusters with descriptive statistics")
    start_time = time.time()
    
    logger.info("  Adding cluster labels to dataframe")
    df_clustered = df.with_columns([
        pl.Series("cluster_label", cluster_labels),
        pl.Series("lang_primary", lang_labels)
    ])
    
    logger.info("  Computing cluster statistics")
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
        logger.info(f"    Cluster {row[0]}: size={row[1]}, avg_likes={row[2]:.2f}, avg_replies={row[3]:.2f}, dominant_lang={row[4]}")
    
    logger.info("  Generating cluster characterization visualization")
    top_clusters_df = top_clusters.to_pandas()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_clusters_df, x="cluster_label", y="size", hue="dominant_lang", palette="Set3")
    ax.set_title("Top 10 Clusters: Size and Dominant Language")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_cluster_characterization.png", dpi=300)
    plt.close()
    logger.info("Saved cluster characterization to output_data/img/phase2_cluster_characterization.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Cluster characterization completed in {elapsed:.2f} seconds")
    
    return cluster_stats

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
        logger.info(f"    Progress: {progress_pct:.1f}%")
    
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_persistent_homology.png", dpi=300)
    plt.close()
    logger.info("Saved persistent homology diagrams to output_data/img/phase2_persistent_homology.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Persistent homology computation completed in {elapsed:.2f} seconds")
    
    return {"diagrams_computed": True}

def build_per_language_gow(df, lang_labels, top_langs=3):
    logger.info(f"Building per-language Graph of Words for top {top_langs} languages")
    start_time = time.time()
    
    lang_counts = Counter(lang_labels)
    valid_langs = [lang for lang, count in lang_counts.most_common(top_langs) if lang != "unknown"]
    
    logger.info(f"Selected languages: {valid_langs}")
    for lang in valid_langs:
        logger.info(f"  {lang}: {lang_counts[lang]} comments")
    
    gow_results = {}
    fig, axes = plt.subplots(1, len(valid_langs), figsize=(6 * len(valid_langs), 5))
    if len(valid_langs) == 1:
        axes = [axes]
    
    for i, lang in enumerate(valid_langs):
        logger.info(f"  Building GoW for {lang}")
        start_lang = time.time()
        
        texts = [str(t) for t in df.filter(pl.col("lang_primary") == lang).select(TEXT_COL).to_numpy().flatten()]
        logger.info(f"    Processing {len(texts)} texts")
        
        vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=500)
        X = vectorizer.fit_transform(texts)
        logger.info(f"    Vocabulary size: {X.shape[1]}")
        
        cooccurrence = (X.T * X).toarray()
        np.fill_diagonal(cooccurrence, 0)
        
        G = nx.from_numpy_array(cooccurrence)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        degrees = [d for n, d in G.degree()]
        gow_results[lang] = {
            "nodes": G.number_of_nodes(), 
            "edges": G.number_of_edges(), 
            "avg_degree": np.mean(degrees)
        }
        
        logger.info(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, avg degree {np.mean(degrees):.2f}")
        
        sns.histplot(degrees, bins=30, kde=True, ax=axes[i], color="teal")
        axes[i].set_title(f"Degree Distribution: {lang}")
        axes[i].set_xlabel("Degree")
        axes[i].set_yscale("log")
        
        elapsed_lang = time.time() - start_lang
        logger.info(f"    {lang} GoW completed in {elapsed_lang:.2f} seconds")
    
    plt.tight_layout()
    plt.savefig("output_data/img/phase3_per_language_gow.png", dpi=300)
    plt.close()
    logger.info("Saved per-language GoW visualizations to output_data/img/phase3_per_language_gow.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Per-language GoW analysis completed in {elapsed:.2f} seconds")
    
    return gow_results

def build_cross_lingual_similarity_graph(embeddings, lang_labels, sample_size=5000):
    logger.info(f"Building cross-lingual similarity graph with sample size {sample_size}")
    start_time = time.time()
    
    np.random.seed(42)
    actual_sample = min(sample_size, len(lang_labels))
    idx = np.random.choice(len(lang_labels), actual_sample, replace=False)
    
    emb_sample = embeddings["embedding_ft"][idx]
    lang_sample = np.array(lang_labels)[idx]
    
    logger.info(f"  Sampled {actual_sample} points")
    logger.info("  Computing cosine similarity matrix")
    
    start_sim = time.time()
    sim_matrix = cosine_similarity(emb_sample)
    np.fill_diagonal(sim_matrix, 0)
    sim_time = time.time() - start_sim
    logger.info(f"  Similarity matrix computation took {sim_time:.2f} seconds")
    
    threshold = np.percentile(sim_matrix[sim_matrix > 0], 90)
    logger.info(f"  Using threshold {threshold:.4f} (90th percentile of positive similarities)")
    
    adj_matrix = (sim_matrix > threshold).astype(int)
    edge_count = adj_matrix.sum()
    logger.info(f"  Created {edge_count} edges (density: {edge_count/(actual_sample*actual_sample):.6f})")
    
    G = nx.from_numpy_array(adj_matrix)
    
    cross_edges = 0
    total_edges = G.number_of_edges()
    for u, v in G.edges():
        if lang_sample[u] != lang_sample[v]:
            cross_edges += 1
    
    logger.info(f"  Total edges: {total_edges}")
    logger.info(f"  Cross-lingual edges: {cross_edges}")
    logger.info(f"  Cross-lingual ratio: {cross_edges/total_edges*100:.2f}%")
    
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": total_edges,
        "cross_lingual_edges": cross_edges,
        "cross_lingual_ratio": cross_edges / total_edges if total_edges > 0 else 0.0
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Total Edges", "Cross-Lingual Edges"], [total_edges, cross_edges], color=["#4C72B0", "#DD8452"])
    ax.set_title("Cross-Lingual Semantic Similarity Graph Edges")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("output_data/img/phase3_cross_lingual_graph.png", dpi=300)
    plt.close()
    logger.info("Saved cross-lingual graph visualization to output_data/img/phase3_cross_lingual_graph.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Cross-lingual graph construction completed in {elapsed:.2f} seconds")
    
    return G, metrics

def build_comment_knn_graph(embeddings, lang_labels, k=15, sample_size=10000):
    logger.info(f"Building comment k-NN graph with k={k}, sample size {sample_size}")
    start_time = time.time()
    
    np.random.seed(42)
    actual_sample = min(sample_size, len(lang_labels))
    idx = np.random.choice(len(lang_labels), actual_sample, replace=False)
    
    emb_sample = embeddings["embedding"][idx]
    np.array(lang_labels)[idx]
    
    logger.info(f"  Sampled {actual_sample} points")
    logger.info(f"  Computing {k} nearest neighbors")
    
    start_knn = time.time()
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(emb_sample)
    distances, indices = nbrs.kneighbors(emb_sample)
    knn_time = time.time() - start_knn
    logger.info(f"  k-NN computation took {knn_time:.2f} seconds")
    
    rows = np.repeat(np.arange(actual_sample), k)
    cols = indices[:, 1:].flatten()
    data = np.ones_like(rows, dtype=float)
    
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(actual_sample, actual_sample))
    adj_matrix = adj_matrix.maximum(adj_matrix.T)
    
    G = nx.from_scipy_sparse_array(adj_matrix)
    
    degrees = [d for n, d in G.degree()]
    components = list(nx.connected_components(G))
    largest_component_size = len(max(components, key=len)) if components else 0
    
    logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"  Average degree: {np.mean(degrees):.2f}")
    logger.info(f"  Number of connected components: {len(components)}")
    logger.info(f"  Largest component size: {largest_component_size} ({largest_component_size/actual_sample*100:.1f}%)")
    
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": np.mean(degrees),
        "num_components": len(components),
        "largest_component_size": largest_component_size
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(degrees, bins=50, kde=True, ax=ax, color="purple")
    ax.set_title(f"Comment k-NN Graph Degree Distribution (k={k})")
    ax.set_xlabel("Degree")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("output_data/img/phase3_comment_knn_graph.png", dpi=300)
    plt.close()
    logger.info("Saved k-NN graph visualization to output_data/img/phase3_comment_knn_graph.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Comment k-NN graph construction completed in {elapsed:.2f} seconds")
    
    return G, metrics

def compute_graph_metrics(G):
    logger.info("Computing advanced graph metrics")
    start_time = time.time()
    
    if G.number_of_nodes() == 0:
        logger.warning("Graph has no nodes, returning empty metrics")
        return {}
    
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_largest = G.subgraph(largest_cc)
    
    logger.info(f"  Largest component: {len(largest_cc)} nodes")
    
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G_largest),
        "num_components": len(components),
        "largest_component_size": len(largest_cc)
    }
    
    logger.info(f"  Graph density: {metrics['density']:.6f}")
    logger.info(f"  Average clustering coefficient: {metrics['avg_clustering']:.4f}")
    
    try:
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(G_largest)
        logger.info(f"  Average shortest path length: {metrics['avg_shortest_path']:.2f}")
    except Exception as e:
        metrics["avg_shortest_path"] = float('inf')
        logger.info(f"  Could not compute average shortest path: {str(e)}")
    
    elapsed = time.time() - start_time
    logger.info(f"Graph metrics computation completed in {elapsed:.2f} seconds")
    
    return metrics

def spectral_clustering_laplacian(embeddings, lang_labels, n_clusters=10, sample_size=10000):
    logger.info(f"Performing spectral clustering with {n_clusters} clusters, sample size {sample_size}")
    start_time = time.time()
    
    np.random.seed(42)
    actual_sample = min(sample_size, len(lang_labels))
    idx = np.random.choice(len(lang_labels), actual_sample, replace=False)
    
    emb_sample = embeddings["embedding_ft"][idx]
    lang_sample = np.array(lang_labels)[idx]
    
    logger.info(f"  Sampled {actual_sample} points")
    logger.info("  Building k-NN graph with k=15")
    
    k = 15
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(emb_sample)
    distances, indices = nbrs.kneighbors(emb_sample)
    
    rows = np.repeat(np.arange(actual_sample), k)
    cols = indices[:, 1:].flatten()
    data = np.exp(-distances[:, 1:].flatten() / 0.5)
    
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(actual_sample, actual_sample))
    adj_matrix = adj_matrix.maximum(adj_matrix.T)
    
    logger.info("  Computing Laplacian matrix")
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    D = diags(degrees)
    L = D - adj_matrix
    
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees + 1e-8))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    
    logger.info(f"  Computing {n_clusters} smallest eigenvalues and eigenvectors")
    start_eig = time.time()
    eigenvalues, eigenvectors = eigsh(L_sym, k=n_clusters, which='SM')
    eig_time = time.time() - start_eig
    logger.info(f"  Eigenvalue computation took {eig_time:.2f} seconds")
    logger.info(f"  Eigenvalues: {eigenvalues[:5]}")
    
    logger.info("  Applying spectral clustering")
    start_cluster = time.time()
    spectral_clusters = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        random_state=42
    ).fit_predict(adj_matrix.toarray())
    cluster_time = time.time() - start_cluster
    logger.info(f"  Spectral clustering took {cluster_time:.2f} seconds")
    
    cluster_sizes = Counter(spectral_clusters)
    logger.info("  Spectral cluster sizes:")
    for cluster_id in sorted(cluster_sizes.keys()):
        logger.info(f"    Cluster {cluster_id}: {cluster_sizes[cluster_id]} points ({cluster_sizes[cluster_id]/actual_sample*100:.1f}%)")
    
    logger.info("  Generating spectral clustering visualizations")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.scatterplot(
        x=eigenvectors[:, 1], y=eigenvectors[:, 2], hue=spectral_clusters, 
        palette="tab10", s=15, alpha=0.7, ax=axes[0], legend=False
    )
    axes[0].set_title("Spectral Embedding (Eigenvectors 1 & 2)")
    
    cluster_lang_counts = Counter(zip(spectral_clusters, lang_sample))
    cluster_df = pl.DataFrame([
        {"cluster": c, "lang": l, "count": count} 
        for (c, l), count in cluster_lang_counts.items()
    ])
    
    pivot_df = cluster_df.pivot(index="cluster", columns="lang", values="count", aggregate_function="sum").fill_null(0).to_pandas()
    pivot_df.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
    axes[1].set_title("Language Composition of Spectral Clusters")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("output_data/img/phase3_spectral_clustering.png", dpi=300)
    plt.close()
    logger.info("Saved spectral clustering visualization to output_data/img/phase3_spectral_clustering.png")
    
    elapsed = time.time() - start_time
    logger.info(f"Spectral clustering completed in {elapsed:.2f} seconds")
    
    return spectral_clusters, eigenvalues

def run_phase_2(df, embeddings, lang_labels):
    logger.info("=" * 60)
    logger.info("STARTING PHASE 2: Embedding Analysis & Clustering")
    logger.info("=" * 60)
    
    phase_start = time.time()
    
    logger.info("Phase 2 - Task 1/5: PCA Analysis")
    pca_results = analyze_embedding_pca(embeddings)
    logger.info("Phase 2 - Task 1/5 completed")
    
    logger.info("Phase 2 - Task 2/5: Embedding Projections")
    projection_results = analyze_embedding_projections(embeddings, lang_labels)
    logger.info("Phase 2 - Task 2/5 completed")
    
    logger.info("Phase 2 - Task 3/5: HDBSCAN Clustering")
    cluster_labels, emb_reduced = cluster_embeddings_hdbscan(embeddings, lang_labels)
    logger.info("Phase 2 - Task 3/5 completed")
    
    logger.info("Phase 2 - Task 4/5: Cluster Characterization")
    cluster_stats = characterize_clusters(df, cluster_labels, lang_labels)
    logger.info("Phase 2 - Task 4/5 completed")
    
    logger.info("Phase 2 - Task 5/5: Persistent Homology")
    homology_results = compute_persistent_homology(embeddings)
    logger.info("Phase 2 - Task 5/5 completed")
    
    total_time = time.time() - phase_start
    logger.info(f"PHASE 2 COMPLETED in {total_time:.2f} seconds")
    logger.info("=" * 60)
    
    return {
        "pca": pca_results,
        "projections": projection_results,
        "cluster_labels": cluster_labels,
        "cluster_stats": cluster_stats,
        "homology": homology_results
    }

def run_phase_3(df, embeddings, lang_labels):
    logger.info("=" * 60)
    logger.info("STARTING PHASE 3: Graph-Based & Spectral Analysis")
    logger.info("=" * 60)
    
    phase_start = time.time()
    
    logger.info("Phase 3 - Task 1/5: Per-Language Graph of Words")
    gow_results = build_per_language_gow(df, lang_labels)
    logger.info("Phase 3 - Task 1/5 completed")
    
    logger.info("Phase 3 - Task 2/5: Cross-Lingual Similarity Graph")
    cross_lingual_G, cross_lingual_metrics = build_cross_lingual_similarity_graph(embeddings, lang_labels)
    logger.info("Phase 3 - Task 2/5 completed")
    
    logger.info("Phase 3 - Task 3/5: Comment k-NN Graph")
    knn_G, knn_metrics = build_comment_knn_graph(embeddings, lang_labels)
    logger.info("Phase 3 - Task 3/5 completed")
    
    logger.info("Phase 3 - Task 4/5: Advanced Graph Metrics")
    graph_metrics = compute_graph_metrics(knn_G)
    logger.info("Phase 3 - Task 4/5 completed")
    
    logger.info("Phase 3 - Task 5/5: Spectral Clustering")
    spectral_clusters, eigenvalues = spectral_clustering_laplacian(embeddings, lang_labels)
    logger.info("Phase 3 - Task 5/5 completed")
    
    total_time = time.time() - phase_start
    logger.info(f"PHASE 3 COMPLETED in {total_time:.2f} seconds")
    logger.info("=" * 60)
    
    return {
        "gow": gow_results,
        "cross_lingual_graph": {"graph": cross_lingual_G, "metrics": cross_lingual_metrics},
        "knn_graph": {"graph": knn_G, "metrics": knn_metrics},
        "graph_metrics": graph_metrics,
        "spectral": {"clusters": spectral_clusters, "eigenvalues": eigenvalues}
    }

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info(f"MULTILINGUAL COMMENT ANALYSIS PIPELINE (Phase 2 & 3) STARTING at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    overall_start = time.time()
    
    logger.info("Loading data...")
    df_raw, X_scaled, embeddings, scaler = load_data()
    
    logger.info("Adding language labels...")
    df_lang = add_language_labels(df_raw)
    lang_labels = df_lang["lang_primary"].to_list()
    logger.info(f"Extracted {len(lang_labels)} language labels")
    
    unique_langs = set(lang_labels)
    logger.info(f"Unique languages detected: {len(unique_langs)}")
    for lang in sorted(unique_langs)[:10]:
        count = lang_labels.count(lang)
        logger.info(f"  {lang}: {count} ({count/len(lang_labels)*100:.1f}%)")
    
    logger.info("Running Phase 2 analysis...")
    phase_2_results = run_phase_2(df_lang, embeddings, lang_labels)
    
    logger.info("Running Phase 3 analysis...")
    phase_3_results = run_phase_3(df_lang, embeddings, lang_labels)
    
    overall_time = time.time() - overall_start
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE - FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    logger.info(f"PCA Variance Explained (embedding_ft first 50 components): {np.sum(phase_2_results['pca']['embedding_ft'][:50]):.4f}")
    logger.info(f"Number of HDBSCAN clusters: {len(set(phase_2_results['cluster_labels'])) - (1 if -1 in phase_2_results['cluster_labels'] else 0)}")
    logger.info(f"Noise points in HDBSCAN: {np.sum(np.array(phase_2_results['cluster_labels']) == -1)}")
    logger.info(f"Cross-Lingual Graph Ratio: {phase_3_results['cross_lingual_graph']['metrics']['cross_lingual_ratio']:.4f}")
    logger.info("KNN Graph Metrics:")
    for key, value in phase_3_results['graph_metrics'].items():
        logger.info(f"  {key}: {value}")
    logger.info(f"Spectral Clusters: {len(set(phase_3_results['spectral']['clusters']))}")
    logger.info("=" * 80)
    logger.info("Visualizations saved to output_data/img/ directory")
    logger.info(f"Pipeline completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)