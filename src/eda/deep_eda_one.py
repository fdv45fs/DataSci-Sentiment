import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import csr_matrix, tril
import umap
import hdbscan
import networkx as nx
import igraph as ig
import ripser
import persim
import re
import langdetect
from collections import Counter
import warnings
import os

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

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
    df = pl.read_parquet(
        COMBINED_DATA_PATH,
        columns=NUMERIC_COLS + EMBEDDING_COLS + [TEXT_COL, "published_at", "crawled_at", "source_query"]
    )
    X_numeric = df.select(NUMERIC_COLS).to_numpy().astype(np.float32)
    embeddings = {}
    for col in EMBEDDING_COLS:
        embeddings[col] = np.stack(df[col].to_numpy())
    X_all = np.hstack([X_numeric] + list(embeddings.values()))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    return df, X_scaled, embeddings, scaler

def detect_language(text):
    try:
        return langdetect.detect(text)
    except Exception as e:
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
        except Exception as e:
            langs.append("unknown")
    if len(langs) < 2:
        return 0.0
    switches = sum(1 for i in range(len(langs)-1) if langs[i] != langs[i+1] and langs[i] != "unknown" and langs[i+1] != "unknown")
    return switches / (len(langs) - 1)

def apply_language_architecture(df):
    df = df.with_columns([
        pl.col(TEXT_COL).map_elements(detect_language, return_dtype=pl.Utf8).alias("lang_primary"),
        pl.col(TEXT_COL).map_elements(detect_script, return_dtype=pl.Utf8).alias("script_type"),
        pl.col(TEXT_COL).map_elements(detect_code_switching, return_dtype=pl.Float64).alias("code_switch_ratio")
    ])
    df = df.with_columns([
        pl.when(pl.col("code_switch_ratio") > 0.3).then(pl.lit(True)).otherwise(pl.lit(False)).alias("is_code_switched"),
        pl.when(pl.col("lang_primary") != pl.col("lang_primary").filter(pl.col("script_type") == "Latin").mode().first())
          .then(pl.lit(True)).otherwise(pl.lit(False)).alias("lang_script_mismatch")
    ])
    return df

def handle_nulls_stratified(df):
    df = df.filter(pl.col(TEXT_COL).is_not_null())
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
    return df

def compute_multilingual_descriptive_stats(df):
    lang_counts = df.group_by("lang_primary").agg(pl.len().alias("count")).filter(pl.col("count") > 1000)
    valid_langs = lang_counts["lang_primary"].to_list()
    df_filtered = df.filter(pl.col("lang_primary").is_in(valid_langs))
    
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
    return stats

def perform_dimensionality_reduction(embeddings, lang_labels, sample_size=10000):
    np.random.seed(42)
    n_samples = min(sample_size, len(lang_labels))
    idx = np.random.choice(len(lang_labels), n_samples, replace=False)
    lang_sample = np.array(lang_labels)[idx]
    
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    for i, (name, emb) in enumerate(embeddings.items()):
        emb_sample = emb[idx]
        
        pca = PCA(n_components=0.95)
        pca.fit(emb_sample)
        results[f"{name}_pca_var"] = pca.explained_variance_ratio_
        
        reducer_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=2, random_state=42)
        emb_umap = reducer_umap.fit_transform(emb_sample)
        results[f"{name}_umap"] = emb_umap
        
        sns.scatterplot(x=emb_umap[:, 0], y=emb_umap[:, 1], hue=lang_sample, palette="tab20", s=15, alpha=0.7, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f"UMAP Projection: {name}")
        axes[i//2, i%2].legend([], [], frameon=False)
        
    plt.tight_layout()
    plt.savefig("output_data/img/p1_embedding_umap_projections.png", dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, emb in embeddings.items():
        emb_sample = emb[idx]
        pca = PCA().fit(emb_sample)
        ax.plot(np.cumsum(pca.explained_variance_ratio_), label=name)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Cumulative Variance by Embedding Type")
    ax.legend()
    plt.savefig("output_data/img/p1_pca_cumulative_variance.png", dpi=300)
    plt.close()
    
    return results

def verify_cross_lingual_alignment(embeddings, df, lang_labels):
    anchor_mask = df.filter(pl.col("emoji_count") > 0).select(pl.col("comment_text").str.contains(r"[😀-🙏]")).to_numpy().flatten()
    anchor_indices = np.where(anchor_mask)[0]
    if len(anchor_indices) < 100:
        return {"alignment_score": 0.0, "clsc": 0.0}
    
    np.random.seed(42)
    sample_idx = np.random.choice(anchor_indices, min(500, len(anchor_indices)), replace=False)
    
    emb_base = embeddings["embedding"][sample_idx]
    emb_ft = embeddings["embedding_ft"][sample_idx]
    
    R, s = orthogonal_procrustes(emb_base, emb_ft)
    aligned_ft = emb_ft @ R
    
    sim_before = np.diag(cosine_similarity(emb_base, emb_ft)).mean()
    sim_after = np.diag(cosine_similarity(emb_base, aligned_ft)).mean()
    
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
    
    return {"sim_before": sim_before, "sim_after": sim_after, "alignment_matrix": R}

def estimate_local_intrinsic_dimensionality(embeddings, lang_labels, k=20):
    results = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, emb in embeddings.items():
        nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(emb)
        distances, _ = nbrs.kneighbors(emb)
        distances = distances[:, 1:]
        
        lid_scores = -1.0 / np.mean(np.log(distances[:, -1:] / distances[:, :-1] + 1e-8), axis=1)
        lid_scores = np.clip(lid_scores, 0, 100)
        results[name] = lid_scores
        
        sns.kdeplot(lid_scores, label=name, ax=ax, fill=True, alpha=0.3)
        
    ax.set_xlabel("Local Intrinsic Dimensionality (LID)")
    ax.set_ylabel("Density")
    ax.set_title("LID Distribution Across Embedding Spaces")
    ax.legend()
    plt.savefig("output_data/img/p1_lid_distributions.png", dpi=300)
    plt.close()
    return results

def compute_persistent_homology(embeddings, sample_size=2000):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (name, emb) in enumerate(embeddings.items()):
        np.random.seed(42)
        idx = np.random.choice(len(emb), min(sample_size, len(emb)), replace=False)
        emb_sample = emb[idx]
        
        diagrams = ripser.ripser(emb_sample, maxdim=1)["dgms"]
        persim.plot_diagrams(diagrams, show=False, ax=axes[i])
        axes[i].set_title(f"Persistent Homology: {name}")
        
    plt.tight_layout()
    plt.savefig("output_data/img/p1_persistent_homology.png", dpi=300)
    plt.close()
    return {"diagrams_computed": True}

def perform_semantic_clustering(embeddings, df, lang_labels):
    emb_ft = embeddings["embedding_ft"]
    
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=10, random_state=42)
    emb_reduced = reducer.fit_transform(emb_ft)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(emb_reduced)
    
    df_clustered = df.with_columns([
        pl.Series("cluster_label", labels),
        pl.Series("lang_primary", lang_labels)
    ])
    
    cluster_stats = df_clustered.group_by("cluster_label").agg([
        pl.len().alias("size"),
        pl.col("like_count").mean().alias("avg_likes"),
        pl.col("reply_count").mean().alias("avg_replies"),
        pl.col("lang_primary").mode().first().alias("dominant_lang")
    ]).filter(pl.col("cluster_label") != -1).sort("size", descending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=emb_reduced[:, 0], y=emb_reduced[:, 1], hue=labels, palette="tab20", s=10, alpha=0.6, ax=axes[0])
    axes[0].set_title("UMAP + HDBSCAN Semantic Clusters")
    axes[0].legend([], [], frameon=False)
    
    top_clusters = cluster_stats.head(10).to_pandas()
    sns.barplot(data=top_clusters, x="cluster_label", y="size", hue="dominant_lang", ax=axes[1], palette="Set2")
    axes[1].set_title("Top 10 Cluster Sizes & Dominant Language")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("output_data/img/p1_semantic_clustering.png", dpi=300)
    plt.close()
    
    return df_clustered, cluster_stats

def build_and_analyze_graphs(df, embeddings, lang_labels, sample_size=5000):
    np.random.seed(42)
    idx = np.random.choice(len(df), min(sample_size, len(df)), replace=False)
    texts = df.select(TEXT_COL).to_numpy().flatten()[idx]
    emb_sample = embeddings["embedding"][idx]
    lang_sample = np.array(lang_labels)[idx]
    
    sim_matrix = cosine_similarity(emb_sample)
    adj_matrix = (sim_matrix > 0.75).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    
    G = nx.from_numpy_array(adj_matrix)
    components = list(nx.connected_components(G))
    giant_component = max(components, key=len)
    G_giant = G.subgraph(giant_component)
    
    degrees = [d for n, d in G_giant.degree()]
    avg_clustering = nx.average_clustering(G_giant)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(degrees, bins=50, kde=True, ax=axes[0], color="teal")
    axes[0].set_title("Comment-Comment Graph Degree Distribution")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale("log")
    
    lang_edges = 0
    total_edges = G_giant.number_of_edges()
    for u, v in G_giant.edges():
        if lang_sample[u] == lang_sample[v]:
            lang_edges += 1
    assortativity = lang_edges / total_edges if total_edges > 0 else 0
    
    metrics = {
        "nodes": G_giant.number_of_nodes(),
        "edges": total_edges,
        "avg_clustering": avg_clustering,
        "language_assortativity": assortativity,
        "components": len(components)
    }
    
    axes[1].bar(metrics.keys(), metrics.values(), color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    axes[1].set_title("Graph Topology Metrics")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("output_data/img/p1_comment_graph_analysis.png", dpi=300)
    plt.close()
    
    return metrics

def run_p0_pipeline(df):
    df_lang = apply_language_architecture(df)
    df_clean = handle_nulls_stratified(df_lang)
    stats = compute_multilingual_descriptive_stats(df_clean)
    return df_clean, stats

def run_p1_pipeline(df_clean, embeddings, lang_labels):
    dim_results = perform_dimensionality_reduction(embeddings, lang_labels)
    align_results = verify_cross_lingual_alignment(embeddings, df_clean, lang_labels)
    lid_results = estimate_local_intrinsic_dimensionality(embeddings, lang_labels)
    homology_results = compute_persistent_homology(embeddings)
    df_clustered, cluster_stats = perform_semantic_clustering(embeddings, df_clean, lang_labels)
    graph_metrics = build_and_analyze_graphs(df_clean, embeddings, lang_labels)
    return {
        "dimensionality": dim_results,
        "alignment": align_results,
        "lid": lid_results,
        "homology": homology_results,
        "clustering": {"df": df_clustered, "stats": cluster_stats},
        "graph": graph_metrics
    }

if __name__ == "__main__":
    df_raw, X_scaled, embeddings, scaler = load_data()
    lang_labels_raw = ["en"] * len(df_raw)
    
    df_p0, p0_stats = run_p0_pipeline(df_raw)
    lang_labels = df_p0["lang_primary"].to_list()
    
    p1_results = run_p1_pipeline(df_p0, embeddings, lang_labels)
    
    print("P0 & P1 Analysis Complete. Visualizations saved to current directory.")
    print(f"Descriptive Stats Shape: {p0_stats.shape}")
    print(f"Alignment Score Improvement: {p1_results['alignment']['sim_before']:.3f} -> {p1_results['alignment']['sim_after']:.3f}")
    print(f"Graph Metrics: {p1_results['graph']}")