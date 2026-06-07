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
        columns=NUMERIC_COLS + EMBEDDING_COLS + [TEXT_COL, "published_at", "crawled_at", "source_query", "post_id"]
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
        return langdetect.detect(str(text))
    except Exception:
        return "unknown"

def add_language_labels(df):
    return df.with_columns(
        pl.col(TEXT_COL).map_elements(detect_language, return_dtype=pl.Utf8).alias("lang_primary")
    )

def analyze_embedding_pca(embeddings):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    pca_results = {}
    for i, (name, emb) in enumerate(embeddings.items()):
        pca = PCA()
        pca.fit(emb)
        pca_results[name] = pca.explained_variance_ratio_
        sns.lineplot(
            x=np.arange(1, len(pca.explained_variance_ratio_) + 1),
            y=np.cumsum(pca.explained_variance_ratio_),
            ax=axes[i],
            color=sns.color_palette()[i]
        )
        axes[i].set_title(f"Cumulative Explained Variance: {name}")
        axes[i].set_xlabel("Number of Components")
        axes[i].set_ylabel("Cumulative Variance Ratio")
        axes[i].axhline(0.95, color='red', linestyle='--', label='95% Threshold')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_pca_variance.png", dpi=300)
    plt.close()
    return pca_results

def analyze_embedding_projections(embeddings, lang_labels, sample_size=10000):
    np.random.seed(42)
    n_samples = min(sample_size, len(lang_labels))
    idx = np.random.choice(len(lang_labels), n_samples, replace=False)
    lang_sample = np.array(lang_labels)[idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    projection_results = {}
    
    for i, (name, emb) in enumerate(embeddings.items()):
        emb_sample = emb[idx]
        
        reducer_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=2, random_state=42)
        emb_umap = reducer_umap.fit_transform(emb_sample)
        projection_results[f"{name}_umap"] = emb_umap
        
        sns.scatterplot(
            x=emb_umap[:, 0], y=emb_umap[:, 1], hue=lang_sample, 
            palette="tab20", s=15, alpha=0.7, ax=axes[i], legend=False
        )
        axes[i].set_title(f"UMAP Projection (Cross-Lingual): {name}")
        
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_umap_projections.png", dpi=300)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, name in enumerate(["embedding", "embedding_ft"]):
        emb_sample = embeddings[name][idx]
        reducer_tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42, init='pca')
        emb_tsne = reducer_tsne.fit_transform(emb_sample)
        projection_results[f"{name}_tsne"] = emb_tsne
        
        sns.scatterplot(
            x=emb_tsne[:, 0], y=emb_tsne[:, 1], hue=lang_sample, 
            palette="tab20", s=15, alpha=0.7, ax=axes[i], legend=False
        )
        axes[i].set_title(f"t-SNE Projection: {name}")
        
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_tsne_projections.png", dpi=300)
    plt.close()
    
    return projection_results

def cluster_embeddings_hdbscan(embeddings, lang_labels):
    emb_ft = embeddings["embedding_ft"]
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=10, random_state=42)
    emb_reduced = reducer.fit_transform(emb_ft)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(emb_reduced)
    
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
    
    return labels, emb_reduced

def characterize_clusters(df, cluster_labels, lang_labels):
    df_clustered = df.with_columns([
        pl.Series("cluster_label", cluster_labels),
        pl.Series("lang_primary", lang_labels)
    ])
    
    cluster_stats = df_clustered.group_by("cluster_label").agg([
        pl.len().alias("size"),
        pl.col("like_count").mean().alias("avg_likes"),
        pl.col("reply_count").mean().alias("avg_replies"),
        pl.col("lang_primary").mode().first().alias("dominant_lang")
    ]).filter(pl.col("cluster_label") != -1).sort("size", descending=True)
    
    top_clusters = cluster_stats.head(10).to_pandas()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_clusters, x="cluster_label", y="size", hue="dominant_lang", palette="Set3")
    ax.set_title("Top 10 Clusters: Size and Dominant Language")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("output_data/img/phase2_cluster_characterization.png", dpi=300)
    plt.close()
    
    return cluster_stats

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
    plt.savefig("output_data/img/phase2_persistent_homology.png", dpi=300)
    plt.close()
    return {"diagrams_computed": True}

def build_per_language_gow(df, lang_labels, top_langs=3):
    lang_counts = Counter(lang_labels)
    valid_langs = [lang for lang, count in lang_counts.most_common(top_langs) if lang != "unknown"]
    
    gow_results = {}
    fig, axes = plt.subplots(1, len(valid_langs), figsize=(6 * len(valid_langs), 5))
    if len(valid_langs) == 1:
        axes = [axes]
        
    for i, lang in enumerate(valid_langs):
        texts = [str(t) for t in df.filter(pl.col("lang_primary") == lang).select(TEXT_COL).to_numpy().flatten()]
        vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=500)
        X = vectorizer.fit_transform(texts)
        
        cooccurrence = (X.T * X).toarray()
        np.fill_diagonal(cooccurrence, 0)
        
        G = nx.from_numpy_array(cooccurrence)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        degrees = [d for n, d in G.degree()]
        gow_results[lang] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "avg_degree": np.mean(degrees)}
        
        sns.histplot(degrees, bins=30, kde=True, ax=axes[i], color="teal")
        axes[i].set_title(f"Degree Distribution: {lang}")
        axes[i].set_xlabel("Degree")
        axes[i].set_yscale("log")
        
    plt.tight_layout()
    plt.savefig("output_data/img/phase3_per_language_gow.png", dpi=300)
    plt.close()
    
    return gow_results

def build_cross_lingual_similarity_graph(embeddings, lang_labels, sample_size=5000):
    np.random.seed(42)
    idx = np.random.choice(len(lang_labels), min(sample_size, len(lang_labels)), replace=False)
    emb_sample = embeddings["embedding_ft"][idx]
    lang_sample = np.array(lang_labels)[idx]
    
    sim_matrix = cosine_similarity(emb_sample)
    np.fill_diagonal(sim_matrix, 0)
    
    threshold = np.percentile(sim_matrix[sim_matrix > 0], 90)
    adj_matrix = (sim_matrix > threshold).astype(int)
    
    G = nx.from_numpy_array(adj_matrix)
    
    cross_edges = 0
    total_edges = G.number_of_edges()
    for u, v in G.edges():
        if lang_sample[u] != lang_sample[v]:
            cross_edges += 1
            
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
    
    return G, metrics

def build_comment_knn_graph(embeddings, lang_labels, k=15, sample_size=10000):
    np.random.seed(42)
    idx = np.random.choice(len(lang_labels), min(sample_size, len(lang_labels)), replace=False)
    emb_sample = embeddings["embedding"][idx]
    lang_sample = np.array(lang_labels)[idx]
    
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(emb_sample)
    distances, indices = nbrs.kneighbors(emb_sample)
    
    rows = np.repeat(np.arange(sample_size), k)
    cols = indices[:, 1:].flatten()
    data = np.ones_like(rows, dtype=float)
    
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(sample_size, sample_size))
    adj_matrix = adj_matrix.maximum(adj_matrix.T)
    
    G = nx.from_scipy_sparse_array(adj_matrix)
    
    degrees = [d for n, d in G.degree()]
    components = list(nx.connected_components(G))
    
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": np.mean(degrees),
        "num_components": len(components),
        "largest_component_size": len(max(components, key=len))
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(degrees, bins=50, kde=True, ax=ax, color="purple")
    ax.set_title(f"Comment k-NN Graph Degree Distribution (k={k})")
    ax.set_xlabel("Degree")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("output_data/img/phase3_comment_knn_graph.png", dpi=300)
    plt.close()
    
    return G, metrics

def compute_graph_metrics(G):
    if G.number_of_nodes() == 0:
        return {}
    
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_largest = G.subgraph(largest_cc)
    
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G_largest),
        "num_components": len(components),
        "largest_component_size": len(largest_cc)
    }
    
    try:
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(G_largest)
    except Exception:
        metrics["avg_shortest_path"] = float('inf')
        
    return metrics

def spectral_clustering_laplacian(embeddings, lang_labels, n_clusters=10, sample_size=10000):
    np.random.seed(42)
    idx = np.random.choice(len(lang_labels), min(sample_size, len(lang_labels)), replace=False)
    emb_sample = embeddings["embedding_ft"][idx]
    lang_sample = np.array(lang_labels)[idx]
    
    k = 15
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(emb_sample)
    distances, indices = nbrs.kneighbors(emb_sample)
    
    rows = np.repeat(np.arange(sample_size), k)
    cols = indices[:, 1:].flatten()
    data = np.exp(-distances[:, 1:].flatten() / 0.5)
    
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(sample_size, sample_size))
    adj_matrix = adj_matrix.maximum(adj_matrix.T)
    
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    D = diags(degrees)
    L = D - adj_matrix
    
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees + 1e-8))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    
    eigenvalues, eigenvectors = eigsh(L_sym, k=n_clusters, which='SM')
    
    spectral_clusters = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        random_state=42
    ).fit_predict(adj_matrix.toarray())
    
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
    
    return spectral_clusters, eigenvalues

def run_phase_2(df, embeddings, lang_labels):
    pca_results = analyze_embedding_pca(embeddings)
    projection_results = analyze_embedding_projections(embeddings, lang_labels)
    cluster_labels, emb_reduced = cluster_embeddings_hdbscan(embeddings, lang_labels)
    cluster_stats = characterize_clusters(df, cluster_labels, lang_labels)
    homology_results = compute_persistent_homology(embeddings)
    
    return {
        "pca": pca_results,
        "projections": projection_results,
        "cluster_labels": cluster_labels,
        "cluster_stats": cluster_stats,
        "homology": homology_results
    }

def run_phase_3(df, embeddings, lang_labels):
    gow_results = build_per_language_gow(df, lang_labels)
    cross_lingual_G, cross_lingual_metrics = build_cross_lingual_similarity_graph(embeddings, lang_labels)
    knn_G, knn_metrics = build_comment_knn_graph(embeddings, lang_labels)
    graph_metrics = compute_graph_metrics(knn_G)
    spectral_clusters, eigenvalues = spectral_clustering_laplacian(embeddings, lang_labels)
    
    return {
        "gow": gow_results,
        "cross_lingual_graph": {"graph": cross_lingual_G, "metrics": cross_lingual_metrics},
        "knn_graph": {"graph": knn_G, "metrics": knn_metrics},
        "graph_metrics": graph_metrics,
        "spectral": {"clusters": spectral_clusters, "eigenvalues": eigenvalues}
    }

if __name__ == "__main__":
    df_raw, X_scaled, embeddings, scaler = load_data()
    df_lang = add_language_labels(df_raw)
    lang_labels = df_lang["lang_primary"].to_list()
    
    phase_2_results = run_phase_2(df_lang, embeddings, lang_labels)
    phase_3_results = run_phase_3(df_lang, embeddings, lang_labels)
    
    print("Phase 2 & 3 Analysis Complete. Visualizations saved to current directory.")
    print(f"PCA Variance Explained (embedding_ft): {np.sum(phase_2_results['pca']['embedding_ft'][:50]):.4f}")
    print(f"Cross-Lingual Graph Ratio: {phase_3_results['cross_lingual_graph']['metrics']['cross_lingual_ratio']:.4f}")
    print(f"KNN Graph Metrics: {phase_3_results['graph_metrics']}")