import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import orthogonal_procrustes
from langdetect import detect_langs
from collections import Counter
import umap
import hdbscan
import networkx as nx

COMBINED_DATA_PATH = "../data/combined_embeddings.parquet"

numeric_cols = [
    "like_count", "reply_count", "char_count", "word_count",
    "avg_word_length", "uppercase_ratio", "exclamation_count",
    "question_count", "hashtag_count", "mention_count",
    "emoji_count", "like_count_log"
]

embedding_cols = ["embedding", "embedding_char", "embedding_word", "embedding_ft"]
comment_text_col = ["comment_text"]
meta_cols = ["published_at", "crawled_at", "source_query"]

def load_and_prepare_data():
    df = pl.read_parquet(
        COMBINED_DATA_PATH,
        columns=numeric_cols + embedding_cols + comment_text_col + meta_cols
    )
    
    X_numeric = df.select(numeric_cols).to_numpy().astype(np.float32)
    
    embeddings = {}
    for col in embedding_cols:
        embeddings[col] = np.stack(df[col].to_numpy())
        
    X_all = np.hstack([X_numeric] + list(embeddings.values()))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    return df, embeddings, X_scaled

def identify_language(text):
    if not text:
        return "unknown"
    try:
        return detect_langs(text)[0].lang
    except Exception:
        return "unknown"

def detect_script(text):
    if not text:
        return "Unknown"
    scripts = re.findall(r'\p{Script=Latin}|\p{Script=Cyrillic}|\p{Script=Arabic}|\p{Script=Han}', text)
    if not scripts:
        return "Unknown"
    return max(set(scripts), key=scripts.count)

def process_language_quality(df):
    df = df.with_columns(
        pl.col("comment_text").map_elements(identify_language, return_dtype=pl.String).alias("language"),
        pl.col("comment_text").map_elements(detect_script, return_dtype=pl.String).alias("script")
    )
    return df

def handle_nulls(df):
    df = df.filter(pl.col("comment_text").is_not_null())
    df = df.with_columns(
        pl.col("like_count").fill_null(pl.col("like_count").median().over("language")),
        pl.col("published_at").fill_null(pl.col("crawled_at"))
    )
    return df

def compute_stratified_stats(df):
    top_langs = df["language"].value_counts().sort("count", descending=True).head(10)["language"].to_list()
    df_top = df.filter(pl.col("language").is_in(top_langs))
    
    stats = df_top.group_by("language").agg([
        pl.col("char_count").mean().alias("avg_chars"),
        pl.col("word_count").mean().alias("avg_words"),
        (pl.col("emoji_count") / pl.col("char_count")).mean().alias("emoji_density"),
        pl.col("uppercase_ratio").mean().alias("avg_caps"),
        pl.col("like_count").mean().alias("avg_likes")
    ])
    return stats

def plot_language_distributions(df):
    top_langs = df["language"].value_counts().sort("count", descending=True).head(5)["language"].to_list()
    df_top = df.filter(pl.col("language").is_in(top_langs))
    df_pdf = df_top.with_columns((pl.col("emoji_count") / pl.col("char_count")).alias("emoji_density")).to_pandas()
    
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 2, 1)
    sns.histplot(data=df_pdf, x="char_count", hue="language", element="step", stat="density", common_norm=False)
    plt.title("Text Length Distribution by Language")
    
    plt.subplot(2, 2, 2)
    sns.histplot(data=df_pdf, x="emoji_density", hue="language", element="step", stat="density", common_norm=False)
    plt.title("Emoji Density Distribution")
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df_pdf, x="language", y="like_count")
    plt.yscale("log")
    plt.title("Engagement Ratios (Likes)")
    
    plt.subplot(2, 2, 4)
    sns.countplot(data=df_pdf, x="language", hue="source_query")
    plt.title("Source Query Distribution")
    
    plt.tight_layout()
    plt.show()

def plot_cross_lingual_heatmap(df):
    top_langs = df["language"].value_counts().sort("count", descending=True).head(10)["language"].to_list()
    df_top = df.filter(pl.col("language").is_in(top_langs))
    
    pivot_df = df_top.group_by("language").agg([
        pl.col(c).mean() for c in numeric_cols
    ])
    
    data_matrix = pivot_df.select(numeric_cols).to_numpy()
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", xticklabels=numeric_cols, yticklabels=numeric_cols)
    plt.title("Cross-Lingual Feature Correlation (Top 10 Languages)")
    plt.show()

def analyze_embedding_structure(embeddings, df, sample_size=10000):
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    
    plt.figure(figsize=(20, 10))
    plot_idx = 1
    
    for col, emb in embeddings.items():
        if col != "embedding_ft":
            continue
            
        emb_sample = emb[sample_idx]
        lang_sample = df["language"].to_numpy()[sample_idx]
        
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb_sample)
        
        plt.subplot(1, 2, plot_idx)
        sns.scatterplot(x=emb_pca[:, 0], y=emb_pca[:, 1], hue=lang_sample, palette="tab10", s=10, alpha=0.5, legend=False)
        plt.title(f"{col} - PCA")
        plot_idx += 1
        
        umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=2)
        emb_umap = umap_model.fit_transform(emb_sample)
        
        plt.subplot(1, 2, plot_idx)
        sns.scatterplot(x=emb_umap[:, 0], y=emb_umap[:, 1], hue=lang_sample, palette="tab10", s=10, alpha=0.5)
        plt.title(f"{col} - UMAP")
        plot_idx += 1

    plt.tight_layout()
    plt.show()

def verify_cross_lingual_alignment(embeddings, df, lang1="en", lang2="es"):
    emb = embeddings["embedding_ft"]
    
    mask1 = df["language"].to_numpy() == lang1
    mask2 = df["language"].to_numpy() == lang2
    
    idx1 = np.where(mask1)[0]
    idx2 = np.where(mask2)[0]
    
    sample_size = min(len(idx1), len(idx2), 5000)
    if sample_size == 0:
        return 0.0
        
    idx1_sample = np.random.choice(idx1, sample_size, replace=False)
    idx2_sample = np.random.choice(idx2, sample_size, replace=False)
    
    emb1 = emb[idx1_sample]
    emb2 = emb[idx2_sample]
    
    pca = PCA(n_components=50)
    emb1_pca = pca.fit_transform(emb1)
    emb2_pca = pca.transform(emb2)
    
    R, scale = orthogonal_procrustes(emb1_pca, emb2_pca)
    aligned_emb2 = np.dot(emb2_pca, R)
    
    similarities = np.sum(emb1_pca * aligned_emb2, axis=1) / (np.linalg.norm(emb1_pca, axis=1) * np.linalg.norm(aligned_emb2, axis=1))
    return similarities.mean()

def compute_embedding_geometry(embeddings, df):
    sample_idx = np.random.choice(len(df), 5000, replace=False)
    emb = embeddings["embedding_ft"][sample_idx]
    
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(emb)
    distances, _ = nn.kneighbors(emb)
    
    m = distances.shape[1] - 1
    sum_log_r = np.sum(np.log(distances[:, 1:] / distances[:, 1:2]))
    lid = -m / sum_log_r
    return lid.mean()

def semantic_clustering_pipeline(embeddings, df, sample_size=20000):
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    emb = embeddings["embedding_ft"][sample_idx]
    
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=10)
    emb_reduced = reducer.fit_transform(emb)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10, metric="euclidean")
    labels = clusterer.fit_predict(emb_reduced)
    
    return sample_idx, labels

def characterize_clusters(df, sample_idx, labels):
    df_sample = df[sample_idx]
    df_sample = df_sample.with_columns(pl.Series("cluster", labels))
    
    cluster_stats = df_sample.group_by("cluster").agg([
        pl.col("language").value_counts().alias("lang_dist"),
        pl.col("like_count").mean().alias("avg_likes"),
        pl.col("source_query").value_counts().alias("source_dist")
    ])
    return cluster_stats

def build_monolingual_gow(texts, lang="en"):
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    
    try:
        stop_words = set(stopwords.words("english" if lang == "en" else lang))
    except Exception:
        stop_words = set()
        
    word_counts = Counter()
    edges = Counter()
    
    for text in texts:
        if not text:
            continue
        words = [w.lower() for w in re.findall(r'\w+', text) if w.lower() not in stop_words]
        word_counts.update(words)
        for i in range(len(words) - 1):
            edges[(words[i], words[i+1])] += 1
            
    G = nx.Graph()
    for word, count in word_counts.most_common(100):
        G.add_node(word, weight=count)
        
    for (w1, w2), weight in edges.most_common(200):
        if w1 in G and w2 in G:
            G.add_edge(w1, w2, weight=weight)
            
    return G

def build_cross_lingual_graph(embeddings, df, threshold=0.6, sample_size=5000):
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    emb = embeddings["embedding_ft"][sample_idx]
    langs = df["language"].to_numpy()[sample_idx]
    
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / norms
    sim_matrix = np.dot(emb_norm, emb_norm.T)
    np.fill_diagonal(sim_matrix, 0)
    
    G = nx.Graph()
    for i in range(sample_size):
        G.add_node(i, lang=langs[i])
        
    rows, cols = np.where(sim_matrix > threshold)
    for r, c in zip(rows, cols):
        if langs[r] != langs[c]:
            G.add_edge(r, c, weight=sim_matrix[r, c])
            
    return G

def build_comment_graph(embeddings, df, sample_size=10000, threshold=0.8):
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    emb = embeddings["embedding"][sample_idx]
    
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / norms
    sim_matrix = np.dot(emb_norm, emb_norm.T)
    
    np.fill_diagonal(sim_matrix, 0)
    
    G = nx.Graph()
    G.add_nodes_from(range(sample_size))
    
    rows, cols = np.where(sim_matrix > threshold)
    edges = list(zip(rows, cols))
    G.add_edges_from(edges)
    
    return G

def execute_pipeline():
    df, embeddings, X_scaled = load_and_prepare_data()
    
    df = process_language_quality(df)
    df = handle_nulls(df)
    
    stats = compute_stratified_stats(df)
    
    plot_language_distributions(df)
    plot_cross_lingual_heatmap(df)
    
    analyze_embedding_structure(embeddings, df)
    
    alignment_score = verify_cross_lingual_alignment(embeddings, df)
    print(f"Cross-Lingual Alignment Score: {alignment_score}")
    
    lid_score = compute_embedding_geometry(embeddings, df)
    print(f"Local Intrinsic Dimensionality: {lid_score}")
    
    sample_idx, cluster_labels = semantic_clustering_pipeline(embeddings, df)
    cluster_stats = characterize_clusters(df, sample_idx, cluster_labels)
    
    en_texts = df.filter(pl.col("language") == "en")["comment_text"].to_list()
    G_gow = build_monolingual_gow(en_texts, "english")
    
    G_cross = build_cross_lingual_graph(embeddings, df)
    
    G_comments = build_comment_graph(embeddings, df)
    
    return df, stats, G_gow, G_cross, G_comments

if __name__ == "__main__":
    execute_pipeline()