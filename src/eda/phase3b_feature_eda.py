import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from pathlib import Path
from datetime import datetime
from scipy.stats import f_oneway, kruskal, shapiro, pointbiserialr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_READY = Path("output_data/parquet/model_ready_features.parquet")
FINAL_FEATURE_NAMES = Path("output_data/parquet/final_feature_names.parquet")
CLEANED_CORPUS = Path("output_data/parquet/cleaned_corpus.parquet")
OUTPUT_PARQUET = Path("output_data/parquet")
OUTPUT_IMG = Path("output_data/img")

LABEL_ORDER = ["positive", "neutral", "negative"]
LABEL_COLORS = {"positive": "#55A868", "neutral": "#4C72B0", "negative": "#C44E52"}
NORMALITY_SAMPLE = 5000
PCA_N_COMPONENTS = 50
UMAP_SAMPLE = 20000
EMBEDDING_COLS = ["embedding", "embedding_char", "embedding_word", "embedding_ft"]
SHAPIRO_ALPHA = 0.05


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pl.DataFrame, list[str]]:
    logger.info("=" * 70)
    logger.info("PHASE 3B — Loading model-ready features")
    logger.info("=" * 70)

    if not MODEL_READY.exists():
        raise FileNotFoundError(f"Run phase3_feature_eng.py first. Missing: {MODEL_READY}")

    t0 = time.time()
    df = pl.read_parquet(MODEL_READY)
    logger.info(f"  Loaded model-ready features: {len(df):,} rows × {len(df.columns)} cols in {time.time() - t0:.2f}s")

    if FINAL_FEATURE_NAMES.exists():
        feature_names = pl.read_parquet(FINAL_FEATURE_NAMES)["feature_name"].to_list()
        feature_names = [f for f in feature_names if f in df.columns]
        logger.info(f"  Loaded {len(feature_names)} final feature names")
    else:
        feature_names = [c for c in df.columns if c not in {"label", "comment_id"}]
        logger.warning("  feature_names file not found — using all non-label columns")

    return df, feature_names


def test_feature_normality(vals: np.ndarray) -> bool:
    n = len(vals)
    if n < 20:
        return False
    sample_size = min(NORMALITY_SAMPLE, n)
    rng = np.random.default_rng(42)
    sample = rng.choice(vals, sample_size, replace=False)
    try:
        _, p_val = shapiro(sample)
        return p_val > SHAPIRO_ALPHA
    except Exception:
        return False


def compute_feature_discriminative_power(
    df: pl.DataFrame, feature_names: list[str]
) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3B — Table 9a: Feature Discriminative Power (ANOVA / Kruskal-Wallis)")
    logger.info("=" * 70)

    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())

    rows = []
    for feat in tqdm(feature_names, desc="Feature discriminative power", ncols=80):
        if feat not in df.columns:
            continue

        col_vals = df[feat].to_numpy().astype(float)
        is_binary = set(np.unique(col_vals[~np.isnan(col_vals)])).issubset({0, 1, 0.0, 1.0})

        groups = []
        for label_idx in range(len(le.classes_)):
            mask = y == label_idx
            vals = col_vals[mask]
            vals = vals[~np.isnan(vals) & ~np.isinf(vals)]
            groups.append(vals)

        if any(len(g) < 5 for g in groups):
            continue

        if is_binary:
            try:
                corr, p_val = pointbiserialr(col_vals, y)
                stat = corr
                test_name = "pointbiserial"
                eta_sq = corr ** 2
            except Exception:
                continue
        else:
            is_normal = all(test_feature_normality(g) for g in groups)
            if is_normal:
                try:
                    f_stat, p_val = f_oneway(*groups)
                    stat = f_stat
                    test_name = "ANOVA_F"
                    total_var = np.var(col_vals[~np.isnan(col_vals)])
                    group_means = [g.mean() for g in groups]
                    global_mean = np.mean([v for g in groups for v in g])
                    ss_between = sum(len(g) * (m - global_mean) ** 2 for g, m in zip(groups, group_means))
                    n_total = sum(len(g) for g in groups)
                    ss_total = n_total * total_var
                    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
                except Exception:
                    continue
            else:
                try:
                    h_stat, p_val = kruskal(*groups)
                    stat = h_stat
                    test_name = "KruskalWallis_H"
                    n_total = sum(len(g) for g in groups)
                    eta_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))
                    eta_sq = max(0.0, eta_sq)
                except Exception:
                    continue

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        keep = "keep" if (p_val < 0.05 and eta_sq > 0.01) else "review"

        rows.append({
            "feature_name": feat,
            "test": test_name,
            "stat": round(float(stat), 4),
            "p_value": round(float(p_val), 8),
            "significance": sig,
            "eta_squared": round(float(eta_sq), 6),
            "is_binary": is_binary,
            "recommendation": keep,
        })

    disc_table = pl.DataFrame(rows).sort("eta_squared", descending=True)

    top_feats = disc_table.head(15)
    logger.info(f"  Top 15 features by effect size (η²):")
    for row in top_feats.iter_rows(named=True):
        logger.info(
            f"    {row['feature_name']:40s}: η²={row['eta_squared']:.4f}  "
            f"{row['test']}={row['stat']:.2f}  p={row['p_value']:.2e}  {row['significance']}"
        )

    keep_count = disc_table.filter(pl.col("recommendation") == "keep").height
    logger.info(f"  Features recommended to keep: {keep_count} / {len(disc_table)}")

    return disc_table


def compute_feature_distribution_by_label(
    df: pl.DataFrame, feature_names: list[str]
) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3B — Table 9b: Feature Distribution by Label")
    logger.info("=" * 70)

    rows = []
    numeric_feats = [
        f for f in feature_names
        if f in df.columns and df[f].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int8]
    ]

    logger.info(f"  Computing stats for {len(numeric_feats)} numeric features across {len(LABEL_ORDER)} labels...")
    for feat in tqdm(numeric_feats, desc="Feature dist by label", ncols=80):
        for label in LABEL_ORDER:
            vals = df.filter(pl.col("label") == label)[feat].drop_nulls().to_numpy().astype(float)
            if len(vals) < 5:
                continue
            rows.append({
                "feature": feat,
                "label": label,
                "n": len(vals),
                "mean": round(float(vals.mean()), 5),
                "median": round(float(np.median(vals)), 5),
                "std": round(float(vals.std()), 5),
                "q25": round(float(np.percentile(vals, 25)), 5),
                "q75": round(float(np.percentile(vals, 75)), 5),
            })

    dist_table = pl.DataFrame(rows)
    logger.info(f"  Distribution table: {len(dist_table):,} rows")
    return dist_table


def compute_categorical_association(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3B — Table 9c: Categorical Feature Association (Cramér's V)")
    logger.info("=" * 70)

    from scipy.stats import chi2_contingency

    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())

    categorical_feats = []
    for col in df.columns:
        if col in {"label", "comment_id"}:
            continue
        if df[col].dtype in [pl.Utf8, pl.Categorical]:
            categorical_feats.append(col)
        elif col in {"engagement_tier_ordinal", "era_flag", "is_weekend", "day_of_week", "hour_of_day"}:
            categorical_feats.append(col)
        elif col.startswith("lang_"):
            categorical_feats.append(col)

    logger.info(f"  Categorical features to test: {categorical_feats}")

    rows = []
    for feat in categorical_feats:
        if feat not in df.columns:
            continue
        try:
            feat_vals = df[feat].to_numpy()
            unique_vals = np.unique(feat_vals[~(feat_vals == None)])
            contingency = np.zeros((len(unique_vals), len(le.classes_)), dtype=int)
            for i, val in enumerate(unique_vals):
                for j in range(len(le.classes_)):
                    contingency[i, j] = int(np.sum((feat_vals == val) & (y == j)))

            chi2, p_val, _, _ = chi2_contingency(contingency)
            n = contingency.sum()
            k = min(contingency.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else 0.0

            rows.append({
                "feature": feat,
                "cramers_v": round(cramers_v, 4),
                "chi2": round(chi2, 3),
                "p_value": round(p_val, 8),
                "n_categories": len(unique_vals),
            })
            logger.info(
                f"  {feat:40s}: Cramér's V={cramers_v:.4f}  p={p_val:.4f}"
            )
        except Exception as e:
            logger.warning(f"  Failed for {feat}: {e}")

    return pl.DataFrame(rows).sort("cramers_v", descending=True)


def compute_embedding_silhouette(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3B — Silhouette scores for embedding spaces")
    logger.info("=" * 70)

    corpus_path = Path("output_data/parquet/cleaned_corpus.parquet")
    if not corpus_path.exists():
        logger.warning("  cleaned_corpus.parquet not found — skipping embedding silhouette")
        return pl.DataFrame()

    t0 = time.time()
    df_full = pl.read_parquet(corpus_path)

    le = LabelEncoder()
    y = le.fit_transform(df_full["label"].to_numpy())

    rng = np.random.default_rng(42)
    sample_size = min(UMAP_SAMPLE, len(df_full))
    idx = rng.choice(len(df_full), sample_size, replace=False)
    y_sample = y[idx]

    rows = []
    for col in EMBEDDING_COLS:
        if col not in df_full.columns:
            logger.warning(f"  Embedding '{col}' not found — skipping")
            continue

        logger.info(f"  Processing embedding: {col}")
        try:
            emb_full = np.stack(df_full[col].to_numpy())
            emb_sample = emb_full[idx]

            pca = PCA(n_components=min(PCA_N_COMPONENTS, emb_sample.shape[1]))
            emb_pca = pca.fit_transform(emb_sample)

            sil_score = silhouette_score(emb_pca, y_sample, metric="cosine", sample_size=5000, random_state=42)
            explained_var = float(pca.explained_variance_ratio_.sum())

            logger.info(
                f"    PCA-{PCA_N_COMPONENTS}: silhouette={sil_score:.4f}  "
                f"explained_var={explained_var:.3f}"
            )

            try:
                import umap
                reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)
                emb_umap = reducer.fit_transform(emb_pca)
                sil_umap = silhouette_score(emb_umap, y_sample, metric="euclidean", sample_size=5000, random_state=42)
                logger.info(f"    UMAP-2D: silhouette={sil_umap:.4f}")
            except Exception as e:
                sil_umap = None
                logger.warning(f"    UMAP silhouette failed for {col}: {e}")

            rows.append({
                "embedding": col,
                "pca_silhouette": round(sil_score, 4),
                "umap_silhouette": round(sil_umap, 4) if sil_umap is not None else None,
                "pca_n_components": PCA_N_COMPONENTS,
                "pca_explained_var": round(explained_var, 4),
                "sample_size": sample_size,
            })
        except Exception as e:
            logger.error(f"  Failed for {col}: {e}")

    logger.info(f"  Embedding silhouette completed in {time.time() - t0:.1f}s")
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def check_target_encoding_leakage(df: pl.DataFrame) -> str:
    logger.info("=" * 70)
    logger.info("PHASE 3B — Target encoding leakage check")
    logger.info("=" * 70)

    if "source_query_target_encoded" not in df.columns:
        return "source_query_target_encoded not present — leakage check skipped"

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())

    for train_idx, val_idx in split.split(np.zeros(len(df)), y):
        enc_train = df["source_query_target_encoded"].to_numpy()[train_idx]
        enc_val = df["source_query_target_encoded"].to_numpy()[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        from sklearn.feature_selection import mutual_info_classif
        mi_train = mutual_info_classif(enc_train.reshape(-1, 1), y_train, discrete_features=False, random_state=42)[0]
        mi_val = mutual_info_classif(enc_val.reshape(-1, 1), y_val, discrete_features=False, random_state=42)[0]

    leakage_ratio = mi_val / mi_train if mi_train > 0 else 0.0
    result = (
        f"MI(target_enc → label): train={mi_train:.4f}  val={mi_val:.4f}  "
        f"ratio={leakage_ratio:.3f}  "
        f"{'⚠ LEAKAGE SUSPECTED' if leakage_ratio > 1.5 else '✓ OK'}"
    )
    logger.info(f"  {result}")
    return result


def plot_eta_squared_ranking(disc_table: pl.DataFrame):
    logger.info("Generating η² effect size ranking chart")

    top30 = disc_table.head(30)
    feats = top30["feature_name"].to_list()
    eta = top30["eta_squared"].to_list()
    recs = top30["recommendation"].to_list()

    colors = ["#55A868" if r == "keep" else "#DD8452" for r in recs]

    fig, ax = plt.subplots(figsize=(12, max(6, len(feats) * 0.35)))
    bars = ax.barh(range(len(feats)), eta, color=colors, edgecolor="white")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1, label="η²=0.01 threshold")
    ax.axvline(0.06, color="orange", linestyle="--", linewidth=1, label="η²=0.06 medium")
    ax.set_xlabel("Effect Size (η²)")
    ax.set_title("Feature Discriminative Power — Effect Size Ranking", fontweight="bold")
    ax.legend(fontsize=8)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#55A868", label="Keep"),
        Patch(facecolor="#DD8452", label="Review"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t9_feature_eta_squared_ranking.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't9_feature_eta_squared_ranking.png'}")


def plot_silhouette_scores(sil_df: pl.DataFrame):
    if len(sil_df) == 0:
        return

    logger.info("Generating embedding silhouette chart")

    fig, ax = plt.subplots(figsize=(10, 5))
    embs = sil_df["embedding"].to_list()
    pca_sils = sil_df["pca_silhouette"].to_list()
    umap_sils = sil_df["umap_silhouette"].to_list() if "umap_silhouette" in sil_df.columns else [None] * len(embs)

    x = np.arange(len(embs))
    width = 0.35
    ax.bar(x - width / 2, pca_sils, width, label=f"PCA-{PCA_N_COMPONENTS}", color="#4C72B0", alpha=0.85)
    if any(s is not None for s in umap_sils):
        umap_vals = [s if s is not None else 0.0 for s in umap_sils]
        ax.bar(x + width / 2, umap_vals, width, label="UMAP-2D", color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(embs, rotation=20, ha="right")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Label Cluster Quality in Embedding Spaces", fontweight="bold")
    ax.legend()
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t9_embedding_silhouette_scores.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't9_embedding_silhouette_scores.png'}")


def plot_cramers_v_heatmap(cat_assoc: pl.DataFrame):
    if len(cat_assoc) == 0:
        return

    logger.info("Generating categorical association Cramér's V chart")

    fig, ax = plt.subplots(figsize=(10, max(4, len(cat_assoc) * 0.5)))
    feats = cat_assoc["feature"].to_list()
    cramers = cat_assoc["cramers_v"].to_list()

    colors = plt.cm.viridis(np.array(cramers) / max(cramers + [0.001]))
    bars = ax.barh(feats, cramers, color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.axvline(0.1, color="red", linestyle="--", linewidth=1, label="V=0.1 (weak)")
    ax.axvline(0.3, color="orange", linestyle="--", linewidth=1, label="V=0.3 (moderate)")
    ax.set_xlabel("Cramér's V")
    ax.set_title("Categorical Feature → Label Association (Cramér's V)", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t9_categorical_association.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't9_categorical_association.png'}")


def save_outputs(
    disc_table: pl.DataFrame,
    dist_table: pl.DataFrame,
    cat_assoc: pl.DataFrame,
    sil_df: pl.DataFrame,
    leakage_result: str,
):
    logger.info("=" * 70)
    logger.info("PHASE 3B — Saving outputs")
    logger.info("=" * 70)

    disc_table.write_parquet(OUTPUT_PARQUET / "t9_feature_discriminative_power.parquet")
    logger.info(f"  Discriminative power → {OUTPUT_PARQUET / 't9_feature_discriminative_power.parquet'}")

    dist_table.write_parquet(OUTPUT_PARQUET / "t9_feature_distribution_by_label.parquet")
    logger.info(f"  Distribution by label → {OUTPUT_PARQUET / 't9_feature_distribution_by_label.parquet'}")

    if len(cat_assoc) > 0:
        cat_assoc.write_parquet(OUTPUT_PARQUET / "t9_categorical_association.parquet")
        logger.info(f"  Categorical assoc    → {OUTPUT_PARQUET / 't9_categorical_association.parquet'}")

    if len(sil_df) > 0:
        sil_df.write_parquet(OUTPUT_PARQUET / "t9_embedding_silhouette.parquet")
        logger.info(f"  Silhouette scores    → {OUTPUT_PARQUET / 't9_embedding_silhouette.parquet'}")

    pl.DataFrame([{"leakage_check_result": leakage_result}]).write_parquet(
        OUTPUT_PARQUET / "t9_target_encoding_leakage_check.parquet"
    )
    logger.info(f"  Leakage check        → {OUTPUT_PARQUET / 't9_target_encoding_leakage_check.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 3B FEATURE EDA — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()
    df, feature_names = load_data()

    disc_table = compute_feature_discriminative_power(df, feature_names)
    dist_table = compute_feature_distribution_by_label(df, feature_names)
    cat_assoc = compute_categorical_association(df)
    sil_df = compute_embedding_silhouette(df)
    leakage_result = check_target_encoding_leakage(df)

    plot_eta_squared_ranking(disc_table)
    plot_silhouette_scores(sil_df)
    plot_cramers_v_heatmap(cat_assoc)

    save_outputs(disc_table, dist_table, cat_assoc, sil_df, leakage_result)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 3B COMPLETE — elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)
