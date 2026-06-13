import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import os
import joblib
from pathlib import Path
from datetime import datetime, UTC
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLEANED_CORPUS = Path("output_data/parquet/cleaned_corpus.parquet")
CORPUS_WITH_EMOJI = Path("output_data/parquet/corpus_with_emoji_types.parquet")
CORPUS_WITH_LANG = Path("output_data/parquet/corpus_with_language.parquet")
CORPUS_WITH_ENGAGE = Path("output_data/parquet/corpus_with_engagement.parquet")
OUTPUT_PARQUET = Path("output_data/parquet")
OUTPUT_IMG = Path("output_data/img")
OUTPUT_MODELS = Path("output_data/models")

LABEL_ORDER = ["positive", "neutral", "negative"]
LABEL_COLORS = {"positive": "#55A868", "neutral": "#4C72B0", "negative": "#C44E52"}
TOP_N_LANGUAGES = 5
VIF_THRESHOLD = 10.0
ENGAGEMENT_FEATURES = [
    "like_count_log", "reply_count_log", "reply_to_like_ratio",
    "likes_per_day", "replies_per_day",
]
TEXT_FEATURES = [
    "char_count", "word_count", "avg_word_length", "uppercase_ratio",
    "exclamation_count", "question_count", "hashtag_count",
    "mention_count", "emoji_count", "emoji_density",
    "emoji_face_positive_count", "emoji_face_negative_count",
    "emoji_face_neutral_count", "emoji_symbol_positive_count",
    "emoji_symbol_negative_count", "emoji_other_count",
]


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)


def load_best_corpus() -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3 — Loading most enriched corpus available")
    logger.info("=" * 70)

    for path, name in [
        (CORPUS_WITH_LANG, "corpus_with_language"),
        (CORPUS_WITH_EMOJI, "corpus_with_emoji_types"),
        (CORPUS_WITH_ENGAGE, "corpus_with_engagement"),
        (CLEANED_CORPUS, "cleaned_corpus"),
    ]:
        if path.exists():
            t0 = time.time()
            df = pl.read_parquet(path)
            logger.info(f"  Loaded {name}: {len(df):,} rows × {len(df.columns)} cols in {time.time() - t0:.2f}s")
            return df

    raise FileNotFoundError("No corpus found. Run phase0_cleaning.py first.")


def build_text_features(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3 — Building text features (Group A)")
    logger.info("=" * 70)

    if "emoji_density" not in df.columns:
        df = df.with_columns(
            (pl.col("emoji_count") / pl.col("char_count").clip(lower_bound=1)).alias("emoji_density")
        )
        logger.info("  Computed emoji_density")

    for col in TEXT_FEATURES:
        if col not in df.columns:
            logger.warning(f"  Feature '{col}' not present — will be 0-filled")
            df = df.with_columns(pl.lit(0).cast(pl.Float32).alias(col))

    logger.info(f"  Text features ready: {[c for c in TEXT_FEATURES if c in df.columns]}")
    return df


def build_engagement_features(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3 — Building engagement features (Group B)")
    logger.info("=" * 70)

    df = df.with_columns([
        pl.col("like_count").log1p().alias("like_count_log"),
        pl.col("reply_count").log1p().alias("reply_count_log"),
        ((pl.col("reply_count") + 1) / (pl.col("like_count") + 1)).alias("reply_to_like_ratio"),
    ])

    if "days_since_publish" in df.columns:
        df = df.with_columns([
            (pl.col("like_count") / pl.col("days_since_publish").clip(lower_bound=1)).alias("likes_per_day"),
            (pl.col("reply_count") / pl.col("days_since_publish").clip(lower_bound=1)).alias("replies_per_day"),
        ])
    else:
        if "published_at" in df.columns and "crawled_at" in df.columns:
            crawled_expr = pl.col("crawled_at")
            if df["crawled_at"].dtype == pl.String:
                crawled_expr = crawled_expr.str.to_datetime(time_zone="UTC")
            elif df["crawled_at"].dtype.time_zone is None:
                crawled_expr = crawled_expr.dt.replace_time_zone("UTC")

            published_expr = pl.col("published_at")
            if df["published_at"].dtype == pl.String:
                published_expr = published_expr.str.to_datetime(time_zone="UTC")
            elif df["published_at"].dtype.time_zone is None:
                published_expr = published_expr.dt.replace_time_zone("UTC")

            df = df.with_columns(
                (
                    (crawled_expr - published_expr)
                    .dt.total_seconds() / 86400.0
                ).alias("days_since_publish")
            )
            df = df.with_columns([
                (pl.col("like_count") / pl.col("days_since_publish").clip(lower_bound=1)).alias("likes_per_day"),
                (pl.col("reply_count") / pl.col("days_since_publish").clip(lower_bound=1)).alias("replies_per_day"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0.0).alias("likes_per_day"),
                pl.lit(0.0).alias("replies_per_day"),
            ])

    if "engagement_tier" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("like_count") < 10).then(pl.lit(0))
            .when(pl.col("like_count") < 100).then(pl.lit(1))
            .when(pl.col("like_count") < 1000).then(pl.lit(2))
            .when(pl.col("like_count") < 10000).then(pl.lit(3))
            .otherwise(pl.lit(4))
            .alias("engagement_tier_ordinal")
        )
    else:
        tier_map = {"Micro (<10)": 0, "Small (10-100)": 1, "Medium (100-1K)": 2, "Large (1K-10K)": 3, "Viral (>=10K)": 4}
        df = df.with_columns(
            pl.col("engagement_tier").replace(tier_map, default=0).alias("engagement_tier_ordinal")
        )

    logger.info(f"  Engagement features built: like_count_log, reply_count_log, ratio, velocity, tier_ordinal")
    return df


def build_metadata_features(df: pl.DataFrame, train_mask: np.ndarray) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3 — Building metadata features (Group C)")
    logger.info("=" * 70)

    if "published_at" in df.columns:
        df = df.with_columns([
            pl.col("published_at").dt.hour().alias("hour_of_day"),
            pl.col("published_at").dt.weekday().alias("day_of_week"),
            pl.col("published_at").dt.month().alias("month"),
            (pl.col("published_at").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        ])

        if "days_since_publish" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("days_since_publish"))

        published_expr = pl.col("published_at")
        if df["published_at"].dtype == pl.String:
            published_expr = published_expr.str.to_datetime(time_zone="UTC")
        elif df["published_at"].dtype.time_zone is None:
            published_expr = published_expr.dt.replace_time_zone("UTC")

        df = df.with_columns(
            pl.when(published_expr < pl.lit(datetime(2020, 1, 1, tzinfo=UTC))).then(pl.lit(0))
            .when(published_expr <= pl.lit(datetime(2021, 12, 31, tzinfo=UTC))).then(pl.lit(1))
            .otherwise(pl.lit(2))
            .alias("era_flag")
        )
        logger.info("  Temporal features: hour_of_day, day_of_week, month, is_weekend, era_flag, days_since_publish")
    else:
        for feat in ["hour_of_day", "day_of_week", "month", "is_weekend", "era_flag", "days_since_publish"]:
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias(feat))
        logger.warning("  published_at not available — zero-filling temporal features")

    if "source_query" in df.columns:
        le_label = LabelEncoder()
        labels_encoded = le_label.fit_transform(df["label"].to_numpy())

        train_queries = df["source_query"].to_numpy()[train_mask]
        train_labels = labels_encoded[train_mask]

        query_target_map = {}
        query_freq_map = {}
        for query in np.unique(train_queries):
            mask_q = train_queries == query
            query_target_map[query] = float(train_labels[mask_q].mean())
            query_freq_map[query] = int(mask_q.sum())

        global_mean = float(train_labels.mean())
        k = 10

        all_queries = df["source_query"].to_numpy()
        smoothed_enc = np.array([
            (query_target_map.get(q, global_mean) * query_freq_map.get(q, 0) + k * global_mean) /
            (query_freq_map.get(q, 0) + k)
            for q in all_queries
        ], dtype=np.float32)

        freq_enc = np.array([
            query_freq_map.get(q, 0)
            for q in all_queries
        ], dtype=np.float32)

        df = df.with_columns([
            pl.Series("source_query_target_encoded", smoothed_enc, dtype=pl.Float32),
            pl.Series("source_query_freq_encoded", freq_enc, dtype=pl.Float32),
        ])
        logger.info(f"  Source query target encoding computed (smoothing k={k})")
        logger.info(f"  Query map: {len(query_target_map)} unique queries")
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("source_query_target_encoded"),
            pl.lit(0.0).alias("source_query_freq_encoded"),
        ])

    if "language_group" in df.columns or "primary_language" in df.columns:
        lang_col = "language_group" if "language_group" in df.columns else "primary_language"
        lang_vals = df[lang_col].to_numpy()
        lang_counts = {}
        for l in lang_vals:
            lang_counts[l] = lang_counts.get(l, 0) + 1
        top_langs = sorted(lang_counts, key=lang_counts.get, reverse=True)[:TOP_N_LANGUAGES]

        for lang in top_langs:
            safe_name = re.sub(r"[^a-zA-Z0-9]", "_", str(lang))
            df = df.with_columns(
                (pl.col(lang_col) == lang).cast(pl.Int8).alias(f"lang_{safe_name}")
            )
            logger.info(f"  One-hot: lang_{safe_name}")
    else:
        logger.warning("  No language column found — skipping language one-hot encoding")

    return df


def build_feature_inventory(df: pl.DataFrame, all_features: list[str]) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3 — Building Feature Inventory Table")
    logger.info("=" * 70)

    feature_meta = {
        "char_count": ("text", "int", "len(comment_text)", "StandardScaler"),
        "word_count": ("text", "int", "space-split token count", "StandardScaler"),
        "avg_word_length": ("text", "float", "mean(token lengths)", "StandardScaler"),
        "uppercase_ratio": ("text", "float", "n_upper / char_count", "StandardScaler"),
        "exclamation_count": ("text", "int", "count('!')", "RobustScaler"),
        "question_count": ("text", "int", "count('?')", "RobustScaler"),
        "hashtag_count": ("text", "int", "count('#word')", "RobustScaler"),
        "mention_count": ("text", "int", "count('@word')", "RobustScaler"),
        "emoji_count": ("text", "int", "unicode emoji count", "RobustScaler"),
        "emoji_density": ("text", "float", "emoji_count / char_count", "RobustScaler"),
        "like_count_log": ("engagement", "float", "log1p(like_count)", "RobustScaler"),
        "reply_count_log": ("engagement", "float", "log1p(reply_count)", "RobustScaler"),
        "reply_to_like_ratio": ("engagement", "float", "(reply+1)/(like+1)", "RobustScaler"),
        "engagement_tier_ordinal": ("engagement", "int", "ordinal: Micro=0..Viral=4", "passthrough"),
        "likes_per_day": ("engagement", "float", "like_count / days_since_publish", "RobustScaler"),
        "replies_per_day": ("engagement", "float", "reply_count / days_since_publish", "RobustScaler"),
        "source_query_target_encoded": ("metadata", "float", "smoothed mean label per query (train only)", "StandardScaler"),
        "source_query_freq_encoded": ("metadata", "float", "frequency of query in train set", "StandardScaler"),
        "hour_of_day": ("metadata", "int", "published_at.hour", "StandardScaler"),
        "day_of_week": ("metadata", "int", "published_at.weekday()", "StandardScaler"),
        "month": ("metadata", "int", "published_at.month", "StandardScaler"),
        "is_weekend": ("metadata", "binary", "weekday >= 5", "passthrough"),
        "days_since_publish": ("metadata", "float", "crawled_at - published_at (days)", "RobustScaler"),
        "era_flag": ("metadata", "int", "0=pre-COVID, 1=COVID, 2=post-COVID", "passthrough"),
    }

    for etype in ["face_positive", "face_negative", "face_neutral", "symbol_positive", "symbol_negative", "other"]:
        cnt_col = f"emoji_{etype}_count"
        pres_col = f"emoji_{etype}_present"
        feature_meta[cnt_col] = ("text_emoji", "int", f"count of {etype} emojis", "RobustScaler")
        feature_meta[pres_col] = ("text_emoji", "binary", f"has {etype} emoji", "passthrough")

    rows = []
    for feat in all_features:
        if feat in df.columns:
            category, dtype, logic, scaling = feature_meta.get(feat, ("other", "unknown", "derived", "StandardScaler"))
            null_count = df[feat].is_null().sum()
            rows.append({
                "feature_name": feat,
                "category": category,
                "dtype": dtype,
                "derivation_logic": logic,
                "intended_scaling_method": scaling,
                "null_count": null_count,
                "in_dataframe": True,
            })

    inventory = pl.DataFrame(rows).sort(["category", "feature_name"])
    logger.info(f"  Feature inventory: {len(inventory)} features across categories")
    for cat in inventory["category"].unique().to_list():
        n = inventory.filter(pl.col("category") == cat).height
        logger.info(f"    {cat:20s}: {n} features")

    return inventory


def compute_vif_iterative(
    feature_matrix: np.ndarray, feature_names: list[str]
) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3 — VIF Audit & Multicollinearity Remediation")
    logger.info("=" * 70)

    try:
        print("StatsModel import")
    except ImportError:
        logger.error("  statsmodels not installed — installing fallback VIF via correlation")
        logger.warning("  Skipping VIF computation — install statsmodels for full audit")
        rows = [{"feature": f, "VIF_before": -1.0, "action": "skipped", "VIF_after": -1.0, "rationale": "statsmodels unavailable"} for f in feature_names]
        return pl.DataFrame(rows), feature_matrix, feature_names

    sample_size = min(50000, feature_matrix.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(feature_matrix.shape[0], sample_size, replace=False)
    X_sample = feature_matrix[idx]

    nan_mask = np.isnan(X_sample).any(axis=0)
    inf_mask = np.isinf(X_sample).any(axis=0)
    bad_cols = np.where(nan_mask | inf_mask)[0]
    if len(bad_cols) > 0:
        logger.warning(f"  Replacing NaN/Inf in {len(bad_cols)} columns before VIF")
        X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)

    std_mask = X_sample.std(axis=0) > 1e-8
    zero_var_cols = [feature_names[i] for i in range(len(feature_names)) if not std_mask[i]]
    if zero_var_cols:
        logger.warning(f"  Zero-variance columns (dropped): {zero_var_cols}")

    active_idx = [i for i in range(len(feature_names)) if std_mask[i]]
    X_active = X_sample[:, active_idx]
    names_active = [feature_names[i] for i in active_idx]

    vif_before = {}
    logger.info(f"  Computing initial VIF for {len(names_active)} features...")
    for i in tqdm(range(X_active.shape[1]), desc="VIF computation", ncols=80):
        try:
            vif_before[names_active[i]] = variance_inflation_factor(X_active, i)
        except Exception:
            vif_before[names_active[i]] = float("inf")

    high_vif = [(n, v) for n, v in vif_before.items() if v > VIF_THRESHOLD]
    logger.info(f"  Features with VIF > {VIF_THRESHOLD}: {len(high_vif)}")
    for name, vif in sorted(high_vif, key=lambda x: x[1], reverse=True):
        logger.info(f"    {name:40s}: VIF = {vif:.2f}")

    actions = {}
    dropped_features = set()

    known_collinear_pairs = [
        ("char_count", "word_count", "word_density = word_count / char_count"),
        ("like_count_log", "likes_per_day", "keep both — different semantic"),
        ("reply_count_log", "replies_per_day", "keep both — different semantic"),
    ]

    for feat_a, feat_b, rationale in known_collinear_pairs:
        if feat_a in vif_before and feat_b in vif_before:
            vif_a = vif_before[feat_a]
            vif_b = vif_before[feat_b]
            if vif_a > VIF_THRESHOLD and vif_b > VIF_THRESHOLD:
                if "keep both" in rationale.lower():
                    actions[feat_a] = ("kept", rationale)
                    actions[feat_b] = ("kept", rationale)
                else:
                    weaker = feat_a if vif_a < vif_b else feat_b
                    stronger = feat_b if weaker == feat_a else feat_a
                    actions[weaker] = ("dropped", f"high VIF pair with {stronger}: {rationale}")
                    dropped_features.add(weaker)

    remaining_high_vif = [
        (n, v) for n, v in vif_before.items()
        if v > VIF_THRESHOLD and n not in actions and n not in dropped_features
    ]
    for name, vif in sorted(remaining_high_vif, key=lambda x: x[1], reverse=True):
        actions[name] = ("dropped", f"VIF={vif:.2f} > {VIF_THRESHOLD}, no known pair rule")
        dropped_features.add(name)
        logger.info(f"    Dropping {name} (VIF={vif:.2f})")

    final_idx = [
        i for i, n in enumerate(names_active)
        if n not in dropped_features
    ]
    X_final = X_active[:, final_idx]
    names_final = [names_active[i] for i in final_idx]

    vif_after = {}
    logger.info(f"  Computing post-remediation VIF for {len(names_final)} features...")
    for i in tqdm(range(X_final.shape[1]), desc="VIF post-check", ncols=80):
        try:
            vif_after[names_final[i]] = variance_inflation_factor(X_final, i)
        except Exception:
            vif_after[names_final[i]] = float("inf")

    remaining_high = [(n, v) for n, v in vif_after.items() if v > VIF_THRESHOLD]
    if remaining_high:
        logger.warning(f"  Still {len(remaining_high)} features with VIF > {VIF_THRESHOLD} after remediation:")
        for n, v in remaining_high:
            logger.warning(f"    {n}: VIF={v:.2f}")
    else:
        logger.info(f"  All features now have VIF <= {VIF_THRESHOLD}")

    rows = []
    all_names = list(set(list(vif_before.keys()) + zero_var_cols))
    for name in all_names:
        action, rationale = actions.get(name, ("kept", "VIF within threshold"))
        if name in zero_var_cols:
            action, rationale = "dropped", "zero variance"
        rows.append({
            "feature": name,
            "VIF_before": round(vif_before.get(name, 0.0), 3),
            "action": action,
            "VIF_after": round(vif_after.get(name, 0.0), 3) if name in vif_after else 0.0,
            "rationale": rationale,
        })

    vif_table = pl.DataFrame(rows).sort("VIF_before", descending=True)

    full_idx = [feature_names.index(n) for n in names_final if n in feature_names]
    X_remediated = feature_matrix[:, full_idx]

    logger.info(f"  Features after VIF remediation: {len(names_final)} (was {len(names_active)})")
    return vif_table, X_remediated, names_final


def compute_correlation_matrix(X: np.ndarray, feature_names: list[str]) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 3 — Computing post-remediation correlation matrix")
    logger.info("=" * 70)

    corr = np.corrcoef(X.T)
    rows = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            rows.append({
                "feature_1": feature_names[i],
                "feature_2": feature_names[j],
                "pearson_r": round(float(corr[i, j]), 4),
                "abs_r": round(abs(float(corr[i, j])), 4),
            })

    corr_df = pl.DataFrame(rows).sort("abs_r", descending=True)
    high_corr = corr_df.filter(pl.col("abs_r") > 0.8)
    logger.info(f"  Feature pairs with |r| > 0.8: {len(high_corr)}")
    for row in high_corr.head(10).iter_rows(named=True):
        logger.info(f"    {row['feature_1']:30s} × {row['feature_2']:30s}: r={row['pearson_r']:.4f}")

    return corr_df


def fit_and_apply_scalers(
    X: np.ndarray,
    feature_names: list[str],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[dict], dict]:
    logger.info("=" * 70)
    logger.info("PHASE 3 — Fitting scalers (StandardScaler / RobustScaler)")
    logger.info("=" * 70)

    ROBUST_FEATURES = {
        "like_count_log", "reply_count_log", "reply_to_like_ratio",
        "likes_per_day", "replies_per_day", "exclamation_count",
        "question_count", "hashtag_count", "mention_count",
        "emoji_count", "emoji_density",
    }

    X_scaled = X.copy()
    registry = []
    scalers = {}

    for i, feat in enumerate(tqdm(feature_names, desc="Scaling features", ncols=80)):
        col = X[:, i]
        train_col = col[train_mask]

        if feat in ROBUST_FEATURES:
            scaler = RobustScaler()
            scaler_type = "RobustScaler"
        else:
            scaler = StandardScaler()
            scaler_type = "StandardScaler"

        scaler.fit(train_col.reshape(-1, 1))
        X_scaled[:, i] = scaler.transform(col.reshape(-1, 1)).flatten()
        scalers[feat] = scaler

        if scaler_type == "RobustScaler":
            params = f"center={scaler.center_[0]:.4f}, scale={scaler.scale_[0]:.4f}"
        else:
            params = f"mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}"

        registry.append({
            "feature": feat,
            "scaler_type": scaler_type,
            "params": params,
        })

    scaling_registry = pl.DataFrame(registry)
    logger.info(f"  Scaled {len(feature_names)} features")
    robust_count = sum(1 for r in registry if r["scaler_type"] == "RobustScaler")
    standard_count = len(registry) - robust_count
    logger.info(f"  RobustScaler: {robust_count}  StandardScaler: {standard_count}")

    return X_scaled, registry, scalers


def plot_vif_before_after(vif_table: pl.DataFrame):
    logger.info("Generating VIF before/after comparison chart")

    kept = vif_table.filter(pl.col("action") == "kept").sort("VIF_before", descending=True).head(20)
    dropped = vif_table.filter(pl.col("action") == "dropped").sort("VIF_before", descending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(kept) * 0.4)))

    feats_k = kept["feature"].to_list()
    vif_before_k = kept["VIF_before"].to_list()
    vif_after_k = kept["VIF_after"].to_list()
    y_k = range(len(feats_k))

    axes[0].barh(y_k, vif_before_k, 0.4, label="VIF Before", color="#C44E52", alpha=0.8)
    axes[0].barh([y + 0.4 for y in y_k], vif_after_k, 0.4, label="VIF After", color="#55A868", alpha=0.8)
    axes[0].set_yticks([y + 0.2 for y in y_k])
    axes[0].set_yticklabels(feats_k, fontsize=8)
    axes[0].axvline(VIF_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"VIF={VIF_THRESHOLD}")
    axes[0].set_title("VIF Before vs After Remediation (Kept Features)", fontweight="bold")
    axes[0].set_xlabel("VIF Score")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()

    action_counts = vif_table.group_by("action").agg(pl.len().alias("count")).to_dicts()
    ax_labels = [r["action"] for r in action_counts]
    ax_counts = [r["count"] for r in action_counts]
    colors = ["#55A868" if a == "kept" else "#C44E52" if a == "dropped" else "#DD8452" for a in ax_labels]
    axes[1].bar(ax_labels, ax_counts, color=colors, edgecolor="white")
    for i, (l, c) in enumerate(zip(ax_labels, ax_counts)):
        axes[1].text(i, c + 0.3, str(c), ha="center", fontweight="bold")
    axes[1].set_title("Feature Actions (VIF Audit)", fontweight="bold")
    axes[1].set_ylabel("Count")

    plt.suptitle("VIF Multicollinearity Audit", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t8_vif_audit.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't8_vif_audit.png'}")


def plot_correlation_heatmap(corr_df: pl.DataFrame, feature_names: list[str]):
    logger.info("Generating post-remediation correlation heatmap")

    if len(feature_names) > 30:
        feature_names_plot = feature_names[:30]
        logger.info(f"  Truncating heatmap to first 30 features for readability")
    else:
        feature_names_plot = feature_names

    n = len(feature_names_plot)
    corr_matrix = np.zeros((n, n))

    for row in corr_df.iter_rows(named=True):
        if row["feature_1"] in feature_names_plot and row["feature_2"] in feature_names_plot:
            i = feature_names_plot.index(row["feature_1"])
            j = feature_names_plot.index(row["feature_2"])
            corr_matrix[i, j] = row["pearson_r"]
            corr_matrix[j, i] = row["pearson_r"]
    np.fill_diagonal(corr_matrix, 1.0)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), max(10, n * 0.4)))
    sns.heatmap(
        corr_matrix,
        xticklabels=feature_names_plot,
        yticklabels=feature_names_plot,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=n <= 20, fmt=".2f" if n <= 20 else "",
        linewidths=0.3 if n <= 20 else 0, ax=ax,
    )
    ax.set_title("Post-Remediation Feature Correlation Matrix", fontweight="bold")
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t8_final_correlation_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / 't8_final_correlation_matrix.png'}")


def save_outputs(
    df_features: pl.DataFrame,
    feature_inventory: pl.DataFrame,
    vif_table: pl.DataFrame,
    corr_df: pl.DataFrame,
    scaling_registry: pl.DataFrame,
    final_feature_names: list[str],
    scalers: dict,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
):
    logger.info("=" * 70)
    logger.info("PHASE 3 — Saving outputs")
    logger.info("=" * 70)

    feature_inventory.write_parquet(OUTPUT_PARQUET / "t8_feature_inventory.parquet")
    logger.info(f"  Feature inventory     → {OUTPUT_PARQUET / 't8_feature_inventory.parquet'}")

    vif_table.write_parquet(OUTPUT_PARQUET / "t8_vif_audit.parquet")
    logger.info(f"  VIF audit             → {OUTPUT_PARQUET / 't8_vif_audit.parquet'}")

    corr_df.write_parquet(OUTPUT_PARQUET / "t8_final_correlation_matrix.parquet")
    logger.info(f"  Correlation matrix    → {OUTPUT_PARQUET / 't8_final_correlation_matrix.parquet'}")

    scaling_registry.write_parquet(OUTPUT_PARQUET / "t8_scaling_registry.parquet")
    logger.info(f"  Scaling registry      → {OUTPUT_PARQUET / 't8_scaling_registry.parquet'}")

    save_cols = [c for c in final_feature_names if c in df_features.columns]
    save_cols += ["label", "comment_id"]
    save_cols = list(dict.fromkeys([c for c in save_cols if c in df_features.columns]))
    df_features.select(save_cols).write_parquet(OUTPUT_PARQUET / "model_ready_features.parquet")
    logger.info(f"  Model-ready features  → {OUTPUT_PARQUET / 'model_ready_features.parquet'}")

    split_df = pl.DataFrame({
        "is_train": pl.Series(train_mask.astype(bool)),
        "is_test": pl.Series(test_mask.astype(bool)),
    })
    split_df.write_parquet(OUTPUT_PARQUET / "train_test_split.parquet")
    logger.info(f"  Train/test split      → {OUTPUT_PARQUET / 'train_test_split.parquet'}")

    joblib.dump(scalers, OUTPUT_MODELS / "scaling_pipeline.joblib")
    logger.info(f"  Scaler pipeline       → {OUTPUT_MODELS / 'scaling_pipeline.joblib'}")

    pl.DataFrame({"feature_name": final_feature_names}).write_parquet(
        OUTPUT_PARQUET / "final_feature_names.parquet"
    )
    logger.info(f"  Final feature names   → {OUTPUT_PARQUET / 'final_feature_names.parquet'}")


import re as _re


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 3 FEATURE ENGINEERING — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()
    df = load_best_corpus()

    labels = df["label"].to_numpy()
    le_global = LabelEncoder()
    y_global = le_global.fit_transform(labels)

    logger.info("  Creating 80/20 stratified train-test split...")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y_global, random_state=42
    )
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask
    logger.info(f"  Train: {train_mask.sum():,}  Test: {test_mask.sum():,}")

    df = build_text_features(df)
    df = build_engagement_features(df)
    df = build_metadata_features(df, train_mask)

    all_features = []
    for feat in TEXT_FEATURES:
        if feat in df.columns:
            all_features.append(feat)
    for feat in ENGAGEMENT_FEATURES + ["engagement_tier_ordinal"]:
        if feat in df.columns and feat not in all_features:
            all_features.append(feat)
    for feat in ["source_query_target_encoded", "source_query_freq_encoded",
                 "hour_of_day", "day_of_week", "month", "is_weekend",
                 "days_since_publish", "era_flag"]:
        if feat in df.columns and feat not in all_features:
            all_features.append(feat)
    for col in df.columns:
        if col.startswith("lang_") and col not in all_features:
            all_features.append(col)
    for etype in ["face_positive", "face_negative", "face_neutral", "symbol_positive", "symbol_negative", "other"]:
        for suffix in ["_count", "_present"]:
            col = f"emoji_{etype}{suffix}"
            if col in df.columns and col not in all_features:
                all_features.append(col)

    logger.info(f"  Total features assembled: {len(all_features)}")

    feature_inventory = build_feature_inventory(df, all_features)

    X_raw = np.column_stack([
        df[f].fill_null(0).cast(pl.Float64).to_numpy()
        for f in all_features
    ]).astype(np.float32)
    logger.info(f"  Raw feature matrix: {X_raw.shape}")

    vif_table, X_remediated, final_feature_names = compute_vif_iterative(X_raw, all_features)
    corr_df = compute_correlation_matrix(X_remediated, final_feature_names)
    X_scaled, registry, scalers = fit_and_apply_scalers(X_remediated, final_feature_names, train_mask)
    scaling_registry = pl.DataFrame(registry)

    plot_vif_before_after(vif_table)
    plot_correlation_heatmap(corr_df, final_feature_names)

    save_outputs(
        df, feature_inventory, vif_table, corr_df,
        scaling_registry, final_feature_names, scalers,
        train_mask, test_mask,
    )

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"PHASE 3 COMPLETE — {len(final_feature_names)} final features — elapsed: {elapsed:.1f}s")
    logger.info("=" * 70)
