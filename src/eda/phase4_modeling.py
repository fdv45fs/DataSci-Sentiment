import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import os
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_READY = Path("output_data/parquet/model_ready_features.parquet")
FINAL_FEATURE_NAMES = Path("output_data/parquet/final_feature_names.parquet")
TRAIN_TEST_SPLIT = Path("output_data/parquet/train_test_split.parquet")
CLEANED_CORPUS = Path("output_data/parquet/cleaned_corpus.parquet")
OUTPUT_PARQUET = Path("output_data/parquet")
OUTPUT_IMG = Path("output_data/img")
OUTPUT_MODELS = Path("output_data/models")

LABEL_ORDER = ["positive", "neutral", "negative"]
LABEL_COLORS = {"positive": "#55A868", "neutral": "#4C72B0", "negative": "#C44E52"}
N_FOLDS = 5
OPTUNA_N_TRIALS = 50
OPTUNA_PATIENCE = 3
RANDOM_STATE = 42
SHAP_SAMPLE = 3000


def setup_output_dirs():
    OUTPUT_PARQUET.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], LabelEncoder]:
    logger.info("=" * 70)
    logger.info("PHASE 4 — Loading model-ready features and train/test split")
    logger.info("=" * 70)

    for required in [MODEL_READY, CLEANED_CORPUS]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}. Run preceding phases first.")

    t0 = time.time()
    df_feat = pl.read_parquet(MODEL_READY)
    df_corpus = pl.read_parquet(CLEANED_CORPUS)
    logger.info(f"  Features loaded: {len(df_feat):,} rows × {len(df_feat.columns)} cols in {time.time() - t0:.2f}s")

    if FINAL_FEATURE_NAMES.exists():
        feature_names = pl.read_parquet(FINAL_FEATURE_NAMES)["feature_name"].to_list()
        feature_names = [f for f in feature_names if f in df_feat.columns]
    else:
        feature_names = [c for c in df_feat.columns if c not in {"label", "comment_id"}]

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(df_feat["label"].to_numpy())

    X = np.column_stack([
        df_feat[f].fill_null(0).cast(pl.Float64).to_numpy()
        for f in feature_names
    ]).astype(np.float32)

    if TRAIN_TEST_SPLIT.exists():
        split_df = pl.read_parquet(TRAIN_TEST_SPLIT)
        train_mask = split_df["is_train"].to_numpy().astype(bool)
    else:
        logger.warning("  train_test_split.parquet not found — recreating split")
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[train_idx] = True

    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]

    logger.info(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    for label_idx, label_name in enumerate(le.classes_):
        n_tr = (y_train == label_idx).sum()
        n_te = (y_test == label_idx).sum()
        logger.info(f"    {label_name:10s}: train={n_tr:>7,}  test={n_te:>6,}")

    texts_train = df_corpus["comment_text"].to_numpy()[train_mask]
    texts_test = df_corpus["comment_text"].to_numpy()[~train_mask]

    return X_train, X_test, y_train, y_test, feature_names, le, texts_train, texts_test


def compute_class_weights(y: np.ndarray, n_classes: int) -> dict:
    counts = np.bincount(y, minlength=n_classes)
    n_total = len(y)
    weights = {i: n_total / (n_classes * cnt) for i, cnt in enumerate(counts)}
    logger.info(f"  Class weights: {weights}")
    return weights


def build_cv_fold_summary(
    X: np.ndarray, y: np.ndarray, le: LabelEncoder
) -> tuple[pl.DataFrame, StratifiedKFold]:
    logger.info("=" * 70)
    logger.info(f"PHASE 4 — Table 10a: StratifiedKFold(k={N_FOLDS}) on training data")
    logger.info("=" * 70)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        n_train = len(tr_idx)
        n_val = len(val_idx)
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        class_weights = compute_class_weights(y_tr, len(le.classes_))
        weights_str = " | ".join(f"{le.classes_[i]}={w:.3f}" for i, w in class_weights.items())

        pct_pos_train = (y_tr == le.transform(["positive"])[0]).sum() / n_train * 100
        pct_pos_val = (y_val == le.transform(["positive"])[0]).sum() / n_val * 100

        rows.append({
            "fold": fold_idx,
            "train_n": n_train,
            "val_n": n_val,
            "pct_positive_train": round(pct_pos_train, 2),
            "pct_positive_val": round(pct_pos_val, 2),
            "class_weights": weights_str,
        })

        logger.info(
            f"  Fold {fold_idx}: train={n_train:,}  val={n_val:,}  "
            f"pos_train={pct_pos_train:.1f}%  pos_val={pct_pos_val:.1f}%"
        )

    return pl.DataFrame(rows), skf


def build_smote_pipeline(estimator, class_weights: dict):
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        pipe = ImbPipeline([
            ("smote", smote),
            ("clf", estimator),
        ])
        logger.info(f"  SMOTE pipeline built for {type(estimator).__name__}")
        return pipe
    except ImportError:
        logger.warning("  imbalanced-learn not installed — SMOTE not available, using class_weight instead")
        return estimator


def evaluate_model_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
    le: LabelEncoder,
    model_name: str,
    use_smote: bool = False,
) -> dict:
    logger.info(f"  Evaluating {model_name} with {N_FOLDS}-fold CV...")
    fold_f1s = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if use_smote:
            pipe = build_smote_pipeline(model, compute_class_weights(y_tr, len(le.classes_)))
            try:
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_val)
            except Exception as e:
                logger.warning(f"    Fold {fold_idx} SMOTE failed: {e} — using plain fit")
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

        fold_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_f1s.append(fold_f1)
        logger.info(f"    Fold {fold_idx}: macro_F1={fold_f1:.4f}")

    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))
    logger.info(f"  {model_name}: CV macro_F1={mean_f1:.4f} ± {std_f1:.4f}")

    return {"model_name": model_name, "cv_mean_f1": mean_f1, "cv_std_f1": std_f1, "fold_f1s": fold_f1s}


def train_baseline_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    texts_train: np.ndarray,
    texts_test: np.ndarray,
    skf: StratifiedKFold,
    le: LabelEncoder,
) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 4 — Baseline Models Training & CV Evaluation")
    logger.info("=" * 70)

    class_weights = compute_class_weights(y_train, len(le.classes_))

    models_config = {
        "LogReg_L2_engineered": LogisticRegression(
            C=1.0, class_weight=class_weights, max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs", multi_class="auto"
        ),
        "LinearSVC_engineered": LinearSVC(
            C=0.1, class_weight=class_weights, max_iter=5000, random_state=RANDOM_STATE
        ),
    }

    leaderboard_rows = []

    for model_name, model in models_config.items():
        logger.info(f"\n  ─── {model_name} ───")
        cv_result = evaluate_model_cv(model, X_train, y_train, skf, le, model_name, use_smote=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

        logger.info(f"  Test acc={acc:.4f}  macro_P={macro_p:.4f}  macro_R={macro_r:.4f}  macro_F1={macro_f1:.4f}")
        for i, cls in enumerate(le.classes_):
            logger.info(f"    {cls:10s} F1={per_class_f1[i]:.4f}")

        row = {
            "model": model_name,
            "features": "engineered",
            "cv_macro_f1": round(cv_result["cv_mean_f1"], 4),
            "cv_std_f1": round(cv_result["cv_std_f1"], 4),
            "test_accuracy": round(acc, 4),
            "test_macro_P": round(macro_p, 4),
            "test_macro_R": round(macro_r, 4),
            "test_macro_F1": round(macro_f1, 4),
        }
        for i, cls in enumerate(le.classes_):
            row[f"test_F1_{cls}"] = round(float(per_class_f1[i]), 4)
        leaderboard_rows.append(row)

        joblib.dump(model, OUTPUT_MODELS / f"{model_name}_baseline.joblib")
        logger.info(f"  Saved model → {OUTPUT_MODELS / f'{model_name}_baseline.joblib'}")

    logger.info("\n  Training TF-IDF baselines on raw text...")
    tfidf_models = {
        "LogReg_TFIDF": LogisticRegression(
            C=1.0, class_weight=class_weights, max_iter=1000, random_state=RANDOM_STATE
        ),
        "LinearSVC_TFIDF": LinearSVC(
            C=0.1, class_weight=class_weights, max_iter=5000, random_state=RANDOM_STATE
        ),
    }

    tfidf_vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=100000,
        sublinear_tf=True,
    )
    X_tfidf_train = tfidf_vec.fit_transform(
        [t if isinstance(t, str) else "" for t in texts_train]
    )
    X_tfidf_test = tfidf_vec.transform(
        [t if isinstance(t, str) else "" for t in texts_test]
    )
    logger.info(f"  TF-IDF matrix: train={X_tfidf_train.shape}  test={X_tfidf_test.shape}")

    for model_name, model in tfidf_models.items():
        logger.info(f"\n  ─── {model_name} ───")

        model.fit(X_tfidf_train, y_train)
        y_pred = model.predict(X_tfidf_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

        logger.info(f"  Test acc={acc:.4f}  macro_F1={macro_f1:.4f}")

        row = {
            "model": model_name,
            "features": "tfidf_1-2gram",
            "cv_macro_f1": None,
            "cv_std_f1": None,
            "test_accuracy": round(acc, 4),
            "test_macro_P": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "test_macro_R": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "test_macro_F1": round(macro_f1, 4),
        }
        for i, cls in enumerate(le.classes_):
            row[f"test_F1_{cls}"] = round(float(per_class_f1[i]), 4)
        leaderboard_rows.append(row)

    leaderboard = pl.DataFrame(leaderboard_rows).sort("test_macro_F1", descending=True)
    logger.info("\n  === MODEL LEADERBOARD ===")
    for row in leaderboard.iter_rows(named=True):
        logger.info(
            f"  {row['model']:35s}: macro_F1={row['test_macro_F1']:.4f}  acc={row['test_accuracy']:.4f}"
        )

    return leaderboard


def run_optuna_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    skf: StratifiedKFold,
    le: LabelEncoder,
) -> tuple[pl.DataFrame, dict]:
    logger.info("=" * 70)
    logger.info(f"PHASE 4 — Optuna Hyperparameter Optimization ({OPTUNA_N_TRIALS} trials)")
    logger.info("=" * 70)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("  optuna not installed — skipping optimization")
        logger.error("  Install with: pip install optuna")
        return pl.DataFrame(), {}

    class_weights = compute_class_weights(y_train, len(le.classes_))
    best_f1_history = []
    no_improve_count = 0

    def objective(trial):
        nonlocal no_improve_count, best_f1_history

        model_type = trial.suggest_categorical("model_type", ["LogisticRegression", "LinearSVC"])

        if model_type == "LogisticRegression":
            C = trial.suggest_float("C", 1e-3, 100.0, log=True)
            solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
            model = LogisticRegression(
                C=C, solver=solver, class_weight=class_weights,
                max_iter=1000, random_state=RANDOM_STATE, multi_class="auto",
            )
        else:
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            model = LinearSVC(
                C=C, class_weight=class_weights,
                max_iter=5000, random_state=RANDOM_STATE,
            )

        fold_f1s = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            pipe = build_smote_pipeline(model, class_weights)
            try:
                pipe.fit(X_train[tr_idx], y_train[tr_idx])
                y_pred = pipe.predict(X_train[val_idx])
            except Exception:
                model.fit(X_train[tr_idx], y_train[tr_idx])
                y_pred = model.predict(X_train[val_idx])
            fold_f1s.append(f1_score(y_train[val_idx], y_pred, average="macro", zero_division=0))

        mean_f1 = float(np.mean(fold_f1s))

        if not best_f1_history or mean_f1 > max(best_f1_history):
            no_improve_count = 0
        else:
            no_improve_count += 1

        best_f1_history.append(mean_f1)
        return mean_f1

    def early_stopping_callback(study, trial):
        if no_improve_count >= OPTUNA_PATIENCE:
            logger.info(f"  Early stopping triggered at trial {trial.number} (patience={OPTUNA_PATIENCE})")
            study.stop()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

    logger.info(f"  Running {OPTUNA_N_TRIALS} trials with early stopping patience={OPTUNA_PATIENCE}...")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, callbacks=[early_stopping_callback])

    best_params = study.best_params
    best_value = study.best_value
    logger.info(f"  Best CV macro_F1: {best_value:.4f}")
    logger.info(f"  Best params: {best_params}")

    optuna_rows = [{"param": k, "best_value": str(v)} for k, v in best_params.items()]
    optuna_rows.append({"param": "best_cv_macro_f1", "best_value": str(round(best_value, 6))})
    optuna_rows.append({"param": "n_trials_completed", "best_value": str(len(study.trials))})

    return pl.DataFrame(optuna_rows), best_params


def compute_shap_importance(
    model, X_train: np.ndarray, X_test: np.ndarray, feature_names: list[str]
) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 4 — SHAP Feature Importance")
    logger.info("=" * 70)

    try:
        import shap
    except ImportError:
        logger.error("  shap not installed — skipping SHAP analysis")
        logger.error("  Install with: pip install shap")
        return pl.DataFrame()

    sample_size = min(SHAP_SAMPLE, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[idx]

    try:
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_train[:1000])
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            logger.info(f"  SHAP LinearExplainer computed on {sample_size} test samples")
        else:
            logger.warning("  Model does not have coef_ — using KernelExplainer (slow)")
            explainer = shap.KernelExplainer(model.predict, X_train[:200])
            shap_values = explainer.shap_values(X_sample[:200])
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
    except Exception as e:
        logger.error(f"  SHAP computation failed: {e}")
        return pl.DataFrame()

    shap_df = pl.DataFrame({
        "feature": feature_names[:len(mean_abs_shap)],
        "mean_abs_shap": mean_abs_shap.tolist(),
    }).sort("mean_abs_shap", descending=True)

    logger.info("  Top 15 features by mean |SHAP|:")
    for row in shap_df.head(15).iter_rows(named=True):
        logger.info(f"    {row['feature']:40s}: {row['mean_abs_shap']:.5f}")

    return shap_df


def run_mcnemar_tests(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 4 — McNemar's Tests Between Models")
    logger.info("=" * 70)

    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        logger.error("  statsmodels not installed — skipping McNemar tests")
        return pl.DataFrame()

    model_names = list(predictions.keys())
    rows = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            correct_1 = predictions[m1] == y_test
            correct_2 = predictions[m2] == y_test

            b = int(np.sum(correct_1 & ~correct_2))
            c = int(np.sum(~correct_1 & correct_2))

            contingency = np.array([[0, b], [c, 0]])

            try:
                result = mcnemar(contingency, exact=False)
                rows.append({
                    "model_1": m1,
                    "model_2": m2,
                    "b": b,
                    "c": c,
                    "chi2_statistic": round(float(result.statistic), 4),
                    "p_value": round(float(result.pvalue), 6),
                    "significant": result.pvalue < 0.05,
                })
                logger.info(
                    f"  {m1} vs {m2}: chi2={result.statistic:.3f}  "
                    f"p={result.pvalue:.4f}  {'*' if result.pvalue < 0.05 else ''}"
                )
            except Exception as e:
                logger.warning(f"  McNemar failed for {m1} vs {m2}: {e}")

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def compute_error_analysis(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    df_test_meta: pl.DataFrame,
) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 4 — Error Analysis")
    logger.info("=" * 70)

    true_labels = le.inverse_transform(y_test)
    pred_labels = le.inverse_transform(y_pred)

    errors_mask = true_labels != pred_labels
    logger.info(f"  Total test errors: {errors_mask.sum():,} / {len(y_test):,} ({errors_mask.mean()*100:.2f}%)")

    error_rows = []
    for true_l in LABEL_ORDER:
        for pred_l in LABEL_ORDER:
            if true_l == pred_l:
                continue
            mask = (true_labels == true_l) & (pred_labels == pred_l)
            count = mask.sum()
            if count == 0:
                continue
            error_rows.append({
                "true_label": true_l,
                "pred_label": pred_l,
                "count": int(count),
                "pct_of_true_class": round(count / (true_labels == true_l).sum() * 100, 2),
            })

    error_df = pl.DataFrame(error_rows).sort("count", descending=True)
    logger.info("  Confusion pairs:")
    for row in error_df.iter_rows(named=True):
        logger.info(
            f"    {row['true_label']:10s} → {row['pred_label']:10s}: "
            f"{row['count']:>6,} ({row['pct_of_true_class']:.1f}% of true class)"
        )

    return error_df


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, le: LabelEncoder, model_name: str):
    logger.info(f"Generating confusion matrix for {model_name}")

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mat, title, fmt in [
        (axes[0], cm, "Raw Counts", "d"),
        (axes[1], cm_normalized, "Normalized (Row %)", ".2f"),
    ]:
        sns.heatmap(
            mat, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle(f"Confusion Matrix — {model_name}", fontweight="bold", fontsize=12)
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    plt.savefig(OUTPUT_IMG / f"t10_confusion_matrix_{safe_name}.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / f't10_confusion_matrix_{safe_name}.png'}")


def plot_shap_beeswarm(shap_df: pl.DataFrame, model_name: str):
    if len(shap_df) == 0:
        return

    logger.info("Generating SHAP importance bar chart")

    top20 = shap_df.head(20)
    feats = top20["feature"].to_list()
    vals = top20["mean_abs_shap"].to_list()

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(feats)))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(feats)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title(f"Top 20 Features by SHAP Importance — {model_name}", fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    plt.savefig(OUTPUT_IMG / f"t10_shap_importance_{safe_name}.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {OUTPUT_IMG / f't10_shap_importance_{safe_name}.png'}")


def plot_optimization_history(optuna_df: pl.DataFrame):
    if len(optuna_df) == 0:
        return

    logger.info("Saving Optuna best config table (visualization requires study object)")


def save_outputs(
    cv_fold_summary: pl.DataFrame,
    leaderboard: pl.DataFrame,
    optuna_df: pl.DataFrame,
    shap_df: pl.DataFrame,
    error_df: pl.DataFrame,
    mcnemar_df: pl.DataFrame,
):
    logger.info("=" * 70)
    logger.info("PHASE 4 — Saving outputs")
    logger.info("=" * 70)

    cv_fold_summary.write_parquet(OUTPUT_PARQUET / "t10_cv_fold_summary.parquet")
    logger.info(f"  CV fold summary  → {OUTPUT_PARQUET / 't10_cv_fold_summary.parquet'}")

    leaderboard.write_parquet(OUTPUT_PARQUET / "t10_model_leaderboard.parquet")
    logger.info(f"  Model leaderboard → {OUTPUT_PARQUET / 't10_model_leaderboard.parquet'}")

    if len(optuna_df) > 0:
        optuna_df.write_parquet(OUTPUT_PARQUET / "t10_optuna_best_config.parquet")
        logger.info(f"  Optuna config    → {OUTPUT_PARQUET / 't10_optuna_best_config.parquet'}")

    if len(shap_df) > 0:
        shap_df.write_parquet(OUTPUT_PARQUET / "t10_shap_feature_importance.parquet")
        logger.info(f"  SHAP importance  → {OUTPUT_PARQUET / 't10_shap_feature_importance.parquet'}")

    if len(error_df) > 0:
        error_df.write_parquet(OUTPUT_PARQUET / "t10_error_analysis.parquet")
        logger.info(f"  Error analysis   → {OUTPUT_PARQUET / 't10_error_analysis.parquet'}")

    if len(mcnemar_df) > 0:
        mcnemar_df.write_parquet(OUTPUT_PARQUET / "t10_mcnemar_tests.parquet")
        logger.info(f"  McNemar tests    → {OUTPUT_PARQUET / 't10_mcnemar_tests.parquet'}")


if __name__ == "__main__":
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PHASE 4 MODELING — started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    setup_output_dirs()

    X_train, X_test, y_train, y_test, feature_names, le, texts_train, texts_test = load_data()

    cv_fold_summary, skf = build_cv_fold_summary(X_train, y_train, le)
    leaderboard = train_baseline_models(
        X_train, X_test, y_train, y_test, texts_train, texts_test, skf, le
    )

    optuna_df, best_params = run_optuna_optimization(X_train, y_train, skf, le)

    best_model_name = leaderboard["model"][0]
    best_model_path = OUTPUT_MODELS / f"{best_model_name}_baseline.joblib"

    shap_df = pl.DataFrame()
    error_df = pl.DataFrame()
    mcnemar_df = pl.DataFrame()
    predictions = {}

    if best_model_path.exists():
        best_model = joblib.load(best_model_path)
        y_pred_best = best_model.predict(X_test)
        predictions[best_model_name] = y_pred_best

        shap_df = compute_shap_importance(best_model, X_train, X_test, feature_names)
        plot_confusion_matrix(y_test, y_pred_best, le, best_model_name)
        plot_shap_beeswarm(shap_df, best_model_name)

        df_test_meta = pl.DataFrame({"label": le.inverse_transform(y_test)})
        error_df = compute_error_analysis(y_test, y_pred_best, le, df_test_meta)

    for model_path in OUTPUT_MODELS.glob("*_baseline.joblib"):
        mname = model_path.stem.replace("_baseline", "")
        if mname not in predictions:
            m = joblib.load(model_path)
            if hasattr(m, "predict"):
                try:
                    if "TFIDF" in mname:
                        pass
                    else:
                        predictions[mname] = m.predict(X_test)
                except Exception:
                    pass

    if len(predictions) > 1:
        mcnemar_df = run_mcnemar_tests(y_test, predictions)

    save_outputs(cv_fold_summary, leaderboard, optuna_df, shap_df, error_df, mcnemar_df)

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info("PHASE 4 FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Best model: {leaderboard['model'][0]}")
    logger.info(f"  Best test macro_F1: {leaderboard['test_macro_F1'][0]:.4f}")
    logger.info(f"  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)
