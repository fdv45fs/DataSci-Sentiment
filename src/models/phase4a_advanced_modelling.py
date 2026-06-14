from __future__ import annotations

from datetime import datetime
import json
import logging
import math
import os
import random
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =========================
# Paths / constants
# =========================
MODEL_READY = Path("output_data/parquet/model_ready_features.parquet")
TRAIN_TEST_SPLIT = Path("output_data/parquet/train_test_split.parquet")
CLEANED_CORPUS = Path("output_data/parquet/cleaned_corpus.parquet")
OUTPUT_PARQUET = Path("output_data/model_parquet")
OUTPUT_IMG = Path("output_data/model_parquet")
OUTPUT_MODELS = Path("output_data/models")
OUTPUT_REPORTS = Path("output_data/reports")

LABEL_ORDER = ["positive", "neutral", "negative"]
RANDOM_STATE = 42
N_FOLDS = 3
SHAP_SAMPLE = 512
MLP_BATCH_SIZE = 256
MLP_MAX_EPOCHS = 5
MLP_PATIENCE = 2
XGB_EARLY_STOPPING_ROUNDS = 12
SOURCE_QUERY_FEATURES = {
    "source_query_target_encoded",
    "source_query_freq_encoded",
}
EMBEDDING_COLUMNS = [
    "embedding",
    "embedding_char",
    "embedding_word",
    "embedding_ft",
]


# =====================================
# Utility / reproducibility
# =====================================
def seed_everything(seed: int = RANDOM_STATE) -> None:
    logger.info("Setting random seed to %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        num_threads = os.cpu_count() or 1
        torch.set_num_threads(num_threads)
        logger.info("Configured PyTorch to use %s CPU threads", num_threads)
    except Exception as exc:
        logger.warning("Could not configure PyTorch runtime: %s", exc)


def setup_output_dirs() -> None:
    for path in [OUTPUT_PARQUET, OUTPUT_IMG, OUTPUT_MODELS, OUTPUT_REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoadedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    le: LabelEncoder
    embedding_train: np.ndarray
    embedding_test: np.ndarray
    embedding_char_train: np.ndarray
    embedding_char_test: np.ndarray
    embedding_word_train: np.ndarray
    embedding_word_test: np.ndarray
    embedding_ft_train: np.ndarray
    embedding_ft_test: np.ndarray
    all_embeddings_train: np.ndarray
    all_embeddings_test: np.ndarray
    combined_dense_train: np.ndarray
    combined_dense_test: np.ndarray
    combined_dense_names: list[str]
    corpus_train: pl.DataFrame
    corpus_test: pl.DataFrame


# =====================================
# Data loading
# =====================================
def _stack_embedding_column(df: pl.DataFrame, column: str) -> np.ndarray:
    values = df[column].to_list()
    first = next((value for value in values if value is not None), None)
    if first is None:
        raise ValueError(f"Embedding column '{column}' is entirely null")
    dim = len(first)
    stacked = np.vstack([
        np.asarray(value if value is not None else np.zeros(dim), dtype=np.float32)
        for value in values
    ])
    return stacked.astype(np.float32, copy=False)


def load_data(drop_source_query: bool = True) -> LoadedData:
    logger.info("=" * 70)
    logger.info("PHASE 4A — Loading aligned feature and embedding data")
    logger.info("=" * 70)

    for required in [MODEL_READY, CLEANED_CORPUS]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    t0 = time.time()
    df_feat = pl.read_parquet(MODEL_READY)
    df_corpus = pl.read_parquet(CLEANED_CORPUS)
    logger.info("Read parquet inputs in %.2fs", time.time() - t0)

    logger.info("Aligning cleaned corpus to feature row order using comment_id")
    df_corpus = df_feat.select(["comment_id"]).join(df_corpus, on="comment_id", how="left")
    if len(df_corpus) != len(df_feat):
        raise ValueError("Aligned corpus row count does not match feature row count")

    missing_after_join = {
        column: int(df_corpus[column].null_count())
        for column in ["comment_text", *EMBEDDING_COLUMNS]
        if column in df_corpus.columns
    }
    missing_after_join = {key: value for key, value in missing_after_join.items() if value > 0}
    if missing_after_join:
        raise ValueError(f"Missing corpus data after alignment: {missing_after_join}")

    feat_cols = [column for column in df_feat.columns if column not in {"label", "comment_id"}]
    if drop_source_query:
        feat_cols = [column for column in feat_cols if column not in SOURCE_QUERY_FEATURES]
        logger.info("Dropping source_query leakage features: %s", sorted(SOURCE_QUERY_FEATURES))

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(df_feat["label"].to_numpy())

    X = np.column_stack([
        df_feat[column].fill_null(0).cast(pl.Float64).to_numpy() for column in feat_cols
    ]).astype(np.float32)

    if TRAIN_TEST_SPLIT.exists():
        split_df = pl.read_parquet(TRAIN_TEST_SPLIT)
        train_mask = split_df["is_train"].to_numpy().astype(bool)
        logger.info("Loaded pre-computed train/test split mask")
    else:
        logger.warning("train_test_split.parquet not found; recreating split")
        idx_train, idx_test = train_test_split(
            np.arange(len(df_feat)),
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        train_mask = np.zeros(len(df_feat), dtype=bool)
        train_mask[idx_train] = True

    logger.info("Stacking pre-computed embeddings in RAM")
    t_stack = time.time()
    embedding = _stack_embedding_column(df_corpus, "embedding")
    embedding_char = _stack_embedding_column(df_corpus, "embedding_char")
    embedding_word = _stack_embedding_column(df_corpus, "embedding_word")
    embedding_ft = _stack_embedding_column(df_corpus, "embedding_ft")
    all_embeddings = np.hstack([embedding, embedding_char, embedding_word, embedding_ft]).astype(np.float32)
    logger.info("Embeddings stacked in %.2fs", time.time() - t_stack)

    engineered_scaler = StandardScaler()
    X_train_scaled = engineered_scaler.fit_transform(X[train_mask]).astype(np.float32)
    X_test_scaled = engineered_scaler.transform(X[~train_mask]).astype(np.float32)

    combined_dense_train = np.hstack([
        embedding[train_mask],
        embedding_char[train_mask],
        embedding_word[train_mask],
        embedding_ft[train_mask],
        X_train_scaled,
    ]).astype(np.float32)
    combined_dense_test = np.hstack([
        embedding[~train_mask],
        embedding_char[~train_mask],
        embedding_word[~train_mask],
        embedding_ft[~train_mask],
        X_test_scaled,
    ]).astype(np.float32)

    combined_dense_names = (
        [f"embedding_{i}" for i in range(embedding.shape[1])]
        + [f"embedding_char_{i}" for i in range(embedding_char.shape[1])]
        + [f"embedding_word_{i}" for i in range(embedding_word.shape[1])]
        + [f"embedding_ft_{i}" for i in range(embedding_ft.shape[1])]
        + feat_cols
    )

    logger.info(
        "Loaded engineered=%s, transformer=%s, char=%s, word=%s, fasttext=%s, combined_dense=%s",
        X.shape[1],
        embedding.shape[1],
        embedding_char.shape[1],
        embedding_word.shape[1],
        embedding_ft.shape[1],
        combined_dense_train.shape[1],
    )

    return LoadedData(
        X_train=X[train_mask],
        X_test=X[~train_mask],
        y_train=y[train_mask],
        y_test=y[~train_mask],
        feature_names=feat_cols,
        le=le,
        embedding_train=embedding[train_mask],
        embedding_test=embedding[~train_mask],
        embedding_char_train=embedding_char[train_mask],
        embedding_char_test=embedding_char[~train_mask],
        embedding_word_train=embedding_word[train_mask],
        embedding_word_test=embedding_word[~train_mask],
        embedding_ft_train=embedding_ft[train_mask],
        embedding_ft_test=embedding_ft[~train_mask],
        all_embeddings_train=all_embeddings[train_mask],
        all_embeddings_test=all_embeddings[~train_mask],
        combined_dense_train=combined_dense_train,
        combined_dense_test=combined_dense_test,
        combined_dense_names=combined_dense_names,
        corpus_train=df_corpus.filter(pl.Series(train_mask)),
        corpus_test=df_corpus.filter(pl.Series(~train_mask)),
    )


# =====================================
# Reporting helpers
# =====================================
def compute_class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=n_classes)
    total = len(y)
    return {i: total / (n_classes * max(1, counts[i])) for i in range(n_classes)}


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    model_name: str,
    split_name: str,
) -> pl.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=le.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    rows: list[dict[str, Any]] = []
    for label_name, stats in report.items():
        if isinstance(stats, dict):
            rows.append({"row_name": label_name, **{k: float(v) for k, v in stats.items()}})

    df_report = pl.DataFrame(rows)
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    out_path = OUTPUT_PARQUET / f"t10_classification_report_{split_name}_{safe_name}.parquet"
    df_report.write_parquet(out_path)
    logger.info("Saved classification report -> %s", out_path)
    return df_report


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    le: LabelEncoder,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        metrics["log_loss"] = log_loss(y_true, y_proba)
        metrics["auc_roc_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        metrics["log_loss"] = math.nan
        metrics["auc_roc_macro"] = math.nan

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    for index, class_name in enumerate(le.classes_):
        metrics[f"f1_{class_name}"] = float(per_class_f1[index])
        metrics[f"recall_{class_name}"] = float(per_class_recall[index])
        metrics[f"precision_{class_name}"] = float(per_class_precision[index])
    return metrics


def tune_thresholds(y_true: np.ndarray, y_probas: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    def objective(weights: np.ndarray) -> float:
        clipped = np.clip(weights, 1e-4, 1e4)
        pred = np.argmax(y_probas * clipped[None, :], axis=1)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        recall = recall_score(y_true, pred, average="macro", zero_division=0)
        return -(0.7 * f1 + 0.3 * recall)

    result = minimize(objective, np.ones(y_probas.shape[1]), method="Nelder-Mead", options={"maxiter": 250})
    weights = np.clip(result.x, 1e-4, 1e4)
    return weights / weights.sum()


def decision_scores_to_proba(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


# =====================================
# Feature builders
# =====================================
def concatenate_feature_blocks(data: LoadedData, mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if mode == "engineered_only":
        return data.X_train, data.X_test, data.feature_names[:]

    if mode == "engineered_plus_transformer":
        names = data.feature_names[:] + [f"embedding_{i}" for i in range(data.embedding_train.shape[1])]
        return (
            np.hstack([data.X_train, data.embedding_train]).astype(np.float32),
            np.hstack([data.X_test, data.embedding_test]).astype(np.float32),
            names,
        )

    if mode == "engineered_plus_all_embeddings":
        names = (
            data.feature_names[:]
            + [f"embedding_{i}" for i in range(data.embedding_train.shape[1])]
            + [f"embedding_char_{i}" for i in range(data.embedding_char_train.shape[1])]
            + [f"embedding_word_{i}" for i in range(data.embedding_word_train.shape[1])]
            + [f"embedding_ft_{i}" for i in range(data.embedding_ft_train.shape[1])]
        )
        return (
            np.hstack([data.X_train, data.all_embeddings_train]).astype(np.float32),
            np.hstack([data.X_test, data.all_embeddings_test]).astype(np.float32),
            names,
        )

    raise ValueError(f"Unknown feature mode: {mode}")


# =====================================
# Single-draw XGBoost
# =====================================
def sample_xgb_params(rng: np.random.Generator, n_classes: int) -> dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": int(rng.integers(120, 241)),
        "max_depth": int(rng.integers(3, 6)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.08), np.log(0.25)))),
        "min_child_weight": float(np.exp(rng.uniform(np.log(1.0), np.log(6.0)))),
        "subsample": float(rng.uniform(0.8, 1.0)),
        "colsample_bytree": float(rng.uniform(0.75, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.75, 1.0)),
        "gamma": float(rng.uniform(0.0, 1.0)),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(0.8)))),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))),
        "max_delta_step": float(rng.uniform(0.0, 1.0)),
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }


def build_fast_xgb_baseline(n_classes: int) -> dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": 160,
        "max_depth": 4,
        "learning_rate": 0.12,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "colsample_bylevel": 0.85,
        "gamma": 0.0,
        "reg_alpha": 1e-3,
        "reg_lambda": 1.0,
        "max_delta_step": 0.0,
        "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }


def initialize_xgboost_params(data: LoadedData) -> tuple[dict[str, Any], np.ndarray]:
    logger.info("=" * 70)
    logger.info("PHASE 4A — Single random initialization for XGBoost")
    logger.info("=" * 70)

    n_classes = len(data.le.classes_)
    rng = np.random.default_rng(RANDOM_STATE)
    params = {**build_fast_xgb_baseline(n_classes), **sample_xgb_params(rng, n_classes)}
    best_weights = np.ones(n_classes, dtype=np.float32) / float(n_classes)
    with open(OUTPUT_REPORTS / "t10_best_xgb_params.json", "w", encoding="utf-8") as handle:
        json.dump(params, handle, ensure_ascii=False, indent=2)
    logger.info("Initialized XGBoost with one random parameter draw: %s", params)
    return params, best_weights


def fit_final_xgboost(
    data: LoadedData,
    best_params: dict[str, Any],
    best_weights: np.ndarray,
) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 4A — Fitting final optimized XGBoost")
    logger.info("=" * 70)

    final_params = {**best_params, "n_jobs": -1}
    train_idx, eval_idx = train_test_split(
        np.arange(len(data.combined_dense_train)),
        test_size=0.1,
        stratify=data.y_train,
        random_state=RANDOM_STATE,
    )
    model = XGBClassifier(**final_params)
    model.fit(
        data.combined_dense_train[train_idx],
        data.y_train[train_idx],
        eval_set=[(data.combined_dense_train[eval_idx], data.y_train[eval_idx])],
        verbose=False,
    )

    test_proba = model.predict_proba(data.combined_dense_test).astype(np.float32)
    raw_pred = np.argmax(test_proba, axis=1)
    tuned_pred = np.argmax(test_proba * best_weights[None, :], axis=1)

    save_classification_report(data.y_test, raw_pred, data.le, "XGBoost_optimized", "test_raw")
    save_classification_report(data.y_test, tuned_pred, data.le, "XGBoost_optimized", "test_tuned")

    metrics = compute_metrics(data.y_test, tuned_pred, test_proba, data.le)
    joblib.dump(
        {
            "model": model,
            "threshold_weights": best_weights,
            "feature_names": data.combined_dense_names,
            "params": final_params,
        },
        OUTPUT_MODELS / "XGBoost_optimized.joblib",
    )

    run_shap_analysis(model, data.combined_dense_test, data.combined_dense_names)
    return pl.DataFrame([{"model": "XGBoost_optimized", **metrics}])


def run_shap_analysis(
    model: XGBClassifier,
    X_test: np.ndarray,
    feature_names: list[str],
) -> None:
    logger.info("Running SHAP on final optimized XGBoost model")
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping SHAP analysis")
        return

    sample_size = min(SHAP_SAMPLE, len(X_test))
    X_sample = X_test[:sample_size]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(value) for value in shap_values], axis=(0, 1))
    elif np.asarray(shap_values).ndim == 3:
        mean_abs_shap = np.abs(np.asarray(shap_values)).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(np.asarray(shap_values)).mean(axis=0)

    importance = pl.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap.astype(float),
    }).sort("mean_abs_shap", descending=True)
    importance.write_parquet(OUTPUT_PARQUET / "t10_xgboost_shap_importance.parquet")

    top = importance.head(30).sort("mean_abs_shap")
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"].to_list(), top["mean_abs_shap"].to_list(), color="#4C72B0")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Top 30 SHAP Features — Optimized XGBoost")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG / "t10_xgboost_shap_importance.png", dpi=200)
    plt.close()
    logger.info("Saved SHAP feature importance outputs")


# =====================================
# Fast embedding MLP
# =====================================
def train_fast_embedding_mlp(data: LoadedData) -> pl.DataFrame:
    logger.info("=" * 70)
    logger.info("PHASE 4A — FastEmbeddingMLP")
    logger.info("=" * 70)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        logger.warning("PyTorch not installed; skipping FastEmbeddingMLP")
        return pl.DataFrame()

    class TabularDataset(Dataset):
        def __init__(self, features: np.ndarray, labels: np.ndarray):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            return self.features[index], self.labels[index]

    class FastEmbeddingMLP(nn.Module):
        def __init__(self, input_dim: int, n_classes: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 112),
                nn.BatchNorm1d(112),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(112, 48),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(48, n_classes),
            )

        def forward(self, features: Any) -> Any:
            return self.network(features)

    X_train = data.combined_dense_train
    X_test = data.combined_dense_test
    y_train = data.y_train
    y_test = data.y_test

    train_idx, val_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=0.15,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    X_fit, X_val = X_train[train_idx], X_train[val_idx]
    y_fit, y_val = y_train[train_idx], y_train[val_idx]

    train_loader = DataLoader(TabularDataset(X_fit, y_fit), batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=MLP_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=MLP_BATCH_SIZE, shuffle=False)

    device = "cpu"
    model = FastEmbeddingMLP(X_train.shape[1], len(data.le.classes_)).to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info("FastEmbeddingMLP trainable parameters: %s", f"{param_count:,}")

    class_weights = compute_class_weights(y_train, len(data.le.classes_))
    weight_tensor = torch.tensor(
        [class_weights[index] for index in range(len(data.le.classes_))],
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state: dict[str, Any] | None = None
    best_score = -np.inf
    best_weights: np.ndarray | None = None
    wait = 0

    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        t_epoch = time.time()
        model.train()
        train_losses: list[float] = []

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_logits: list[np.ndarray] = []
        with torch.no_grad():
            for batch_features, _ in val_loader:
                logits = model(batch_features.to(device))
                val_logits.append(logits.cpu().numpy())

        val_proba = decision_scores_to_proba(np.vstack(val_logits).astype(np.float32))
        weights = tune_thresholds(y_val, val_proba)
        val_pred = np.argmax(val_proba * weights[None, :], axis=1)
        macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_val, val_pred, average="macro", zero_division=0)
        score = 0.7 * macro_f1 + 0.3 * macro_recall

        logger.info(
            "Epoch %02d/%02d loss=%.4f val_macro_F1=%.4f val_macro_R=%.4f time=%.2fs",
            epoch,
            MLP_MAX_EPOCHS,
            float(np.mean(train_losses)),
            macro_f1,
            macro_recall,
            time.time() - t_epoch,
        )

        if score > best_score:
            best_score = score
            wait = 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            best_weights = weights.copy()
        else:
            wait += 1
            if wait >= MLP_PATIENCE:
                logger.info("Early stopping FastEmbeddingMLP at epoch %s", epoch)
                break

    assert best_state is not None and best_weights is not None
    model.load_state_dict(best_state)

    model.eval()
    test_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch_features, _ in test_loader:
            logits = model(batch_features.to(device))
            test_logits.append(logits.cpu().numpy())

    test_proba = decision_scores_to_proba(np.vstack(test_logits).astype(np.float32))
    raw_pred = np.argmax(test_proba, axis=1)
    tuned_pred = np.argmax(test_proba * best_weights[None, :], axis=1)

    save_classification_report(y_test, raw_pred, data.le, "FastEmbeddingMLP", "test_raw")
    save_classification_report(y_test, tuned_pred, data.le, "FastEmbeddingMLP", "test_tuned")

    metrics = compute_metrics(y_test, tuned_pred, test_proba, data.le)
    torch.save(
        {
            "state_dict": best_state,
            "threshold_weights": best_weights,
            "input_dim": X_train.shape[1],
            "classes": data.le.classes_.tolist(),
            "feature_names": data.combined_dense_names,
        },
        OUTPUT_MODELS / "FastEmbeddingMLP.pt",
    )
    return pl.DataFrame([{"model": "FastEmbeddingMLP", **metrics}])


# =====================================
# Main
# =====================================
def main() -> None:
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 4A ADVANCED MODELING — started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    seed_everything()
    setup_output_dirs()
    data = load_data(drop_source_query=True)

    # Skipping LinearSVC reruns because that baseline has already been completed.
    # classic = run_classic_models(data)
    # classic.write_parquet(OUTPUT_PARQUET / "t10_revised_classic_leaderboard.parquet")

    best_xgb_params, best_xgb_weights = initialize_xgboost_params(data)

    xgb_results = fit_final_xgboost(data, best_xgb_params, best_xgb_weights)
    xgb_results.write_parquet(OUTPUT_PARQUET / "t10_xgboost_optimized_results.parquet")

    mlp_results = train_fast_embedding_mlp(data)
    if len(mlp_results) > 0:
        mlp_results.write_parquet(OUTPUT_PARQUET / "t10_fast_embedding_mlp_results.parquet")

    leaderboard_parts = [xgb_results]
    if len(mlp_results) > 0:
        leaderboard_parts.append(mlp_results)
    leaderboard = pl.concat(leaderboard_parts, how="diagonal").sort("f1_macro", descending=True)
    leaderboard.write_parquet(OUTPUT_PARQUET / "t10_advanced_model_leaderboard.parquet")

    elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info("PHASE 4A ADVANCED MODELING COMPLETE — Total time: %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
