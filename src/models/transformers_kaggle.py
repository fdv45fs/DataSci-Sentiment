"""
Transformer Late-Fusion Model — Kaggle GPU Notebook Scaffold

⚠ NOT INTENDED FOR LOCAL EXECUTION ⚠
This file is a scaffold for running DistilBERT and RoBERTa on Kaggle GPU instances.
It requires:
  - GPU runtime (T4, P100, or better)
  - transformers >= 4.40.0
  - torch >= 2.0.0
  - Install in Kaggle notebook: !pip install -q transformers accelerate

Architecture:
  DistilBERT / RoBERTa [CLS] token → MLP head
  Engineered feature vector         → Linear projection
  Late fusion: concat(CLS_repr, feat_proj) → 3-class classifier

Usage:
  1. Upload model_ready_features.parquet and cleaned_corpus.parquet to Kaggle dataset
  2. Run this notebook with GPU enabled
  3. Results will be saved to kaggle output parquet files
"""

import numpy as np
import polars as pl
from pathlib import Path


DISTILBERT_MODEL = "distilbert-base-multilingual-cased"
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
N_EPOCHS = 3
WARMUP_RATIO = 0.1
LABEL_ORDER = ["positive", "neutral", "negative"]
OUTPUT_DIR = Path("kaggle_output")

ENGINEERED_FEATURE_DIM = 64
MLP_HIDDEN_DIM = 128


def build_late_fusion_model(
    transformer_model_name: str,
    n_engineered_features: int,
    n_classes: int = 3,
    dropout_p: float = 0.2,
):
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class LateFusionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = AutoModel.from_pretrained(transformer_model_name)
            hidden_size = self.transformer.config.hidden_size

            self.feat_projection = nn.Sequential(
                nn.Linear(n_engineered_features, MLP_HIDDEN_DIM),
                nn.LayerNorm(MLP_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size + MLP_HIDDEN_DIM, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(256, n_classes),
            )

        def forward(self, input_ids, attention_mask, engineered_features):
            transformer_out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            cls_repr = transformer_out.last_hidden_state[:, 0, :]

            feat_repr = self.feat_projection(engineered_features)

            fused = torch.cat([cls_repr, feat_repr], dim=-1)
            logits = self.classifier(fused)
            return logits

    return LateFusionClassifier()


def build_dataset(
    texts: list[str],
    engineered_features: np.ndarray,
    labels: np.ndarray,
    tokenizer,
    max_length: int,
):
    import torch
    from torch.utils.data import Dataset

    class CommentDataset(Dataset):
        def __init__(self):
            self.encodings = tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.features = torch.tensor(engineered_features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "engineered_features": self.features[idx],
                "labels": self.labels[idx],
            }

    return CommentDataset()


def get_linear_warmup_cosine_scheduler(optimizer, n_warmup_steps: int, n_total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step: int):
        if current_step < n_warmup_steps:
            return float(current_step) / float(max(1, n_warmup_steps))
        progress = float(current_step - n_warmup_steps) / float(max(1, n_total_steps - n_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    import torch
    import torch.nn as nn

    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch["engineered_features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask, features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"    Batch {batch_idx}/{len(dataloader)}  loss={loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, device):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score, accuracy_score

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["engineered_features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, features)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return macro_f1, acc


def run_transformer_pipeline(
    model_name: str,
    texts_train: list[str],
    texts_val: list[str],
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
):
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = build_dataset(texts_train, X_train, y_train, tokenizer, MAX_SEQ_LEN)
    val_dataset = build_dataset(texts_val, X_val, y_val, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = build_late_fusion_model(model_name, n_engineered_features=X_train.shape[1])
    model.to(device)

    n_total_steps = len(train_loader) * N_EPOCHS
    n_warmup_steps = int(n_total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_warmup_cosine_scheduler(optimizer, n_warmup_steps, n_total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_f1 = 0.0
    best_epoch = 0
    history = []

    for epoch in range(N_EPOCHS):
        print(f"\n[Epoch {epoch+1}/{N_EPOCHS}]")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_f1, val_acc = evaluate_epoch(model, val_loader, device)

        print(f"  train_loss={train_loss:.4f}  val_macro_F1={val_f1:.4f}  val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1, "val_acc": val_acc})

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), OUTPUT_DIR / f"{model_name.replace('/', '_')}_best.pt")
            print(f"  ✓ New best model saved (val_F1={best_f1:.4f})")

    print(f"\n[DONE] Best val_F1={best_f1:.4f} at epoch {best_epoch}")
    return history, best_f1


if __name__ == "__main__":
    print("=" * 70)
    print("TRANSFORMER KAGGLE PIPELINE")
    print("This file is a Kaggle GPU scaffold — do not run locally.")
    print("=" * 70)
    print(f"  DistilBERT model : {DISTILBERT_MODEL}")
    print(f"  RoBERTa model    : {ROBERTA_MODEL}")
    print(f"  Max seq len      : {MAX_SEQ_LEN}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Learning rate    : {LEARNING_RATE}")
    print(f"  N epochs         : {N_EPOCHS}")
    print(f"  Warmup ratio     : {WARMUP_RATIO}")
    print()
    print("Architecture: [CLS] + Engineered Features → Late Fusion MLP → 3-class head")
    print("Scheduler   : Linear warmup + cosine decay")
    print("Optimizer   : AdamW (weight_decay=0.01)")
    print("Mixed prec  : torch.cuda.amp.GradScaler")
    print()
    print("Upload model_ready_features.parquet + cleaned_corpus.parquet to Kaggle,")
    print("then call run_transformer_pipeline() from your notebook cell.")
