import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler

COMBINED_DATA_PATH = "../data/combined_embeddings.parquet"

numeric_cols = [
    "like_count", "reply_count", "char_count", "word_count",
    "avg_word_length", "uppercase_ratio", "exclamation_count",
    "question_count", "hashtag_count", "mention_count",
    "emoji_count", "like_count_log"
]
embedding_cols = ["embedding", "embedding_char", "embedding_word", "embedding_ft"]
comment_text_col = ["comment_text"]

df = pl.read_parquet(
    COMBINED_DATA_PATH,
    columns=numeric_cols + embedding_cols
)

X_numeric = df.select(numeric_cols).to_numpy().astype(np.float32)

embeddings = {}
for col in embedding_cols:
    embeddings[col] = np.stack(df[col].to_numpy())

X_all = np.hstack([X_numeric] + list(embeddings.values()))
print(f"Combined feature matrix shape: {X_all.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)