## Input data:
### **data/combined_embeddings.parquet**:
```
column_name	column_type	null	key	default	extra
comment_id	VARCHAR	YES			
post_id	VARCHAR	YES			
comment_text	VARCHAR	YES			
published_at	VARCHAR	YES			
like_count	BIGINT	YES			
reply_count	BIGINT	YES			
author_id	VARCHAR	YES			
author_name	VARCHAR	YES			
title_youtube	VARCHAR	YES			
source_query	VARCHAR	YES			
crawled_at	VARCHAR	YES			
char_count	BIGINT	YES			
word_count	BIGINT	YES			
avg_word_length	DOUBLE	YES			
uppercase_ratio	DOUBLE	YES			
exclamation_count	BIGINT	YES			
question_count	BIGINT	YES			
hashtag_count	BIGINT	YES			
mention_count	BIGINT	YES			
emoji_count	BIGINT	YES			
like_count_log	DOUBLE	YES			
embedding	FLOAT[]	YES			
embedding_char	FLOAT[]	YES			
embedding_word	FLOAT[]	YES			
embedding_ft	FLOAT[]	YES			
```
### **data/combined_labeled.parquet**:
```
column_name	column_type	null	key	default	extra
comment_id	VARCHAR	YES			
label	VARCHAR	YES			
source_file	VARCHAR	YES			
source_row	BIGINT	YES			
post_id	VARCHAR	YES			
```

### Some more notes:
1. For data that can be plot using charts, consider outputting them using reasonable metrics in the form of table (parquet), so we can later use them to further analyze if we need. Sometimes chart does not fully pose the problems.
2. Code should not contain any comments inside it, clean code, modular structure and carefully put together. Mind your business deeply.

## Performance notes:
1. If there are tasks that can be accelerated, please try to accelerate it as much as you can. We have a 12 cores CPU and 48 freaking GBs of RAM, so perfomance should be pushed. Use polars and eradicate pandas completely. Calculations should prioritized using numpy, polars, scipy, etc.. (optimized libraries for perfomance).

## Table 0 — Compact Data Dictionary & Cleaning Rules
*Baseline reference for all downstream work. Saves tokens by collapsing schema + cleaning into one view.*

| Column | Description | Null / Corruption Rule |
| :--- | :--- | :--- |
| `comment_id` | Primary key (YouTube comment ID) | Drop row if NULL; deduplicate exact duplicates keeping first |
| `post_id` | YouTube video identifier | Drop row if NULL |
| `comment_text` | Raw multilingual comment body | Drop row if NULL or whitespace-only; fix `char_count=0` when text is non-empty |
| `published_at` | UTC publication timestamp | Drop row if NULL; drop future dates |
| `like_count` | Social endorsement count | Impute NULL → 0; negative values → 0 |
| `reply_count` | Conversational "share/virality" proxy | Impute NULL → 0; negative values → 0 |
| `author_id` | Anonymized author identifier | Flag "unknown" if NULL |
| `author_name` | Display name | Keep as-is |
| `title_youtube` | Video title at crawl time | Keep NULL; used for context enrichment only |
| `source_query` | Search query that surfaced the video | Keep NULL; target-encode later |
| `crawled_at` | Data collection timestamp | Keep NULL; compute `lag_days = crawled_at − published_at` |
| `char_count` | Character length | Recompute from `comment_text` |
| `word_count` | Space-split token count | Recompute from `comment_text` |
| `avg_word_length` | Mean token length | Recompute from `comment_text` |
| `uppercase_ratio` | Proportion of uppercase chars | Recompute from `comment_text` |
| `exclamation_count` | Count of `!` | Recompute from `comment_text` |
| `question_count` | Count of `?` | Recompute from `comment_text` |
| `hashtag_count` | Count of `#` tokens | Recompute from `comment_text` |
| `mention_count` | Count of `@` tokens | Recompute from `comment_text` |
| `emoji_count` | Count of Unicode emojis | Recompute from `comment_text`; will be expanded to **emoji types** |
| `like_count_log` | `log1p(like_count)` | Recompute |
| `embedding` | Sentence-transformer dense vector | Recompute if NULL |
| `embedding_char` | Small n-gram char embedding | Recompute if NULL |
| `embedding_word` | Big n-gram word embedding | Recompute if NULL |
| `embedding_ft` | fastText multilingual embedding | Recompute if NULL |

---

## Table 1 — Phase 0: Corpus Matching & Deduplication
*Goal: produce the exact cleaned cardinality table that every later phase depends on.*

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Deduplicate & inner-merge** | Inner join `combined_embeddings.parquet` with `combined_labeled.parquet` on `comment_id` + `post_id`; validate strict 1:1 label mapping (one `label` per `comment_id`); drop exact duplicates on `comment_id` and on `comment_text`; remove rows with NULL `comment_text` or `comment_id`; resolve mismatched `post_id` between tables by dropping mismatches. | **Table: Cleaned Corpus Cardinality** — rows retained, % of raw 395k, unique comments, unique videos, unique authors, time span. |
| **Null audit & feature recalculation** | Recompute all count/ratio features (`char_count`, `word_count`, `uppercase_ratio`, `exclamation_count`, `question_count`, `hashtag_count`, `mention_count`, `emoji_count`, `like_count_log`) directly from cleaned `comment_text` to eliminate drift; verify all 4 embedding vectors are non-empty; flag rows with any NULL embedding for recomputation. | **Table: Null Pattern Matrix** — column, null_count, null_%, imputation_action, post-cleaning_null_count. |

---

## Table 2 — Phase 1: Label-Directed Baseline Statistics
*All numbers must be stratified by the 3 labels. This is the "directed" foundation the prior EDA missed.*

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Label distribution & imbalance diagnostics** | Compute absolute count and percentage for Positive, Neutral, Negative; calculate imbalance ratios (Negative:Positive, Neutral:Positive); compute Gini coefficient across labels; verify no label leakage by checking that `source_file` / `source_row` ranges do not perfectly predict label. | **Table: Label Distribution** — label, count, %, cumulative_%, imbalance_ratio_vs_Positive. |
| **Stratified descriptive statistics** | Per label, compute mean / median / std of `char_count`, `word_count`, `like_count`, `reply_count`, `emoji_count`, `uppercase_ratio`, `exclamation_count`, `question_count`; compute label distribution per `source_query` and per quarter-year. | **Table: Stratified Baseline Stats** — label × (feature_mean, feature_median, feature_std); **Table: Label × Source Query** cross-tab (% per row). |

---

## Table 3 — Phase 2A: Emoji Deep Analysis (Type-Level, Not Just Count)
*Teacher requirement: "Thêm emoji type vào emoji_count" + "Thống kê tương quan emoji theo nhãn"*

### Notes:
1. Instead of doing the assumption that positive icons mean 'face_positive', we should do a small survey on the data with labels. We should output a correlation matrix between types of icons and their labels to see which corresponds to face_positive, face_neutral or face_negative. The same with symbols.

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Emoji type extraction & vectorization** | Extract full emoji list per comment via Unicode regex; classify into 6 types: `face_positive` (😂❤️😍🥰), `face_negative` (😡😢😠😤), `face_neutral` (😐🤔😶), `symbol_positive` (👍🙏💪🔥), `symbol_negative` (👎💔), `other` (flags, objects, animals); create 6 binary presence columns + 6 count columns; compute `emoji_density = emoji_count / char_count`; identify emoji-only comments (zero text tokens). | **Table: Emoji Type Inventory** — type, total_occurrences, %_of_corpus, top_5_emojis_in_type, emoji-only_comment_count. |
| **Emoji–label correlation & statistical testing** | Cross-tabulate each emoji type against label (Pos/Neu/Neg); run Chi-square test of independence and compute Cramér's V per emoji type; compute mean `emoji_density` per label; compute mutual information (MI) between individual emojis and label; analyze whether emoji-only comments skew toward a specific sentiment. | **Table: Emoji-Label Correlation** — emoji_type, chi2_stat, p_value, Cramers_V, mean_density_Pos, mean_density_Neu, mean_density_Neg; **Table: Top Discriminating Emojis** — emoji, MI_score, most_common_label. |

---

## Table 4 — Phase 2B: Engagement Semantics (Like & Reply Meaning)
*Teacher requirement: "Giải thích tác dụng của like, share" + temporal context (COVID-19)*

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Engagement semantics & controversy proxy** | Define `like_count` as **social endorsement** and `reply_count` as **conversational virality/share proxy** (YouTube comments lack native shares; replies are the debate metric); compute `reply_to_like_ratio = (reply_count + 1) / (like_count + 1)`; define engagement tiers: Micro (<10 likes), Small (10–100), Medium (100–1k), Large (1k–10k), Viral (≥10k); test hypothesis: high-reply + low-like correlates with Negative (controversy), while high-like + low-reply correlates with Positive (agreement). | **Table: Engagement Semantics by Label** — label, mean_like, median_like, mean_reply, median_reply, mean_ratio, %_controversial (reply>like), %_endorsed (like>reply), tier_distribution_per_label. |
| **Temporal engagement contextualization** | Stratify by `published_at` into pre-COVID, COVID-era, post-COVID (or by year/quarter); compute monthly global median `like_count` as inflation baseline; calculate `like_inflation_index = like_count / monthly_median`; compute `engagement_velocity = like_count / days_since_publish` (using `crawled_at − published_at`); compare velocity and inflation-adjusted likes across labels and eras to detect whether a "like" meant more during low-activity periods. | **Table: Temporal Engagement Baseline** — era, monthly_median_likes, label_%_per_era, mean_velocity_per_label, inflation_index_per_label. |

---

## Table 5 — Phase 2C: Multilingual Stratification
*Teacher requirement: multilingual focus*

### **EXTREMELY IMPORTANT NOTES**:
1. We need a fast method to detect languages, the languages in here are mostly common ones: en, pt, zh, jp, vi, etc... Like the most commonly used ones in the world. They are not the weird languages of some kind. The last time I run the code with deep_eda_one.py, it costs us roughly half a day to process. 

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Language identification & confidence scoring** | Run `fastText lid.176` + `langdetect` ensemble on `comment_text`; assign `primary_language` and `confidence_score`; use fastText as tie-breaker; group into top 5 languages + "other"; flag code-switching if secondary language covers ≥10% of tokens or if script blocks switch mid-comment; record `script_family` (Latin, CJK, Cyrillic, Arabic, Devanagari, Thai). | **Table: Language Distribution** — language, comment_count, %_of_corpus, unique_authors, unique_videos, avg_confidence, script_family. |
| **Cross-lingual label & engagement stratification** | For top 5 languages, compute label distribution (% Pos/Neu/Neg), mean `char_count`, mean `word_count`, `emoji_density`, mean `like_count`, mean `reply_count`; test homogeneity of label distribution across languages (Chi-square); flag languages with statistically higher Negative rates. | **Table: Multilingual Stratification** — language, N, %_Pos, %_Neu, %_Neg, mean_chars, mean_words, emoji_density, mean_likes, mean_replies, code_switching_%. |

---

## Table 6 — Phase 2D: N-Gram Discriminative Analysis (1-gram vs 2-gram vs 3-gram)
*Teacher requirement: "So sánh các gram (1, 2, 3) để chọn feature tốt" + "Thống kê cụ thể về gram để tránh scale up sai"*

### Notes:
- We also need to compare bigger grams (4-7, even bigger ones).

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **N-gram extraction & relative frequency** | Extract unigrams (1-gram), bigrams (2-gram), trigrams (3-gram) after cleaning (lower, remove URLs, keep emojis as separate tokens); compute **document frequency (DF)** and **relative frequency per 1,000 comments per label** to avoid raw-count scale-up errors; filter n-grams with DF < 5 (too rare) and DF > 50% corpus (too common, e.g., "the", "this video"); compute TF-IDF per label class. | **Table: N-gram Frequency Profile** — n_gram_order, total_unique_grams, avg_grams_per_comment, top_20_grams_per_label_with_rel_freq_per_1k. |
| **N-gram discriminative power comparison** | Compute mutual information (MI) and Chi-square between n-gram presence and label; rank by MI per order; calculate **mean MI_score per n-gram order** to objectively compare which order separates labels best; compute **coverage** (% of comments in each label covered by top 100 n-grams of that order); recommend which order(s) to retain for feature engineering. | **Table: N-gram Discriminative Power** — n_gram_order, mean_MI, top_3_grams_by_MI, coverage_%_Pos, coverage_%_Neu, coverage_%_Neg; **Recommendation** — which orders to vectorize. |

---

## Table 7 — Phase 2E: Source Query & Temporal Context
*Teacher requirement: source query importance*

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Source query bias & intent profiling** | Treat `source_query` as search-intent proxy; per query compute: comment count, label distribution (% Pos/Neu/Neg), mean `like_count`, mean `reply_count`, mean text length, dominant language; flag queries with >80% single-label dominance as **sampling-biased**; cross-tab `source_query` × label and `source_query` × language. | **Table: Source Query Profile** — query, N, %_Pos, %_Neu, %_Neg, avg_likes, avg_replies, avg_length, dominant_lang, bias_flag. |
| **Temporal volume & sentiment dynamics** | Aggregate volume by month; compute label distribution per quarter; compute median lag (`crawled_at − published_at`) per month; identify temporal anomalies (spikes in negative sentiment or engagement). | **Table: Temporal Baseline** — month, volume, %_Pos, %_Neu, %_Neg, median_lag_days, median_likes, anomaly_flag. |

---

## Table 8 — Phase 3: Feature Engineering & Multicollinearity Audit
*Bridge from EDA to model-ready features. Integrates likes, replies, emoji types, source queries, languages.*

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Curated feature creation** | **(A) Text:** `char_count`, `word_count`, `avg_word_length`, `uppercase_ratio`, `exclamation_count`, `question_count`, `hashtag_count`, `mention_count`, `emoji_count`, `emoji_density`, 6 `emoji_type_count` vectors, `text_sentiment_lexicon_score` (multilingual VADER or similar). **(B) Engagement:** `like_count_log`, `reply_count_log`, `reply_to_like_ratio`, `engagement_tier` (ordinal encoded), `likes_per_day`, `replies_per_day`. **(C) Metadata:** `source_query_target_encoded` (smoothed mean label per query), `source_query_freq_encoded`, `primary_language` one-hot (top 5 + other), `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `days_since_publish`, `era_flag` (pre/during/post-COVID). **(D) Embeddings:** `embedding`, `embedding_char`, `embedding_word`, `embedding_ft` (float32 arrays). | **Table: Feature Inventory** — feature_name, category, dtype, derivation_logic, intended_scaling_method. |
| **Multicollinearity detection & remediation** | Compute Pearson/Spearman matrix for all numeric features; iteratively compute VIF; for pairs with VIF > 10 (e.g., `char_count` vs `word_count`), replace with ratio feature (`word_density = char_count / word_count`) or drop the weaker predictor; for `like_count` vs `like_count_log`, retain log for linear models, retain both for tree models; document all actions. | **Table: VIF Audit** — feature, VIF_before, action (kept/dropped/combined), VIF_after, rationale; **Table: Final Feature Correlation Matrix** (post-remediation). |
| **Feature scaling & encoding registry** | Apply `StandardScaler` to normal numeric features; `RobustScaler` to engagement features (outlier-heavy); one-hot encode language and engagement tier; keep embedding vectors as `float32`; serialize pipeline. | **Table: Scaling Registry** — feature, scaler_type, params. |

---

## Table 9 — Phase 3B: EDA of Engineered Features
*Validate that the features you built actually separate the 3 labels.*

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Post-engineering feature validation** | For each engineered feature, compute ANOVA F-statistic (or Kruskal-Wallis H if non-normal) against label; compute effect size (η²); for embedding spaces, compute silhouette score of label clusters in PCA-reduced space (50 components) and UMAP space; compute point-biserial correlation for binary features vs label. | **Table: Feature Discriminative Power** — feature_name, F_stat/H_stat, p_value, eta_squared, silhouette_if_embedding, keep/drop_recommendation. |
| **Feature distribution by label** | Compute mean ± std, median, IQR for every numeric feature stratified by label; compute Cramér's V for categorical features vs label; verify that `source_query_target_encoded` does not overfit (check correlation with label on hold-out fold). | **Table: Feature Distribution by Label** — feature × label matrix of means/medians; **Table: Categorical Association** — feature, Cramers_V, p_value. |

---

## Table 10 — Phase 4: Modeling Pipeline, Optimization & Interpretability
*Teacher requirements: CV=5, RoBERTa, DistilBERT, SHAP, Optuna, SMOTE, multicollinearity fix.*

### Notes:
1. Leave the implementation of DistillBERT and RoBERTa in seperate code (because this should be run on Jupyter Notebook inside Kaggle, not here).
2. Have train-test division strategy before Cross-Validation or anything else. If we train on test data, that is a violation of policy.

| Work | Full Description | Goal |
| :--- | :--- | :--- |
| **Cross-validation & imbalance strategy** | `StratifiedKFold(k=5)` preserving global label distribution in every fold; compute `class_weights = inverse_frequency`; optionally apply **SMOTE or ADASYN only inside training folds** (never on validation); store fold indices for exact reproducibility. | **Table: CV Fold Summary** — fold, train_N, val_N, %_Pos_train, %_Pos_val, class_weight_vector. |
| **Model architecture design** | **(A) Baseline:** Logistic Regression / Linear SVM on selected n-gram TF-IDF + engineered features (post-VIF). **(B) DistilBERT:** Fine-tune `distilbert-base-uncased` (or multilingual variant); append late-fusion MLP head that concatenates `[CLS]` token with top engineered features (engagement + emoji types + source_query encoding). **(C) RoBERTa:** Fine-tune `cardiffnlp/twitter-roberta-base-sentiment-latest` (domain-aligned for social text) with identical late-fusion head. **(D) Optional XLM-RoBERTa** for multilingual backbone. All use AdamW, linear warmup, cosine decay. | **Table: Model Architecture Summary** — model_name, backbone, input_modalities, fusion_strategy, head_structure, param_count. |
| **Hyperparameter optimization (Optuna)** | Search: `learning_rate` (1e-5 to 5e-5), `batch_size` (16, 32), `dropout` (0.1–0.5), `weight_decay` (0.01–0.1), MLP hidden dims (64, 128, 256), `class_weight_intensity` (0.5×–2× inverse frequency), frozen backbone epochs (0–2); optimize for **macro-F1**; early stopping patience=3; save best checkpoint per trial. | **Table: Optuna Best Config** — param, best_value; **Table: Optimization History** — trial, macro_f1, params. |
| **SHAP interpretability & error analysis** | For MLP head / tree meta-learner: compute **SHAP** values for engineered features; for transformer input: use **Integrated Gradients** or attention rollout to highlight influential tokens; aggregate SHAP by category (text, engagement, emoji, metadata); analyze misclassifications grouped by language, source_query, and emoji presence to find systematic failure modes. | **Table: SHAP Feature Importance** — feature, mean_abs_SHAP, category; **Table: Error Analysis** — true_label, pred_label, count, common_patterns, dominant_language, dominant_source_query. |
| **Evaluation & statistical comparison** | Report Accuracy, Macro Precision, Macro Recall, Macro F1, per-class Precision/Recall/F1; normalized confusion matrix; model leaderboard; **McNemar's test** for statistical significance between best models. | **Table: Model Leaderboard** — model, accuracy, macro_P, macro_R, macro_F1, Pos_F1, Neu_F1, Neg_F1; **Confusion Matrix** (normalized); **McNemar's p-value matrix**. |

---

### Execution Order Summary

1. **Table 0** → Establish schema truth.  
2. **Table 1** → Clean and merge; lock corpus size.  
3. **Table 2** → Baseline numbers directed by label.  
4. **Tables 3–7** (parallelizable) → Deep EDA: emoji types, engagement semantics, multilingual, n-gram comparison, source query, temporal.  
5. **Table 8** → Build features from EDA insights; fix multicollinearity (VIF < 10).  
6. **Table 9** → Validate that engineered features actually separate labels.  
7. **Table 10** → Model with DistilBERT / RoBERTa, late-fusion features, Optuna, SHAP, SMOTE inside CV folds only.