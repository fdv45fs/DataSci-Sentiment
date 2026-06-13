# NMDS Deep EDA, Feature Engineering & Modeling Pipeline

## Context

The project has two existing EDA files:
- `notebooks/youtube_eda.py` — broad exploratory analysis using **pandas**, covers basic distributions, engagement, temporal patterns, and naive sentiment analysis. **Not label-directed**. Uses pandas (to be replaced with polars everywhere new).
- `src/eda/deep_eda_one.py` and `deep_eda_two.py` — deep structure analysis: UMAP, HDBSCAN, PCA, persistent homology, graph of words, cross-lingual similarity. Language detection via `langdetect` (single-threaded, very slow — the half-day problem).

**What must NOT be re-done** (already covered):
- Generic univariate distributions (histograms, box plots, violin plots of raw numeric cols)
- Basic engagement stats (mean/median likes, replies by source_query)
- Word clouds
- Day-of-week / hour-of-day heatmaps
- Labeler model analysis
- UMAP/HDBSCAN/PCA/persistent homology of raw embeddings
- Basic n-gram word clouds (per sentiment)
- Basic top-unigram bar charts (unlabeled)
- Basic cross-lingual similarity graphs and graph-of-words

**What is required per `LastDay.md`** (Tables 0 through 10):
- Canonical deduplication + inner merge + null audit (Table 0 & 1)
- Label-directed stratified stats (Table 2)
- Emoji type extraction + Chi²/Cramér's V correlation with labels (Table 3)
- Engagement semantics: controversy proxy, temporal COVID contextualization (Table 4)
- Multilingual stratification using **fast** FastText LID (Table 5)
- N-gram discriminative power comparison 1–7+ grams (Table 6)
- Source query bias + temporal dynamics (Table 7)
- Full feature engineering: text, engagement, metadata, time, encoding registry (Table 8)
- VIF audit + multicollinearity remediation (Table 8)
- Post-engineering feature validation (ANOVA/KW, SHAP, silhouette) (Table 9)
- CV scaffold + SMOTE + Optuna baseline models + error analysis (Table 10; DistilBERT/RoBERTa excluded to Kaggle)

---

## Critical Notes Captured from LastDay.md

> [!IMPORTANT]
> **No pandas** — use polars exclusively. No comments inside code. Clean, modular structure.

> [!IMPORTANT]
> **FastText LID** (`lid.176.bin`) must replace `langdetect` for language detection. The previous `langdetect` approach took half a day. FastText LID processes ~400k rows in seconds via batch Python calls or ctypes, not via `map_elements` one-by-one.

> [!IMPORTANT]
> **Emoji type classification must be data-driven.** Do NOT assume `face_positive=😂❤️`. Instead, output a correlation matrix between emoji types and labels first, then validate/adjust the classification accordingly.

> [!IMPORTANT]
> **Train-test split FIRST**, then StratifiedKFold(k=5) **only on train**. SMOTE applied only inside training folds, never on validation or test.

> [!IMPORTANT]
> **DistilBERT and RoBERTa implementations go in a separate file** (meant for Kaggle GPU notebook, not local execution).

> [!NOTE]
> All numeric outputs must be dual-output: chart (PNG, saved to `output_data/img/`) **and** parquet table (saved to `output_data/parquet/`). This is a hard requirement from Note 1 of LastDay.md.

> [!NOTE]
> N-gram comparison must include 1-gram through 7+ grams (note in Table 6: "We also need to compare bigger grams (4-7, even bigger ones)").

> [!NOTE]
> `like_count` = social endorsement; `reply_count` = conversational virality / share proxy. This must be documented and tested statistically.

---

## Open Questions

> [!IMPORTANT]
> **FastText LID model file**: Does `data/fasttext_avg/` contain the `lid.176.bin` model, or does it need to be downloaded? I'll check the directory and handle either case.

> [!NOTE]
> The `data/13gram/13gram.parquet` and `data/37gram/37gram.parquet` files (438MB each) appear to contain precomputed n-gram data. These will be inspected and reused if their schema matches what Table 6 requires, avoiding recomputation.

---

## Proposed Changes

### Architecture Overview

```
src/
├── eda/
│   ├── deep_eda_one.py          [EXISTING — not modified]
│   ├── deep_eda_two.py          [EXISTING — not modified]
│   ├── phase0_cleaning.py       [NEW] Table 0 & 1: dedup, inner merge, null audit, feature recompute
│   ├── phase1_label_stats.py    [NEW] Table 2: label distribution, stratified stats, imbalance
│   ├── phase2a_emoji.py         [NEW] Table 3: emoji type extraction, Chi², Cramér's V, MI
│   ├── phase2b_engagement.py    [NEW] Table 4: engagement semantics, controversy proxy, temporal COVID
│   ├── phase2c_multilingual.py  [NEW] Table 5: FastText LID ensemble, cross-lingual stratification
│   ├── phase2d_ngrams.py        [NEW] Table 6: n-gram 1–7+ discriminative power, MI, TF-IDF
│   ├── phase2e_source_temporal.py [NEW] Table 7: source query bias, temporal dynamics
│   ├── phase3_feature_eng.py    [NEW] Table 8: feature creation, VIF audit, scaling registry
│   ├── phase3b_feature_eda.py   [NEW] Table 9: ANOVA/KW, effect size, silhouette, Cramér's V
│   └── phase4_modeling.py       [NEW] Table 10: CV scaffold, SMOTE, Optuna, baseline models, SHAP
├── features/
│   └── emoji_classifier.py      [NEW] Emoji Unicode → type mapping + data-driven survey
└── models/
    └── transformers_kaggle.py   [NEW] DistilBERT + RoBERTa late-fusion (Kaggle-only stub)
```

---

### Phase 0 & 1 — Corpus Cleaning

#### [NEW] [phase0_cleaning.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase0_cleaning.py)

**Table 0 outputs:**
- `output_data/parquet/t0_schema_truth.parquet` — column, null_count, null_%, imputation_action, post_cleaning_null_count
- `output_data/parquet/t1_corpus_cardinality.parquet` — rows_retained, pct_of_raw, unique_comments, unique_videos, unique_authors, time_span

**Logic:**
1. Load `combined_embeddings.parquet` (all columns including all 4 embedding vectors)
2. Load `combined_labeled.parquet`
3. Inner join on `comment_id` + `post_id`; validate strict 1:1 label mapping
4. Drop exact duplicates on `comment_id` (keep first), then on `comment_text`
5. Drop rows with NULL `comment_text`, `comment_id`, `post_id`, `published_at`; drop future dates
6. Impute `like_count` NULL → 0; clip negatives → 0; same for `reply_count`
7. Recompute all 9 count/ratio features from `comment_text` in polars (vectorized, no row-wise Python)
8. Recompute `like_count_log = log1p(like_count)`
9. Flag rows with any NULL embedding for downstream recompute warning
10. Save cleaned corpus as `output_data/parquet/cleaned_corpus.parquet`
11. Emit Null Pattern Matrix table

**Performance:** All ops in polars lazy mode; parallelism via polars thread pool (12 cores). No pandas.

---

### Phase 1 — Label-Directed Baseline Statistics

#### [NEW] [phase1_label_stats.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase1_label_stats.py)

**Table 2 outputs:**
- `output_data/parquet/t2_label_distribution.parquet` — label, count, %, cumulative_%, imbalance_ratio_vs_Positive, Gini_coefficient
- `output_data/parquet/t2_stratified_baseline_stats.parquet` — label × (feature_mean, median, std) for all 8 numeric features
- `output_data/parquet/t2_label_by_source_query.parquet` — cross-tab source_query × label (% per row)
- `output_data/parquet/t2_label_by_quarter.parquet` — quarter × label distribution
- `output_data/img/t2_*.png` — label distribution bar+pie, stratified violin grids, source query × label heatmap

**Logic:**
- Gini coefficient computed from label counts
- Source query leakage check: compute mutual information between `source_file` / `source_row` rank vs label; flag if MI > threshold
- All stats in polars group_by + agg; visualizations in matplotlib/seaborn (no pandas in computation)

---

### Phase 2A — Emoji Deep Analysis

#### [NEW] [phase2a_emoji.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase2a_emoji.py)
#### [NEW] [emoji_classifier.py](file:///mnt/seagate_320/assets/NMDS/src/features/emoji_classifier.py)

**Table 3 outputs:**
- `output_data/parquet/t3_emoji_type_inventory.parquet` — type, total_occurrences, %_corpus, top_5_emojis, emoji_only_count
- `output_data/parquet/t3_emoji_label_correlation.parquet` — emoji_type, chi2_stat, p_value, Cramers_V, mean_density_Pos/Neu/Neg
- `output_data/parquet/t3_top_discriminating_emojis.parquet` — emoji, MI_score, most_common_label
- `output_data/img/t3_*.png` — correlation matrix heatmap, density violin per label, top MI emojis bar

**Logic:**
1. **Data-driven survey first**: extract all emojis; cross-tab each emoji vs label to see which labels it appears in most; output raw correlation matrix before any classification
2. Classify into 6 types based on survey findings: `face_positive`, `face_negative`, `face_neutral`, `symbol_positive`, `symbol_negative`, `other`
3. Create 12 new columns: 6 `_presence` (bool) + 6 `_count` (int) — all vectorized in polars using `str.extract_all` + Unicode regex
4. `emoji_density = emoji_count / char_count` (clip char_count at 1)
5. Flag emoji-only comments (`word_count == 0` and `emoji_count > 0`)
6. Chi² + Cramér's V: computed via scipy, per emoji type
7. Mutual information per individual emoji vs label: `sklearn.feature_selection.mutual_info_classif` on presence booleans

---

### Phase 2B — Engagement Semantics

#### [NEW] [phase2b_engagement.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase2b_engagement.py)

**Table 4 outputs:**
- `output_data/parquet/t4_engagement_semantics_by_label.parquet` — label, mean_like, median_like, mean_reply, median_reply, mean_ratio, %_controversial, %_endorsed, tier_dist
- `output_data/parquet/t4_temporal_engagement_baseline.parquet` — era, monthly_median_likes, label_%_per_era, mean_velocity, inflation_index_per_label
- `output_data/img/t4_*.png` — controversy vs endorsement stacked bars, velocity timeline, engagement tier by label heatmap

**Logic:**
1. Compute `reply_to_like_ratio = (reply_count + 1) / (like_count + 1)` in polars
2. Define 5 engagement tiers using polars `when/then`
3. Controversy hypothesis test: Mann-Whitney U for `reply_to_like_ratio` between Negative vs Positive; report p-value
4. COVID eras: pre=before 2020-01, during=2020-01 to 2021-12, post=2022-01+
5. Monthly median baseline computed in polars; `like_inflation_index = like_count / monthly_median` via join
6. `engagement_velocity = like_count / days_since_publish` (requires `crawled_at - published_at`)
7. Statistical test: Kruskal-Wallis H across labels for each engagement metric; η² effect size via scipy

---

### Phase 2C — Multilingual Stratification

#### [NEW] [phase2c_multilingual.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase2c_multilingual.py)

**Table 5 outputs:**
- `output_data/parquet/t5_language_distribution.parquet` — language, count, %, unique_authors, unique_videos, avg_confidence, script_family
- `output_data/parquet/t5_multilingual_stratification.parquet` — language, N, %_Pos, %_Neu, %_Neg, mean_chars, mean_words, emoji_density, mean_likes, mean_replies, code_switching_%
- `output_data/img/t5_*.png` — language distribution bars, label distribution per language heatmap, emoji density by language

**Logic:**
1. **Fast FastText LID**: load `lid.176.bin` via the `fasttext` Python package (`fasttext.load_model`). Call `model.predict(texts, k=2)` in **batches of 10,000** using Python list comprehension (not `map_elements`), then wrap back into polars via `pl.Series`. This should process 400k rows in <60 seconds.
2. Confidence score = probability of top-1 prediction
3. Ensemble tie-break: if fasttext confidence < 0.5, use `langdetect` as secondary (only for low-confidence subset)
4. Script family detection via Unicode range regex in polars `str.contains`
5. Code-switching flag: secondary language covers ≥10% of tokens OR script blocks switch mid-comment
6. Top 5 languages + "other" grouping in polars
7. Chi-square homogeneity test across languages for label distribution; flag languages with statistically higher Negative rates

---

### Phase 2D — N-Gram Discriminative Analysis

#### [NEW] [phase2d_ngrams.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase2d_ngrams.py)

**Table 6 outputs:**
- `output_data/parquet/t6_ngram_freq_profile.parquet` — n_gram_order, total_unique_grams, avg_grams_per_comment, top_20_grams_per_label_rel_freq_per_1k
- `output_data/parquet/t6_ngram_discriminative_power.parquet` — n_gram_order, mean_MI, top_3_grams_by_MI, coverage_%_Pos/Neu/Neg, recommendation
- `output_data/img/t6_*.png` — MI comparison across gram orders (bar), top-gram relative freq heatmap per label

**Logic:**
1. **Inspect existing `data/13gram/` and `data/37gram/` parquets** — if they contain token sequences with label columns, reuse; otherwise recompute
2. Text preprocessing: lowercase, strip URLs (`re.sub`), keep emoji tokens (re-tokenized as `__EMOJI_xxx__`), drop stopwords (NLTK multilingual)
3. N-gram orders: **1, 2, 3, 4, 5, 6, 7** (and optionally 8-10 if 7 shows strong signal) using `sklearn.feature_extraction.text.CountVectorizer` with appropriate `ngram_range`
4. Filter: DF < 5 dropped, DF > 50% corpus dropped
5. TF-IDF per label class: fit vectorizer on each label's corpus separately
6. Relative frequency per 1,000 comments per label (avoids raw-count scale-up errors)
7. MI and Chi² via `sklearn.feature_selection.mutual_info_classif` and `chi2`
8. Mean MI per gram order → recommendation table
9. Coverage: % of comments in each label covered by top-100 n-grams of that order
10. All heavy vectorization uses `scipy.sparse` to stay memory-efficient with 48GB RAM

---

### Phase 2E — Source Query & Temporal Context

#### [NEW] [phase2e_source_temporal.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase2e_source_temporal.py)

**Table 7 outputs:**
- `output_data/parquet/t7_source_query_profile.parquet` — query, N, %_Pos/Neu/Neg, avg_likes, avg_replies, avg_length, dominant_lang, bias_flag
- `output_data/parquet/t7_temporal_baseline.parquet` — month, volume, %_Pos/Neu/Neg, median_lag_days, median_likes, anomaly_flag
- `output_data/img/t7_*.png` — source bias divergence bars, temporal sentiment line chart, lag distribution

**Logic:**
1. Per-query stats in polars group_by
2. Bias flag: query where single label > 80% of its comments
3. Monthly aggregation in polars using `dt.truncate("1mo")`
4. Temporal anomaly: month where label % deviates > 2σ from global mean
5. `lag_days = crawled_at - published_at` in polars using duration arithmetic

---

### Phase 3 — Feature Engineering & Multicollinearity Audit

#### [NEW] [phase3_feature_eng.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase3_feature_eng.py)

**Table 8 outputs:**
- `output_data/parquet/t8_feature_inventory.parquet` — feature_name, category, dtype, derivation_logic, intended_scaling_method
- `output_data/parquet/t8_vif_audit.parquet` — feature, VIF_before, action, VIF_after, rationale
- `output_data/parquet/t8_final_correlation_matrix.parquet` — post-remediation Pearson matrix (long format)
- `output_data/parquet/t8_scaling_registry.parquet` — feature, scaler_type, params
- `output_data/parquet/model_ready_features.parquet` — final feature-engineered dataset (without embeddings, saved separately)
- `output_data/img/t8_*.png` — VIF before/after bar, correlation heatmap before/after

**Feature groups (per LastDay.md Table 8):**
- **(A) Text**: recomputed char_count, word_count, avg_word_length, uppercase_ratio, exclamation_count, question_count, hashtag_count, mention_count, emoji_count, emoji_density, 6 emoji_type_count vectors, `text_sentiment_lexicon_score` (multilingual VADER / SentiWordNet)
- **(B) Engagement**: like_count_log, reply_count_log (log1p), reply_to_like_ratio, engagement_tier (ordinal), likes_per_day, replies_per_day
- **(C) Metadata**: source_query_target_encoded (smoothed mean label per query on training set only), source_query_freq_encoded, primary_language one-hot (top 5 + other), hour_of_day, day_of_week, month, is_weekend, days_since_publish, era_flag
- **(D) Embeddings**: stored as float32 arrays, referenced by name

**VIF logic:**
1. Compute Pearson + Spearman matrix for all numeric features (numpy / polars)
2. Iterative VIF via `statsmodels.stats.outliers_influence.variance_inflation_factor`
3. For pairs VIF > 10: create ratio feature or drop weaker predictor; document all actions
4. Re-run VIF post-remediation until all < 10

**Scaling:**
- StandardScaler on normal numerics
- RobustScaler on engagement features (like_count_log, reply_count_log, ratios)
- One-hot: language, engagement_tier
- Serialize pipeline via `joblib` to `output_data/models/scaling_pipeline.joblib`

---

### Phase 3B — EDA of Engineered Features

#### [NEW] [phase3b_feature_eda.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase3b_feature_eda.py)

**Table 9 outputs:**
- `output_data/parquet/t9_feature_discriminative_power.parquet` — feature_name, F_stat/H_stat, p_value, eta_squared, silhouette_if_embedding, keep/drop_recommendation
- `output_data/parquet/t9_feature_distribution_by_label.parquet` — feature × label matrix of means/medians
- `output_data/parquet/t9_categorical_association.parquet` — feature, Cramers_V, p_value
- `output_data/img/t9_*.png` — η² ranked bar chart, silhouette scores per embedding, Cramér's V categorical heatmap

**Logic:**
1. ANOVA (scipy `f_oneway`) if feature passes Shapiro-Wilk normality (subsample); else Kruskal-Wallis H (`kruskal`)
2. η² = SS_between / SS_total (computed from ANOVA output)
3. Silhouette score on PCA-50 reduced embedding; and UMAP-2D reduced embedding (subsample 20k for speed)
4. Point-biserial correlation for binary presence features
5. Cramér's V for categorical features (language, engagement_tier, era_flag) vs label
6. Target encoding leakage check for `source_query_target_encoded`: compute correlation with label on held-out fold (use 20% stratified split)

---

### Phase 4 — Modeling Pipeline

#### [NEW] [phase4_modeling.py](file:///mnt/seagate_320/assets/NMDS/src/eda/phase4_modeling.py)

**Table 10 outputs:**
- `output_data/parquet/t10_cv_fold_summary.parquet` — fold, train_N, val_N, %_Pos_train, %_Pos_val, class_weight_vector
- `output_data/parquet/t10_model_leaderboard.parquet` — model, accuracy, macro_P, macro_R, macro_F1, Pos_F1, Neu_F1, Neg_F1
- `output_data/parquet/t10_optuna_best_config.parquet` — param, best_value
- `output_data/parquet/t10_shap_feature_importance.parquet` — feature, mean_abs_SHAP, category
- `output_data/parquet/t10_error_analysis.parquet` — true_label, pred_label, count, dominant_language, dominant_source_query
- `output_data/img/t10_*.png` — confusion matrices (normalized), SHAP beeswarm, optimization history

**Logic:**
1. **Train-test split first**: `train_test_split(stratify=label, test_size=0.2, random_state=42)` — test set locked away until final evaluation
2. `StratifiedKFold(n_splits=5)` on train only
3. Class weights: inverse frequency
4. SMOTE applied **only inside training fold** (never on validation): use `imblearn.pipeline.Pipeline` wrapping SMOTE + model
5. Baseline models: Logistic Regression + LinearSVC on TF-IDF unigrams + top engineered features (post-VIF)
6. Optuna optimization: 50 trials, macro-F1 objective, early stopping patience=3; search space per LastDay.md Table 10
7. SHAP: `shap.LinearExplainer` for linear models; `shap.TreeExplainer` for any tree meta-learner
8. McNemar's test between best models using `statsmodels.stats.contingency_tables.mcnemar`
9. Error analysis: group misclassifications by language, source_query, emoji presence

#### [NEW] [transformers_kaggle.py](file:///mnt/seagate_320/assets/NMDS/src/models/transformers_kaggle.py)

- Stub / scaffold for DistilBERT and RoBERTa late-fusion (to be run on Kaggle GPU)
- Contains architecture definitions, AdamW + linear warmup + cosine decay setup
- Accepts engineered feature vectors as late-fusion input to `[CLS]` token MLP head
- **Not meant to run locally**; clearly documented at top of file

---

## Execution Order

Per `LastDay.md` Section "Execution Order Summary":

1. `phase0_cleaning.py` → locked cleaned corpus
2. `phase1_label_stats.py` → baseline stratified stats
3. `phase2a_emoji.py`, `phase2b_engagement.py`, `phase2c_multilingual.py`, `phase2d_ngrams.py`, `phase2e_source_temporal.py` → **parallelizable** (run concurrently via subagents)
4. `phase3_feature_eng.py` → feature creation + VIF
5. `phase3b_feature_eda.py` → validate feature power
6. `phase4_modeling.py` → CV + SMOTE + Optuna + SHAP

---

## Verification Plan

### Automated Checks
- Each script runs to completion and writes expected parquet files
- Schema validation: check column names and dtypes match spec
- Cardinality check: cleaned corpus rows < 395k (dedup), > 300k (sanity)
- VIF audit: post-remediation max VIF < 10
- SMOTE: verify SMOTE is only called inside `imblearn.pipeline.Pipeline` training loop
- FastText LID: total processing time < 5 minutes for 400k rows

### Manual Verification
- Review all saved parquet files have meaningful non-null values
- Review all saved PNG charts for visual correctness
- Confirm no test set data leakage in target encoding (leakage check in phase3b)
- Confirm label distribution in each CV fold is stratified (table t10_cv_fold_summary)
