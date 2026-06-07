# combined_embeddings.parquet
|    column_name    | column_type | null | key  | default | extra |
|-------------------|-------------|------|------|---------|-------|
| comment_id        | VARCHAR     | YES  | NULL | NULL    | NULL  |
| post_id           | VARCHAR     | YES  | NULL | NULL    | NULL  |
| comment_text      | VARCHAR     | YES  | NULL | NULL    | NULL  |
| published_at      | VARCHAR     | YES  | NULL | NULL    | NULL  |
| like_count        | BIGINT      | YES  | NULL | NULL    | NULL  |
| reply_count       | BIGINT      | YES  | NULL | NULL    | NULL  |
| author_id         | VARCHAR     | YES  | NULL | NULL    | NULL  |
| author_name       | VARCHAR     | YES  | NULL | NULL    | NULL  |
| title_youtube     | VARCHAR     | YES  | NULL | NULL    | NULL  |
| source_query      | VARCHAR     | YES  | NULL | NULL    | NULL  |
| crawled_at        | VARCHAR     | YES  | NULL | NULL    | NULL  |
| char_count        | BIGINT      | YES  | NULL | NULL    | NULL  |
| word_count        | BIGINT      | YES  | NULL | NULL    | NULL  |
| avg_word_length   | DOUBLE      | YES  | NULL | NULL    | NULL  |
| uppercase_ratio   | DOUBLE      | YES  | NULL | NULL    | NULL  |
| exclamation_count | BIGINT      | YES  | NULL | NULL    | NULL  |
| question_count    | BIGINT      | YES  | NULL | NULL    | NULL  |
| hashtag_count     | BIGINT      | YES  | NULL | NULL    | NULL  |
| mention_count     | BIGINT      | YES  | NULL | NULL    | NULL  |
| emoji_count       | BIGINT      | YES  | NULL | NULL    | NULL  |
| like_count_log    | DOUBLE      | YES  | NULL | NULL    | NULL  |
| embedding         | FLOAT[]     | YES  | NULL | NULL    | NULL  |
| embedding_char    | FLOAT[]     | YES  | NULL | NULL    | NULL  |
| embedding_word    | FLOAT[]     | YES  | NULL | NULL    | NULL  |
| embedding_ft      | FLOAT[]     | YES  | NULL | NULL    | NULL  |

1. This is the table that we are gonna analyze. It is full of youtube comments. Combine the analyze methods like above and suggest me more EDA & Data Science methods to dig in those. I want deep analysis, t-SNE, PCA, further and further. Deep into data. This is multilingual data so please suggest me ways to analyze them. I want to analyze multilingual, identify their languages, make use of them and understand deeply. We are not going with the classification models here, just analysis. Go from potentially normal to Graph models, such as Graph Laplacian, Graph Clustering methods, linking using video titles, using comments, using etc... I need deeply integrated methods instead of just on the surface. Dig as deep as you can, as deep as you can. For now, we do not care about machine learning models for prediction. We only have 3 labels for comments: positive, neutral & negative. For now, we also do not care much about those labels. Only care about deep data analysis. Open and open further out to new methods, to advanced methods given these data
2. Types of embedding: char = small n-gram, word = big n-gram, ft=fasttext multilingual, embedding = sentence transformers dense embedding.
3. Data is NULL=YES, so we need to figure out ways to eliminate the rows with null values => but do not eliminate too many of them. Most important: comment_id & comment_text.

# Prior work:
Here is an exhaustive, deeply detailed breakdown of the capstone project report you provided. This guide is structured to give you a complete understanding of the methodology, architecture, and implementation details for YouTube comment sentiment analysis.

# Comprehensive Guide to YouTube Comment Sentiment Analysis
## Based on the IT4142E Capstone Project Report

---

## Part 1: Project Overview & Core Philosophy

### 1.1 What is Being Solved?
- **Problem:** Social media comments (YouTube) are noisy, informal, context-dependent, and massive in scale. Manual analysis is impossible.
- **Goal:** Build an automated, end-to-end system that can crawl YouTube comments, classify them as Positive, Neutral, or Negative, and present insights.
- **Domain:** English-language entertainment content (movies, music).

### 1.2 Key Constraints (Scope)
- **Only text:** No images or video analysis.
- **Three classes only:** Positive, Neutral, Negative (no fine-grained emotions like "anger" or "joy").
- **No multilingual analysis.**
- **Top-level comments only** (replies are mostly excluded).
- **YouTube only** (not Facebook, TikTok, etc., due to API access).

---

## Part 2: The Data Pipeline (End-to-End)

This is the most critical part. The report uses a **three-stage intelligent labeling pipeline**, not just simple manual labeling.

### Stage 1: Data Collection (YouTube API v3)

**Source:** YouTube Data API v3 (not web scraping, to avoid terms-of-service violations).

**Search Strategy (Appendix A):**
- **Positive/Nuanced:** Official music videos, live concerts, OSTs, movie trailers, reactions, trending hits (Adele, Taylor Swift, etc.).
- **Negative/Controversial (intentionally added):** "box office flop", "worst movie", "bad music review", "rap beef", "diss track", "controversial music video".

**What is collected per comment (Table 2):**
- `comment_id`, `comment_text`
- `like_count`, `reply_count`
- `author_id`, `author_name`
- `title_youtube` (video title - used as context!)
- `source_query` (which search term found it)
- Timestamps, etc.

**Output:** 48,205 comments after deduplication.

### Stage 2: Preprocessing & Feature Engineering

**Text Cleaning (Preserves sentiment cues):**
- Remove hyperlinks, HTML tags, user mentions.
- **Keep emojis** (separate with whitespace).
- **Keep punctuation** (! and ? are isolated, others normalized).
- **Keep casing** (but create `uppercase_ratio` feature for emphasis).
- Collapse multiple spaces.

**Derived Numeric Features (from text):**
- Character count, word count, average word length
- Emoji count, mention count, hashtag count, punctuation count
- Uppercase ratio

**Engagement Features:**
- `like_count` and `log_like_count` (log transform reduces outlier impact).

### Stage 3: The Intelligent Labeling Strategy (Most Sophisticated Part)

This is a **cascading architecture** with confidence thresholds, model agreement, weighted voting, and human review.

**Two AI Models Used for Labeling (Appendix C):**
- `gemini-2.5-flash` (fast, cheap, lower quality)
- `gemini-2.5-pro` (expert, slower, higher quality)

**Parameters (Table 5):**
- `CONF_FAST_ACCEPT = 0.985` → Flash model must be 98.5% confident to accept alone.
- `AUDIT_RATE = 0.12` → 12% of accepted samples are randomly sent to expert anyway (probabilistic auditing).
- `WEIGHT_FAST = 1.0`, `WEIGHT_PRO = 2.0` (expert has double weight in voting).
- `MARGIN_THRESHOLD = 0.2` → If top two scores are within 20%, send to human.

**The Process (Figure 1 in report):**

1. **Fast model** (`gemini-2.5-flash`) predicts with probability distribution.
2. **If** `max(probabilities) >= 0.985` **AND** not randomly audited → **Accept fast label.**
3. **Else** → Send to **expert model** (`gemini-2.5-pro`).
4. **If** fast and expert agree → **Accept by agreement.**
5. **If they disagree** → Weighted soft voting:
   ```
   S(label) = (1.0 * P_fast + 2.0 * P_pro) / 3.0
   ```
6. **Margin check:** `Δ = S(1st) - S(2nd)`. If `Δ >= 0.2` → Accept. Else → **Human review.**
7. **Output stored:** `sentiment_label`, `margin` (confidence gap), `strategy` (how label was obtained).

**Why this matters:** No single model is perfect. This system balances cost, speed, and accuracy while detecting uncertainty explicitly.

---

## Part 3: Exploratory Data Analysis (EDA) Key Findings

### Data Quality (Figure 2)
- `parent_comment_id` 99.9% missing (by design - top-level only).
- `margin` 98.6% missing (only recorded when disagreement occurred).
- 504 duplicate comment_ids removed, 1,354 duplicate texts (templated comments).

### Text Length (Figure 3)
- **Right-skewed:** Most comments are short (<20 words).
- 99th percentile used for outlier removal (not 95th, to keep more data).
- **Negative comments are longest** (users explain criticism).

### Sentiment Distribution (Figure 4)
- Positive: 47.7% (22,980)
- Neutral: 30.1% (14,500)
- Negative: 22.2% (10,725)
- **Imbalance ratio:** 0.467 minority-to-majority → Use class-weighted loss, not aggressive resampling.

### Word Frequency by Sentiment (Figure 6)
- **Positive:** "love", "best", "good", "great", "beautiful", "amazing"
- **Negative:** "bad", "think", "would", "never" (more explanatory/argumentative)
- **Neutral:** "know", "anyone", "time" (factual, no strong emotion)
- **Key insight:** Unigram frequency alone is insufficient → need context-aware models.

### Bigrams (Figure 7)
- Common across all: "feel like", "sounds like", "looks like" (sentiment-agnostic)
- **Conclusion:** TF-IDF is better than raw bag-of-words because it down-weights common non-discriminative phrases.

### Engagement Analysis (Figures 8 & 9)
- Like counts are zero-inflated and long-tailed (most comments have 0-1 likes).
- Weak correlation between sentiment and likes → engagement is driven by visibility, not polarity.

### Correlation Matrix (Figure 10)
- `char_count` and `word_count` are almost perfectly correlated (r≈0.99) → redundant.
- Expressive features (emoji_count, uppercase_ratio) are independent of length.
- **Implication:** Remove redundant features, keep diverse groups.

---

## Part 4: All Methods Used (Full Breakdown)

### Method 1: Logistic Regression (Baseline Linear)

**Input:** TF-IDF vectors + numerical features (log_like_count, emoji_count, uppercase ratio, negation indicator).

**Architecture:**
- One-vs-Rest (OvR) for 3 classes.
- Sigmoid activation: `h(x) = 1/(1 + e^(-θ^T x))`
- L2 regularization (Ridge).

**Loss function:**
```
J(θ) = -1/m Σ [y log(h(x)) + (1-y) log(1-h(x))] + (λ/2m) Σ θ²
```

**Hyperparameters:** Grid search with 10-fold CV, `C` in [0.01, 100], liblinear solver.

**Result:** 69.99% accuracy, 69.56% F1.

---

### Method 2: Linear Support Vector Machine (Linear SVM)

**Why linear?** TF-IDF creates >13,000 features that are near-linearly separable. Nonlinear kernels would overfit.

**Objective (Squared Hinge Loss + L2):**
```
min ½ w^T w + C Σ max(0, 1 - y_i(w^T x_i + b))²
```

**Key difference from Logistic Regression:** Maximizes margin, not just likelihood. Squared hinge loss imposes stronger penalty on misclassifications.

**Result:** 69.78% accuracy, 69.78% F1 (more balanced precision/recall than LR).

---

### Method 3: Bi-LSTM (Bidirectional Long Short-Term Memory)

**Why Bi-LSTM?** Unidirectional LSTM only sees past context. Bi-LSTM sees both past and future via two parallel LSTM layers.

**Forward LSTM:** `→h_t` (left to right)
**Backward LSTM:** `←h_t` (right to left)
**Final representation:** `h_t = [→h_t; ←h_t]` (concatenation)

**LSTM Gating Mechanism (per direction):**
- Forget gate: `f_t = σ(W_f x_t + U_f h_{t-1} + b_f)`
- Input gate: `i_t = σ(W_i x_t + U_i h_{t-1} + b_i)`
- Candidate cell: `c̃_t = tanh(W_c x_t + U_c h_{t-1} + b_c)`
- Output gate: `o_t = σ(W_o x_t + U_o h_{t-1} + b_o)`
- Cell update: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`
- Hidden state: `h_t = o_t ⊙ tanh(c_t)`

**Embeddings:** 300d fastText (pretrained), frozen for 1 epoch then fine-tuned.

**Preprocessing for Bi-LSTM:**
- Replace URLs, mentions, hashtags with `<URL>`, `<USER>`, `<HASHTAG>`
- Keep emojis as tokens
- Vocabulary size: 30,000 most frequent
- Sequence length: 95th percentile (clipped 16-256 tokens)

**Architecture (after Bayesian Optimization):**
- Hidden dimension: 128 (not 256)
- Number of LSTM layers: 2 (not 3)
- Dropout: 0.4
- Batch size: 128
- Learning rate: 4.01e-4
- Max sequence length: 128
- Max-pooling for sequence-level classification

**Optimization:** AdamW, ReduceLROnPlateau scheduler, gradient clipping at 1.0, early stopping (patience 3).

**Result:** 73.89% accuracy, 71.52% F1.

---

### Method 4: RoBERTa (Transformer-Based)

**Two variants used:**

**Variant A:** `xlm-roberta-base` (multilingual, 100+ languages)
**Variant B:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (best performer)

**Architecture (Transformer):**
- Multi-head self-attention: `Attention(Q,K,V) = softmax(QK^T/√d_k) V`
- Masked Language Modeling (MLM) objective only (no Next Sentence Prediction like BERT).
- Dynamic masking (different tokens masked each epoch).

**RoBERTa advantages over BERT:**
- Larger batch sizes
- Longer training
- More data
- Removed NSP objective

**Input representation (minimal preprocessing):**
- Keep emojis, informal text (model can handle noise)
- Tokenizer.encode_plus → `{input_ids, attention_mask, labels}`
- Max sequence length: 256 tokens (tested 128 and 512, 256 was optimal)

**Training config:**
- Batch size: 32 (GPU memory constrained)
- Epochs: 2 (fine-tuning only)
- Optimizer: AdamW
- Learning rate: 2e-5
- Cosine scheduler with 10% warmup

**Why no feature engineering?** RoBERTa learns contextualized representations directly from raw tokens.

**Result (twitter-roberta):** 84.86% accuracy, 84.76% macro F1.

---

## Part 5: Training & Evaluation Methodology

### Data Splitting (Different for each model)

| Model | Train | Validation | Test |
|-------|-------|------------|------|
| LR, SVM | 80% | (CV only) | 20% |
| Bi-LSTM | 80% | 10% | 10% |
| RoBERTa | 70% | 15% | 15% |

**All splits use stratified sampling** to preserve class distribution.

### Hyperparameter Tuning Methods

**For LR & SVM:** Grid Search with 10-fold Cross-Validation.

**For Bi-LSTM:** Bayesian Optimization with Optuna.
- 50 trials
- Median Pruner for early stopping
- Objective: maximize validation accuracy
- Search spaces: LR (1e-4 to 1e-3 log scale), dropout (0.3-0.5), hidden dim (128/256), layers (2/3), batch size (128/256), seq len (96/128)

**For RoBERTa:** Minimal tuning (fixed architecture, only sequence length tested).

### Evaluation Metrics (All macro-averaged)
- **Accuracy:** (TP+TN)/(Total)
- **Precision:** TP/(TP+FP)
- **Recall:** TP/(TP+FN)
- **F1-Score:** 2 * (Precision * Recall)/(Precision + Recall)

---

## Part 6: Deployment Architecture

**Model deployed:** `twitter-roberta-base-sentiment-latest` (best performance: 84.86% accuracy)

**Frontend:** React
**Backend:** Django (RESTful APIs)
**AI Insights:** Gemini API (generates natural language summaries of sentiment distribution)

**Features:**
1. Input YouTube URL + number of comments
2. Real-time crawl via YouTube Data API
3. Sentiment classification per comment
4. Video-level aggregation (percentages per class)
5. History page (store past analyses)
6. AI-generated insight module (e.g., "Audience response is generally positive with 65% positive comments...")

**Deployment environment:** Local (development/evaluation only, not cloud).

---

## Part 7: Results Summary Table

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 69.99% | 69.51% | 69.99% | 69.56% |
| Linear SVM | 69.78% | 69.30% | 69.78% | 69.78% |
| Bi-LSTM | 73.89% | 71.49% | 71.55% | 71.52% |
| xlm-roberta-base | 79.45% | 79.37% | 79.45% | 79.33% |
| **twitter-roberta-base-sentiment-latest** | **84.86%** | **84.87%** | **84.86%** | **84.76%** |

**Key insight:** Domain-aligned pretraining (Twitter data) + sentiment fine-tuning (TweetEval) yields ~15% absolute gain over linear baselines.

---

## Part 8: Future Work Directions

1. **Finer-grained classes:** Add specific emotions (anger, joy, sarcasm) or feedback types (content request, criticism).
2. **Better models:** Ensembles, domain-adaptive pretraining, larger transformers.
3. **MLOps integration:** MLflow for experiment tracking, CI/CD for continuous training/deployment.
4. **Production deployment:** Cloud hosting (vs local only).

---

## Part 9: Practical Lessons for Your Own Implementation

### If you build this yourself:

1. **Don't label everything manually.** Use a cascading system (fast model → expert model → margin check → human). It's cheaper and faster.

2. **Keep emojis and punctuation.** They carry sentiment. The report explicitly preserves them.

3. **Use log transform for like counts.** Raw likes are zero-inflated and long-tailed.

4. **For linear models:** TF-IDF + handcrafted features (uppercase ratio, negation indicator, emoji count) is better than raw counts.

5. **For deep learning:** Bi-LSTM is a good middle ground, but transformer (RoBERTa) is significantly better if you have GPU.

6. **Don't use BERT.** Use RoBERTa or Twitter-RoBERTa for social media text. The report specifically chose RoBERTa over BERT because RoBERTa removes Next Sentence Prediction and uses dynamic masking.

7. **Context matters.** The labeling system uses video title and source_query, not just comment text. A comment saying "terrible" could be positive if the video is about "terrible movie" (sarcasm/context).

8. **The 98.5% threshold for fast model acceptance is very high.** This ensures high precision but may reject many samples. Adjust based on your needs.

9. **Margin threshold of 0.2** means if the top two predicted classes are within 20% probability, send to human. This catches ambiguity.

10. **Class-weighted loss** (not oversampling) is preferred for this level of imbalance (0.467 ratio).

---

## Part 10: Critical Definitions to Remember

| Term | Definition |
|------|------------|
| **Cascading labeling** | Fast model → expert model → weighted voting → human review |
| **Margin** | Difference between top two predicted probabilities |
| **Macro F1** | Average of F1 scores per class (not weighted by class size) |
| **TF-IDF** | Term Frequency-Inverse Document Frequency (down-weights common words) |
| **Bi-LSTM** | Two LSTM layers (forward + backward) concatenated |
| **RoBERTa** | Robustly optimized BERT approach (larger batch, dynamic masking, no NSP) |
| **Stratified sampling** | Preserve class percentages across train/val/test splits |

---

This breakdown gives you the complete blueprint. The report's key innovation is the **cascading labeling system with confidence thresholds and margin checks**, not just the final RoBERTa model. If you implement this, focus on that pipeline first, then benchmark models.