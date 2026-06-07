**Document: Phase I — Data Cleaning & Descriptive Statistics Baseline**
**Dataset:** 400K Multilingual YouTube Comments (`combined_embeddings.parquet`)
**Objective:** Establish numerical ground truth before any embedding or graph analysis.

---

## Section 1: Data Cleaning Audit Log
*Before analysis numbers, document what was removed or fixed.*

| Table 1.1: Cleaning Actions Ledger |
| :--- |
| **Action ID** | **Action Description** | **Records Affected** | **% of Raw Corpus** | **Rationale** |
| D-01 | Remove rows where `comment_id` is NULL | [n] | [%] | Primary key violation |
| D-02 | Remove rows where `comment_text` is NULL or empty string | [n] | [%] | Unanalyzable |
| D-03 | Remove rows where `comment_text` is whitespace-only | [n] | [%] | Unanalyzable |
| D-04 | Deduplicate exact `comment_text` duplicates (keep first) | [n] | [%] | Spam/templates |
| D-05 | Deduplicate exact `comment_id` duplicates | [n] | [%] | API crawl overlap |
| D-06 | Flag rows with NULL `published_at` (impute from `crawled_at`) | [n] | [%] | Temporal analysis |
| D-07 | Flag rows with NULL `like_count` (impute 0) | [n] | [%] | Engagement analysis |
| D-08 | Flag rows with NULL `reply_count` (impute 0) | [n] | [%] | Engagement analysis |
| D-09 | Flag rows with NULL `author_id` | [n] | [%] | Authorship graph |
| D-10 | Remove rows where `char_count` = 0 but `comment_text` not NULL (encoding error) | [n] | [%] | Data corruption |
| **Final** | **Clean Corpus Size** | **[N_final]** | **[% of raw]** | **Working dataset** |

---

## Section 2: Corpus Grand Totals
*Single-row tables establishing the absolute baseline.*

| Table 2.1: Corpus Cardinality |
| :--- |
| Metric | Value |
| Raw records ingested | 400,000 |
| Records after cleaning | [N_final] |
| Unique comments (`comment_id`) | [n] |
| Unique videos (`post_id`) | [n] |
| Unique authors (`author_id`) | [n] |
| Unique author names (`author_name`) | [n] |
| Unique source queries (`source_query`) | [n] |
| Time span (first `published_at` → last `published_at`) | [YYYY-MM-DD] to [YYYY-MM-DD] |
| Time span (first `crawled_at` → last `crawled_at`) | [YYYY-MM-DD] to [YYYY-MM-DD] |

| Table 2.2: Text Volume Totals |
| :--- |
| Metric | Value |
| Total characters (all comments) | [sum] |
| Total words (all comments) | [sum] |
| Average characters per comment | [mean ± std] |
| Average words per comment | [mean ± std] |
| Median characters per comment | [median] |
| Median words per comment | [median] |
| 95th percentile char count | [p95] |
| 99th percentile char count | [p99] |
| 95th percentile word count | [p95] |
| 99th percentile word count | [p99] |
| Total emojis across corpus | [sum] |
| Total hashtags across corpus | [sum] |
| Total mentions across corpus | [sum] |
| Total exclamation marks | [sum] |
| Total question marks | [sum] |

---

## Section 3: Language Distribution
*The multilingual backbone. Every number here stratifies everything downstream.*

| Table 3.1: Primary Language Distribution (Top 20 + Others) |
| :--- |
| **Rank** | **Language** | **ISO Code** | **Comment Count** | **% of Clean Corpus** | **Cumulative %** | **Unique Authors** | **Unique Videos** |
| 1 | English | en | [n] | [%] | [%] | [n] | [n] |
| 2 | Spanish | es | [n] | [%] | [%] | [n] | [n] |
| 3 | Portuguese | pt | [n] | [%] | [%] | [n] | [n] |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 20 | [Language] | [xx] | [n] | [%] | [%] | [n] | [n] |
| — | Others (aggregated) | misc | [n] | [%] | [%] | [n] | [n] |
| — | **Unidentified / Low Confidence** | unk | [n] | [%] | [%] | [n] | [n] |
| **Total** | | | **[N_final]** | **100%** | | | |

| Table 3.2: Language Identification Confidence |
| :--- |
| **Confidence Bucket** | **Comment Count** | **% of Corpus** |
| ≥ 0.99 (Very High) | [n] | [%] |
| 0.95 – 0.99 (High) | [n] | [%] |
| 0.90 – 0.95 (Medium) | [n] | [%] |
| 0.80 – 0.90 (Low) | [n] | [%] |
| < 0.80 (Very Low / Failed) | [n] | [%] |

| Table 3.3: Script Distribution (Orthographic Analysis) |
| :--- |
| **Script Family** | **Comment Count** | **% of Corpus** | **Top Languages Within** |
| Latin (Basic + Extended) | [n] | [%] | EN, ES, PT, FR, DE, ... |
| Cyrillic | [n] | [%] | RU, UK, BG, ... |
| CJK Unified | [n] | [%] | ZH, JA, KO |
| Arabic / Hebrew | [n] | [%] | AR, FA, UR, HE |
| Devanagari | [n] | [%] | HI, MR, NE |
| Thai | [n] | [%] | TH |
| Mixed / Multiple Scripts | [n] | [%] | Code-switching flag |
| Emoji-only / Symbolic | [n] | [%] | — |

| Table 3.4: Code-Switching Detection Summary |
| :--- |
| **Category** | **Comment Count** | **% of Corpus** |
| Monolingual (single language ≥ 95% tokens) | [n] | [%] |
| Bilingual (two languages, each ≥ 5% tokens) | [n] | [%] |
| Trilingual+ (three+ languages detected) | [n] | [%] |
| Intra-word mixing (e.g., Romanized Arabic) | [n] | [%] |
| English + Other (most common pair) | [n] | [%] |

| Table 3.5: Cross-Lingual Video Penetration |
| :--- |
| **Metric** | **Count** |
| Videos with comments in ≥ 1 language | [n] |
| Videos with comments in ≥ 2 languages | [n] |
| Videos with comments in ≥ 3 languages | [n] |
| Videos with comments in ≥ 5 languages | [n] |
| Maximum languages on a single video | [n] |
| Average languages per video | [mean] |

---

## Section 4: Textual Feature Statistics
*Per-language and overall distributions of the engineered features.*

| Table 4.1: Text Length by Language (Top 10 Languages) |
| :--- |
| **Language** | **N** | **Char Count Mean** | **Char Count Std** | **Char Median** | **Word Count Mean** | **Word Count Std** | **Word Median** | **Avg Word Length** |
| English | | | | | | | | |
| Spanish | | | | | | | | |
| Portuguese | | | | | | | | |
| ... | | | | | | | | |
| **Global** | | | | | | | | |

| Table 4.2: Expressive / Stylistic Feature Distribution (Overall) |
| :--- |
| **Feature** | **Mean** | **Std** | **Median** | **Min** | **Max** | **P95** | **P99** | **% of Comments = 0** |
| `uppercase_ratio` | | | | | | | | |
| `exclamation_count` | | | | | | | | |
| `question_count` | | | | | | | | |
| `hashtag_count` | | | | | | | | |
| `mention_count` | | | | | | | | |
| `emoji_count` | | | | | | | | |

| Table 4.3: Expressive Features by Language (Top 5 Languages) |
| :--- |
| **Language** | **Uppercase Ratio Mean** | **Emoji Count Mean** | **Exclamation Mean** | **Question Mean** | **Hashtag Mean** | **Mention Mean** |
| English | | | | | | |
| Spanish | | | | | | |
| Portuguese | | | | | | |
| Korean | | | | | | |
| Japanese | | | | | | |

| Table 4.4: Emoji Usage Statistics |
| :--- |
| **Metric** | **Value** |
| Comments containing ≥ 1 emoji | [n] ([%]) |
| Comments containing ≥ 5 emojis | [n] ([%]) |
| Comments with emoji-only text | [n] ([%]) |
| Unique distinct emojis in corpus | [n] |
| Top 10 most frequent emojis | [emoji] [count] [rank] |
| Emoji density per 100 chars (global) | [mean] |
| Emoji density per 100 chars by language (Top 5) | [table] |

| Table 4.5: Punctuation & Capitalization Extremes |
| :--- |
| **Metric** | **Value** |
| Comments with `uppercase_ratio` > 0.5 (shouting) | [n] ([%]) |
| Comments with `exclamation_count` > 5 | [n] ([%]) |
| Comments with `question_count` > 3 | [n] ([%]) |
| Comments with `exclamation_count` = 0 AND `question_count` = 0 AND `emoji_count` = 0 (plain text) | [n] ([%]) |

---

## Section 5: Engagement & Reaction Statistics
*Likes and replies are the social signal. Must understand their nulls and skew.*

| Table 5.1: Engagement Raw Distribution |
| :--- |
| **Metric** | **Like Count** | **Reply Count** |
| Total sum across corpus | [sum] | [sum] |
| Mean | [mean] | [mean] |
| Std Dev | [std] | [std] |
| Median | [median] | [median] |
| Min | [min] | [min] |
| Max | [max] | [max] |
| P95 | [p95] | [p95] |
| P99 | [p99] | [p99] |
| % Zero values | [%] | [%] |
| % Single value (exactly 1) | [%] | [%] |
| % Values > 100 | [%] | [%] |
| % Values > 1000 | [%] | [%] |

| Table 5.2: Engagement by Language (Top 10 Languages) |
| :--- |
| **Language** | **N** | **Like Mean** | **Like Median** | **Like Sum** | **Reply Mean** | **Reply Median** | **Reply Sum** | **Likes per Word Ratio** |
| English | | | | | | | | |
| Spanish | | | | | | | | |
| ... | | | | | | | | |

| Table 5.3: Engagement Tiers (Global) |
| :--- |
| **Tier** | **Like Count Range** | **Comment Count** | **% of Corpus** | **Cumulative %** |
| Zero | 0 | [n] | [%] | [%] |
| Micro | 1 | [n] | [%] | [%] |
| Low | 2 – 10 | [n] | [%] | [%] |
| Medium | 11 – 100 | [n] | [%] | [%] |
| High | 101 – 1000 | [n] | [%] | [%] |
| Viral | > 1000 | [n] | [%] | [%] |

| Table 5.4: `like_count_log` Verification |
| :--- |
| **Metric** | **Value** |
| Comments where `like_count_log` could not be computed (original = 0 or NULL) | [n] |
| Correlation(Pearson) between `like_count` and `like_count_log` | [r] |
| Correlation(Spearman) between `like_count` and `like_count_log` | [ρ] |

| Table 5.5: Reply-to-Like Ratio Analysis |
| :--- |
| **Metric** | **Value** |
| Comments with replies > likes | [n] ([%]) |
| Comments with replies = 0, likes = 0 | [n] ([%]) |
| Comments with replies = 0, likes > 0 | [n] ([%]) |
| Comments with replies > 0, likes = 0 | [n] ([%]) |
| Mean reply/like ratio (excluding zeros) | [mean] |
| Median reply/like ratio | [median] |

---

## Section 6: Temporal Statistics
*When were comments posted? When were they crawled?*

| Table 6.1: Temporal Span Summary |
| :--- |
| **Metric** | **`published_at`** | **`crawled_at`** |
| Earliest timestamp | [datetime] | [datetime] |
| Latest timestamp | [datetime] | [datetime] |
| Total span (days) | [n] | [n] |
| Records with valid timestamp | [n] | [n] |
| Records with NULL timestamp | [n] | [n] |

| Table 6.2: Comment Volume by Hour of Day (UTC) |
| :--- |
| **Hour (UTC)** | **Comment Count** | **% of Daily Volume** |
| 00 | [n] | [%] |
| 01 | [n] | [%] |
| ... | ... | ... |
| 23 | [n] | [%] |

| Table 6.3: Comment Volume by Day of Week |
| :--- |
| **Day** | **Comment Count** | **% of Weekly Volume** |
| Monday | [n] | [%] |
| Tuesday | [n] | [%] |
| ... | ... | ... |
| Sunday | [n] | [%] |

| Table 6.4: Monthly Volume Distribution |
| :--- |
| **Month** | **Comment Count** | **% of Corpus** | **Unique Videos** |
| YYYY-MM | [n] | [%] | [n] |
| ... | ... | ... | ... |

| Table 6.5: Lag Analysis (Crawled vs. Published) |
| :--- |
| **Lag Bucket** | **Comment Count** | **% of Corpus** |
| Same day (0 days) | [n] | [%] |
| 1 – 7 days | [n] | [%] |
| 8 – 30 days | [n] | [%] |
| 31 – 90 days | [n] | [%] |
| 91 – 365 days | [n] | [%] |
| > 365 days | [n] | [%] |
| Negative lag (crawled before published — data error) | [n] | [%] |

---

## Section 7: Metadata & Source Statistics
*What videos? What queries? Who commented?*

| Table 7.1: Video-Level Statistics |
| :--- |
| **Metric** | **Value** |
| Total unique videos (`post_id`) | [n] |
| Videos with exactly 1 comment | [n] ([%]) |
| Videos with 2 – 10 comments | [n] ([%]) |
| Videos with 11 – 100 comments | [n] ([%]) |
| Videos with 101 – 1000 comments | [n] ([%]) |
| Videos with > 1000 comments | [n] ([%]) |
| Mean comments per video | [mean] |
| Median comments per video | [median] |
| Gini coefficient (comment inequality across videos) | [G] |

| Table 7.2: Source Query Distribution |
| :--- |
| **Rank** | **`source_query`** | **Comment Count** | **% of Corpus** | **Unique Videos** | **Unique Authors** | **Top Language** | **Avg Likes** |
| 1 | [query] | [n] | [%] | [n] | [n] | [lang] | [mean] |
| 2 | [query] | [n] | [%] | [n] | [n] | [lang] | [mean] |
| ... | ... | ... | ... | ... | ... | ... | ... |
| Top 20 | | | | | | | |
| Others | (aggregated) | [n] | [%] | [n] | [n] | — | [mean] |

| Table 7.3: Author Activity Distribution |
| :--- |
| **Activity Tier** | **# Comments per Author** | **# Unique Authors** | **% of Authors** | **% of Total Comments** |
| Single-comment | 1 | [n] | [%] | [%] |
| Casual | 2 – 5 | [n] | [%] | [%] |
| Regular | 6 – 20 | [n] | [%] | [%] |
| Active | 21 – 100 | [n] | [%] | [%] |
| Power | > 100 | [n] | [%] | [%] |
| **Max comments by single author** | | [n] | — | [%] |

| Table 7.4: Author Name Analysis |
| :--- |
| **Metric** | **Value** |
| Unique `author_name` values | [n] |
| Unique `author_id` values | [n] |
| Authors with name = "Anonymous" or empty | [n] ([%]) |
| Authors with numeric-only names | [n] ([%]) |
| Authors with emoji in name | [n] ([%]) |
| `author_id` to `author_name` mapping ratio (names per ID) | [mean] |

---

## Section 8: Feature Correlation & Redundancy Matrix
*Numerical relationships between the engineered features.*

| Table 8.1: Pearson Correlation Matrix (Top Features) |
| :--- |
| | **char_count** | **word_count** | **avg_word_length** | **uppercase_ratio** | **exclamation_count** | **question_count** | **emoji_count** | **like_count** | **reply_count** |
| **char_count** | 1.000 | [r] | [r] | [r] | [r] | [r] | [r] | [r] | [r] |
| **word_count** | [r] | 1.000 | [r] | [r] | [r] | [r] | [r] | [r] | [r] |
| **avg_word_length** | [r] | [r] | 1.000 | [r] | [r] | [r] | [r] | [r] | [r] |
| **uppercase_ratio** | [r] | [r] | [r] | 1.000 | [r] | [r] | [r] | [r] | [r] |
| **exclamation_count** | [r] | [r] | [r] | [r] | 1.000 | [r] | [r] | [r] | [r] |
| **question_count** | [r] | [r] | [r] | [r] | [r] | 1.000 | [r] | [r] | [r] |
| **emoji_count** | [r] | [r] | [r] | [r] | [r] | [r] | 1.000 | [r] | [r] |
| **like_count** | [r] | [r] | [r] | [r] | [r] | [r] | [r] | 1.000 | [r] |
| **reply_count** | [r] | [r] | [r] | [r] | [r] | [r] | [r] | [r] | 1.000 |

| Table 8.2: Spearman Rank Correlation (Same Features) |
| *(Same structure as 8.1, but Spearman ρ — captures non-linear monotonic relationships)* |

| Table 8.3: High Redundancy Flags |
| :--- |
| **Feature Pair** | **Pearson r** | **Spearman ρ** | **Action Taken** |
| `char_count` vs `word_count` | [r] | [ρ] | [Keep both / Drop one / Create ratio] |
| `like_count` vs `like_count_log` | [r] | [ρ] | [Keep both / Drop linear] |
| `emoji_count` vs `exclamation_count` | [r] | [ρ] | [Flag for collinearity] |

---

## Section 9: Data Quality Flags & Anomaly Inventory
*Numbers that expose data problems or interesting edge cases.*

| Table 9.1: Null Pattern Matrix |
| :--- |
| **Column** | **Total NULL** | **% NULL** | **NULL by Top Language** | **Imputation Strategy** |
| `comment_id` | [n] | [%] | [lang: %] | Dropped |
| `post_id` | [n] | [%] | [lang: %] | Dropped |
| `comment_text` | [n] | [%] | [lang: %] | Dropped |
| `published_at` | [n] | [%] | [lang: %] | Imputed from `crawled_at` |
| `like_count` | [n] | [%] | [lang: %] | Imputed 0 |
| `reply_count` | [n] | [%] | [lang: %] | Imputed 0 |
| `author_id` | [n] | [%] | [lang: %] | Flagged |
| `author_name` | [n] | [%] | [lang: %] | Flagged |
| `title_youtube` | [n] | [%] | [lang: %] | None |
| `source_query` | [n] | [%] | [lang: %] | None |
| `crawled_at` | [n] | [%] | [lang: %] | None |
| `char_count` | [n] | [%] | [lang: %] | Recomputed from text |
| `word_count` | [n] | [%] | [lang: %] | Recomputed from text |
| `avg_word_length` | [n] | [%] | [lang: %] | Recomputed |
| `uppercase_ratio` | [n] | [%] | [lang: %] | Recomputed |
| `exclamation_count` | [n] | [%] | [lang: %] | Recomputed |
| `question_count` | [n] | [%] | [lang: %] | Recomputed |
| `hashtag_count` | [n] | [%] | [lang: %] | Recomputed |
| `mention_count` | [n] | [%] | [lang: %] | Recomputed |
| `emoji_count` | [n] | [%] | [lang: %] | Recomputed |
| `like_count_log` | [n] | [%] | [lang: %] | Recomputed |
| `embedding` | [n] | [%] | [lang: %] | Recomputed if text exists |
| `embedding_char` | [n] | [%] | [lang: %] | Recomputed if text exists |
| `embedding_word` | [n] | [%] | [lang: %] | Recomputed if text exists |
| `embedding_ft` | [n] | [%] | [lang: %] | Recomputed if text exists |

| Table 9.2: Anomaly Cases Detected |
| :--- |
| **Anomaly Type** | **Count** | **Example / Description** |
| `char_count` = 0 but `comment_text` not empty | [n] | Encoding corruption |
| `word_count` = 0 but `char_count` > 0 | [n] | No whitespace (e.g., CJK) |
| `like_count` negative | [n] | Data corruption |
| `reply_count` > `like_count` by extreme margin (>100x) | [n] | Possible controversy or data error |
| `published_at` > `crawled_at` by > 5 years | [n] | Stale crawl or timestamp error |
| `comment_text` length > 10,000 chars (YouTube limit is 10K) | [n] | Possible concatenation error |
| Duplicate `comment_id` with different `comment_text` | [n] | API version drift |
| Same `author_id` with > 5 different `author_name` | [n] | Name changers or data inconsistency |

| Table 9.3: Embedding Completeness |
| :--- |
| **Embedding Type** | **Present** | **Missing** | **% Missing** | **Recomputed** | **Failed Recompute** |
| `embedding` | [n] | [n] | [%] | [n] | [n] |
| `embedding_char` | [n] | [n] | [%] | [n] | [n] |
| `embedding_word` | [n] | [n] | [%] | [n] | [n] |
| `embedding_ft` | [n] | [n] | [%] | [n] | [n] |

---

## Section 10: Cross-Tabulation Matrices
*Joint distributions that reveal structure.*

| Table 10.1: Language × Source Query (Top 10 × Top 10) |
| :--- |
| | **Query A** | **Query B** | **Query C** | ... | **Row Total** |
| **English** | [n] | [n] | [n] | | |
| **Spanish** | [n] | [n] | [n] | | |
| **Portuguese** | [n] | [n] | [n] | | |
| ... | | | | | |
| **Col Total** | | | | | **[N_final]** |

| Table 10.2: Language × Engagement Tier |
| :--- |
| | **Zero Likes** | **Micro** | **Low** | **Medium** | **High** | **Viral** |
| **English** | [n] | [n] | [n] | [n] | [n] | [n] |
| **Spanish** | [n] | [n] | [n] | [n] | [n] | [n] |
| ... | | | | | | |

| Table 10.3: Language × Temporal Period |
| :--- |
| | **Q1** | **Q2** | **Q3** | **Q4** | **Out of Range** |
| **English** | [n] | [n] | [n] | [n] | [n] |
| **Spanish** | [n] | [n] | [n] | [n] | [n] |
| ... | | | | | |

| Table 10.4: Video Popularity × Dominant Language |
| :--- |
| **Video Tier (by comment count)** | **Dominant Language** | **Count** | **% within Tier** |
| 1 comment | English | [n] | [%] |
| 1 comment | Spanish | [n] | [%] |
| 2-10 comments | English | [n] | [%] |
| 2-10 comments | Spanish | [n] | [%] |
| ... | ... | ... | ... |

---

## Section 11: Sentiment Label Distribution (If Available)
*Since you mentioned 3 labels exist but are not the focus, still baseline them.*

| Table 11.1: Sentiment Distribution (Overall) |
| :--- |
| **Label** | **Count** | **% of Labeled Data** |
| Positive | [n] | [%] |
| Neutral | [n] | [%] |
| Negative | [n] | [%] |
| **Unlabeled** | [n] | [%] |
| **Total** | [N_final] | 100% |

| Table 11.2: Sentiment by Language (Top 5 Languages) |
| :--- |
| **Language** | **Positive %** | **Neutral %** | **Negative %** | **N** |
| English | [%] | [%] | [%] | [n] |
| Spanish | [%] | [%] | [%] | [n] |
| ... | | | | |

| Table 11.3: Sentiment by Engagement Tier |
| :--- |
| **Engagement Tier** | **Positive %** | **Neutral %** | **Negative %** | **N** |
| Zero likes | [%] | [%] | [%] | [n] |
| Micro (1) | [%] | [%] | [%] | [n] |
| Low (2-10) | [%] | [%] | [%] | [n] |
| Medium (11-100) | [%] | [%] | [%] | [n] |
| High (101-1000) | [%] | [%] | [%] | [n] |
| Viral (>1000) | [%] | [%] | [%] | [n] |

---

## Section 12: Summary Statistics Dashboard (One-Page Reference)
*The final output: a single table containing all key baseline numbers.*

| Table 12.1: The Baseline Numbers |
| :--- |
| **Category** | **Metric** | **Value** |
| **Corpus** | Clean records | [N_final] |
| | Raw → Clean reduction | [%] |
| | Unique videos | [n] |
| | Unique authors | [n] |
| | Time span | [n days] |
| **Language** | Languages detected | [n] |
| | Top language | [lang] ([%]) |
| | Top 3 languages cover | [%] |
| | Code-switched comments | [%] |
| **Text** | Avg chars / comment | [mean] |
| | Avg words / comment | [mean] |
| | Comments with emoji | [%] |
| | Comments with hashtags | [%] |
| | Comments with mentions | [%] |
| **Engagement** | Comments with 0 likes | [%] |
| | Comments with 0 replies | [%] |
| | Mean likes (including zeros) | [mean] |
| | Mean replies (including zeros) | [mean] |
| | Top 1% likes threshold | [p99] |
| **Quality** | Records with imputed values | [n] ([%]) |
| | Records flagged anomalous | [n] ([%]) |
| | Embedding completeness | [%] |

---

**End of Phase I Planning Document**