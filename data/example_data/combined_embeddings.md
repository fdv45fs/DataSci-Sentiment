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

# combined_labeled.parquet
| column_name | column_type | null | key  | default | extra |
|-------------|-------------|------|------|---------|-------|
| comment_id  | VARCHAR     | YES  | NULL | NULL    | NULL  |
| label       | VARCHAR     | YES  | NULL | NULL    | NULL  |
| source_file | VARCHAR     | YES  | NULL | NULL    | NULL  |
| source_row  | BIGINT      | YES  | NULL | NULL    | NULL  |
| post_id     | VARCHAR     | YES  | NULL | NULL    | NULL  |
