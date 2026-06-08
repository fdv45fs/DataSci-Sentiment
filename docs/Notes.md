# Teacher Nguyen Duc Anh's opinion
Cần làm:

    Thống kê tương quan emoji theo nhãn
    Giải thích tác dụng của like, share
    So sánh các gram (1, 12, 123) để chọn feature tốt
    Dùng Roberta và DistilBERT
    Thêm emoji type vào emoji count
    Thống kê cụ thể về gram để tránh scale up sai

# Notes:
1. Extensive cleaning and filtering to get the final number of data after matching the two files. Deduplicate is strongly encouraged.
2. The EDA part should extensively focus on: emoji, multilingual, like & share meaning, label distribution, n-gram distribution, source query. We should focus on other values such as timestamps (for example during COVID-19, likes and share worth more).
3. The EDA and feature engineering should be a reflection of labelled comments, with the 3 labels: positive, neutral & negative.
4. We should focus on great statistics. Feature engineering also needs to integrate like, share, emojis (not their count), etc... More and more efforts should be put in extensive EDA and curated a great set of features.
5. For models, use Cross_Validation = 5, use BERT and also find ways to integrate the features correctly. Also source query is very important. All of these aboves, word, source queries, emojis, like count, etc... all contributed to our model. Integrate SHAP, optuna, etc.. and even do further optimization to return the best results. We also try to solve the problem of inequality in labelling, maybe using data augmentation (SMOT, ...). We may also detect multicolinearity, and find some fixes to it. 

# Goal:
1. Firstly, I will provide you an already good EDA Python code on this data. But it is missing quite a few important things: the first phase of EDA does not point towards the labelled data, which is undirected on the goal of classification with 3 labels: positive, neutral & negative. But that is okay, without problems.
2. And also, up above, you already generated a bunch of guideline towards implementation of EDA and descriptive statistics. It is really enormous, but we should now nagivate it down to fewer and more centralized set of analysis. 
3. In this phase, you should still generate me the list of implementations (no Python code, just list of what to be done). Integrate both descriptive stats (this time make smaller table with col_name & description only, or another form of table to save tokens) and deep EDA analysis (emoji, multilingual, like & share meaning, label distribution, n-gram distribution, source query). This time, also cover the path from EDA to feature engineering, and also EDA the created features.

# Already-good-EDA code:
``` py

```