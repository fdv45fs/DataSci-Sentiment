# Teacher Nguyen Duc Anh's opinion
Cần làm:

    Thống kê tương quan emoji theo nhãn
    Giải thích tác dụng của like, share
    So sánh các gram (1, 12, 123) để chọn feature tốt
    Dùng Roberta và DistilBERT
    Thêm emoji type vào emoji count
    Thống kê cụ thể về gram để tránh scale up sai

# Data Cleaning Notes:
1. Extensive cleaning and filtering to get the final number of data after matching the two files. Deduplicate is strongly encouraged.
2. The EDA part should extensively focus on: emoji, multilingual, like & share meaning, label distribution, n-gram distribution, source query. We should focus on other values such as timestamps (for example during COVID-19, likes and share worth more).
3. The EDA and feature engineering should be a reflection of labelled comments, with the 3 labels: positive, neutral & negative.
4. We should focus on great statistics. Feature engineering also needs to integrate like, share, emojis (not their count), etc... More and more efforts should be put in extensive EDA and curated a great set of features.
5. For models, use Cross_Validation = 5, use BERT and also find ways to integrate the features correctly. Also source query is very important. All of these aboves, word, source queries, emojis, like count, etc... all contributed to our model. Integrate SHAP, optuna, etc.. and even do further optimization to return the best results. We also try to solve the problem of inequality in labelling, maybe using data augmentation (SMOT, ...). We may also detect multicolinearity, and find some fixes to it. 