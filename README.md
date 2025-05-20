# 📰 Fake News Detector

Detect fake and real news articles using machine learning and natural language processing (NLP).

![accuracy-badge](https://img.shields.io/badge/accuracy-98.9%25-brightgreen)  
![python-badge](https://img.shields.io/badge/Python-3.10-blue) ![sklearn-badge](https://img.shields.io/badge/scikit--learn-1.4.2-orange)

---

## 🚀 Project Overview

This project classifies news articles as **Fake** or **Real** using text-based features. It uses TF-IDF vectorization and Logistic Regression to train a lightweight yet high-performing model.

> Achieved **98.9% accuracy** on a Kaggle dataset of over 44,000 news articles.

---

## 📁 Dataset

[Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- `Fake.csv`: 23,481 fake news articles  
- `True.csv`: 21,417 real news articles

---

## ⚙️ Tech Stack

- Python (Pandas, NumPy)
- Scikit-learn (TF-IDF, Logistic Regression, Evaluation)
- NLTK (Text preprocessing)
- Google Colab (Development Environment)

---

## 🔬 Workflow

1. **Data Loading**  
   Load and label fake (0) and real (1) news data.

2. **Preprocessing**  
   - Lowercasing  
   - Removing punctuation and stopwords  
   - Cleaned text stored as `clean_text`

3. **Vectorization**  
   - TF-IDF vectorizer with 5,000 features

4. **Model Training**  
   - Logistic Regression with 80/20 train-test split

5. **Evaluation**  
   - Accuracy: **98.9%**  
   - F1-Score: 0.99  
   - Confusion Matrix:
     ```
     [[4672   59]
      [  36 4213]]
     ```

---

## 📊 Sample Output

| Metric     | Value   |
|------------|---------|
| Accuracy   | 98.94%  |
| Precision  | 99%     |
| Recall     | 99%     |
| F1-Score   | 99%     |

---

## 🛠️ Future Improvements

- [ ] Add GridSearchCV for hyperparameter tuning  
- [ ] Try alternative models (Naive Bayes, SVM)  
- [ ] Visualize feature importance and confusion matrix  
- [ ] Build an interactive [Streamlit](https://streamlit.io/) demo  
- [ ] Deploy via Flask + Docker

---

## 📎 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Pranay Paka**  
[GitHub](https://github.com/Pranaypaka) | 

---

