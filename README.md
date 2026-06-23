# Zomato Restaurant Review Sentiment Analysis

EDA + NLP sentiment classification on Zomato restaurant reviews (Hyderabad).

## Problem Statement

Predict review sentiment (Positive / Neutral / Negative) from review text, and uncover patterns in rating, cost, cuisine, and review timing that drive customer satisfaction.

## Dataset

- `Zomato Restaurant reviews.csv` — 10,000 reviews, 100 restaurants.
- `Zomato Restaurant names and Metadata.csv` — cost, cuisines, collections, timings for 105 restaurants.

Both files are placed in `data/` folder.

## Approach

1. Data cleaning & merging (rating fix, metadata parsing, cost cleanup).
2. EDA — 15 charts following Univariate / Bivariate / Multivariate rule.
3. Hypothesis testing — ANOVA, t-test, Pearson correlation.
4. NLP preprocessing — contraction expansion, cleaning, stopword removal, lemmatization.
5. TF-IDF vectorization.
6. Models — Logistic Regression, Multinomial Naive Bayes, Random Forest, each tuned with GridSearchCV / RandomizedSearchCV.
7. Model explainability via Logistic Regression coefficients.
8. Final model saved with `joblib` for deployment.

## Results

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Logistic Regression (Tuned) | ~0.77 | ~0.79 |
| Naive Bayes (Tuned) | ~0.80 | ~0.77 |
| Random Forest (Tuned) | ~0.80 | ~0.77 |

Final model: **Tuned Logistic Regression** — best F1-weighted score, fast, interpretable.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Zomato_Sentiment_Analysis_Project.ipynb
```

Run all cells top to bottom. No manual steps needed.

## Repo Structure

```
├── Zomato_Sentiment_Analysis_Project.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── Zomato Restaurant reviews.csv
│   └── Zomato Restaurant names and Metadata.csv
└── model_artifacts/
    ├── label_encoder.joblib
    ├── sentiment_model.joblib
    └── tfidf_vectorizer.joblib
```

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, NLTK, Scikit-learn, WordCloud, Joblib.
