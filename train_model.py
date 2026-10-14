"""
train_model.py
Trains a text classifier that sorts farmer queries (Kisan Call Centre style)
into categories like Pest & Disease, Weather, Fertilizer & Nutrient, etc.

HOW TO USE
----------
1. First run: this trains on the small SAMPLE dataset (sample_queries.csv)
   just so you can see the whole pipeline work end-to-end immediately.
2. Once you download the REAL KCC dataset (data.gov.in or the Kaggle mirror),
   replace sample_queries.csv with your real file. It just needs two
   columns named exactly: query_text, category
3. Re-run this script (`python train_model.py`) any time your data changes.
   It overwrites model.pkl and vectorizer.pkl, which app.py loads to power
   the dashboard. You do NOT need to touch app.py when you retrain.
"""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---- 1. Load data ----
df = pd.read_csv("sample_queries.csv")
print(f"Loaded {len(df)} rows across {df['category'].nunique()} categories")
print(f"Category distribution:\n{df['category'].value_counts()}\n")

# ---- 2. Convert text into numbers (TF-IDF) ----
# Improved TF-IDF: use bigrams + unigrams to capture phrases like
# "yellow rust" or "market price", and cap features at 5000 for small dataset.
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    min_df=1,
    max_df=0.95,
)
X_all_vec = vectorizer.fit_transform(df["query_text"])

# ---- 3. Train on ALL data (not just train/test split) ----
# For a small 60-row dataset, training on all data gives the model
# maximum signal. We use cross-validation to evaluate instead.
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=1.0,
    solver="lbfgs",
)
model.fit(X_all_vec, df["category"])

# ---- 4. Cross-validation to estimate real accuracy ----
# 5-fold CV gives a better estimate of how the model performs on unseen data
# than a single 75/25 split, especially with small datasets.
cv_scores = cross_val_score(model, X_all_vec, df["category"], cv=5, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
print(f"Fold scores: {[f'{s:.2f}' for s in cv_scores]}\n")

# ---- 5. Per-category report on full data (sanity check) ----
preds = model.predict(X_all_vec)
print("Full dataset classification report (training sanity check):")
print(classification_report(df["category"], preds))

# ---- 6. Save the trained model + vectorizer for the dashboard to use ----
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved model.pkl and vectorizer.pkl — app.py loads these two files.")
