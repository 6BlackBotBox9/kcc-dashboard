"""
train_scheme_model.py

Trains a TF-IDF retrieval model on the KCC Schemes dataset
so that AgroRunner6 can instantly find the most relevant scheme
via cosine similarity (no re-transformation needed at runtime).

Saves:
  - scheme_vectorizer.pkl
  - scheme_matrix.pkl       (pre-computed TF-IDF for all schemes)
  - scheme_data.pkl         (the raw dataframe, for reference)
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ---- 1. Load data ----
df = pd.read_csv("datasetsQueries/datasetsAnalysis/schemes.csv")
print(f"Loaded {len(df)} scheme rows from schemes.csv")

# ---- 2. Build combined text column for TF-IDF ----
# We combine name, short title, tags, and brief description for best matching
df["combined_text"] = (
    df["scheme_name"].fillna("") + " " +
    df["short_title"].fillna("") + " " +
    df["tags"].fillna("") + " " +
    df["brief_description"].fillna("")
)

# ---- 3. Train TF-IDF vectorizer on combined text ----
vectorizer = TfidfVectorizer(stop_words="english")
scheme_matrix = vectorizer.fit_transform(df["combined_text"])

# ---- 4. Save vectorizer + pre-computed matrix ----
joblib.dump(vectorizer, "scheme_vectorizer.pkl")
joblib.dump(scheme_matrix, "scheme_matrix.pkl")
joblib.dump(df, "scheme_data.pkl")

print("Saved scheme_vectorizer.pkl, scheme_matrix.pkl, and scheme_data.pkl")
print("Runtime will use cosine_similarity(user_vec, scheme_matrix) for instant lookup")