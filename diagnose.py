"""Diagnose the AgroRunner6 pipeline issue"""
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
category_names = model.classes_

print("=" * 60)
print("CATEGORY DISTRIBUTION in sample_queries.csv")
print("=" * 60)
df = pd.read_csv("sample_queries.csv")
print(df["category"].value_counts())
print()

print("=" * 60)
print("QUERY: 'Which crops require less water?'")
print("=" * 60)

# Transform and predict
vec = vectorizer.transform(["Which crops require less water?"])
pred = model.predict(vec)[0]
probs = model.predict_proba(vec)[0]

print(f"Predicted category: {pred}")
print()

print("=" * 60)
print("PROBABILITIES for ALL CATEGORIES")
print("=" * 60)
sorted_idx = probs.argsort()[::-1]
for i in sorted_idx:
    cat = category_names[i]
    score = float(probs[i])
    marker = " <-- predicted" if cat == pred else ""
    print(f"  {cat}: {score:.4f}{marker}")
print()

# Confusion matrix on test split
X_train, X_test, y_train, y_test = train_test_split(
    df["query_text"], df["category"],
    test_size=0.25, random_state=42, stratify=df["category"]
)
X_test_vec = vectorizer.transform(X_test)
preds = model.predict(X_test_vec)
cm = confusion_matrix(y_test, preds, labels=category_names)
print("=" * 60)
print("CONFUSION MATRIX (on 60-sample test split)")
print("=" * 60)
print("      " + "  ".join([f"{c[:10]}" for c in category_names]))
for i, c in enumerate(category_names):
    row = cm[i]
    print(f"{c[:10]:10} " + "  ".join([f"{row[j]:5d}" for j in range(len(category_names))]))
print()

# Per-class report
print("=" * 60)
print("CLASSIFICATION REPORT (per-class on test split)")
print("=" * 60)
print(classification_report(y_test, preds, labels=category_names))

# Now check the query in the Q&A dataset
print("\n" + "=" * 60)
print("STEP 3: Check Q&A retrieval for this query")
print("=" * 60)

from sklearn.metrics.pairwise import cosine_similarity

qa_vectorizer = joblib.load("qa_vectorizer.pkl")
qa_matrix = joblib.load("qa_matrix.pkl")
qa_data = joblib.load("qa_data.pkl")

# Test Q&A retrieval WITHOUT category filtering
user_vec = qa_vectorizer.transform(["Which crops require less water?"])
sims = cosine_similarity(user_vec, qa_matrix).flatten()
top_indices = sims.argsort()[::-1][:5]

print("Top 5 Q&A matches (no category filtering):")
for i in top_indices:
    sim = float(sims[i])
    row = qa_data.iloc[i]
    print(f"  Similarity: {sim:.4f} | Question: {row['questions'][:80]}...")
    print(f"  Answer: {str(row['answers'])[:80]}...")
    print()

# Now check with category filtering (Government Scheme)
print("=" * 60)
print("STEP 4: Check retrieval WITH category filtering")
print("=" * 60)

# Check if the query matches Government Scheme in Q&A
# The Q&A dataset may not have a category column, so let's just check all
print("All Q&A matches sorted by similarity:")
for i in range(len(qa_data)):
    sim = float(cosine_similarity(user_vec, qa_matrix)[0, i])
    if sim > 0.1:  # only show significant matches
        row = qa_data.iloc[i]
        print(f"  sim={sim:.4f}: Q: {row['questions'][:60]}...")

# Check Government Scheme keywords in the query
print("\n" + "=" * 60)
print("STEP 5: Check if query triggers scheme keywords")
print("=" * 60)
scheme_keywords = ["scheme", "yojana", "subsidy", "pmfby"]
query_lower = "which crops require less water".lower()
hits = [kw for kw in scheme_keywords if kw in query_lower]
print(f"Scheme keywords found in query: {hits}")
print(f"Query lowercased: {query_lower}")

# Check what "Government Scheme" training queries look like
print("\n" + "=" * 60)
print("STEP 6: Government Scheme training examples")
print("=" * 60)
gs_examples = df[df["category"] == "Government Scheme"]["query_text"].values
print(f"Number of Gov Scheme examples: {len(gs_examples)}")
for i, ex in enumerate(gs_examples[:5]):
    print(f"  {i+1}. {ex}")

# Check TF-IDF settings
print("\n" + "=" * 60)
print("STEP 7: TF-IDF vectorizer settings")
print("=" * 60)
print(f"ngram_range: {vectorizer.ngram_range}")
print(f"max_features: {vectorizer.max_features}")
print(f"stop_words: {vectorizer.stop_words}")
print(f"vocab size: {len(vectorizer.vocabulary_)}")

# Check key terms
feature_names = vectorizer.get_feature_names_out()
print("\nTop features for 'crops' check:")
for term in ["crops", "water", "less", "irrigation", "crop"]:
    if term in vectorizer.vocabulary_:
        idx = vectorizer.vocabulary_[term]
        print(f"  '{term}' -> index {idx}")
    else:
        print(f"  '{term}' -> NOT in vocabulary")