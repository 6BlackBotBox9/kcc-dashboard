"""Step-by-step diagnosis of AgroRunner6 issue"""
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("DIAGNOSIS: Query 'Which crops require less water?'")
print("=" * 60)

# Step 1: Category distribution
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
category_names = model.classes_
df = pd.read_csv("sample_queries.csv")

print("\n1. CATEGORY DISTRIBUTION:")
print(df["category"].value_counts().to_string())
print()

# Step 2: Query prediction probabilities
query = "Which crops require less water?"
vec = vectorizer.transform([query])
pred = model.predict(vec)[0]
probs = model.predict_proba(vec)[0]

print("\n2. QUERY PREDICTION PROBABILITIES:")
print(f"   Query: {query}")
print(f"   Predicted category: {pred}")
sorted_idx = probs.argsort()[::-1]
for i in sorted_idx:
    cat = category_names[i]
    score = float(probs[i])
    marker = " <-- PREDICTED" if cat == pred else ""
    print(f"   {cat:20s}: {score:.4f}{marker}")

# Lowest probability category is the predicted one - very concerning
print(f"\n   Note: Predicted has only {probs[sorted_idx[0]]:.4f} probability")
print(f"   All probabilities range from {probs.min():.4f} to {probs.max():.4f}")
print()

# Step 3: Q&A retrieval top 5 (no filtering)
qa_vectorizer = joblib.load("qa_vectorizer.pkl")
qa_matrix = joblib.load("qa_matrix.pkl")
qa_data = joblib.load("qa_data.pkl")
user_vec = qa_vectorizer.transform([query])
sims = cosine_similarity(user_vec, qa_matrix).flatten()
top_indices = sims.argsort()[::-1][:5]

print("\n3. Q&A RETRIEVAL TOP 5 (no category filtering):")
for i in top_indices:
    sim = float(sims[i])
    row = qa_data.iloc[i]
    q_txt = row["questions"][:60]
    a_txt = str(row["answers"])[:60]
    print(f"   sim={sim:.4f} | Q: {q_txt}...")
    print(f"           A: {a_txt}...")

print()

# Step 4: Scheme keywords check
scheme_kw = ["scheme", "yojana", "subsidy", "pmfby"]
hits = [kw for kw in scheme_kw if kw in query.lower()]

print("\n4. SCHEME KEYWORD CHECK:")
print(f"   Query: {query}")
print(f"   Keywords searched: {scheme_kw}")
print(f"   Keywords found in query: {hits}")
print()

# Step 5: Government Scheme training examples
gs_examples = df[df["category"] == "Government Scheme"]["query_text"].values
print("\n5. GOVERNMENT SCHEME TRAINING EXAMPLES:")
print(f"   Count: {len(gs_examples)}")
for i, ex in enumerate(gs_examples):
    print(f"   {i+1}. {ex}")
print()

# Step 6: TF-IDF settings
print("\n6. TF-IDF VECTORIZER SETTINGS:")
print(f"   ngram_range: {vectorizer.ngram_range}")
print(f"   max_features: {vectorizer.max_features}")
print(f"   stop_words: {vectorizer.stop_words}")
feature_names = vectorizer.get_feature_names_out()
for term in ["crops", "water", "less", "irrigation", "crop"]:
    if term in vectorizer.vocabulary_:
        idx = vectorizer.vocabulary_[term]
        print(f"   '{term}' in vocabulary at index {idx}")
    else:
        print(f"   '{term}' NOT in vocabulary")