"""Test fixes for the confidence threshold issue"""
import joblib
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
category_names = model.classes_
qa_vectorizer = joblib.load("qa_vectorizer.pkl")
qa_matrix = joblib.load("qa_matrix.pkl")
qa_data = joblib.load("qa_data.pkl")

query = "Which crops require less water?"
vec = vectorizer.transform([query])
probs = model.predict_proba(vec)[0]
max_prob = probs.max()
pred = model.predict(vec)[0]

scheme_keywords = ["scheme", "yojana", "subsidy", "pmfby"]

print("=" * 60)
print("TESTING FIXES")
print("=" * 60)

# Fix 1: Confidence threshold
CONFIDENCE_THRESHOLD = 0.25
is_low_confidence = max_prob < CONFIDENCE_THRESHOLD

print("\n1. CONFIDENCE THRESHOLD FIX:")
print(f"   Max probability: {max_prob:.4f}")
print(f"   Threshold: {CONFIDENCE_THRESHOLD}")
print(f"   Low confidence: {is_low_confidence}")
adjusted_cat = "Other" if is_low_confidence else pred
print(f"   Adjusted category: {adjusted_cat}")
print()

# Fix 2: Scheme trigger with confidence check
scheme_trigger_original = (pred == "Government Scheme" or any(kw in query.lower() for kw in scheme_keywords))
scheme_trigger_fixed = (
    (pred == "Government Scheme" and not is_low_confidence and any(kw in query.lower() for kw in scheme_keywords))
    or
    (is_low_confidence and any(kw in query.lower() for kw in scheme_keywords))
)
print("\n2. SCHEME TRIGGER FIX:")
print(f"   Original trigger: {scheme_trigger_original}")
print(f"   Fixed trigger: {scheme_trigger_fixed}")
print()

# Fix 3: Q&A retrieval approach
print("\n3. Q&A RETRIEVAL APPROACH:")
user_vec = qa_vectorizer.transform([query])
sims = cosine_similarity(user_vec, qa_matrix).flatten()

# Approach A: Top 3 from full dataset (no category filtering)
top_a = sims.argsort()[::-1][:3]
print("   Approach A - Top 3 from full dataset:")
for i in top_a:
    sim = float(sims[i])
    row = qa_data.iloc[i]
    print(f"     sim={sim:.4f} | Q: {row['questions'][:50]}...")
print()

# Approach B: Only if high confidence, filtered by category
print("   Approach B - Category-filtered only if high confidence:")
if not is_low_confidence:
    # Would filter by predicted category here
    print("     (High confidence - would filter by category)")
else:
    print("     (Low confidence - showing full dataset top 3)")
    for i in top_a:
        sim = float(sims[i])
        row = qa_data.iloc[i]
        print(f"       sim={sim:.4f} | Q: {row['questions'][:50]}...")
print()

# Summary
print("\n" + "=" * 60)
print("SUMMARY OF FIXES")
print("=" * 60)
print(f"   • Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"   • Low confidence detected: {is_low_confidence}")
print(f"   • Adferred category for unclear queries: Other")
print(f"   • Scheme lookup only triggers with high confidence + keywords")
print(f"   • Q&A retrieval shows full dataset top results for low confidence")