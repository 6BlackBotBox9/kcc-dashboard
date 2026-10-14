"""
train_qa_model.py

Trains Q&A retrieval models on the KCC Q&A dataset:
1. TF-IDF vectorizer + matrix (for backward compatibility)
2. Cleaned corpus with noise-word filtering
3. BM25 model for high-precision entity and lexical retrieval

Saves:
  - qa_vectorizer.pkl
  - qa_matrix.pkl
  - qa_data.pkl
  - data/qa_data_cleaned.pkl
  - models/bm25_model.pkl
"""

import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print('---- 1. Loading data ----')
csv_path = 'datasetsQueries/Question-Answer/questionsv4kaggle.csv'
df = pd.read_csv(csv_path)
print(f'Loaded {len(df)} Q&A pairs from {csv_path}')

# ---- 2. Define custom agricultural noise stop-words ----
NOISE_WORDS = {
    'asking', 'about', 'query', 'regarding', 'details', 'information',
    'today', 'sir', 'please', 'tell', 'want', 'kisan', 'problem',
    'give', 'solution', 'help', 'need', 'kindly'
}

def clean_question(text):
    if not isinstance(text, str):
        return ''
    # Lowercase
    text = text.lower()
    # Remove numbers and special characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove noise words and extra whitespace
    words = [w for w in text.split() if w not in NOISE_WORDS]
    return ' '.join(words)

print('---- 2. Cleaning questions & building tokenized corpus ----')
df['cleaned_questions'] = df['questions'].apply(clean_question)
tokenized_corpus = [q.split() for q in df['cleaned_questions']]

print('---- 3. Training BM25 model ----')
bm25_model = BM25Okapi(tokenized_corpus)

print('---- 4. Training TF-IDF vectorizer on questions ----')
vectorizer = TfidfVectorizer(stop_words='english')
qa_matrix = vectorizer.fit_transform(df['questions'])

print('---- 5. Saving model artifacts ----')
joblib.dump(vectorizer, 'qa_vectorizer.pkl')
joblib.dump(qa_matrix, 'qa_matrix.pkl')
joblib.dump(df, 'qa_data.pkl')
joblib.dump(df, 'data/qa_data_cleaned.pkl')
joblib.dump(bm25_model, 'models/bm25_model.pkl')

print('Successfully saved:')
print('  - qa_vectorizer.pkl')
print('  - qa_matrix.pkl')
print('  - qa_data.pkl')
print('  - data/qa_data_cleaned.pkl')
print('  - models/bm25_model.pkl')
