import pandas as pd
import numpy as np
import os
import random
from rank_bm25 import BM25Okapi

data_path = r"C:\Users\Priyaranjan\Downloads\kcc_dashboard\kcc_dashboard\datasetsQueries\Question-Answer\questionsv4kaggle.csv"

if not os.path.exists(data_path):
    print(f"Error: Could not find file at {data_path}")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(data_path).dropna(subset=['questions', 'answers'])

print("Building BM25 index...")
NOISE_WORDS = {'asking', 'about', 'query', 'regarding', 'details', 'information', 'today', 'sir', 'please', 'tell', 'want', 'kisan', 'problem', 'give', 'solution', 'help', 'need', 'kindly'}

def clean_text(text):
    words = str(text).lower().split()
    return [w for w in words if w not in NOISE_WORDS and w.isalnum()]

corpus = [clean_text(q) for q in df['questions']]
bm25 = BM25Okapi(corpus)

# Increased sample size for a more stable metric
sample_size = min(1000, len(df))
print(f"Evaluating Semantic Top-1 Retrieval Accuracy on {sample_size} sample queries...")

top1_hits = 0
scores = []

sample_indices = random.sample(range(len(df)), sample_size)

for idx in sample_indices:
    original_query = df.iloc[idx]['questions']
    expected_answer = str(df.iloc[idx]['answers']).strip().lower()
    query_tokens = clean_text(original_query)
    
    if not query_tokens:
        continue
        
    doc_scores = bm25.get_scores(query_tokens)
    top_match_idx = np.argmax(doc_scores)
    retrieved_answer = str(df.iloc[top_match_idx]['answers']).strip().lower()
    
    scores.append(doc_scores[top_match_idx])
    
    # HIT LOGIC: It's a hit if it finds the exact row OR the exact same answer text
    if top_match_idx == idx or expected_answer == retrieved_answer:
        top1_hits += 1

hit1_accuracy = (top1_hits / sample_size) * 100
avg_score = np.mean(scores)

print("\n" + "="*60)
print("       AGRORUNNER6 BM25 SEMANTIC EVALUATION REPORT           ")
print("="*60)
print(f"Total Evaluated Samples : {sample_size}")
print(f"True Hit Rate (Hit@1)   : {hit1_accuracy:.2f}%")
print(f"Average Top Match Score : {avg_score:.2f}")
print("="*60 + "\n")