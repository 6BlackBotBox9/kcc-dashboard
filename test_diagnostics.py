# test_diagnostics.py
"""
Standalone diagnostic script for AgroRunner6 Q&A retrieval pipeline.
Evaluates accuracy, similarity/retrieval metrics, and edge-case queries using:
1. Two-Stage BM25 Retrieval with Entity Masking (retrieve_answer.py)
2. Fallback to TF-IDF baseline
"""

import os
import random
import re
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from retrieve_answer import get_best_answer, preprocess_query, load_retrieval_models

# Agricultural terms synonym map
SYNONYMS = {
    'wheat': 'gehun',
    'gehun': 'wheat',
    'subsidy': 'scheme',
    'scheme': 'subsidy',
    'paddy': 'rice',
    'rice': 'paddy'
}

def rephrase_question(question: str) -> str:
    """Apply a realistic transformation: lowercase, drop filler/word, or swap synonym."""
    if not isinstance(question, str):
        return ''
    
    choice = random.choice(['lowercase', 'drop_word', 'swap_synonym'])
    
    if choice == 'lowercase':
        return question.lower()
    
    elif choice == 'drop_word':
        words = question.split()
        long_words = [w for w in words if len(w) > 3]
        if not long_words:
            return question.lower()
        word_to_drop = random.choice(long_words)
        pattern = re.compile(r'\b' + re.escape(word_to_drop) + r'\b', flags=re.IGNORECASE)
        return pattern.sub('', question).strip()
        
    elif choice == 'swap_synonym':
        res = question
        for src, dest in SYNONYMS.items():
            if re.search(r'\b' + src + r'\b', res, re.IGNORECASE):
                res = re.sub(r'\b' + src + r'\b', dest, res, flags=re.IGNORECASE)
                break
        return res

def main():
    random.seed(42)
    print('Loading dataset and models for diagnostic benchmark...')
    df, bm25 = load_retrieval_models()
    
    total_rows = len(df)
    # Check BM25 vocabulary size
    vocab_size = len(bm25.idf) if hasattr(bm25, 'idf') else len(bm25.doc_freqs)

    # -------------------------------------------------------------------
    # Automated Benchmark Test (100 Random Samples)
    # -------------------------------------------------------------------
    sample_df = df.sample(n=min(100, total_rows), random_state=42)
    scores = []
    low_confidence_count = 0
    total_mismatch_count = 0

    for original_q in sample_df['questions'].tolist():
        rephrased = rephrase_question(original_q)
        results = get_best_answer(rephrased, top_k=1)
        if results:
            top_res = results[0]
            score = top_res['confidence']
            scores.append(score)
            if score < 0.25:
                low_confidence_count += 1
        else:
            scores.append(0.0)
            total_mismatch_count += 1
            low_confidence_count += 1

    avg_score = float(np.mean(scores)) if scores else 0.0
    low_conf_pct = 100.0 * low_confidence_count / len(scores) if scores else 0.0
    mismatch_pct = 100.0 * total_mismatch_count / len(scores) if scores else 0.0

    # -------------------------------------------------------------------
    # Edge-Case Manual Test Suite (5 Real-World Queries)
    # -------------------------------------------------------------------
    edge_queries = [
        'wheat seed subsidy details',
        'yellow spots on paddy leaves',
        'pmfby claim status not showing',
        'fertiliser application rate for sugarcane',
        'today price of onion in mandi',
    ]

    edge_results = []
    for q in edge_queries:
        res = get_best_answer(q, top_k=1)
        if res:
            retrieved_q = str(res[0]['question']).strip().replace('\n', ' ')[:45]
            score_str = f"{res[0]['score']:.2f} (conf {res[0]['confidence']*100:.0f}%)"
            edge_results.append((q, retrieved_q, score_str))
        else:
            edge_results.append((q, 'No match found', '0.00 (conf 0%)'))

    # -------------------------------------------------------------------
    # Terminal Report Output
    # -------------------------------------------------------------------
    print('=' * 65)
    print('AGRORUNNER6 Q&A RETRIEVAL DIAGNOSTIC REPORT')
    print('=' * 65)
    print(f'Total dataset rows loaded   : {total_rows}')
    print(f'BM25 index vocabulary size  : {vocab_size}')
    print()
    print('Benchmark Metrics (100 random re-phrased queries):')
    print(f'  Average Top-1 Confidence   : {avg_score:.4f}')
    print(f'  Low-Confidence (<0.25) Rate : {low_conf_pct:.2f}%')
    print(f'  Total Mismatch (0.0) Rate  : {mismatch_pct:.2f}%')
    print()
    print('Edge-Case Manual Test Suite Results:')
    print(f"{'Query':42} | {'Top-1 Retrieved Match':45} | Score")
    print('-' * 115)
    for qry, ret_q, sc in edge_results:
        print(f"{qry:42} | {ret_q:45} | {sc}")
    print('=' * 65)

if __name__ == '__main__':
    main()
