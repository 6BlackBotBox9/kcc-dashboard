"""
retrieve_answer.py

Entity Locking & Two-Stage Search Logic for AgroRunner6.
Uses BM25 scoring with strict crop entity filtering to eliminate crop-mismatch errors.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

# Core agricultural entities
CROPS = {
    'wheat', 'paddy', 'rice', 'onion', 'sugarcane', 'cabbage',
    'cotton', 'arecanut', 'maize', 'potato', 'tomato', 'mustard',
    'chilli', 'banana', 'mango', 'groundnut', 'soyabean'
}

# Agricultural noise stop-words
NOISE_WORDS = {
    'asking', 'about', 'query', 'regarding', 'details', 'information',
    'today', 'sir', 'please', 'tell', 'want', 'kisan', 'problem',
    'give', 'solution', 'help', 'need', 'kindly'
}

# Cache loaded models in memory
_CACHED_DATA = None
_CACHED_BM25 = None

def load_retrieval_models():
    global _CACHED_DATA, _CACHED_BM25
    if _CACHED_DATA is None or _CACHED_BM25 is None:
        if os.path.exists('data/qa_data_cleaned.pkl'):
            _CACHED_DATA = joblib.load('data/qa_data_cleaned.pkl')
        elif os.path.exists('qa_data.pkl'):
            _CACHED_DATA = joblib.load('qa_data.pkl')
        else:
            raise FileNotFoundError('Cleaned QA dataset not found. Run train_qa_model.py first.')

        if os.path.exists('models/bm25_model.pkl'):
            _CACHED_BM25 = joblib.load('models/bm25_model.pkl')
        elif os.path.exists('bm25_model.pkl'):
            _CACHED_BM25 = joblib.load('bm25_model.pkl')
        else:
            raise FileNotFoundError('BM25 model not found. Run train_qa_model.py first.')

    return _CACHED_DATA, _CACHED_BM25

def preprocess_query(query: str):
    """Lowercase, strip non-alphabetic chars, and remove noise words."""
    if not isinstance(query, str):
        return '', []
    text = query.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [w for w in text.split() if w not in NOISE_WORDS]
    return ' '.join(tokens), tokens

def detect_crops(text_or_tokens):
    """Detect any crop entities present in the query."""
    if isinstance(text_or_tokens, str):
        tokens = set(text_or_tokens.lower().split())
    else:
        tokens = set(text_or_tokens)
    found = tokens.intersection(CROPS)
    return found

def get_best_answer(user_query: str, top_k: int = 3):
    """
    Retrieve best Q&A answers using BM25 and Entity Masking.
    
    Args:
        user_query: Farmer's query string.
        top_k: Number of top results to return.
        
    Returns:
        List of dicts: [
            {
                'question': str,
                'answer': str,
                'score': float,
                'confidence': float, # Normalized percentage (0.0 to 1.0)
                'matched_crop': str or None
            }, ...
        ]
    """
    df, bm25 = load_retrieval_models()
    cleaned_query, query_tokens = preprocess_query(user_query)
    
    if not query_tokens:
        return []

    # Detect crop in user query
    detected_crops = detect_crops(query_tokens)
    
    # Calculate raw BM25 scores
    scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)

    # Apply Entity Masking if a crop is detected
    if detected_crops:
        # Pre-extract cleaned tokens if not cached
        target_crop = next(iter(detected_crops))
        # Handle 'rice'/'paddy' equivalence
        equivalent_crops = {target_crop}
        if target_crop in {'rice', 'paddy'}:
            equivalent_crops = {'rice', 'paddy'}

        # Prepare masks
        # Rows containing target crop get 1.0 multiplier
        # Rows containing conflicting crop get 0.0 multiplier
        # Check against cleaned_questions
        cleaned_col = df['cleaned_questions'] if 'cleaned_questions' in df.columns else df['questions'].astype(str).str.lower()
        
        # Check presence of any target crop token in the row
        has_target_crop = cleaned_col.apply(lambda text: any(c in text.split() for c in equivalent_crops)).values
        
        # Check presence of other conflicting crops
        other_crops = CROPS - equivalent_crops
        has_conflicting_crop = cleaned_col.apply(lambda text: any(c in text.split() for c in other_crops)).values

        # Apply entity mask
        # If row has target crop -> maintain full score (or boost)
        # If row has conflicting crop without target crop -> set score to 0.0
        # If row has neither -> strongly penalize (0.05) since user explicitly specified a crop
        mask = np.zeros_like(scores)
        mask[has_target_crop] = 1.0
        mask[~has_target_crop & has_conflicting_crop] = 0.0
        mask[~has_target_crop & ~has_conflicting_crop] = 0.05

        scores = scores * mask

    # If all scores are 0, return empty
    max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
    if max_score <= 0:
        # Fallback to general BM25 if strict entity masking leaves nothing
        # But if entity was detected, we must never return a wrong crop
        if not detected_crops:
            scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
            max_score = float(np.max(scores)) if len(scores) > 0 else 0.0

    if max_score <= 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        row = df.iloc[idx]
        
        # Calculate confidence normalized against typical BM25 top score
        confidence = min(score / max(max_score, 1.0), 1.0)
        
        results.append({
            'question': row['questions'],
            'answer': row['answers'],
            'score': score,
            'confidence': confidence,
            'matched_crop': next(iter(detected_crops)) if detected_crops else None
        })

    return results
