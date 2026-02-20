import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optimization: Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentiment(text):
    """
    Simple heuristic-based sentiment analysis. 
    For better results, consider using 'textblob' or 'vaderSentiment'.
    """
    text = str(text).lower()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'helpful', 'satisfied', 'perfect', 'yes']
    negative_words = ['bad', 'poor', 'unhelpful', 'slow', 'boring', 'confusing', 'no', 'disappointed']
    
    pos_score = sum(1 for word in positive_words if word in text)
    neg_score = sum(1 for word in negative_words if word in text)
    
    if pos_score > neg_score: return "Positive"
    if neg_score > pos_score: return "Negative"
    return "Neutral"

def summarize_feedback(series):
    ignore_list = ['na', 'n/a', 'no', 'nothing', 'none', 'nil', '.', 'good', 'ok', 'nice', 'yes']
    raw_entries = series.dropna().astype(str).tolist()
    all_points = []
    
    for entry in raw_entries:
        parts = [p.strip() for p in re.split(r'\n|\||•', entry)]
        all_points.extend([p for p in parts if len(p.split()) >= 2])

    valid_points = [p for p in all_points if p.lower() not in ignore_list]
    
    if not valid_points:
        return ["No detailed suggestions provided."]

    embeddings = model.encode(valid_points)
    final_summary = []
    summary_embeddings = []

    for i, point in enumerate(valid_points):
        is_duplicate = False
        if final_summary:
            sim_matrix = cosine_similarity(embeddings[i].reshape(1, -1), np.vstack(summary_embeddings))
            if np.max(sim_matrix) > 0.92: 
                is_duplicate = True
        
        if not is_duplicate:
            final_summary.append(point)
            summary_embeddings.append(embeddings[i])

    return final_summary[:10]