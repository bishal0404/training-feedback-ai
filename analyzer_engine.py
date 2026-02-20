import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
# Import model from nlp_processor to avoid double-loading in memory
from nlp_processor import model 

def classify_columns(df):
    rating_cols = []
    text_cols = []
    ignored_cols = []
    
    # Precise keywords for metadata
    metadata_keywords = ['name', 'id', 'email', 'phone', 'contact', 'mobile', 'date', 'timestamp', 'designation']
    
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        
        # 1. Force Rating if starts with "Number." (e.g., "1. How was...")
        if re.match(r'^\d+\.', col_str):
            numeric_check = pd.to_numeric(df[col], errors='coerce').dropna()
            if not numeric_check.empty and numeric_check.max() <= 100:
                rating_cols.append(col)
                continue

        # 2. Metadata check - FIXED: Use word boundaries so "provided" isn't flagged as "id"
        is_metadata = False
        for key in metadata_keywords:
            if re.search(rf'\b{key}\b', col_lower):
                is_metadata = True
                break
        
        if is_metadata:
            ignored_cols.append(col)
            continue
            
        # 3. Standard Rating check (Numeric, small range of values)
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 15 and df[col].max() <= 100: 
                rating_cols.append(col)
            else:
                ignored_cols.append(col)
        
        # 4. Text Feedback check (Non-numeric, longer average word count)
        elif pd.api.types.is_object_dtype(df[col]):
            valid_rows = df[col].dropna()
            if not valid_rows.empty:
                avg_len = valid_rows.astype(str).str.split().str.len().mean()
                if avg_len > 2:
                    text_cols.append(col)
                else:
                    ignored_cols.append(col)
            else:
                ignored_cols.append(col)
                
    return rating_cols, text_cols, ignored_cols

def compute_stats(df, columns):
    stats = {}
    for col in columns:
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        data = data[(data >= 1) & (data <= 5)]
        if len(data) == 0: continue
            
        counts = data.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).to_dict()
        stats[col] = {
            "mean": round(float(np.mean(data)), 2), 
            "std": round(float(np.std(data)), 2),
            "counts": counts
        }
    return stats

def group_into_sections(questions):
    if not questions: return {}
    themes = {
        "Trainer Rating": "instructor knowledge, delivery, teaching style, quality of delivery",
        "Training Effectiveness": "expectations, apply knowledge, objectives, content organized, materials",
        "Infrastructure": "room, infrastructure, time provided, allotment, sufficient time",
        "Engagement": "interaction, participation, encouragement, questions and discussion",
        "Overall Satisfaction": "overall rating, recommend, value"
    }
    theme_names = list(themes.keys())
    q_embeddings = model.encode(questions)
    t_embeddings = model.encode(list(themes.values()))
    similarity = cosine_similarity(q_embeddings, t_embeddings)
    groups = {name: [] for name in theme_names}
    for i, q in enumerate(questions):
        best_theme_idx = np.argmax(similarity[i])
        groups[theme_names[best_theme_idx]].append(q)
    return {k: v for k, v in groups.items() if v}