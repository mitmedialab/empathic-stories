import itertools
import numpy as np
from numpy import dot
from numpy.linalg import norm
import faiss
import pandas as pd

def bin_label(label):
    if label <= 2.5:
        return 0
    else:
        return 1
    
def normalize(values, actual_bounds, desired_bounds):
    return [desired_bounds[0] + (x - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0]) for x in values]

def get_cosine_similarity(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def get_all_cos_similarities(df, text_column_name, embedding_column_name):
    idx_pairs = list(itertools.combinations(list(df.index), 2))
    cos_sims = []
    for pair in idx_pairs:
        text_a = df[text_column_name][pair[0]]
        text_b = df[text_column_name][pair[1]]
        embedding_a = eval(df[embedding_column_name][pair[0]])
        embedding_b = eval(df[embedding_column_name][pair[1]])
        cos_sim = get_cosine_similarity(embedding_a, embedding_b)
        cos_sims.append(cos_sim)
    return pd.DataFrame({"pairs": idx_pairs, "similarity": cos_sims})
    
def get_all_cos_similarities_faiss(df, column_name, similarity_col_name):
    k = len(df)
    embeddings = np.array([list(eval(_)) for _ in df[column_name]], dtype=np.float32)

    d = len(embeddings[0])
    index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    D, I = index.search(embeddings, k) 
    
    pairs = {}
    i = 0
    for scores, indices in zip(D, I):
        for idx, j in enumerate(indices):
            if i != j:
                pair_idx = tuple(sorted([i, j]))
                if pair_idx not in pairs:
                    pairs[pair_idx] = D[i][idx]
        i+=1
    idx_pairs = []
    cos_sims = []
    for item in pairs:
        idx_pairs.append(item)
        cos_sims.append(pairs[item])
    return pd.DataFrame({"pairs": idx_pairs, similarity_col_name: cos_sims}).sort_values(by="pairs", ascending=True).reset_index(drop=True)
