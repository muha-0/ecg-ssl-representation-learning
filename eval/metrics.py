import numpy as np
import torch
from sklearn.metrics import silhouette_score
import torch.nn.functional as F

def auc_roc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    order = np.argsort(-y_score)
    y_true = y_true[order]

    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    if P == 0 or N == 0:
        return float("nan")

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    tpr = tps / P
    fpr = fps / N
    return float(np.trapz(tpr, fpr))

def average_precision(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    order = np.argsort(-y_score)
    y_true = y_true[order]

    P = (y_true == 1).sum()
    if P == 0:
        return float("nan")

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / P

    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

# For Clustering Scores
def compute_metrics(embeddings, labels, k_list=[1, 5]):
    # Normalize for cosine similarity calculation
    feat = torch.from_numpy(embeddings)
    feat = F.normalize(feat, p=2, dim=1)
    
    # Compute Cosine Similarity Matrix [N_total, N_total]
    sim_matrix = torch.mm(feat, feat.t()).cpu().numpy()
    np.fill_diagonal(sim_matrix, -1e9) # Don't match with self
    
    metrics = {}
    
    # Recall@K Calculation
    for k in k_list:
        correct = 0
        for i in range(len(labels)):
            # Get indices of top K similarities
            top_k_idx = np.argsort(-sim_matrix[i])[:k]
            # Check if any of the top K belong to the same patient
            if any(labels[idx] == labels[i] for idx in top_k_idx):
                correct += 1
        metrics[f"Recall@{k}"] = correct / len(labels)
    
    # Silhouette Score using Cosine Distance
    # distance = 1 - similarity
    metrics["Silhouette"] = silhouette_score(embeddings, labels, metric='cosine')
    
    return metrics