"""
src/evaluation/metrics.py
--------------------------
All evaluation metrics.

Mandatory:
    - RMSE  (Root Mean Squared Error)
    - MAP@K (Mean Average Precision @ K)

Optional:
    - MAE, Precision@K, Recall@K, NDCG@K, Hit Rate, Coverage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Rating Prediction Metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — lower is better."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — lower is better."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


# ---------------------------------------------------------------------------
# Ranking Metrics
# ---------------------------------------------------------------------------

def average_precision_at_k(
    recommended: List[int],
    relevant: set,
    k: int = 10,
) -> float:
    """
    Compute Average Precision @ K for a single user.

    Parameters
    ----------
    recommended : ordered list of recommended item indices (top-K)
    relevant    : set of truly relevant item indices for this user
    k           : cutoff

    Returns
    -------
    AP@K score ∈ [0, 1]
    """
    if not relevant:
        return 0.0

    recommended = recommended[:k]
    hits = 0
    precision_sum = 0.0

    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant), k)


def map_at_k(
    user_recommendations: Dict[int, List[int]],
    user_relevant: Dict[int, set],
    k: int = 10,
) -> float:
    """
    Mean Average Precision @ K across all users.

    Parameters
    ----------
    user_recommendations : {user_idx → [movie_idx, ...]} (ordered, top-K)
    user_relevant        : {user_idx → {movie_idx, ...}} (ground truth)
    k                    : cutoff

    Returns
    -------
    MAP@K ∈ [0, 1]
    """
    aps = []
    for user_idx, recs in user_recommendations.items():
        relevant = user_relevant.get(user_idx, set())
        ap = average_precision_at_k(recs, relevant, k)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def precision_at_k(
    user_recommendations: Dict[int, List[int]],
    user_relevant: Dict[int, set],
    k: int = 10,
) -> float:
    """Fraction of top-K recommendations that are relevant, averaged over users."""
    precisions = []
    for user_idx, recs in user_recommendations.items():
        relevant = user_relevant.get(user_idx, set())
        hits = sum(1 for m in recs[:k] if m in relevant)
        precisions.append(hits / k)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(
    user_recommendations: Dict[int, List[int]],
    user_relevant: Dict[int, set],
    k: int = 10,
) -> float:
    """Fraction of relevant items that appear in top-K, averaged over users."""
    recalls = []
    for user_idx, recs in user_recommendations.items():
        relevant = user_relevant.get(user_idx, set())
        if not relevant:
            continue
        hits = sum(1 for m in recs[:k] if m in relevant)
        recalls.append(hits / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(
    user_recommendations: Dict[int, List[int]],
    user_relevant: Dict[int, set],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    def dcg(recs, relevant, k):
        return sum(
            1 / np.log2(rank + 1)
            for rank, item in enumerate(recs[:k], start=1)
            if item in relevant
        )

    ndcgs = []
    for user_idx, recs in user_recommendations.items():
        relevant = user_relevant.get(user_idx, set())
        if not relevant:
            continue
        dcg_score = dcg(recs, relevant, k)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcgs.append(dcg_score / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def hit_rate_at_k(
    user_recommendations: Dict[int, List[int]],
    user_relevant: Dict[int, set],
    k: int = 10,
) -> float:
    """Fraction of users for whom at least one top-K recommendation is relevant."""
    hits = 0
    for user_idx, recs in user_recommendations.items():
        relevant = user_relevant.get(user_idx, set())
        if any(m in relevant for m in recs[:k]):
            hits += 1
    return hits / len(user_recommendations) if user_recommendations else 0.0


def coverage(
    user_recommendations: Dict[int, List[int]],
    n_movies: int,
) -> float:
    """Fraction of the item catalogue that appears in any recommendation list."""
    all_recs = set(m for recs in user_recommendations.values() for m in recs)
    return len(all_recs) / n_movies if n_movies > 0 else 0.0


# ---------------------------------------------------------------------------
# Relevance builder
# ---------------------------------------------------------------------------

def build_relevant_items(
    test_df: pd.DataFrame,
    relevance_threshold: float = 3.5,
) -> Dict[int, set]:
    """
    Build {user_idx → {relevant movie_idx, ...}} from test set.

    A movie is relevant if actual rating ≥ relevance_threshold.
    """
    relevant_map: Dict[int, set] = {}
    for _, row in test_df.iterrows():
        if row["rating"] >= relevance_threshold:
            relevant_map.setdefault(int(row["user_idx"]), set()).add(int(row["movie_idx"]))
    return relevant_map


# ---------------------------------------------------------------------------
# All-in-one summary
# ---------------------------------------------------------------------------

def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    user_recommendations: Dict[int, List[int]],
    user_relevant: Dict[int, set],
    n_movies: int,
    k: int = 10,
) -> pd.DataFrame:
    """
    Compute all metrics and return as a single-row DataFrame.

    Parameters
    ----------
    y_true               : actual ratings (test set)
    y_pred               : predicted ratings (test set)
    user_recommendations : {user_idx → [movie_idx ...]} top-K
    user_relevant        : {user_idx → {relevant_movie_idx ...}}
    n_movies             : total catalogue size (for coverage)
    k                    : cutoff

    Returns
    -------
    pd.DataFrame with metric columns
    """
    results = {
        "RMSE":         rmse(y_true, y_pred),
        "MAE":          mae(y_true, y_pred),
        f"MAP@{k}":     map_at_k(user_recommendations, user_relevant, k),
        f"Prec@{k}":    precision_at_k(user_recommendations, user_relevant, k),
        f"Recall@{k}":  recall_at_k(user_recommendations, user_relevant, k),
        f"NDCG@{k}":    ndcg_at_k(user_recommendations, user_relevant, k),
        f"HitRate@{k}": hit_rate_at_k(user_recommendations, user_relevant, k),
        "Coverage":     coverage(user_recommendations, n_movies),
    }
    return pd.DataFrame([results])
