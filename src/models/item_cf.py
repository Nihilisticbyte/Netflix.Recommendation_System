"""
src/models/item_cf.py
---------------------
Item-Based Collaborative Filtering.

Algorithm:
1. Build user-movie rating matrix (transposed view = item-user).
2. Compute pairwise cosine similarity between items (movies).
3. Predict rating for (user, movie) as the weighted average of
   the user's ratings on movies similar to the target movie.
4. Generate Top-K recommendations by scoring all unseen movies.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    def __init__(self, k_neighbors: int = 50):
        self.k = k_neighbors
        self.rating_matrix = None   # dense [n_users x n_movies]
        self.item_sim = None        # [n_movies x n_movies] cosine similarity

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame):
        n_users  = train_df["user_idx"].max() + 1
        n_movies = train_df["movie_idx"].max() + 1

        sparse = csr_matrix(
            (train_df["rating"].values,
             (train_df["user_idx"].values, train_df["movie_idx"].values)),
            shape=(n_users, n_movies),
        )
        self.rating_matrix = sparse.toarray().astype(np.float32)
        self.item_sim = cosine_similarity(sparse.T)
        np.fill_diagonal(self.item_sim, 0)

        print(f"[ItemCF] Fit complete: {n_users} users x {n_movies} movies")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        if self.rating_matrix is None:
            raise RuntimeError("Call fit() first")
        user_row     = self.rating_matrix[user_idx]
        rated_movies = np.where(user_row != 0)[0]
        if len(rated_movies) == 0:
            return 3.0
        sims          = self.item_sim[movie_idx][rated_movies]
        k             = min(self.k, len(rated_movies))
        top_k_pos     = np.argpartition(sims, -k)[-k:]
        top_k_sims    = sims[top_k_pos]
        top_k_ratings = user_row[rated_movies[top_k_pos]]
        denom         = np.sum(np.abs(top_k_sims)) + 1e-9
        return float(np.clip(np.dot(top_k_sims, top_k_ratings) / denom, 1.0, 5.0))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized batch predict - groups by user so each user's rated-movie
        list is fetched once instead of once per test row. ~100x faster.
        """
        preds      = np.full(len(test_df), 3.0, dtype=np.float32)
        user_idxs  = test_df["user_idx"].values.astype(int)
        movie_idxs = test_df["movie_idx"].values.astype(int)

        user_to_rows = defaultdict(list)
        for pos, u in enumerate(user_idxs):
            user_to_rows[u].append(pos)

        n_users_eval = len(user_to_rows)
        for done, (user_idx, positions) in enumerate(user_to_rows.items()):
            if done % 2000 == 0:
                print(f"  [ItemCF predict] {done}/{n_users_eval} users ...", end="\r")
            user_row     = self.rating_matrix[user_idx]
            rated_movies = np.where(user_row != 0)[0]
            if len(rated_movies) == 0:
                continue
            for pos, movie_idx in zip(positions, movie_idxs[positions]):
                sims          = self.item_sim[movie_idx][rated_movies]
                k             = min(self.k, len(rated_movies))
                top_k_pos     = np.argpartition(sims, -k)[-k:]
                top_k_sims    = sims[top_k_pos]
                top_k_ratings = user_row[rated_movies[top_k_pos]]
                denom         = np.sum(np.abs(top_k_sims)) + 1e-9
                preds[pos]    = float(np.clip(
                    np.dot(top_k_sims, top_k_ratings) / denom, 1.0, 5.0
                ))
        print()
        return preds.astype(float)

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def recommend(self, user_idx: int, top_k: int = 10, seen_movies: set = None) -> list:
        n_movies = self.rating_matrix.shape[1]
        if seen_movies is None:
            seen_movies = set(np.where(self.rating_matrix[user_idx] != 0)[0])
        candidates = [m for m in range(n_movies) if m not in seen_movies]
        scores = [(m, self.predict_rating(user_idx, m)) for m in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
