"""
src/models/user_cf.py
---------------------
User-Based Collaborative Filtering.

Algorithm:
1. Build user–movie rating matrix.
2. Compute pairwise cosine similarity between users.
3. Predict rating for (user, movie) as the weighted average of
   similar users' ratings for that movie.
4. Generate Top-K recommendations by predicting ratings for
   all unseen movies and returning the highest.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    def __init__(self, k_neighbors: int = 50):
        """
        Parameters
        ----------
        k_neighbors : number of nearest neighbors to use for prediction
        """
        self.k = k_neighbors
        self.user_map = None       # user_id  → row index
        self.movie_map = None      # movie_id → col index
        self.idx_to_user = None
        self.idx_to_movie = None
        self.rating_matrix = None  # dense np.ndarray [n_users × n_movies]
        self.sim_matrix = None     # user-user cosine similarity
        self.user_means = None     # per-user mean rating (for bias correction)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame):
        """
        Build rating matrix and compute user-user similarities.

        Parameters
        ----------
        train_df : DataFrame with columns user_idx, movie_idx, rating
        """
        n_users = train_df["user_idx"].max() + 1
        n_movies = train_df["movie_idx"].max() + 1

        # Build sparse matrix then convert to dense for cosine similarity
        sparse = csr_matrix(
            (train_df["rating"].values,
             (train_df["user_idx"].values, train_df["movie_idx"].values)),
            shape=(n_users, n_movies),
        )
        self.rating_matrix = sparse.toarray().astype(np.float32)

        # Per-user mean (ignoring zeros = unrated)
        with np.errstate(invalid="ignore"):
            row_sums = self.rating_matrix.sum(axis=1)
            row_counts = (self.rating_matrix != 0).sum(axis=1)
            self.user_means = np.where(row_counts > 0, row_sums / row_counts, 0)

        # Cosine similarity on sparse matrix (much faster)
        self.sim_matrix = cosine_similarity(sparse)
        np.fill_diagonal(self.sim_matrix, 0)  # exclude self

        print(f"[UserCF] Fit complete: {n_users} users × {n_movies} movies")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        """Predict the rating for a single (user, movie) pair."""
        if self.rating_matrix is None:
            raise RuntimeError("Call fit() first")

        # Similarities of this user with all others
        sims = self.sim_matrix[user_idx]

        # Only consider users who rated this movie
        rated_mask = self.rating_matrix[:, movie_idx] != 0
        if rated_mask.sum() == 0:
            return self.user_means[user_idx] if self.user_means[user_idx] > 0 else 3.0

        sims_filtered = sims * rated_mask
        # Top-K neighbors
        top_k_idx = np.argpartition(sims_filtered, -self.k)[-self.k:]
        top_k_sims = sims_filtered[top_k_idx]
        top_k_ratings = self.rating_matrix[top_k_idx, movie_idx]

        # Mean-centered weighted average
        top_k_means = self.user_means[top_k_idx]
        num = np.dot(top_k_sims, top_k_ratings - top_k_means)
        denom = np.sum(np.abs(top_k_sims)) + 1e-9
        pred = self.user_means[user_idx] + num / denom
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for all rows in test_df — grouped by user so we
        only fetch each user's similarity row once instead of once per row.
        Runs ~100x faster than iterrows on 500k+ test rows.
        """
        from collections import defaultdict

        preds      = np.full(len(test_df), 3.0, dtype=np.float32)
        user_idxs  = test_df["user_idx"].values.astype(int)
        movie_idxs = test_df["movie_idx"].values.astype(int)

        user_to_rows = defaultdict(list)
        for pos, u in enumerate(user_idxs):
            user_to_rows[u].append(pos)

        n_users_eval = len(user_to_rows)
        for done, (user_idx, positions) in enumerate(user_to_rows.items()):
            if done % 2000 == 0:
                print(f"  [UserCF predict] {done}/{n_users_eval} users ...", end="\r")

            sims   = self.sim_matrix[user_idx]
            u_mean = self.user_means[user_idx]
            movies = movie_idxs[positions]

            for pos, movie_idx in zip(positions, movies):
                col        = self.rating_matrix[:, movie_idx]
                rated_mask = col != 0
                if rated_mask.sum() == 0:
                    preds[pos] = u_mean if u_mean > 0 else 3.0
                    continue
                sims_f    = sims * rated_mask
                k         = min(self.k, int(rated_mask.sum()))
                top_k_idx = np.argpartition(sims_f, -k)[-k:]
                top_k_s   = sims_f[top_k_idx]
                top_k_r   = col[top_k_idx]
                top_k_m   = self.user_means[top_k_idx]
                num        = np.dot(top_k_s, top_k_r - top_k_m)
                denom      = np.sum(np.abs(top_k_s)) + 1e-9
                preds[pos] = float(np.clip(u_mean + num / denom, 1.0, 5.0))

        print()
        return preds.astype(float)

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def recommend(self, user_idx: int, top_k: int = 10, seen_movies: set = None) -> list:
        """
        Return Top-K recommended movie indices for a user.

        Parameters
        ----------
        user_idx    : integer index of the user
        top_k       : number of recommendations
        seen_movies : set of movie_idx already rated by the user (to exclude)

        Returns list of (movie_idx, predicted_rating) sorted descending
        """
        n_movies = self.rating_matrix.shape[1]
        if seen_movies is None:
            seen_movies = set(np.where(self.rating_matrix[user_idx] != 0)[0])

        candidates = [m for m in range(n_movies) if m not in seen_movies]
        scores = [
            (m, self.predict_rating(user_idx, m)) for m in candidates
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
