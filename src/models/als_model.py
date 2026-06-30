"""
src/models/als_model.py
-----------------------
Alternating Least Squares (ALS) for implicit / explicit feedback.

We use the `implicit` library which implements fast Cython ALS.
Because `implicit` is designed for implicit feedback, we convert
ratings to confidence values: c = 1 + α * rating.

For explicit rating prediction we fall back to the dot product
of user and item latent factors plus bias.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit

try:
    
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False
    print("[WARNING] `implicit` not installed. ALSModel will use numpy fallback.")


class ALSModel:
    def __init__(
        self,
        factors: int = 100,
        iterations: int = 20,
        regularization: float = 0.01,
        alpha: float = 40.0,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        factors        : number of latent factors
        iterations     : ALS iterations
        regularization : L2 regularization
        alpha          : confidence scaling factor (c = 1 + alpha * rating)
        random_state   : seed
        """
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state

        self.user_factors = None     # [n_users × factors]
        self.item_factors = None     # [n_movies × factors]
        self.rating_matrix = None    # sparse [n_users × n_movies]
        self.n_users = None
        self.n_movies = None
        self._model = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame):
        """
        Train ALS on a DataFrame with columns user_idx, movie_idx, rating.
        """
        self.n_users = train_df["user_idx"].max() + 1
        self.n_movies = train_df["movie_idx"].max() + 1

        # Confidence matrix (item × user for implicit library)
        confidence = (1 + self.alpha * train_df["rating"].values).astype(np.float32)
        self.rating_matrix = csr_matrix(
            (train_df["rating"].values,
             (train_df["user_idx"].values, train_df["movie_idx"].values)),
            shape=(self.n_users, self.n_movies),
        )

        if HAS_IMPLICIT:
            self._model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                iterations=self.iterations,
                regularization=self.regularization,
                random_state=self.random_state,
            )
            # implicit expects item × user matrix
            item_user = csr_matrix(
                (confidence,
                 (train_df["movie_idx"].values, train_df["user_idx"].values)),
                shape=(self.n_movies, self.n_users),
            )
            self._model.fit(item_user, show_progress=True)
            self.user_factors = np.array(self._model.user_factors)
            self.item_factors = np.array(self._model.item_factors)
        else:
            # Pure numpy ALS fallback (slower but dependency-free)
            self._fit_numpy(train_df)

        print(f"[ALS] Fit complete: {self.n_users} users × {self.n_movies} movies "
              f"| factors={self.factors}")
        return self

    def _fit_numpy(self, train_df: pd.DataFrame):
        """Minimal numpy ALS for environments without `implicit`."""
        rng = np.random.default_rng(self.random_state)
        U = rng.normal(0, 0.1, (self.n_users, self.factors)).astype(np.float32)
        V = rng.normal(0, 0.1, (self.n_movies, self.factors)).astype(np.float32)
        lam = self.regularization * np.eye(self.factors)

        R = self.rating_matrix.toarray()
        C = (R > 0).astype(np.float32)  # simple binary confidence

        for it in range(self.iterations):
            # Update user factors
            for u in range(self.n_users):
                Cu = np.diag(C[u])
                U[u] = np.linalg.solve(V.T @ Cu @ V + lam, V.T @ Cu @ R[u])
            # Update item factors
            for i in range(self.n_movies):
                Ci = np.diag(C[:, i])
                V[i] = np.linalg.solve(U.T @ Ci @ U + lam, U.T @ Ci @ R[:, i])
            print(f"  ALS epoch {it+1}/{self.iterations}")

        self.user_factors = U
        self.item_factors = V

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        """Predict rating as dot product of latent factors."""
        score = float(self.item_factors[user_idx] @ self.user_factors[movie_idx])
        return float(np.clip(score, 1.0, 5.0))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for all rows in test_df (Normalized to 1-5 scale)."""
        u = test_df["user_idx"].values
        m = test_df["movie_idx"].values
        
        # Get raw unbounded confidence scores
        raw_scores = np.sum(self.item_factors[u] * self.user_factors[m], axis=1)
        
        # Min-Max Scale the scores to a 1.0 - 5.0 range
        min_score = np.min(raw_scores)
        max_score = np.max(raw_scores)
        
        # Prevent division by zero edge-case
        if max_score > min_score:
            scaled_scores = 1.0 + 4.0 * ((raw_scores - min_score) / (max_score - min_score))
        else:
            scaled_scores = np.full_like(raw_scores, 3.0) # Fallback
            
        return np.clip(scaled_scores, 1.0, 5.0).astype(float)

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def recommend(self, user_idx: int, top_k: int = 10, seen_movies: set = None) -> list:
        """
        Return Top-K recommended movie indices for a user.

        Returns list of (movie_idx, predicted_rating) sorted descending.
        """
        if seen_movies is None:
            seen_movies = set(self.rating_matrix[user_idx].nonzero()[1])

        scores = self.user_factors @ self.item_factors[user_idx]
        for m in seen_movies:
            scores[m] = -np.inf

        top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        return [(int(m), float(np.clip(scores[m], 1.0, 5.0))) for m in top_k_idx]
