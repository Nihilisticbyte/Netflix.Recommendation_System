"""
src/models/svd_model.py
-----------------------
SVD-based Matrix Factorization using the `surprise` library.

Simon Funk's SVD (used in Netflix Prize) factorizes the rating matrix R ≈ P × Q^T
where P is the user-latent-factor matrix and Q is the item-latent-factor matrix.
Bias terms for users and items are included.

We wrap Surprise's SVD so our evaluation pipeline can call it
with the same interface as other models.
"""

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_split


class SVDModel:
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_factors    : number of latent factors
        n_epochs     : SGD epochs
        lr_all       : learning rate for all parameters
        reg_all      : regularization for all parameters
        random_state : seed
        """
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state,
            verbose=True,
        )
        self.trainset = None
        self.rating_matrix = None   # for recommend()
        self.n_users = None
        self.n_movies = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame):
        """
        Train SVD on a DataFrame with columns user_idx, movie_idx, rating.
        We use raw integer indices as user/item IDs inside Surprise.
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_df[["user_idx", "movie_idx", "rating"]], reader
        )
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

        # Store sparse rating matrix for recommendation generation
        self.n_users = train_df["user_idx"].max() + 1
        self.n_movies = train_df["movie_idx"].max() + 1

        from scipy.sparse import csr_matrix
        sparse = csr_matrix(
            (train_df["rating"].values,
             (train_df["user_idx"].values, train_df["movie_idx"].values)),
            shape=(self.n_users, self.n_movies),
        )
        self.rating_matrix = sparse.toarray().astype(np.float32)

        print(f"[SVD] Fit complete: {self.n_users} users × {self.n_movies} movies "
              f"| factors={self.model.n_factors}")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    # In svd_model.py
    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        pred = self.model.predict(int(user_idx), int(movie_idx))
        return float(np.clip(pred.est, 1.0, 5.0))
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in test_df.iterrows():
            est = self.model.predict(int(row["user_idx"]), int(row["movie_idx"])).est
            preds.append(float(np.clip(est, 1.0, 5.0)))
        return np.array(preds)

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def recommend(self, user_idx: int, top_k: int = 10, seen_movies: set = None) -> list:
        """
        Return Top-K recommended movie indices for a user.

        Returns list of (movie_idx, predicted_rating) sorted descending.
        """
        if seen_movies is None and self.rating_matrix is not None:
            seen_movies = set(np.where(self.rating_matrix[user_idx] != 0)[0])
        seen_movies = seen_movies or set()

        candidates = [m for m in range(self.n_movies) if m not in seen_movies]
        scores = [
            (m, self.predict_rating(user_idx, m)) for m in candidates
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
