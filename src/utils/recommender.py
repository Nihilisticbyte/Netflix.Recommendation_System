"""
src/utils/recommender.py
------------------------
High-level utility to generate and display Top-K recommendations,
including movie title lookup, success/failure analysis, and
explainability helpers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class Recommender:
    def __init__(
        self,
        model,
        movies_df: pd.DataFrame,
        train_df: pd.DataFrame,
        idx_to_movie: Dict[int, int],    # movie_idx → movie_id
        relevance_threshold: float = 3.5,
    ):
        """
        Parameters
        ----------
        model              : fitted recommendation model
        movies_df          : DataFrame with columns movie_id, title, year
        train_df           : training data (to know what user has already seen)
        idx_to_movie       : reverse mapping from movie_idx → movie_id
        relevance_threshold: ratings >= this count as positive ground truth
        """
        self.model = model
        self.movies_df = movies_df.set_index("movie_id")
        self.train_df = train_df
        self.idx_to_movie = idx_to_movie
        self.movie_to_idx = {v: k for k, v in idx_to_movie.items()}
        self.relevance_threshold = relevance_threshold

        # Pre-compute seen movies per user
        self._seen: Dict[int, set] = (
            train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        )

    # ------------------------------------------------------------------
    # Core recommendation
    # ------------------------------------------------------------------

    def recommend_for_user(
        self,
        user_idx: int,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Generate Top-K recommendations for a user.

        Returns a DataFrame with: rank, movie_id, title, year, predicted_rating
        """
        seen = self._seen.get(user_idx, set())
        recs = self.model.recommend(user_idx, top_k=top_k, seen_movies=seen)

        rows = []
        for rank, (movie_idx, pred_rating) in enumerate(recs, start=1):
            movie_id = self.idx_to_movie.get(movie_idx)
            if movie_id and movie_id in self.movies_df.index:
                title = self.movies_df.loc[movie_id, "title"]
                year  = self.movies_df.loc[movie_id, "year"]
            else:
                title, year = f"Movie {movie_idx}", None
            rows.append({
                "rank": rank,
                "movie_id": movie_id,
                "movie_idx": movie_idx,
                "title": title,
                "year": year,
                "predicted_rating": round(pred_rating, 3),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # History display
    # ------------------------------------------------------------------

    def get_user_history(
        self,
        user_idx: int,
        n: int = 10,
    ) -> pd.DataFrame:
        """Return the N highest-rated movies a user has seen (from train set)."""
        user_rows = self.train_df[self.train_df["user_idx"] == user_idx]
        top_rated = user_rows.nlargest(n, "rating")

        rows = []
        for _, row in top_rated.iterrows():
            movie_id = self.idx_to_movie.get(int(row["movie_idx"]))
            if movie_id and movie_id in self.movies_df.index:
                title = self.movies_df.loc[movie_id, "title"]
                year  = self.movies_df.loc[movie_id, "year"]
            else:
                title, year = f"Movie {int(row['movie_idx'])}", None
            rows.append({
                "movie_id": movie_id,
                "title": title,
                "year": year,
                "rating": row["rating"],
                "date": row.get("date", None),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Success / failure case analysis
    # ------------------------------------------------------------------

    def analyze_recommendations(
        self,
        user_idx: int,
        test_df: pd.DataFrame,
        top_k: int = 10,
    ) -> dict:
        """
        Compare recommendations with ground-truth test ratings.

        Returns dict with:
            recs_df      : recommendation table
            history_df   : user's top-rated history
            hits         : list of correctly recommended relevant movies
            misses       : relevant movies NOT recommended
            irrelevant   : recommended movies that were not relevant
        """
        recs_df = self.recommend_for_user(user_idx, top_k)
        history_df = self.get_user_history(user_idx)

        # Ground truth from test set
        user_test = test_df[test_df["user_idx"] == user_idx]
        relevant_idx = set(
            user_test[user_test["rating"] >= self.relevance_threshold]["movie_idx"].astype(int)
        )

        rec_idx = set(recs_df["movie_idx"].astype(int))
        hits       = rec_idx & relevant_idx
        misses     = relevant_idx - rec_idx
        irrelevant = rec_idx - relevant_idx

        return {
            "recs_df": recs_df,
            "history_df": history_df,
            "hits": hits,
            "misses": misses,
            "irrelevant": irrelevant,
            "precision": len(hits) / len(rec_idx) if rec_idx else 0,
            "recall": len(hits) / len(relevant_idx) if relevant_idx else 0,
        }

    # ------------------------------------------------------------------
    # Similar items (explainability)
    # ------------------------------------------------------------------

    def similar_movies(self, movie_idx: int, top_k: int = 5) -> pd.DataFrame:
        """
        Return movies most similar to a given movie.
        Works if model has an item_sim matrix (ItemCF) or item_factors (ALS/NCF).
        """
        sim = None

        # ItemCF
        if hasattr(self.model, "item_sim") and self.model.item_sim is not None:
            sim = self.model.item_sim[movie_idx]

        # ALS / NCF: cosine similarity from latent factors
        elif hasattr(self.model, "item_factors") and self.model.item_factors is not None:
            factors = self.model.item_factors
            norms = np.linalg.norm(factors, axis=1, keepdims=True) + 1e-9
            normed = factors / norms
            sim = normed @ normed[movie_idx]

        if sim is None:
            return pd.DataFrame(columns=["rank", "title", "year", "similarity"])

        sim[movie_idx] = -1  # exclude self
        top_idx = np.argpartition(sim, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sim[top_idx])[::-1]]

        rows = []
        for rank, idx in enumerate(top_idx, start=1):
            movie_id = self.idx_to_movie.get(int(idx))
            if movie_id and movie_id in self.movies_df.index:
                title = self.movies_df.loc[movie_id, "title"]
                year  = self.movies_df.loc[movie_id, "year"]
            else:
                title, year = f"Movie {idx}", None
            rows.append({
                "rank": rank,
                "title": title,
                "year": year,
                "similarity": round(float(sim[idx]), 4),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_recommendations(self, user_idx: int, top_k: int = 10):
        recs = self.recommend_for_user(user_idx, top_k)
        hist = self.get_user_history(user_idx, n=5)

        print(f"\n{'='*55}")
        print(f"  User {user_idx} — Top-{top_k} Recommendations")
        print(f"{'='*55}")
        print("\n📽  User's recent favourites:")
        for _, r in hist.iterrows():
            print(f"    ⭐ {r['rating']:.1f}  {r['title']} ({r['year']})")

        print(f"\n🎬  Recommended for you:")
        for _, r in recs.iterrows():
            print(f"    #{r['rank']:2d}  {r['predicted_rating']:.2f}  "
                  f"{r['title']} ({r['year']})")
        print()
