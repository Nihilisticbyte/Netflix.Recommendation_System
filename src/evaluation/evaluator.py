"""
src/evaluation/evaluator.py
---------------------------
Orchestrates evaluation of any recommendation model.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List

from src.evaluation.metrics import (
    rmse, mae, map_at_k, precision_at_k, recall_at_k,
    ndcg_at_k, hit_rate_at_k, coverage,
    build_relevant_items,
)


class Evaluator:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_movies: int,
        k: int = 10,
        relevance_threshold: float = 3.5,
        eval_users_sample: int = 500,
        rmse_sample: int = 50000,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        rmse_sample      : max test rows used for RMSE (sub-sample for slow CF models).
                           Set None to use all rows. SVD/ALS/NCF are fast so they use all.
        eval_users_sample: users to generate Top-K recs for (ranking metrics)
        """
        self.train_df  = train_df
        self.test_df   = test_df
        self.n_movies  = n_movies
        self.k         = k
        self.relevance_threshold = relevance_threshold
        self.random_state        = random_state
        self.rmse_sample         = rmse_sample

        rng = np.random.default_rng(random_state)

        # RMSE test subset
        if rmse_sample and len(test_df) > rmse_sample:
            self.test_rmse = test_df.sample(n=rmse_sample, random_state=random_state).reset_index(drop=True)
        else:
            self.test_rmse = test_df

        # Ground-truth relevant items per user
        self.user_relevant = build_relevant_items(test_df, relevance_threshold)

        # Subset of users for ranking metrics
        all_test_users = sorted(self.user_relevant.keys())
        if eval_users_sample and len(all_test_users) > eval_users_sample:
            self.eval_users = list(
                rng.choice(all_test_users, size=eval_users_sample, replace=False)
            )
        else:
            self.eval_users = all_test_users

        # Train-set movies per user (exclude from recs)
        self._train_seen: Dict[int, set] = (
            train_df.groupby("user_idx")["movie_idx"]
            .apply(set)
            .to_dict()
        )

        print(f"[Evaluator] {len(self.user_relevant):,} users have relevant test items")
        print(f"[Evaluator] RMSE on {len(self.test_rmse):,} rows | "
              f"Ranking on {len(self.eval_users)} users")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model,
        model_name: str = "Model",
        use_full_test_for_rmse: bool = False,
    ) -> tuple:
        t0 = time.time()

        # ---- 1. RMSE / MAE -------------------------------------------
        test_for_rmse = self.test_df if use_full_test_for_rmse else self.test_rmse
        print(f"[Evaluator] [{model_name}] Predicting on {len(test_for_rmse):,} rows for RMSE ...")
        y_pred = model.predict(test_for_rmse)
        y_true = test_for_rmse["rating"].values

        rmse_val = rmse(y_true, y_pred)
        mae_val  = mae(y_true, y_pred)
        print(f"  RMSE={rmse_val:.4f}  MAE={mae_val:.4f}")

        # ---- 2. Top-K recommendations --------------------------------
        print(f"[Evaluator] [{model_name}] Generating Top-{self.k} recs "
              f"for {len(self.eval_users)} users ...")
        user_recommendations: Dict[int, List[int]] = {}

        for i, user_idx in enumerate(self.eval_users):
            if i % 100 == 0:
                print(f"  recs: {i}/{len(self.eval_users)}", end="\r")
            seen = self._train_seen.get(user_idx, set())
            recs = model.recommend(user_idx, top_k=self.k, seen_movies=seen)
            user_recommendations[user_idx] = [m for m, _ in recs]
        print()

        # ---- 3. Ranking metrics --------------------------------------
        map_val  = map_at_k(user_recommendations, self.user_relevant, self.k)
        prec_val = precision_at_k(user_recommendations, self.user_relevant, self.k)
        rec_val  = recall_at_k(user_recommendations, self.user_relevant, self.k)
        ndcg_val = ndcg_at_k(user_recommendations, self.user_relevant, self.k)
        hr_val   = hit_rate_at_k(user_recommendations, self.user_relevant, self.k)
        cov_val  = coverage(user_recommendations, self.n_movies)

        elapsed = time.time() - t0

        results = pd.DataFrame([{
            "Model":             model_name,
            "RMSE":              round(rmse_val, 4),
            "MAE":               round(mae_val, 4),
            f"MAP@{self.k}":     round(map_val, 4),
            f"Prec@{self.k}":    round(prec_val, 4),
            f"Recall@{self.k}":  round(rec_val, 4),
            f"NDCG@{self.k}":    round(ndcg_val, 4),
            f"HitRate@{self.k}": round(hr_val, 4),
            "Coverage":          round(cov_val, 4),
            "Eval_Time_s":       round(elapsed, 1),
        }])

        print(f"\n{'='*60}")
        print(f"  Results for: {model_name}")
        print(results.to_string(index=False))
        print(f"{'='*60}\n")

        return results, user_recommendations

    def compare(self, models: list) -> pd.DataFrame:
        all_results = []
        for model, name in models:
            res, _ = self.evaluate(model, model_name=name)
            all_results.append(res)
        comparison = pd.concat(all_results, ignore_index=True)
        print("\n===== MODEL COMPARISON =====")
        print(comparison.to_string(index=False))
        return comparison
