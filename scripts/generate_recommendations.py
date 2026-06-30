"""
scripts/generate_recommendations.py
------------------------------------
Generate Top-10 recommendations for sample users and print
success cases, failure cases, and similar-item explanations.

Usage:
    python scripts/generate_recommendations.py [--model svd] [--n_users 5]
"""

import sys
import os
import argparse
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.recommender import Recommender

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main(model_name: str = "svd", n_users: int = 5, top_k: int = 10):
    print("=" * 60)
    print(f"  Generating Top-{top_k} Recommendations [{model_name.upper()}]")
    print("=" * 60)

    # ---- Load data and model ---------------------------------------------
    data_path = os.path.join(OUTPUT_DIR, "preprocessed_data.pkl")
    model_path = os.path.join(OUTPUT_DIR, f"{model_name}.pkl")

    if not os.path.exists(data_path):
        print("[ERROR] Run train_models.py first.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    data  = joblib.load(data_path)
    model = joblib.load(model_path)
    movies = joblib.load(os.path.join(OUTPUT_DIR, "movies.pkl"))

    train = data["train"]
    test  = data["test"]

    # idx → movie_id reverse map
    movie_map = data["movie_map"]            # movie_id → movie_idx
    idx_to_movie = {v: k for k, v in movie_map.items()}

    rec = Recommender(
        model=model,
        movies_df=movies,
        train_df=train,
        idx_to_movie=idx_to_movie,
        relevance_threshold=3.5,
    )

    # ---- Pick sample users who have test ratings -------------------------
    test_users = test["user_idx"].unique()
    rng = np.random.default_rng(42)
    sample_users = rng.choice(test_users, size=min(n_users, len(test_users)), replace=False)

    # ---- Generate recommendations ----------------------------------------
    all_recs = []
    success_cases = []
    failure_cases = []

    for user_idx in sample_users:
        analysis = rec.analyze_recommendations(user_idx, test, top_k=top_k)
        rec.print_recommendations(int(user_idx), top_k=top_k)

        # Classify
        precision = analysis["precision"]
        if precision >= 0.4:
            success_cases.append((int(user_idx), precision))
        elif precision < 0.1:
            failure_cases.append((int(user_idx), precision))

        # Collect for CSV
        for _, row in analysis["recs_df"].iterrows():
            row_dict = row.to_dict()
            row_dict["user_idx"] = int(user_idx)
            row_dict["hit"] = row["movie_idx"] in analysis["hits"]
            all_recs.append(row_dict)

    # ---- Save to CSV -------------------------------------------------------
    recs_df = pd.DataFrame(all_recs)
    csv_path = os.path.join(OUTPUT_DIR, f"recommendations_{model_name}.csv")
    recs_df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Recommendations → {csv_path}")

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Summary")
    print(f"{'='*55}")
    print(f"  Users analyzed   : {len(sample_users)}")
    print(f"  Success cases    : {len(success_cases)} (precision ≥ 40%)")
    print(f"  Failure cases    : {len(failure_cases)} (precision < 10%)")

    if success_cases:
        print(f"\n  ✅ Best hit — User {success_cases[0][0]}: "
              f"precision={success_cases[0][1]:.2f}")

    if failure_cases:
        print(f"  ❌ Worst miss — User {failure_cases[0][0]}: "
              f"precision={failure_cases[0][1]:.2f}")

    # ---- Similar-movie explainability (if supported) ----------------------
    sample_movie_idx = int(train["movie_idx"].value_counts().index[0])
    similar = rec.similar_movies(sample_movie_idx, top_k=5)
    if not similar.empty:
        movie_id = idx_to_movie.get(sample_movie_idx)
        movie_title = movies.set_index("movie_id").loc[movie_id, "title"] if movie_id else "Unknown"
        print(f"\n  🔍 Movies similar to '{movie_title}':")
        for _, row in similar.iterrows():
            print(f"    {row['rank']}. {row['title']} ({row['year']}) — sim={row['similarity']:.4f}")

    print("\n✅ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="svd",
                        choices=["user_cf", "item_cf", "svd", "als", "ncf"],
                        help="Which model to use (default: svd)")
    parser.add_argument("--n_users", type=int, default=5,
                        help="Number of sample users (default: 5)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of recommendations per user (default: 10)")
    args = parser.parse_args()
    main(model_name=args.model, n_users=args.n_users, top_k=args.top_k)
