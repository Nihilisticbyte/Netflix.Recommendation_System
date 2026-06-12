"""
scripts/train_models.py
------------------------
Train all recommendation models and save them to outputs/.

Usage:
    python scripts/train_models.py [--sample 0.05] [--model all|svd|ncf|cf]

Models trained:
    1. User-Based Collaborative Filtering
    2. Item-Based Collaborative Filtering
    3. SVD (Matrix Factorization via Surprise)
    4. ALS
    5. Neural Collaborative Filtering (NCF)
"""

import sys
import os
import argparse
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_all
from src.data.preprocessor import preprocess
from src.models.user_cf import UserBasedCF
from src.models.item_cf import ItemBasedCF
from src.models.svd_model import SVDModel
from src.models.als_model import ALSModel
from src.models.ncf_model import NCFModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def save_model(model, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  [Saved] {path}")


def main(sample_fraction: float = 0.05, model_choice: str = "all"):
    print("=" * 60)
    print("  Netflix Recommendation System — Training")
    print("=" * 60)

    # ---- Load & preprocess -----------------------------------------------
    ratings, movies = load_all(sample_fraction=sample_fraction)
    data = preprocess(
        ratings,
        min_user_ratings=20,
        min_movie_ratings=10,
        test_size=0.2,
        split_strategy="random",
    )

    train = data["train"]
    test  = data["test"]

    # Save preprocessed data for evaluation script
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(data, os.path.join(OUTPUT_DIR, "preprocessed_data.pkl"))
    joblib.dump(movies, os.path.join(OUTPUT_DIR, "movies.pkl"))
    print(f"\n[Data saved] outputs/preprocessed_data.pkl")
    

    print(f"\n  Train: {len(train):,}  |  Test: {len(test):,}")
    print(f"  Users: {data['n_users']:,}  |  Movies: {data['n_movies']:,}\n")

    # ---- Train models ----------------------------------------------------
    should_train = lambda name: model_choice == "all" or model_choice == name

    # 1. User-Based CF
    if should_train("ucf") or should_train("cf"):
        print("\n[1/5] User-Based Collaborative Filtering")
        ucf = UserBasedCF(k_neighbors=50)
        ucf.fit(train)
        save_model(ucf, "user_cf")

    # 2. Item-Based CF
    if should_train("icf") or should_train("cf"):
        print("\n[2/5] Item-Based Collaborative Filtering")
        icf = ItemBasedCF(k_neighbors=50)
        icf.fit(train)
        save_model(icf, "item_cf")

    # 3. SVD
    if should_train("svd"):
        print("\n[3/5] SVD (Matrix Factorization)")
        svd = SVDModel(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        svd.fit(train)
        save_model(svd, "svd")

    # 4. ALS
    if should_train("als"):
        print("\n[4/5] ALS (Alternating Least Squares)")
        als = ALSModel(factors=100, iterations=20, regularization=0.01, alpha=40.0)
        als.fit(train)
        save_model(als, "als")

    # 5. NCF
    if should_train("ncf"):
        print("\n[5/5] Neural Collaborative Filtering (NCF)")
        ncf = NCFModel(
            gmf_dim=32,
            mlp_dims=[64, 32, 16],
            lr=1e-3,
            batch_size=1024,
            epochs=10,
        )
        ncf.fit(train)
        save_model(ncf, "ncf")

    print("\n✅ Training complete. Models saved to outputs/")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=0.05,
                        help="Fraction of data to use (default: 0.05)")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "cf", "ucf", "icf", "svd", "als", "ncf"],
                        help="Which model(s) to train")
    args = parser.parse_args()
    main(sample_fraction=args.sample, model_choice=args.model)
