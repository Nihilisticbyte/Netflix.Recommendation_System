"""
src/data/preprocessor.py
------------------------
Cleaning, filtering, encoding, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_active(
    df: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 10,
) -> pd.DataFrame:
    """
    Remove cold-start users and unpopular movies so models have
    enough signal to learn from.
    """
    before = len(df)

    # Iterative filter (removing users can make movies sparse and vice-versa)
    for _ in range(3):
        user_counts = df["user_id"].value_counts()
        movie_counts = df["movie_id"].value_counts()
        df = df[
            df["user_id"].isin(user_counts[user_counts >= min_user_ratings].index) &
            df["movie_id"].isin(movie_counts[movie_counts >= min_movie_ratings].index)
        ]

    after = len(df)
    print(f"[INFO] Filter active: {before:,} → {after:,} ratings "
          f"({after/before*100:.1f}% kept)")
    print(f"       Users: {df['user_id'].nunique():,} | Movies: {df['movie_id'].nunique():,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Integer encoding (maps raw IDs → 0-based indices for matrix ops)
# ---------------------------------------------------------------------------

def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Encode user_id and movie_id to contiguous integer indices.

    Returns
    -------
    df_encoded  : DataFrame with added columns user_idx, movie_idx
    user_map    : {user_id  → user_idx}
    movie_map   : {movie_id → movie_idx}
    """
    users = sorted(df["user_id"].unique())
    movies = sorted(df["movie_id"].unique())

    user_map = {u: i for i, u in enumerate(users)}
    movie_map = {m: i for i, m in enumerate(movies)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["movie_idx"] = df["movie_id"].map(movie_map)

    print(f"[INFO] Encoded {len(user_map):,} users × {len(movie_map):,} movies")
    return df, user_map, movie_map


# ---------------------------------------------------------------------------
# Train / Test split
# ---------------------------------------------------------------------------

def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    strategy: str = "random",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test sets.

    Parameters
    ----------
    strategy : "random"     – simple random split
               "temporal"   – most-recent 20% of each user goes to test
               "leave_one_out" – last rated item per user → test

    Returns (train_df, test_df)
    """
    if strategy == "random":
        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

    elif strategy == "temporal":
        df = df.sort_values("date")
        cutoff = int(len(df) * (1 - test_size))
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff:]

    elif strategy == "leave_one_out":
        test_idx = (
            df.sort_values("date")
            .groupby("user_id")
            .tail(1)
            .index
        )
        test = df.loc[test_idx]
        train = df.drop(index=test_idx)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"[INFO] Train: {len(train):,} | Test: {len(test):,} | "
          f"Split strategy: {strategy}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------

def compute_sparsity(df: pd.DataFrame) -> float:
    n_users = df["user_id"].nunique()
    n_movies = df["movie_id"].nunique()
    n_ratings = len(df)
    sparsity = 1 - n_ratings / (n_users * n_movies)
    print(f"[INFO] Sparsity: {sparsity * 100:.4f}%  "
          f"({n_ratings:,} / {n_users * n_movies:,} possible ratings filled)")
    return sparsity


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 10,
    test_size: float = 0.2,
    split_strategy: str = "random",
    random_state: int = 42,
) -> dict:
    """
    Run the complete preprocessing pipeline.

    Returns a dict with keys:
        train, test, full, user_map, movie_map,
        n_users, n_movies, sparsity
    """
    df_filtered = filter_active(df, min_user_ratings, min_movie_ratings)
    df_encoded, user_map, movie_map = encode_ids(df_filtered)
    sparsity = compute_sparsity(df_encoded)
    train, test = split_train_test(df_encoded, test_size, random_state, split_strategy)

    return {
        "train": train,
        "test": test,
        "full": df_encoded,
        "user_map": user_map,
        "movie_map": movie_map,
        "n_users": len(user_map),
        "n_movies": len(movie_map),
        "sparsity": sparsity,
    }
