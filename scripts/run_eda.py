"""
scripts/run_eda.py
------------------
Run full Exploratory Data Analysis and save all plots to outputs/.

Usage:
    python scripts/run_eda.py [--sample 0.05]
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_all
from src.data.preprocessor import compute_sparsity, encode_ids, filter_active
from src.utils.visualizer import (
    plot_rating_distribution,
    plot_ratings_per_user,
    plot_ratings_per_movie,
    plot_top_movies,
    plot_ratings_over_time,
    plot_sparsity_heatmap,
)


def main(sample_fraction: float = 0.05):
    print("=" * 60)
    print("  Netflix Recommendation System — EDA")
    print("=" * 60)

    # ---- Load data -------------------------------------------------------
    ratings, movies = load_all(sample_fraction=sample_fraction)

    # ---- Basic stats -----------------------------------------------------
    print(f"\n📊 Dataset Overview:")
    print(f"   Total ratings   : {len(ratings):,}")
    print(f"   Unique users    : {ratings['user_id'].nunique():,}")
    print(f"   Unique movies   : {ratings['movie_id'].nunique():,}")
    print(f"   Rating range    : {ratings['rating'].min()} – {ratings['rating'].max()}")
    print(f"   Date range      : {ratings['date'].min().date()} → {ratings['date'].max().date()}")

    print(f"\n⭐ Rating Statistics:")
    print(ratings["rating"].describe().to_string())

    # ---- User activity ---------------------------------------------------
    user_counts = ratings["user_id"].value_counts()
    print(f"\n👤 User Activity:")
    print(f"   Median ratings/user : {user_counts.median():.0f}")
    print(f"   Mean ratings/user   : {user_counts.mean():.1f}")
    print(f"   Max ratings/user    : {user_counts.max():,}")
    print(f"   Users with <5 ratings: {(user_counts < 5).sum():,} "
          f"({(user_counts < 5).mean()*100:.1f}%)")

    # ---- Movie popularity ------------------------------------------------
    movie_counts = ratings["movie_id"].value_counts()
    print(f"\n🎬 Movie Popularity:")
    print(f"   Median ratings/movie : {movie_counts.median():.0f}")
    print(f"   Mean ratings/movie   : {movie_counts.mean():.1f}")
    print(f"   Max ratings/movie    : {movie_counts.max():,}")
    print(f"   Movies with <10 ratings: {(movie_counts < 10).sum():,} "
          f"({(movie_counts < 10).mean()*100:.1f}%)")

    # ---- Sparsity --------------------------------------------------------
    df_filtered = filter_active(ratings)
    df_enc, _, _ = encode_ids(df_filtered)
    compute_sparsity(df_enc)

    # ---- Top movies by year (if year available) --------------------------
    merged = ratings.merge(movies[["movie_id", "year"]], on="movie_id", how="left")
    year_avg = merged.groupby("year")["rating"].agg(["mean", "count"]).dropna()
    decade_best = year_avg[year_avg["count"] > 100].nlargest(10, "mean")
    print(f"\n🏆 Best-rated decades (>100 ratings):")
    print(decade_best.to_string())

    # ---- Plots -----------------------------------------------------------
    print("\n📈 Generating plots ...")
    plot_rating_distribution(ratings)
    plot_ratings_per_user(ratings)
    plot_ratings_per_movie(ratings)
    plot_top_movies(ratings, movies)
    plot_ratings_over_time(ratings)
    plot_sparsity_heatmap(df_enc)

    print("\n✅ EDA complete. Plots saved to outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=0.05,
                        help="Fraction of data to load (0–1). Default: 0.05")
    args = parser.parse_args()
    main(sample_fraction=args.sample)
