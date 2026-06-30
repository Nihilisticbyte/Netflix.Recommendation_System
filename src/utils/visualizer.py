"""
src/utils/visualizer.py
-----------------------
Plotting helpers for EDA and model evaluation results.
All functions save to a specified output directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def _savefig(fig, name: str, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot saved] {path}")


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_rating_distribution(df: pd.DataFrame, save_dir: str = SAVE_DIR):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["rating"].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d")
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    _savefig(fig, "eda_rating_distribution.png", save_dir)


def plot_ratings_per_user(df: pd.DataFrame, save_dir: str = SAVE_DIR):
    counts = df.groupby("user_id").size()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(counts, bins=100, color="#4C72B0", edgecolor="white", log=True)
    axes[0].set_title("Ratings per User (log scale)", fontweight="bold")
    axes[0].set_xlabel("Number of Ratings")
    axes[0].set_ylabel("User Count (log)")

    axes[1].hist(counts[counts < counts.quantile(0.99)], bins=80,
                 color="#DD8452", edgecolor="white")
    axes[1].set_title("Ratings per User (99th percentile clipped)", fontweight="bold")
    axes[1].set_xlabel("Number of Ratings")

    fig.suptitle("User Activity Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "eda_ratings_per_user.png", save_dir)


def plot_ratings_per_movie(df: pd.DataFrame, save_dir: str = SAVE_DIR):
    counts = df.groupby("movie_id").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts, bins=100, color="#55A868", edgecolor="white", log=True)
    ax.set_title("Ratings per Movie (log scale)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Movie Count (log)")
    _savefig(fig, "eda_ratings_per_movie.png", save_dir)


def plot_top_movies(
    df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 20,
    save_dir: str = SAVE_DIR,
):
    merged = df.merge(movies_df[["movie_id", "title"]], on="movie_id")
    top = (
        merged.groupby("title")
        .agg(count=("rating", "count"), mean_rating=("rating", "mean"))
        .nlargest(n, "count")
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Most rated
    sns.barplot(data=top, x="count", y="title", ax=axes[0], palette="Blues_r")
    axes[0].set_title(f"Top {n} Most Rated Movies", fontweight="bold")
    axes[0].set_xlabel("Rating Count")
    axes[0].set_ylabel("")

    # Highest avg rating (min 1000 ratings)
    top_avg = (
        merged.groupby("title")
        .agg(count=("rating", "count"), mean_rating=("rating", "mean"))
        .query("count >= 1000")
        .nlargest(n, "mean_rating")
        .reset_index()
    )
    sns.barplot(data=top_avg, x="mean_rating", y="title", ax=axes[1], palette="Greens_r")
    axes[1].set_title(f"Top {n} Highest Rated Movies (≥1000 ratings)", fontweight="bold")
    axes[1].set_xlabel("Average Rating")
    axes[1].set_xlim(3.5, 5)
    axes[1].set_ylabel("")

    plt.tight_layout()
    _savefig(fig, "eda_top_movies.png", save_dir)


def plot_ratings_over_time(df: pd.DataFrame, save_dir: str = SAVE_DIR):
    if "date" not in df.columns:
        return
    df2 = df.copy()
    df2["year_month"] = df2["date"].dt.to_period("M")
    monthly = df2.groupby("year_month").size().reset_index(name="count")
    monthly["year_month"] = monthly["year_month"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["year_month"], monthly["count"], color="#4C72B0", linewidth=1.5)
    ax.fill_between(range(len(monthly)), monthly["count"], alpha=0.2, color="#4C72B0")
    ax.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    ax.set_xticklabels(monthly["year_month"].iloc[::max(1, len(monthly)//12)], rotation=45)
    ax.set_title("Monthly Rating Volume Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Ratings")
    plt.tight_layout()
    _savefig(fig, "eda_ratings_over_time.png", save_dir)


def plot_sparsity_heatmap(
    df: pd.DataFrame,
    sample_users: int = 100,
    sample_movies: int = 200,
    save_dir: str = SAVE_DIR,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    sample_u = rng.choice(df["user_idx"].unique(), size=min(sample_users, df["user_idx"].nunique()), replace=False)
    sample_m = rng.choice(df["movie_idx"].unique(), size=min(sample_movies, df["movie_idx"].nunique()), replace=False)

    sub = df[df["user_idx"].isin(sample_u) & df["movie_idx"].isin(sample_m)]
    matrix = sub.pivot_table(index="user_idx", columns="movie_idx", values="rating")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        matrix.notna().astype(int),
        ax=ax, cbar=False, cmap="Blues",
        xticklabels=False, yticklabels=False,
    )
    ax.set_title(
        f"Rating Matrix Sparsity (sample: {sample_users} users × {sample_movies} movies)\n"
        f"Blue = rated, White = unrated",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Movies")
    ax.set_ylabel("Users")
    _savefig(fig, "eda_sparsity_heatmap.png", save_dir)


# ---------------------------------------------------------------------------
# Model comparison plots
# ---------------------------------------------------------------------------

def plot_model_comparison(results_df: pd.DataFrame, save_dir: str = SAVE_DIR):
    metrics = ["RMSE", "MAP@10", "NDCG@10", "HitRate@10", "Coverage"]
    metrics = [m for m in metrics if m in results_df.columns]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", len(results_df))

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(results_df["Model"], results_df[metric], color=colors)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "model_comparison.png", save_dir)


def plot_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_dir: str = SAVE_DIR,
):
    errors = np.array(y_pred) - np.array(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(errors, bins=60, color="#4C72B0", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title(f"{model_name}: Prediction Error Distribution", fontweight="bold")
    axes[0].set_xlabel("Predicted − Actual")
    axes[0].set_ylabel("Count")

    axes[1].scatter(y_true, y_pred, alpha=0.1, s=5, color="#4C72B0")
    axes[1].plot([1, 5], [1, 5], "r--")
    axes[1].set_title(f"{model_name}: Predicted vs Actual", fontweight="bold")
    axes[1].set_xlabel("Actual Rating")
    axes[1].set_ylabel("Predicted Rating")
    axes[1].set_xlim(0.5, 5.5)
    axes[1].set_ylim(0.5, 5.5)

    plt.tight_layout()
    _savefig(fig, f"error_{model_name.lower().replace(' ', '_')}.png", save_dir)
