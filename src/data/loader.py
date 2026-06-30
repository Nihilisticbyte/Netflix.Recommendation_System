"""
src/data/loader.py
------------------
Loads and parses the Netflix Prize Dataset.

Raw format of combined_data_X.txt:
    <movie_id>:
    <user_id>,<rating>,<date>
    ...
"""

import os
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

COMBINED_FILES = [
    "combined_data_1.txt",
    "combined_data_2.txt",
    "combined_data_3.txt",
    "combined_data_4.txt",
]
MOVIE_TITLES_FILE = "movie_titles.csv"


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def parse_combined_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single combined_data_X.txt file into a DataFrame.

    Returns columns: movie_id, user_id, rating, date
    """
    records = []
    current_movie_id = None

    with open(filepath, "r") as f:
        for line in tqdm(f, desc=f"Parsing {os.path.basename(filepath)}"):
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_movie_id = int(line[:-1])
            else:
                parts = line.split(",")
                user_id = int(parts[0])
                rating = float(parts[1])
                date = parts[2]
                records.append((current_movie_id, user_id, rating, date))

    df = pd.DataFrame(records, columns=["movie_id", "user_id", "rating", "date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_ratings(
    data_dir: str = DATA_DIR,
    files: list = None,
    sample_fraction: float = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load all (or a subset of) combined rating files.

    Parameters
    ----------
    data_dir      : folder containing combined_data_X.txt files
    files         : list of filenames to load (default: all 4)
    sample_fraction : if set (0 < f <= 1), randomly sample this fraction
    random_state  : seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: movie_id, user_id, rating, date
    """
    if files is None:
        files = COMBINED_FILES

    dfs = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARNING] File not found, skipping: {fpath}")
            continue
        dfs.append(parse_combined_file(fpath))

    if not dfs:
        raise FileNotFoundError(
            f"No data files found in {data_dir}. "
            "Download from: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data"
        )

    df = pd.concat(dfs, ignore_index=True)

    if sample_fraction is not None and 0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_state).reset_index(drop=True)
        print(f"[INFO] Sampled {len(df):,} ratings ({sample_fraction*100:.0f}% of data)")
    else:
        print(f"[INFO] Loaded {len(df):,} ratings total")

    return df


def load_movie_titles(data_dir: str = DATA_DIR) -> pd.DataFrame:
    fpath = os.path.join(data_dir, MOVIE_TITLES_FILE)

    records = []
    with open(fpath, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split(",", 2)   # only split on first 2 commas
            if len(parts) == 3:
                movie_id, year, title = parts
            elif len(parts) == 2:
                movie_id, year = parts
                title = ""
            else:
                continue
            records.append((movie_id.strip(), year.strip(), title.strip()))

    movies = pd.DataFrame(records, columns=["movie_id", "year", "title"])
    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce")
    movies["year"]     = pd.to_numeric(movies["year"],     errors="coerce")
    movies = movies.dropna(subset=["movie_id"]).reset_index(drop=True)
    movies["movie_id"] = movies["movie_id"].astype(int)
    print(f"[INFO] Loaded {len(movies):,} movie titles")
    return movies


def load_all(
    data_dir: str = DATA_DIR,
    sample_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: load ratings + movie titles.

    Returns (ratings_df, movies_df)
    """
    ratings = load_ratings(data_dir, sample_fraction=sample_fraction)
    movies = load_movie_titles(data_dir)
    return ratings, movies
