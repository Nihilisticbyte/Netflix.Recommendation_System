# 🎬 Netflix Prize — Personalized Recommendation System

An end-to-end recommendation system built on the **Netflix Prize dataset** (~100M ratings,
480K users, 17K movies). It covers the full pipeline: parsing the raw data, EDA, five
recommendation models spanning three families, and a unified evaluation suite reporting
both rating-prediction (RMSE/MAE) and ranking (MAP@10, NDCG@10, etc.) metrics.

The work is driven by four standalone scripts that share a reusable `src/` library — no
notebooks required.

> Built for the **AIML Open Project — "Recommendation Systems for Personalized Content
> Discovery."** See [`AIML Open Project 1.pdf`](./AIML%20Open%20Project%201.pdf) for the
> full brief and [`Personalized-Content-Discovery-at-Scale.pdf`](./Personalized-Content-Discovery-at-Scale.pdf)
> for the technical report.

---

## 📑 Contents

- [Background & Motivation](#-background--motivation)
- [Dataset](#-dataset)
- [Models](#-models)
- [Project Structure](#-project-structure)
- [Setup](#️-setup)
- [Usage](#-usage)
- [Pipeline Details](#-pipeline-details)
- [Metrics](#-metrics)
- [Results & Findings](#-results--findings)
- [Engineering Challenges & Fixes](#-engineering-challenges--fixes)
- [Design Decisions](#-design-decisions)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Outputs](#-outputs)
- [Documentation](#-documentation)

---

## 🎯 Background & Motivation

Recommendation systems are among the highest-impact applications of ML — they drive content
discovery, engagement, and retention on Netflix, Amazon, Spotify, YouTube, and more. The
**Netflix Prize** (2006–2009) was a landmark public competition to improve Netflix's movie
recommender, and its dataset remains a standard benchmark for collaborative filtering,
latent-factor modeling, ranking, and cold-start research.

This project steps into the role of an ML engineer on a streaming platform: given historical
user–item interactions, build a system that **learns user preferences**, **predicts ratings
for unseen content**, **generates personalized Top-K recommendations**, and **surfaces
item-to-item similarities** — then rigorously compares approaches across rating-prediction
and ranking objectives.

---

## 📦 Dataset

[**Netflix Prize Dataset**](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) —
one of the most studied datasets in recommender-system research.

| Property | Value |
|----------|-------|
| Ratings | **100,480,507** |
| Users | **480,189** |
| Movies | **17,770** |
| Rating scale | 1–5 stars (integer) |
| Per-rating fields | user ID, movie ID, rating, date |
| Metadata | movie ID, title, release year (`movie_titles.csv`) |

**Extreme sparsity is the core challenge.** Each user has rated well under 0.01% of the
catalogue, so the user–item matrix is >99.99% empty. This is exactly why memory-based
neighborhood methods struggle at full scale and why latent-factor models become attractive
(see [Engineering Challenges](#-engineering-challenges--fixes)). The raw `combined_data_*.txt`
files use a block format where a `movie_id:` header line is followed by its `user,rating,date`
rows — parsed by `src/data/loader.py`.

---

## 🧠 Models

| Model | Family | Implementation | Key hyperparameters |
|-------|--------|----------------|---------------------|
| **User-Based CF** | Memory-based | cosine user–user similarity, mean-centered weighted KNN | `k_neighbors=50` |
| **Item-Based CF** | Memory-based | cosine item–item similarity, weighted KNN over rated items | `k_neighbors=50` |
| **SVD** | Matrix factorization | Funk SVD via [`surprise`](https://surpriselib.com/) (biased, SGD) | `n_factors=100, n_epochs=20, lr=0.005, reg=0.02` |
| **ALS** | Matrix factorization | Alternating Least Squares via [`implicit`](https://github.com/benfred/implicit) (numpy fallback if unavailable) | `factors=100, iterations=20, reg=0.01, alpha=40` |
| **NCF** | Deep learning | NeuMF (GMF ⊕ MLP fusion) in PyTorch, He et al. 2017 | `gmf_dim=32, mlp=[64,32,16], lr=1e-3, epochs=10` |

> **Note on CF at scale.** User-CF and Item-CF are *trained* and saved, but **excluded from
> the evaluation suite**: their O(n²) similarity computation and per-pair prediction are
> infeasible at Netflix scale. Only **SVD, ALS, and NCF** are scored in
> `evaluate_models.py`. The CF `.pkl` files remain in `outputs/` for inspection and for
> generating recommendations on the sampled subset.

---

## 📁 Project Structure

```
Netflix.Recommendation_System/
├── data/                              # Raw Netflix Prize files go here (gitignored)
│   ├── combined_data_1..4.txt
│   └── movie_titles.csv
├── src/
│   ├── data/
│   │   ├── loader.py                  # Parse combined_data_X.txt → DataFrame, load titles
│   │   └── preprocessor.py            # Iterative active-user filter, ID encoding, splits, sparsity
│   ├── models/
│   │   ├── user_cf.py                 # User-Based Collaborative Filtering
│   │   ├── item_cf.py                 # Item-Based Collaborative Filtering
│   │   ├── svd_model.py               # SVD / Matrix Factorization (surprise)
│   │   ├── als_model.py               # ALS (implicit, with numpy fallback)
│   │   └── ncf_model.py               # Neural Collaborative Filtering (NeuMF, PyTorch)
│   ├── evaluation/
│   │   ├── metrics.py                 # RMSE, MAE, MAP@K, Precision/Recall@K, NDCG@K, HitRate, Coverage
│   │   └── evaluator.py               # Orchestrates RMSE + Top-K ranking evaluation per model
│   └── utils/
│       ├── recommender.py            # Top-K generation, history, hit/miss analysis, similar-items
│       └── visualizer.py             # EDA plots + model-comparison / error plots
├── scripts/
│   ├── run_eda.py                     # Exploratory data analysis + plots
│   ├── train_models.py               # Train models, save to outputs/
│   ├── evaluate_models.py            # Score SVD/ALS/NCF, write comparison CSV + plots
│   └── generate_recommendations.py   # Top-10 recs for sample users + success/failure cases
├── outputs/                           # Models, plots, CSVs (created at runtime)
├── Personalized-Content-Discovery-at-Scale.pdf   # Project report
├── AIML Open Project 1.pdf                        # Problem statement
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo and enter it
git clone <repo-url>
cd Netflix.Recommendation_System

# 2. Install dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# 3. Download the dataset
#    https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
#    Place these files inside data/ :
#      combined_data_1.txt, combined_data_2.txt, combined_data_3.txt, combined_data_4.txt
#      movie_titles.csv
```

> `data/` is gitignored — the dataset is ~2 GB and not committed.
> `scikit-surprise` requires a C/C++ build toolchain; if `implicit` fails to install,
> `ALSModel` automatically falls back to a pure-numpy ALS.

---

## 🚀 Usage

All scripts default to a **5% random sample** of the data (`--sample 0.05`) so the full
pipeline runs on a laptop in minutes. Pass `--sample 1.0` to use everything.

```bash
# 1. Exploratory data analysis → plots in outputs/
python scripts/run_eda.py --sample 0.05

# 2. Train models → outputs/*.pkl  (+ preprocessed_data.pkl, movies.pkl)
python scripts/train_models.py --sample 0.05 --model all
#   --model choices: all | cf | ucf | icf | svd | als | ncf

# 3. Evaluate SVD/ALS/NCF → outputs/model_comparison.csv + comparison/error plots
python scripts/evaluate_models.py

# 4. Generate Top-10 recommendations for sample users
python scripts/generate_recommendations.py --model svd --n_users 5 --top_k 10
#   --model choices: user_cf | item_cf | svd | als | ncf
```

Run `train_models.py` **before** `evaluate_models.py` / `generate_recommendations.py` —
they load the saved models and the cached `preprocessed_data.pkl`.

---

## 🔬 Pipeline Details

**Loading** (`src/data/loader.py`) — parses the Netflix `<movie_id>:` block format into a
flat `(movie_id, user_id, rating, date)` table and reads `movie_titles.csv` (latin-1).

**Preprocessing** (`src/data/preprocessor.py`):
- **Active filtering** — iteratively drops users with `< 20` ratings and movies with
  `< 10` ratings (repeated 3× since removing one sparsifies the other).
- **ID encoding** — maps raw IDs to contiguous `user_idx` / `movie_idx` for matrix ops.
- **Splitting** — `random` (default), `temporal`, or `leave_one_out`.
- **Sparsity** reporting on the filtered matrix.

**Evaluation** (`src/evaluation/`) — for each model:
- **RMSE / MAE** on the full test set.
- **Ranking metrics** (MAP@10, Precision@10, Recall@10, NDCG@10, HitRate@10, Coverage) by
  generating Top-10 recs for a sample of 500 test users, with a relevance threshold of
  `rating ≥ 3.5` and train-seen movies excluded from candidates.

---

## 📊 Metrics

| Metric | What it measures |
|--------|------------------|
| **RMSE / MAE** | Rating-prediction accuracy (lower is better) |
| **MAP@10** | Ranking quality across the Top-10 (primary ranking metric) |
| **Precision@10 / Recall@10** | Relevant fraction of recs / relevant items recovered |
| **NDCG@10** | Position-weighted ranking quality |
| **HitRate@10** | Share of users with ≥1 relevant item in Top-10 |
| **Coverage** | Fraction of the catalogue ever recommended |

Results are written to `outputs/model_comparison.csv` and visualized in
`outputs/model_comparison.png`.

> **Why two metric families?** RMSE rewards *accurate rating prediction*, while MAP@10 rewards
> *good ranking of what to show next*. A model can predict ratings well yet rank poorly (and
> vice-versa) — reporting both exposes that trade-off, which is the heart of the problem.

---

## 📈 Results & Findings

Representative results from the reported run on the sampled configuration (reproducible via the
scripts above). `↓` = lower is better, `↑` = higher is better; **bold** = best in row.

| Metric | SVD | ALS | NCF |
|--------|-----|-----|-----|
| RMSE ↓ | 0.934 | 1.356 | **0.917** |
| MAP@10 ↑ | 0.0021 | **0.0066** | 0.0011 |
| NDCG@10 ↑ | 0.0036 | **0.0115** | 0.0031 |
| HitRate@10 ↑ | 0.0140 | **0.0420** | 0.0220 |
| Coverage ↑ | 0.0384 | **0.0998** | 0.0102 |

**Takeaways:**
- **NCF wins on rating accuracy** (RMSE 0.917), with SVD close behind (0.934).
- **ALS dominates every ranking metric** — MAP@10, NDCG@10, HitRate@10, and Coverage — making
  it the strongest Top-K recommender despite the weakest RMSE. Low RMSE does **not** imply good
  ranking.
- No single model wins both objectives, which is why the suite reports them side by side.

### The Unobserved Feedback Problem

Several Top-10 lists scored ~0% exact-match precision against the held-out test split yet, on
inspection, recommended genuinely on-taste titles (e.g. surfacing acclaimed dramas to a user
whose history centers on similar acclaimed dramas). Under >99.99% sparsity, a model is
*penalized* for recommending relevant items the user simply never happened to rate inside the
test window. A valid global MAP@10 confirms the model is statistically sound, while these
qualitative "misses" show it generalizes to latent taste clusters rather than memorizing
popularity. **In production, recommendations like these would be validated with online A/B
testing — not strict offline exact-match precision.** `generate_recommendations.py` prints
these success/failure cases so the behavior is inspectable.

---

## 🛠 Engineering Challenges & Fixes

Real obstacles hit while building this, and how they were resolved:

| Challenge | Root cause | Fix |
|-----------|-----------|-----|
| **O(n²) similarity blowup** | User-CF / Item-CF need billions of pairwise comparisons at full scale | Pivoted evaluation to latent-factor models (SVD/ALS/NCF) that run in O(n·k); CF kept for the sampled subset only |
| **SVD scored 0 on ranking** | Integer IDs silently coerced to strings forced global-average predictions | Enforced explicit integer `user_idx`/`movie_idx` dtypes throughout the pipeline |
| **ALS RMSE looked broken** | `implicit` ALS emits *unbounded* confidence scores, not 1–5 ratings | Added a Min-Max scaling layer (`ALSModel.predict`) to normalize scores into the valid star range before RMSE |
| **CF prediction too slow** | Naive per-row `iterrows` over 500k+ test rows | Grouped predictions by user so each similarity row is fetched once (~100× faster) in `user_cf.py` / `item_cf.py` |
| **Iterative cold-start filtering** | Removing sparse users re-sparsifies movies and vice-versa | Repeat the active-user/active-movie filter 3× until stable (`filter_active`) |

---

## 🧩 Design Decisions

- **Three complementary model families, not one.** SVD (latent factors, RMSE-optimized), ALS
  (implicit-feedback, scalable, ranking-optimized), and NCF (neural, captures *non-linear*
  user–item interactions via an MLP instead of a plain dot product). Together they cover the
  accuracy-vs-ranking spectrum.
- **Sampling by default (`--sample 0.05`).** The full matrix is ~100M ratings; a 5% sample keeps
  the whole pipeline laptop-friendly while preserving the sparsity characteristics that matter.
  Flip to `--sample 1.0` when compute allows.
- **Relevance threshold of 3.5.** Per the problem brief, a movie counts as relevant for ranking
  metrics only if its actual rating is **≥ 3.5** — distinguishing "great" from merely "watched."
- **Configurable splits.** `random` (default), `temporal` (most recent ratings held out, the
  most production-realistic), and `leave_one_out` are all supported in `split_train_test`.
- **Uniform model interface.** Every model exposes `fit` / `predict` / `recommend`, so the
  `Evaluator` and `Recommender` treat them interchangeably — adding a new model is a drop-in.

---

## 🧰 Tech Stack

- **Core:** Python 3.11, NumPy, pandas, SciPy (sparse matrices)
- **Classical ML:** scikit-learn (cosine similarity), [scikit-surprise](https://surpriselib.com/) (SVD)
- **Scalable MF:** [implicit](https://github.com/benfred/implicit) (Cython ALS, numpy fallback)
- **Deep learning:** PyTorch (NeuMF)
- **Viz & utils:** matplotlib, seaborn, tqdm, joblib

---

## 🔮 Future Improvements

- **Hybrid model** — blend latent-factor scores with content metadata (genre/year) to ease
  cold-start for new users and items.
- **Cold-start strategy** — popularity/metadata fallbacks for users and movies with little history.
- **Bias & temporal dynamics** — model rating drift over time and per-user/per-item biases more explicitly.
- **Interactive dashboard / API** — a Streamlit or FastAPI front-end to browse recommendations,
  similar titles, and score breakdowns (scaffolded but optional in `requirements.txt`).
- **Online evaluation** — A/B testing to validate the recommendations that offline exact-match
  metrics unfairly penalize.

---

## 📦 Outputs

After a full run, `outputs/` contains:

- `*.pkl` — trained models (`user_cf`, `item_cf`, `svd`, `als`, `ncf`), plus
  `preprocessed_data.pkl` and `movies.pkl`.
- `model_comparison.csv` — side-by-side metrics for SVD/ALS/NCF.
- `recommendations_<model>.csv` — Top-10 recs with hit/miss flags for sampled users.
- EDA + comparison + per-model error plots (`.png`).

---

## ✅ Assignment Coverage

How this repo maps to the mandatory tasks in the brief:

| Required task | Where |
|---------------|-------|
| Exploratory Data Analysis | `scripts/run_eda.py`, `src/utils/visualizer.py` |
| Recommendation model development | `src/models/*` (5 models, 3 families) |
| Model comparison | `scripts/evaluate_models.py`, `src/evaluation/evaluator.py` |
| Top-K recommendation generation + success/failure analysis | `scripts/generate_recommendations.py`, `src/utils/recommender.py` |
| Evaluation with **RMSE** & **MAP@10** (relevance ≥ 3.5) | `src/evaluation/metrics.py` |
| Explainable / similar-item recommendations *(optional)* | `Recommender.similar_movies` |
| Reproducible, documented repo | this README + the scripts |

---

## 📄 Documentation

- **[`Personalized-Content-Discovery-at-Scale.pdf`](./Personalized-Content-Discovery-at-Scale.pdf)** — full technical report & presentation (problem, EDA, methodology, results, insights).
- **[`AIML Open Project 1.pdf`](./AIML%20Open%20Project%201.pdf)** — original problem statement, deliverables, and grading criteria.
- **`Netflix_RecSys_Project_Explanation (1).docx`** — written project explanation.
