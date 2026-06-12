# 🎬 Netflix Prize — Personalized Recommendation System

A full recommendation system built on the Netflix Prize Dataset, covering EDA, collaborative filtering, matrix factorization, neural collaborative filtering, and evaluation (RMSE + MAP@10).

---

## 📁 Project Structure

```
netflix-recsys/
├── data/                        # Place raw Netflix Prize files here
│   └── README.md
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_Collaborative_Filtering.ipynb
│   ├── 03_Matrix_Factorization.ipynb
│   ├── 04_Neural_CF.ipynb
│   └── 05_Evaluation_Dashboard.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py            # Data loading & parsing
│   │   └── preprocessor.py     # Cleaning, splitting, sampling
│   ├── models/
│   │   ├── user_cf.py           # User-Based Collaborative Filtering
│   │   ├── item_cf.py           # Item-Based Collaborative Filtering
│   │   ├── svd_model.py         # SVD / Matrix Factorization
│   │   ├── als_model.py         # ALS Model
│   │   └── ncf_model.py         # Neural Collaborative Filtering
│   ├── evaluation/
│   │   ├── metrics.py           # RMSE, MAE, MAP@K, NDCG, etc.
│   │   └── evaluator.py         # Full evaluation pipeline
│   └── utils/
│       ├── recommender.py       # Top-K recommendation generator
│       └── visualizer.py        # Plotting helpers
├── scripts/
│   ├── run_eda.py               # Run full EDA and save plots
│   ├── train_models.py          # Train all models
│   ├── evaluate_models.py       # Run evaluation suite
│   └── generate_recommendations.py  # Generate Top-10 for sample users
├── outputs/                     # Saved models, plots, results CSVs
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone / unzip project
cd netflix-recsys

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
# Place the files inside: data/
#   combined_data_1.txt, combined_data_2.txt, combined_data_3.txt, combined_data_4.txt
#   movie_titles.csv

# 4. Run EDA
python scripts/run_eda.py

# 5. Train all models
python scripts/train_models.py

# 6. Evaluate
python scripts/evaluate_models.py

# 7. Generate Top-10 recommendations
python scripts/generate_recommendations.py
```

---

## 📊 Models Compared

| Model | RMSE | MAP@10 | Notes |
|---|---|---|---|
| User-Based CF | ~1.05 | ~0.12 | Simple baseline |
| Item-Based CF | ~0.98 | ~0.15 | Better for sparse data |
| SVD (Matrix Factorization) | ~0.91 | ~0.20 | Strong baseline |
| ALS | ~0.93 | ~0.18 | Scalable |
| Neural CF | ~0.89 | ~0.22 | Best quality |

> Actual values depend on subset size and hyperparameters.

---

## 📏 Evaluation Protocol

- **Train/Test Split**: 80/20 random split, stratified by user
- **Relevance Threshold**: Rating ≥ 3.5 is considered relevant
- **RMSE**: Measured on held-out (user, movie, rating) triples
- **MAP@10**: For each test user, generate Top-10 recommendations and compute average precision against relevant items

---

## 🗓️ 5-Day Plan

| Day | Focus |
|---|---|
| Day 1 | Setup, data loading, EDA |
| Day 2 | User-CF + Item-CF + RMSE baseline |
| Day 3 | SVD + ALS + MAP@10 evaluation |
| Day 4 | Neural CF + full model comparison |
| Day 5 | Report + slides + cleanup |
