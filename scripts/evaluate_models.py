"""
scripts/evaluate_models.py
---------------------------
Load trained models and run full evaluation.
- SVD / ALS / NCF : RMSE on full test set, Top-K on 500 users

Note: User-CF and Item-CF were trained but excluded from evaluation
due to O(n^2) complexity being infeasible at Netflix scale (~100M ratings).
Their trained .pkl files remain in outputs/ for reference.

Usage:
    python scripts/evaluate_models.py
"""

import sys
import os
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.evaluator import Evaluator
from src.utils.visualizer import plot_model_comparison, plot_prediction_error

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_model(name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        print(f"  [SKIP] {name}.pkl not found")
        return None
    print(f"  [Loading] {name} ...")
    model = joblib.load(path)
    print(f"  [Loaded]  {name}")
    return model


def main():
    print("=" * 60)
    print("  Netflix Recommendation System - Evaluation")
    print("=" * 60)

    data_path = os.path.join(OUTPUT_DIR, "preprocessed_data.pkl")
    if not os.path.exists(data_path):
        print("[ERROR] Run train_models.py first.")
        sys.exit(1)

    data     = joblib.load(data_path)
    train    = data["train"]
    test     = data["test"]
    n_movies = data["n_movies"]

    print(f"\n  Train: {len(train):,} | Test: {len(test):,}")
    print(f"  Users: {data['n_users']:,} | Movies: {n_movies:,}\n")
    print("  [INFO] User-CF and Item-CF skipped — O(n^2) infeasible at Netflix scale.\n")

    # Full test set evaluation for SVD / ALS / NCF
    ev = Evaluator(
        train_df=train, test_df=test, n_movies=n_movies,
        k=10, relevance_threshold=3.5,
        eval_users_sample=500,
        rmse_sample=None,        # full test set
    )

    # (model_file_name, display_name)
    model_configs = [
        ("svd", "SVD"),
        ("als", "ALS"),
        ("ncf", "NCF"),
    ]

    all_results = []

    for mfile, mname in model_configs:
        model = load_model(mfile)
        if model is None:
            continue

        print(f"\n{'─'*50}")
        print(f"  Evaluating: {mname}")
        print(f"{'─'*50}")

        res, _ = ev.evaluate(model, model_name=mname, use_full_test_for_rmse=True)
        all_results.append(res)

        # Error plots — full test set
        y_pred = model.predict(test)
        y_true = test["rating"].values
        plot_prediction_error(y_true, y_pred, model_name=mname)

    if not all_results:
        print("[ERROR] No models found. Run train_models.py first.")
        return

    comparison = pd.concat(all_results, ignore_index=True)
    csv_path   = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison.to_csv(csv_path, index=False)

    print(f"\n[Saved] {csv_path}")
    print("\n===== FINAL COMPARISON =====")
    print(comparison.to_string(index=False))

    plot_model_comparison(comparison)

    best_rmse = comparison.loc[comparison["RMSE"].idxmin(), "Model"]
    best_map  = comparison.loc[comparison["MAP@10"].idxmax(), "Model"]
    print(f"\n  Best RMSE   : {best_rmse}")
    print(f"  Best MAP@10 : {best_map}")
    print("\nDone. Results saved to outputs/")

if __name__ == "__main__":
    main()
