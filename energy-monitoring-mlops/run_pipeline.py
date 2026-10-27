"""
run_pipeline.py
----------------
Phase 1 main entry point. Run from the project root:

    cd energy-monitoring-mlops
    python run_pipeline.py

To launch the MLflow UI and view runs:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.load_data import load_raw_data, validate_dataset, print_validation_report
from src.features.feature_engineering import (
    build_feature_matrix, chronological_split,
    get_X_y, print_split_summary,
)
from src.models.train import (
    setup_mlflow, train_baseline, train_linear_regression,
    train_random_forest, train_xgboost,
    build_comparison_table, save_comparison_table,
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "evidence")
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    print("\n" + "=" * 60)
    print("  SMART ENERGY MONITORING ASSISTANT - PHASE 1")
    print("=" * 60)

    # Step 1: Load & validate
    print("\n[1] Loading and validating dataset...")
    df_raw = load_raw_data()
    val = validate_dataset(df_raw)
    print_validation_report(val)

    # Save validation evidence
    val_json = {k: str(v) for k, v in val.items()
                if not isinstance(v, dict)}
    val_json["target_stats"] = val["target_stats"]
    with open(os.path.join(REPORTS_DIR, "dataset_validation.json"), "w") as f:
        json.dump(val_json, f, indent=2, default=str)
    print("  Validation report saved -> reports/evidence/dataset_validation.json")

    # Step 2: Feature engineering
    print("\n[2] Building feature matrix...")
    df_feat = build_feature_matrix(df_raw)
    print(f"  Feature matrix shape : {df_feat.shape}")
    processed_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "processed", "features.csv"
    )
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_feat.to_csv(processed_path, index=False)
    print("  Saved -> data/processed/features.csv")

    # Step 3: Chronological split
    print("\n[3] Splitting data chronologically...")
    train_df, val_df, test_df = chronological_split(df_feat)
    print_split_summary(train_df, val_df, test_df)

    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)
    X_test,  y_test  = get_X_y(test_df)

    # Step 4: MLflow setup
    print("\n[4] Setting up MLflow...")
    setup_mlflow()

    # Step 5: Train all models
    print("\n[5] Training models...")
    r0 = train_baseline(X_val, y_val, X_test, y_test)
    r1 = train_linear_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    r2 = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    r3 = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 6: Compare
    print("\n[6] Model comparison:")
    results = [r0, r1, r2, r3]
    cmp_df = build_comparison_table(results)
    print("\n" + cmp_df.to_string(index=False))
    save_comparison_table(cmp_df)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("  Run: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("  Then open: http://127.0.0.1:5000")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
