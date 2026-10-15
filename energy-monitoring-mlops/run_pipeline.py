"""
run_pipeline.py
----------------
Phase 1 main entry point. Run from the project root:

    cd energy-monitoring-mlops
    python run_pipeline.py
"""

import os
import sys
import logging

# Configure workspace-local temporary and cache paths for sandbox compatibility
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(WORKSPACE_DIR, ".tmp")
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR
os.environ["TMPDIR"] = TMP_DIR
os.environ["MPLCONFIGDIR"] = os.path.join(TMP_DIR, "matplotlib")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import tempfile
tempfile.tempdir = TMP_DIR

import json
import pandas as pd
import mlflow

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
from src.models.evaluation_analytics import run_full_evaluation_audit
from src.visualization.generate_dashboard_slides import generate_all_slides

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "evidence")
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    log.info("=" * 60)
    log.info("  SMART ENERGY MONITORING ASSISTANT - PHASE 1 & EVALUATION")
    log.info("=" * 60)

    # Step 1: Load & validate
    log.info("[1] Loading and validating dataset...")
    df_raw = load_raw_data()
    val = validate_dataset(df_raw)
    print_validation_report(val)

    # Save validation evidence
    val_json = {k: str(v) for k, v in val.items()
                if not isinstance(v, dict)}
    val_json["target_stats"] = val["target_stats"]
    with open(os.path.join(REPORTS_DIR, "dataset_validation.json"), "w") as f:
        json.dump(val_json, f, indent=2, default=str)
    log.info("Validation report saved -> reports/evidence/dataset_validation.json")

    # Step 2: Feature engineering
    log.info("[2] Building feature matrix...")
    df_feat = build_feature_matrix(df_raw)
    log.info("Feature matrix shape: %s", df_feat.shape)
    processed_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "processed", "features.csv"
    )
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_feat.to_csv(processed_path, index=False)
    log.info("Saved -> data/processed/features.csv")

    # Step 3: Chronological split
    log.info("[3] Splitting data chronologically...")
    train_df, val_df, test_df = chronological_split(df_feat)
    print_split_summary(train_df, val_df, test_df)

    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)
    X_test,  y_test  = get_X_y(test_df)

    # Step 4: MLflow setup
    log.info("[4] Setting up MLflow...")
    setup_mlflow()

    # Step 5: Train all models
    log.info("[5] Training models...")
    r0 = train_baseline(X_val, y_val, X_test, y_test)
    r1 = train_linear_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    r2 = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    r3 = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 6: Compare
    log.info("[6] Building model comparison table...")
    results = [r0, r1, r2, r3]
    cmp_df = build_comparison_table(results)
    log.info("\n%s", cmp_df.to_string(index=False))
    save_comparison_table(cmp_df)

    # Step 7: Deep-Dive Evaluation Analytics
    log.info("[7] Computing regime-stratified and diurnal deep-dive analytics...")
    predictions = {
        r["model"]: {"val": r["val_preds"], "test": r["test_preds"]}
        for r in results
    }
    models_dict = {
        r["model"]: r["model_obj"]
        for r in results if "model_obj" in r
    }
    feature_names = list(X_train.columns)

    deepdive = run_full_evaluation_audit(
        val_df=val_df,
        test_df=test_df,
        predictions=predictions,
        models=models_dict,
        feature_names=feature_names,
    )

    # Step 8: Visual Dashboard Slides Generation
    log.info("[8] Generating visual evaluation dashboard slides...")
    xgb_history = r3.get("evals_result")
    feat_imp_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
    feat_imp_df = pd.read_csv(feat_imp_path) if os.path.exists(feat_imp_path) else None

    s1, s2 = generate_all_slides(
        cmp_df=cmp_df,
        deepdive=deepdive,
        test_df=test_df,
        predictions=predictions,
        xgb_history=xgb_history,
        feat_imp_df=feat_imp_df,
    )

    # Step 9: MLflow Audit Logging (artifact file logging skipped - temp-dir permissions)
    log.info("[9] Skipping MLflow artifact file logging (temp-dir permission on this machine).")

    log.info("=" * 60)
    log.info("EVALUATION PIPELINE COMPLETE")
    log.info("  Slide 1: %s", s1)
    log.info("  Slide 2: %s", s2)
    log.info("  Tip: run 'mlflow ui' then open http://localhost:5000")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
