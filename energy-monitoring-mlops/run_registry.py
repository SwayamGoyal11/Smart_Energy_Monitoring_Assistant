"""
run_registry.py
----------------
Phase 2 entry point: train candidate models, evaluate release quality gates,
and register them in the MLflow Model Registry. Run from the project root:

    python run_registry.py                                   # hurdle + linear_regression
    python run_registry.py --models hurdle xgboost

Candidates passing the gates get the 'staging' alias; the best by validation
RMSE also gets 'champion'. View the registry with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from src.data.load_data import load_raw_data  # noqa: E402
from src.features.feature_engineering import (  # noqa: E402
    build_feature_matrix, chronological_split, get_X_y,
)
from src.models.hurdle import HurdleModel  # noqa: E402
from src.models.registry import REGISTERED_MODEL, register_candidate, tracking_uri  # noqa: E402
from src.validation.input_validation import validate_feature_frame  # noqa: E402

CANDIDATES = {
    "hurdle": (HurdleModel, {"active_threshold_wh": 100.0, "decision_threshold": 0.5}),
    "linear_regression": (LinearRegression, {}),
    "random_forest": (RandomForestRegressor, {
        "n_estimators": 100, "max_depth": 15, "min_samples_leaf": 4,
        "random_state": 42, "n_jobs": -1,
    }),
    "xgboost": (XGBRegressor, {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "n_jobs": -1, "tree_method": "hist",
    }),
}


def main():
    parser = argparse.ArgumentParser(description="Train, gate and register candidate models.")
    parser.add_argument(
        "--models", nargs="+", choices=list(CANDIDATES), default=["hurdle", "linear_regression"],
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(tracking_uri())
    print(f"  MLflow tracking URI : {mlflow.get_tracking_uri()}")
    print(f"  Registered model    : {REGISTERED_MODEL}")

    train_df, val_df, test_df = chronological_split(build_feature_matrix(load_raw_data()))
    X_train, y_train = get_X_y(train_df)
    X_train = validate_feature_frame(X_train)

    results = []
    for name in args.models:
        cls, params = CANDIDATES[name]
        print(f"\n  Training {name}...")
        model = cls(**params).fit(X_train, y_train)
        result = register_candidate(model, name, val_df, test_df, params=params)
        results.append(result)

        status = "PASSED" if result.gates_passed else "FAILED"
        print(f"    version {result.version} | gates {status} | aliases {result.aliases or '-'}")
        print(f"    val  MAE {result.metrics['val_mae']:.2f}  RMSE {result.metrics['val_rmse']:.2f}"
              f"  R2 {result.metrics['val_r2']:.3f}")
        print(f"    test MAE {result.metrics['test_mae']:.2f}  RMSE {result.metrics['test_rmse']:.2f}"
              f"  R2 {result.metrics['test_r2']:.3f}")
        for failure in result.gate_failures:
            print(f"    - {failure}")

    client = mlflow.MlflowClient()
    aliases = client.get_registered_model(REGISTERED_MODEL).aliases
    print("\n  Registry aliases now:")
    for alias, version in sorted(aliases.items()):
        print(f"    {alias:9s} -> version {version}")

    # Non-zero exit lets CI fail when nothing passed the gates.
    if not any(r.gates_passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
