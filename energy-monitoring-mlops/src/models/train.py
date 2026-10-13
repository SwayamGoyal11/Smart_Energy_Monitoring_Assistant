"""
src/models/train.py
--------------------
Trains: Persistence Baseline, Linear Regression, Random Forest, XGBoost.
All runs logged to MLflow. Models saved as .joblib artifacts.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.models.evaluate import compute_metrics, print_metrics

warnings.filterwarnings("ignore")

PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR     = os.path.join(PROJECT_ROOT, "reports", "evidence")
MLFLOW_DB       = os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_URI      = f"sqlite:///{MLFLOW_DB}"
EXPERIMENT_NAME = "energy-consumption-prediction"
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"  MLflow tracking URI : {MLFLOW_URI}")
    print(f"  Experiment          : {EXPERIMENT_NAME}")


# Persistence Baseline
def train_baseline(X_val, y_val, X_test, y_test):
    """Naive baseline: predict next value = most recent observed value (lag_1)."""
    val_preds  = X_val["Appliances_lag_1"].values
    test_preds = X_test["Appliances_lag_1"].values

    val_m  = compute_metrics(y_val,  val_preds)
    test_m = compute_metrics(y_test, test_preds)
    print_metrics("Persistence Baseline", val_m,  "Validation")
    print_metrics("Persistence Baseline", test_m, "Test")

    with mlflow.start_run(run_name="Persistence_Baseline"):
        mlflow.log_param("model", "Persistence Baseline")
        mlflow.log_param("description", "y_hat = Appliances_lag_1")
        mlflow.log_metric("val_mae",   val_m["mae"])
        mlflow.log_metric("val_rmse",  val_m["rmse"])
        mlflow.log_metric("val_r2",    val_m["r2"])
        mlflow.log_metric("test_mae",  test_m["mae"])
        mlflow.log_metric("test_rmse", test_m["rmse"])
        mlflow.log_metric("test_r2",   test_m["r2"])

    return {"model": "Persistence Baseline", "val_metrics": val_m, "test_metrics": test_m}


# Linear Regression
def train_linear_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    val_m  = compute_metrics(y_val,  model.predict(X_val))
    test_m = compute_metrics(y_test, model.predict(X_test))
    print_metrics("Linear Regression", val_m,  "Validation")
    print_metrics("Linear Regression", test_m, "Test")

    joblib.dump(model, os.path.join(MODELS_DIR, "linear_regression.joblib"))

    with mlflow.start_run(run_name="Linear_Regression"):
        mlflow.log_param("model", "Linear Regression")
        mlflow.log_param("fit_intercept", True)
        mlflow.log_metric("val_mae",   val_m["mae"])
        mlflow.log_metric("val_rmse",  val_m["rmse"])
        mlflow.log_metric("val_r2",    val_m["r2"])
        mlflow.log_metric("test_mae",  test_m["mae"])
        mlflow.log_metric("test_rmse", test_m["rmse"])
        mlflow.log_metric("test_r2",   test_m["r2"])
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    return {"model": "Linear Regression", "val_metrics": val_m, "test_metrics": test_m, "model_obj": model}


# Random Forest
def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    params = {"n_estimators": 100, "max_depth": 15,
              "min_samples_leaf": 4, "random_state": 42, "n_jobs": -1}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    val_m  = compute_metrics(y_val,  model.predict(X_val))
    test_m = compute_metrics(y_test, model.predict(X_test))
    print_metrics("Random Forest", val_m,  "Validation")
    print_metrics("Random Forest", test_m, "Test")

    joblib.dump(model, os.path.join(MODELS_DIR, "random_forest.joblib"))

    with mlflow.start_run(run_name="Random_Forest"):
        mlflow.log_param("model", "Random Forest")
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("val_mae",   val_m["mae"])
        mlflow.log_metric("val_rmse",  val_m["rmse"])
        mlflow.log_metric("val_r2",    val_m["r2"])
        mlflow.log_metric("test_mae",  test_m["mae"])
        mlflow.log_metric("test_rmse", test_m["rmse"])
        mlflow.log_metric("test_r2",   test_m["r2"])
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    return {"model": "Random Forest", "val_metrics": val_m, "test_metrics": test_m, "model_obj": model}


# XGBoost
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "random_state": 42, "n_jobs": -1, "tree_method": "hist"}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_m  = compute_metrics(y_val,  model.predict(X_val))
    test_m = compute_metrics(y_test, model.predict(X_test))
    print_metrics("XGBoost", val_m,  "Validation")
    print_metrics("XGBoost", test_m, "Test")

    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost.joblib"))

    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_param("model", "XGBoost")
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("val_mae",   val_m["mae"])
        mlflow.log_metric("val_rmse",  val_m["rmse"])
        mlflow.log_metric("val_r2",    val_m["r2"])
        mlflow.log_metric("test_mae",  test_m["mae"])
        mlflow.log_metric("test_rmse", test_m["rmse"])
        mlflow.log_metric("test_r2",   test_m["r2"])
        mlflow.xgboost.log_model(model, artifact_path="model")

    return {"model": "XGBoost", "val_metrics": val_m, "test_metrics": test_m, "model_obj": model}


# Comparison table
def build_comparison_table(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Model":     r["model"],
            "Val MAE":   r["val_metrics"]["mae"],
            "Val RMSE":  r["val_metrics"]["rmse"],
            "Val R2":    r["val_metrics"]["r2"],
            "Test MAE":  r["test_metrics"]["mae"],
            "Test RMSE": r["test_metrics"]["rmse"],
            "Test R2":   r["test_metrics"]["r2"],
        })
    return pd.DataFrame(rows).sort_values("Val MAE")


def save_comparison_table(df: pd.DataFrame) -> str:
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df.to_csv(path, index=False)
    print(f"\n  Comparison table saved -> {path}")
    return path
