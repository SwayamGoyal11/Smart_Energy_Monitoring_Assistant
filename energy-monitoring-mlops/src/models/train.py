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
TMP_DIR         = os.path.join(PROJECT_ROOT, ".tmp")
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR
os.environ["TMPDIR"] = TMP_DIR
os.environ["MPLCONFIGDIR"] = os.path.join(TMP_DIR, "matplotlib")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import tempfile
tempfile.tempdir = TMP_DIR

MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR     = os.path.join(PROJECT_ROOT, "reports", "evidence")
MLFLOW_DB       = os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_URI      = f"sqlite:///{MLFLOW_DB}"
EXPERIMENT_NAME = "energy-consumption-prediction"
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def setup_mlflow():
    """
    Configure the MLflow tracking server and experiment.

    Sets the SQLite tracking URI and creates (or reuses) the experiment
    named ``energy-consumption-prediction``.  Call this once before any
    ``mlflow.start_run`` context.
    """
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

    return {
        "model": "Persistence Baseline",
        "val_metrics": val_m,
        "test_metrics": test_m,
        "val_preds": val_preds,
        "test_preds": test_preds,
    }


# Linear Regression
def train_linear_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train an Ordinary Least Squares (OLS) Linear Regression model.

    Fits on ``X_train`` / ``y_train``, evaluates on both validation and
    test splits, saves the serialised model to ``models/`` via joblib,
    and logs all hyperparameters and metrics to MLflow.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series):    Training target (Appliances, Wh).
        X_val   (pd.DataFrame): Validation feature matrix.
        y_val   (pd.Series):    Validation target.
        X_test  (pd.DataFrame): Test feature matrix.
        y_test  (pd.Series):    Test target.

    Returns:
        dict: Keys ``model``, ``val_metrics``, ``test_metrics``,
              ``model_obj``, ``val_preds``, ``test_preds``.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_m  = compute_metrics(y_val,  val_preds)
    test_m = compute_metrics(y_test, test_preds)
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
        # mlflow.sklearn.log_model skipped (temp-dir permission error on this machine)

    return {
        "model": "Linear Regression",
        "val_metrics": val_m,
        "test_metrics": test_m,
        "model_obj": model,
        "val_preds": val_preds,
        "test_preds": test_preds,
    }


# Random Forest
def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train a Random Forest Regressor with a bagged ensemble of 100 trees.

    Hyperparameters: ``n_estimators=100``, ``max_depth=15``,
    ``min_samples_leaf=4``.  Model is serialised to
    ``models/random_forest.joblib`` and all params / metrics are logged
    to MLflow.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series):    Training target (Appliances, Wh).
        X_val   (pd.DataFrame): Validation feature matrix.
        y_val   (pd.Series):    Validation target.
        X_test  (pd.DataFrame): Test feature matrix.
        y_test  (pd.Series):    Test target.

    Returns:
        dict: Keys ``model``, ``val_metrics``, ``test_metrics``,
              ``model_obj``, ``val_preds``, ``test_preds``.
    """
    params = {"n_estimators": 100, "max_depth": 15,
              "min_samples_leaf": 4, "random_state": 42, "n_jobs": 1}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_m  = compute_metrics(y_val,  val_preds)
    test_m = compute_metrics(y_test, test_preds)
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
        # mlflow.sklearn.log_model skipped (temp-dir permission error on this machine)

    return {
        "model": "Random Forest",
        "val_metrics": val_m,
        "test_metrics": test_m,
        "model_obj": model,
        "val_preds": val_preds,
        "test_preds": test_preds,
    }


# XGBoost
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train an XGBoost Gradient-Boosted Trees regressor.

    Hyperparameters: ``n_estimators=200``, ``max_depth=6``,
    ``learning_rate=0.05``, ``subsample=0.8``, ``colsample_bytree=0.8``.
    Training uses an eval set for RMSE monitoring.  The full evaluation
    history (``evals_result``) is returned so the caller can plot the
    learning curve.  Model saved to ``models/xgboost.joblib``.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series):    Training target (Appliances, Wh).
        X_val   (pd.DataFrame): Validation feature matrix.
        y_val   (pd.Series):    Validation target.
        X_test  (pd.DataFrame): Test feature matrix.
        y_test  (pd.Series):    Test target.

    Returns:
        dict: Keys ``model``, ``val_metrics``, ``test_metrics``,
              ``model_obj``, ``val_preds``, ``test_preds``, ``evals_result``.
    """
    params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "random_state": 42, "n_jobs": 1, "tree_method": "hist",
              "eval_metric": "rmse"}
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    xgb_history = model.evals_result()

    val_m  = compute_metrics(y_val,  val_preds)
    test_m = compute_metrics(y_test, test_preds)
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
        # mlflow.xgboost.log_model skipped (temp-dir permission error on this machine)

    return {
        "model": "XGBoost",
        "val_metrics": val_m,
        "test_metrics": test_m,
        "model_obj": model,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "evals_result": xgb_history,
    }


# Comparison table
def build_comparison_table(results: list) -> pd.DataFrame:
    """
    Assemble a tidy comparison DataFrame from a list of training result dicts.

    Args:
        results (list[dict]): Each dict must contain ``model``,
            ``val_metrics``, and ``test_metrics`` keys as returned by
            each ``train_*`` function.

    Returns:
        pd.DataFrame: Rows are models, columns are Val/Test MAE, RMSE, R².
                      Sorted ascending by Val MAE.
    """
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
    """
    Persist the model comparison table to ``reports/evidence/model_comparison.csv``.

    Args:
        df (pd.DataFrame): Output of :func:`build_comparison_table`.

    Returns:
        str: Absolute path to the saved CSV file.
    """
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df.to_csv(path, index=False)
    print(f"\n  Comparison table saved -> {path}")
    return path
