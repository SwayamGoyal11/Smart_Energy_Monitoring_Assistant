"""
experiments/_common.py
-----------------------
Shared helpers for the baseline-model experiments (05-10).

Everything here mirrors the team's pipeline: same feature matrix
(build_feature_matrix), same chronological 70/15/15 split, same model
hyper-parameters as src/models/train.py. Nothing in src/ is modified.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: F401,E402

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.load_data import load_raw_data                          # noqa: E402
from src.features.feature_engineering import (                        # noqa: E402,F401
    build_feature_matrix, chronological_split, get_X_y,
    ALL_FEATURE_COLS, TARGET_COL,
)
from src.models.evaluate import compute_metrics                       # noqa: E402,F401

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LAG1 = "Appliances_lag_1"
MODEL_NAMES = ["Linear Regression", "Random Forest", "XGBoost"]
ALL_NAMES = ["Persistence"] + MODEL_NAMES


def evidence_dir(name: str) -> str:
    path = os.path.join(PROJECT_ROOT, "reports", "evidence", name)
    os.makedirs(path, exist_ok=True)
    return path


def make_model(name: str, seed: int = 42, n_jobs: int = -1):
    """Same hyper-parameters as src/models/train.py (seed 42, n_jobs=-1)."""
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=4,
            random_state=seed, n_jobs=n_jobs)
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs,
            tree_method="hist")
    raise ValueError(name)


def load_splits():
    """Returns (full feature df, train, val, test) - the team's chronological split."""
    df = build_feature_matrix(load_raw_data())
    train, val, test = chronological_split(df)
    return df, train, val, test


def predict_all(models: dict, X: pd.DataFrame) -> dict:
    """Persistence + every fitted model's predictions on X."""
    preds = {"Persistence": X[LAG1].values}
    preds.update({n: m.predict(X) for n, m in models.items()})
    return preds
