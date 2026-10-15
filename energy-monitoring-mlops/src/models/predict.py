"""
src/models/predict.py
---------------------
Inference module for smart energy consumption forecasting.
Loads serialized model artifacts (.joblib) and generates leak-free predictions
with automated feature schema validation and physical bounds enforcement (energy >= 0).
"""

import os
import sys
import logging
from typing import Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import ALL_FEATURE_COLS

logger = logging.getLogger("energy_monitoring.predict")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))


def load_model_artifact(model_name: str = "random_forest", models_dir: Optional[str] = None):
    """
    Load a trained model artifact (.joblib) from disk.

    Parameters
    ----------
    model_name : str
        Base name of the model ('random_forest', 'xgboost', 'linear_regression').
    models_dir : str, optional
        Custom directory path where models are stored.

    Returns
    -------
    estimator
        The deserialized model object.
    """
    directory = models_dir or DEFAULT_MODELS_DIR
    clean_name = model_name.replace(".joblib", "")
    artifact_path = os.path.join(directory, f"{clean_name}.joblib")

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")

    logger.info("Loading model artifact: %s", artifact_path)
    model = joblib.load(artifact_path)
    return model


def predict(
    df: pd.DataFrame,
    model_name: str = "random_forest",
    models_dir: Optional[str] = None,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Generate appliance energy consumption predictions for input feature data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required feature columns (ALL_FEATURE_COLS).
    model_name : str
        Model to use for inference ('random_forest', 'xgboost', 'linear_regression').
    models_dir : str, optional
        Directory holding the serialized model artifacts.
    clip_negative : bool, default True
        Whether to enforce energy >= 0.0 Wh by clipping negative predictions.

    Returns
    -------
    np.ndarray
        Predicted appliance energy consumption values in Wh.
    """
    # Schema validation
    missing_cols = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing_cols}")

    X = df[ALL_FEATURE_COLS].copy()

    # Null value check
    if X.isnull().any().any():
        null_counts = X.isnull().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Input features contain NaN values: {null_cols}")

    model = load_model_artifact(model_name=model_name, models_dir=models_dir)
    preds = model.predict(X)

    if clip_negative:
        preds = np.clip(preds, a_min=0.0, a_max=None)

    logger.info("Generated %d predictions using model '%s'", len(preds), model_name)
    return preds
