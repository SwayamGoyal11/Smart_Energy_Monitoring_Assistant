"""
src/monitoring/inference_log.py
--------------------------------
Append-only CSV log of served model inputs and predictions. This is the
"current" dataset the drift report compares against the training reference.
"""

import os
import threading
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from src.features.feature_engineering import ALL_FEATURE_COLS

# Defined here, not in drift.py, so the API can log without importing Evidently.
PREDICTION_COL = "prediction"
LOG_COLUMNS = ["logged_at", "target_timestamp", "model_version"] + ALL_FEATURE_COLS + [PREDICTION_COL]


class InferenceLogger:
    """Thread-safe CSV appender; FastAPI runs sync endpoints in a thread pool."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, features: pd.DataFrame, predictions: Sequence[float],
            target_timestamps: Sequence, model_version: str) -> None:
        rows = features[ALL_FEATURE_COLS].copy()
        rows.insert(0, "model_version", model_version)
        rows.insert(0, "target_timestamp", [pd.Timestamp(t).isoformat() for t in target_timestamps])
        rows.insert(0, "logged_at", datetime.now(timezone.utc).isoformat())
        rows[PREDICTION_COL] = list(predictions)
        with self._lock:
            write_header = not os.path.isfile(self.path) or os.path.getsize(self.path) == 0
            rows[LOG_COLUMNS].to_csv(self.path, mode="a", header=write_header, index=False)


def load_inference_log(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Inference log not found: {path}")
    frame = pd.read_csv(path, parse_dates=["logged_at", "target_timestamp"])
    if frame.empty:
        raise ValueError(f"Inference log is empty: {path}")
    return frame
