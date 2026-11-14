"""
src/monitoring/inference_log.py
--------------------------------
Append-only structured JSON Lines log of served model inputs and predictions.
Each line is one self-describing JSON object (one per predicted row), so the
log can be shipped to any log collector as-is and is also the "current"
dataset the drift report compares against the training reference.

Record layout (keys in LOG_COLUMNS order, plus "event"):
    {"event": "inference", "logged_at": "...", "target_timestamp": "...",
     "model_version": "5", "<feature>": <float>, ..., "prediction": 63.2}

load_inference_log() also reads logs written by earlier CSV-based versions.
"""

import json
import math
import os
import threading
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from src.features.feature_engineering import ALL_FEATURE_COLS

# Defined here, not in drift.py, so the API can log without importing Evidently.
PREDICTION_COL = "prediction"
EVENT_TYPE = "inference"
LOG_COLUMNS = ["logged_at", "target_timestamp", "model_version"] + ALL_FEATURE_COLS + [PREDICTION_COL]


def _json_value(value):
    """Plain JSON scalars only: numpy types become Python numbers, NaN/inf become null."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class InferenceLogger:
    """Thread-safe JSON Lines appender; FastAPI runs sync endpoints in a thread pool."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def build_records(self, features: pd.DataFrame, predictions: Sequence[float],
                      target_timestamps: Sequence, model_version: str) -> list:
        if not (len(features) == len(predictions) == len(target_timestamps)):
            raise ValueError("features, predictions and target_timestamps must have equal length")
        logged_at = datetime.now(timezone.utc).isoformat()
        rows = features[ALL_FEATURE_COLS].to_dict(orient="records")
        return [
            {
                "event": EVENT_TYPE,
                "logged_at": logged_at,
                "target_timestamp": pd.Timestamp(ts).isoformat(),
                "model_version": str(model_version),
                **{col: _json_value(row[col]) for col in ALL_FEATURE_COLS},
                PREDICTION_COL: _json_value(float(pred)),
            }
            for row, pred, ts in zip(rows, predictions, target_timestamps)
        ]

    def log(self, features: pd.DataFrame, predictions: Sequence[float],
            target_timestamps: Sequence, model_version: str) -> None:
        records = self.build_records(features, predictions, target_timestamps, model_version)
        payload = "".join(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in records)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(payload)


def _is_legacy_csv(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return not line.lstrip().startswith("{")
    return False


def load_inference_log(path: str) -> pd.DataFrame:
    """Load a JSON Lines inference log (or a legacy CSV one) as a DataFrame in LOG_COLUMNS order."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Inference log not found: {path}")
    if _is_legacy_csv(path):
        frame = pd.read_csv(path)
    else:
        records = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed inference log line {lineno} in {path}: {exc}") from exc
        frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"Inference log is empty: {path}")
    missing = [c for c in LOG_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Inference log {path} is missing columns: {missing}")
    frame = frame[LOG_COLUMNS].copy()
    for col in ("logged_at", "target_timestamp"):
        frame[col] = pd.to_datetime(frame[col], format="ISO8601")
    frame["model_version"] = frame["model_version"].astype(str)
    return frame
