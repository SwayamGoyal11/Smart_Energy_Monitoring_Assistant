"""
src/monitoring/drift.py
------------------------
Evidently drift monitoring: compares a current window of model inputs and
predictions against the training-period reference distribution.

Run from the project root:
    python -m src.monitoring.drift                          # current = test split
    python -m src.monitoring.drift --current data/monitoring/inference_log.csv

Outputs:
    reports/evidence/drift_report.html   Evidently report for humans
    reports/evidence/drift_metrics.json  structured scores for alerting and
                                         the API's model_drift_score gauge

Calendar features are not monitored: production data always comes from a
later date than training, so hour/day/month would drift by construction.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift

from src.data.load_data import load_raw_data
from src.features.feature_engineering import (
    ALL_FEATURE_COLS, ENVIRONMENTAL_FEATURES, LAG_FEATURES, ROLLING_FEATURES, TARGET_COL,
    build_feature_matrix, chronological_split, get_X_y,
)
from src.models.train import PROJECT_ROOT, REPORTS_DIR
from src.monitoring.inference_log import PREDICTION_COL, load_inference_log
from src.validation.input_validation import validate_feature_frame

logger = logging.getLogger(__name__)

MONITORED_FEATURES = LAG_FEATURES + ROLLING_FEATURES + ENVIRONMENTAL_FEATURES

PROCESSED_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
DRIFT_HTML_PATH = os.path.join(REPORTS_DIR, "drift_report.html")
DRIFT_JSON_PATH = os.path.join(REPORTS_DIR, "drift_metrics.json")

# wasserstein: normed distance (distance / reference std), drift when score >= threshold.
# ks: p-value, drift when score < threshold. KS flags tiny shifts on large samples,
# so wasserstein is the default.
METHODS = {"wasserstein": "distance", "ks": "p_value"}


@dataclass(frozen=True)
class DriftConfig:
    method: str = "wasserstein"
    threshold: float = 0.1
    # Dataset-level alert when at least this share of monitored features drift.
    drift_share: float = 0.3
    # Below this many current rows, drift statistics are noise; refuse to judge.
    min_current_rows: int = 200

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}, got '{self.method}'")

    def is_drifted(self, score: float) -> bool:
        if METHODS[self.method] == "distance":
            return score >= self.threshold
        return score < self.threshold


# Data loading

def _load_features(path: str) -> pd.DataFrame:
    """
    Cached feature matrix from run_pipeline.py, rebuilt from the raw data when
    the cache is missing or stale (written before a feature was added).
    """
    if os.path.isfile(path):
        features = pd.read_csv(path, parse_dates=["date"])
        stale = [c for c in ALL_FEATURE_COLS + [TARGET_COL] if c not in features.columns]
        if not stale:
            return features
        logger.warning("%s is stale (missing %s); rebuilding features from raw data", path, stale)
    return build_feature_matrix(load_raw_data())


def load_reference(path: str = PROCESSED_FEATURES_PATH) -> pd.DataFrame:
    """Training split of the feature matrix: the distribution the model learned."""
    train, _, _ = chronological_split(_load_features(path))
    return train.reset_index(drop=True)


def load_test_window(path: str = PROCESSED_FEATURES_PATH) -> pd.DataFrame:
    """Held-out test split, used as a stand-in production window."""
    _, _, test = chronological_split(_load_features(path))
    return test.reset_index(drop=True)


def attach_predictions(frame: pd.DataFrame, model) -> pd.DataFrame:
    """Add a `prediction` column from any model with .predict(feature_frame)."""
    X, _ = get_X_y(frame)
    preds = np.asarray(model.predict(validate_feature_frame(X)), dtype=float).reshape(-1)
    return frame.assign(**{PREDICTION_COL: np.clip(preds, 0.0, None)})


# Drift computation

def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig = DriftConfig(),
    columns: Optional[List[str]] = None,
):
    """
    Run the Evidently report and return (metrics dict, Evidently snapshot).

    Feature drift covers `columns` (default MONITORED_FEATURES). Prediction
    drift is added when both frames have a `prediction` column; target drift
    when both have actual `Appliances` values.
    """
    columns = list(columns or MONITORED_FEATURES)
    missing = {
        name: [c for c in columns if c not in frame.columns]
        for name, frame in (("reference", reference), ("current", current))
    }
    if any(missing.values()):
        raise ValueError(f"Missing monitored columns: {missing}")
    if reference.empty or current.empty:
        raise ValueError("reference and current must both have rows")
    if len(current) < config.min_current_rows:
        raise ValueError(
            f"current has {len(current)} rows; need at least {config.min_current_rows} rows "
            "for a meaningful drift test"
        )

    outputs = [c for c in (PREDICTION_COL, TARGET_COL)
               if c in reference.columns and c in current.columns and current[c].notna().any()]
    keep = columns + outputs
    ref, cur = reference[keep].astype(float), current[keep].astype(float)

    metrics = [DriftedColumnsCount(
        columns=columns, method=config.method, threshold=config.threshold,
        drift_share=config.drift_share,
    )]
    metrics += [ValueDrift(column=c, method=config.method, threshold=config.threshold) for c in keep]
    snapshot = Report(metrics).run(reference_data=ref, current_data=cur)

    scores = {}
    for result in snapshot.dict()["metrics"]:
        cfg = result["config"]
        if cfg["type"].endswith("ValueDrift"):
            scores[cfg["column"]] = float(result["value"])

    features = {c: {"score": round(scores[c], 6), "drifted": config.is_drifted(scores[c])} for c in columns}
    drifted = sorted(c for c, v in features.items() if v["drifted"])
    share = len(drifted) / len(columns)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"method": config.method, "threshold": config.threshold,
                   "drift_share": config.drift_share, "score_type": METHODS[config.method]},
        "reference": {"rows": len(reference)},
        "current": {"rows": len(current)},
        "dataset_drift": {
            "detected": share >= config.drift_share,
            "drifted_columns": drifted,
            "drifted_count": len(drifted),
            "monitored_count": len(columns),
            "share": round(share, 4),
        },
        "features": features,
    }
    for name, col in (("prediction", PREDICTION_COL), ("target", TARGET_COL)):
        if col in outputs:
            result[name] = {"column": col, "score": round(scores[col], 6),
                            "drifted": config.is_drifted(scores[col])}

    result["alerts"] = _alerts(result)
    return result, snapshot


def _alerts(result: dict) -> List[str]:
    alerts = []
    ds = result["dataset_drift"]
    if ds["detected"]:
        alerts.append(
            f"DATASET_DRIFT: {ds['drifted_count']}/{ds['monitored_count']} features drifted "
            f"(share {ds['share']:.2f} >= {result['config']['drift_share']})"
        )
    for name in ("prediction", "target"):
        if result.get(name, {}).get("drifted"):
            alerts.append(f"{name.upper()}_DRIFT: score {result[name]['score']:.4f}")
    return alerts


def save_reports(result: dict, snapshot, html_path: str = DRIFT_HTML_PATH,
                 json_path: str = DRIFT_JSON_PATH) -> None:
    for path in (html_path, json_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    snapshot.save_html(html_path)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)


# CLI

def main(argv=None) -> int:
    from src.api.model_loader import load_serving_model
    from src.models.registry import REGISTERED_MODEL, tracking_uri

    parser = argparse.ArgumentParser(description="Generate Evidently drift reports.")
    parser.add_argument("--current", help="Inference log CSV; defaults to the held-out test split.")
    parser.add_argument("--method", choices=sorted(METHODS), default=DriftConfig.method)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit 2 when any alert fires.")
    args = parser.parse_args(argv)

    threshold = args.threshold if args.threshold is not None else (0.1 if args.method == "wasserstein" else 0.05)
    config = DriftConfig(method=args.method, threshold=threshold)

    loaded = load_serving_model(tracking_uri(), REGISTERED_MODEL,
                                os.path.join(PROJECT_ROOT, "models", "champion"))
    if loaded is None:
        print("No serving model available; cannot compute prediction drift.", file=sys.stderr)
        return 1

    reference = attach_predictions(load_reference(), loaded.model)
    if args.current:
        current, source = load_inference_log(args.current), args.current
    else:
        current, source = attach_predictions(load_test_window(), loaded.model), "test split"

    try:
        result, snapshot = compute_drift(reference, current, config)
    except ValueError as exc:
        print(f"Drift check skipped: {exc}", file=sys.stderr)
        return 1
    result["model"] = {"name": loaded.name, "version": loaded.version, "source": loaded.source}
    result["current"]["source"] = source
    save_reports(result, snapshot)

    ds = result["dataset_drift"]
    print(f"  Reference rows : {len(reference):,} (training split)")
    print(f"  Current rows   : {len(current):,} ({source})")
    print(f"  Method         : {config.method} (threshold {config.threshold})")
    print(f"  Drifted        : {ds['drifted_count']}/{ds['monitored_count']} features -> {ds['drifted_columns']}")
    for name in ("prediction", "target"):
        if name in result:
            print(f"  {name.capitalize():15s}: score {result[name]['score']:.4f} drifted={result[name]['drifted']}")
    print(f"  Alerts         : {result['alerts'] or 'none'}")
    print(f"  Saved -> {DRIFT_HTML_PATH}\n           {DRIFT_JSON_PATH}")
    return 2 if (args.fail_on_drift and result["alerts"]) else 0


if __name__ == "__main__":
    sys.exit(main())
