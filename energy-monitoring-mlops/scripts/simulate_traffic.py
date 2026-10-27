"""
scripts/simulate_traffic.py
----------------------------
Replay real dataset rows through a running API's /predict endpoint, to feed
the Grafana dashboard and give the drift check a realistic traffic window.

    python scripts/simulate_traffic.py                        # 250 train-split rows
    python scripts/simulate_traffic.py --split test --rows 300
    python -m src.monitoring.drift --current data/monitoring/simulated_traffic.csv

Each request carries a row's own sensor readings and the six Appliances
readings before it, exactly as a client would send them. The inputs and
predictions are also written to --log in the API's inference-log format, so
drift can score this replay alone instead of everything the API has served.

Drift is measured against the training split. Rows from the train split
replay the reference distribution and should come back clean; rows from the
later validation and test months show the real seasonal shift.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from src.data.load_data import TARGET_COL, load_raw_data  # noqa: E402
from src.features.feature_engineering import build_feature_matrix, chronological_split  # noqa: E402
from src.monitoring.inference_log import InferenceLogger  # noqa: E402
from src.serving.schemas import HISTORY_LENGTH, PredictionRequest  # noqa: E402

SPLITS = {"train": 0, "val": 1, "test": 2}
SENSOR_FIELDS = [f for f in PredictionRequest.model_fields if f not in ("target_timestamp", "recent_appliances_wh")]
DEFAULT_LOG = os.path.join(PROJECT_ROOT, "data", "monitoring", "simulated_traffic.csv")
# src.monitoring.drift skips windows smaller than this
MIN_DRIFT_ROWS = 200


def build_requests(split: str, rows: int, seed: int) -> list:
    """Sample rows from one chronological split as (payload, actual Wh) pairs."""
    raw = load_raw_data()
    window = chronological_split(build_feature_matrix(raw))[SPLITS[split]]
    position = {date: i for i, date in enumerate(raw["date"])}
    picks = window.sample(n=min(rows, len(window)), random_state=seed).sort_values("date")

    requests = []
    for _, row in picks.iterrows():
        i = position[row["date"]]
        history = raw[TARGET_COL].iloc[i - HISTORY_LENGTH:i].astype(float).tolist()
        payload = {"target_timestamp": row["date"].isoformat(), "recent_appliances_wh": history}
        payload.update({f: float(raw[f].iloc[i]) for f in SENSOR_FIELDS})
        requests.append((payload, float(raw[TARGET_COL].iloc[i])))
    return requests


def post(url: str, payload: dict, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main():
    parser = argparse.ArgumentParser(description="Replay dataset rows through /predict.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--split", choices=list(SPLITS), default="train")
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between requests.")
    parser.add_argument("--log", default=DEFAULT_LOG, help="CSV of sent traffic for drift (overwritten).")
    args = parser.parse_args()

    requests = build_requests(args.split, args.rows, args.seed)
    if os.path.isfile(args.log):
        os.remove(args.log)
    logger = InferenceLogger(args.log)

    errors, actual, predicted = 0, [], []
    for n, (payload, truth) in enumerate(requests, 1):
        try:
            body = post(f"{args.url.rstrip('/')}/predict", payload)
        except (urllib.error.URLError, OSError) as exc:
            errors += 1
            print(f"  [{n}/{len(requests)}] {payload['target_timestamp']} failed: {exc}")
            if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
                sys.exit(f"API not reachable at {args.url}; is the stack running?")
            continue
        pred = body["predicted_appliances_wh"]
        logger.log(PredictionRequest(**payload).to_feature_frame(), [pred],
                   [payload["target_timestamp"]], body["model_version"])
        actual.append(truth)
        predicted.append(pred)
        if args.delay:
            time.sleep(args.delay)

    sent = len(predicted)
    mae = float(np.mean(np.abs(np.array(actual) - np.array(predicted)))) if sent else float("nan")
    print(f"  Replayed {sent}/{len(requests)} {args.split}-split rows to {args.url} ({errors} errors)")
    print(f"  MAE vs actual : {mae:.2f} Wh")
    print(f"  Traffic log   : {args.log}")
    if sent < MIN_DRIFT_ROWS:
        print(f"  Note: drift needs at least {MIN_DRIFT_ROWS} rows; rerun with --rows {MIN_DRIFT_ROWS} or more.")
    print(f"  Next          : python -m src.monitoring.drift --current {os.path.relpath(args.log, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
