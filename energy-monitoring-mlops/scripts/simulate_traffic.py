"""
scripts/simulate_traffic.py
----------------------------
Replay real dataset rows through a running API to feed the Grafana dashboard
and give the drift check a realistic traffic window.

    python scripts/simulate_traffic.py                              # 250 train-split rows, local API
    python scripts/simulate_traffic.py --split test --rows 300 --batch-size 10
    python scripts/simulate_traffic.py --invalid-share 0.1          # also exercise input validation
    python scripts/simulate_traffic.py --url https://<service>.onrender.com
    python -m src.monitoring.drift --current data/monitoring/simulated_traffic.jsonl

Traffic mix
    Valid rows are sampled from one chronological split and sent to
    /predict/batch in chronological batches. Each row is tagged standby or
    active burst by its actual reading (the hurdle model's 100 Wh boundary);
    by default the split's natural mix is kept (about 22% active in training),
    and --active-share resamples to a fixed share instead. A batch size of 1
    sends single /predict requests.

    --invalid-share adds malformed requests (out-of-range sensor, wrong type,
    short history, unknown field) as single /predict calls, so a bad row never
    sinks a valid batch. The API rejects them with 422 and counts them in
    model_input_validation_failures_total; they are not written to the log.

Log
    Accepted inputs and their predictions are written to --log as JSON Lines
    in the API's inference-log format (overwritten each run), so drift can
    score this replay alone instead of everything the API has served.

Drift is measured against the training split. Rows from the train split
replay the reference distribution and should come back clean; rows from the
later validation and test months show the real seasonal shift. Forcing
--active-share away from the natural mix skews the lag features and can
itself trigger drift.

Remote targets (e.g. a Render free instance) may cold-start: the script waits
for /health and retries timeouts and 5xx/429 responses with backoff.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.load_data import TARGET_COL, load_raw_data  # noqa: E402
from src.features.feature_engineering import build_feature_matrix, chronological_split  # noqa: E402
from src.monitoring.inference_log import InferenceLogger  # noqa: E402
from src.serving.schemas import (  # noqa: E402
    HISTORY_LENGTH, MAX_BATCH_SIZE, PredictionRequest, build_feature_frame,
)

SPLITS = {"train": 0, "val": 1, "test": 2}
SENSOR_FIELDS = [f for f in PredictionRequest.model_fields if f not in ("target_timestamp", "recent_appliances_wh")]
DEFAULT_URL = os.environ.get("ENERGY_API_URL", "http://localhost:8000")
DEFAULT_LOG = os.path.join(PROJECT_ROOT, "data", "monitoring", "simulated_traffic.jsonl")
# Same boundary the hurdle model uses to label intervals as active bursts.
ACTIVE_THRESHOLD_WH = 100.0
# src.monitoring.drift skips windows smaller than this
MIN_DRIFT_ROWS = 200
RETRY_STATUSES = {429, 500, 502, 503, 504}

# (name, mutation) pairs; each turns a valid payload into one the API must reject.
INVALID_KINDS = [
    ("out_of_range", lambda p: {**p, "RH_1": 150.0}),
    ("wrong_type", lambda p: {**p, "T1": "warm"}),
    ("short_history", lambda p: {**p, "recent_appliances_wh": p["recent_appliances_wh"][:2]}),
    ("unknown_field", lambda p: {**p, "occupancy": 3}),
]

# post(url, payload) -> (status_code, parsed JSON body or None)
PostFn = Callable[[str, dict], Tuple[int, Optional[dict]]]


@dataclass
class Sample:
    payload: dict
    actual_wh: float

    @property
    def active(self) -> bool:
        return self.actual_wh > ACTIVE_THRESHOLD_WH


@dataclass
class Report:
    sent: int = 0
    accepted: int = 0
    active: int = 0
    rejected_expected: int = 0
    rejected_unexpected: int = 0
    failed: int = 0
    actual: List[float] = field(default_factory=list)
    predicted: List[float] = field(default_factory=list)

    @property
    def mae(self) -> float:
        if not self.predicted:
            return float("nan")
        return float(np.mean(np.abs(np.array(self.actual) - np.array(self.predicted))))


def build_samples(split: str, rows: int, seed: int, active_share: Optional[float] = None) -> List[Sample]:
    """Sample rows from one chronological split, oldest first.

    active_share=None keeps the split's natural standby/active mix; a value in
    [0, 1] draws that share of active-burst rows (with replacement if the split
    has too few).
    """
    raw = load_raw_data()
    window = chronological_split(build_feature_matrix(raw))[SPLITS[split]]
    if active_share is None:
        picks = window.sample(n=min(rows, len(window)), random_state=seed)
    else:
        if not 0.0 <= active_share <= 1.0:
            raise ValueError("active_share must be within [0, 1]")
        is_active = window[TARGET_COL] > ACTIVE_THRESHOLD_WH
        n_active = round(rows * active_share)
        parts = []
        for pool, n in ((window[is_active], n_active), (window[~is_active], rows - n_active)):
            if n:
                parts.append(pool.sample(n=n, replace=n > len(pool), random_state=seed))
        picks = pd.concat(parts)
    picks = picks.sort_values("date")

    position = {date: i for i, date in enumerate(raw["date"])}
    samples = []
    for _, row in picks.iterrows():
        i = position[row["date"]]
        history = raw[TARGET_COL].iloc[i - HISTORY_LENGTH:i].astype(float).tolist()
        payload = {"target_timestamp": row["date"].isoformat(), "recent_appliances_wh": history}
        payload.update({f: float(raw[f].iloc[i]) for f in SENSOR_FIELDS})
        samples.append(Sample(payload, float(raw[TARGET_COL].iloc[i])))
    return samples


def make_invalid(payload: dict, rng: random.Random) -> Tuple[str, dict]:
    name, mutate = rng.choice(INVALID_KINDS)
    return name, mutate(payload)


def plan_requests(samples: List[Sample], batch_size: int, invalid_share: float,
                  seed: int) -> List[Tuple[str, object]]:
    """Ordered ("batch", [Sample, ...]) and ("invalid", (kind, payload)) steps."""
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be within [1, {MAX_BATCH_SIZE}]")
    if not 0.0 <= invalid_share < 1.0:
        raise ValueError("invalid_share must be within [0, 1)")
    rng = random.Random(seed)
    steps = [("batch", samples[i:i + batch_size]) for i in range(0, len(samples), batch_size)]
    # invalid_share is relative to all requests sent, valid rows included.
    n_invalid = round(len(samples) * invalid_share / (1.0 - invalid_share))
    for _ in range(n_invalid):
        base = rng.choice(samples).payload
        steps.insert(rng.randint(0, len(steps)), ("invalid", make_invalid(base, rng)))
    return steps


def http_post(timeout: float, retries: int) -> PostFn:
    def post(url: str, payload: dict) -> Tuple[int, Optional[dict]]:
        data = json.dumps(payload).encode()
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.status, json.load(resp)
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUSES or attempt == retries:
                    try:
                        return exc.code, json.load(exc)
                    except ValueError:
                        return exc.code, None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == retries:
                    raise
            time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")
    return post


def wait_for_health(base_url: str, timeout: float, wait_seconds: float) -> dict:
    """Poll /health until the model is loaded; remote free tiers can take a minute to wake."""
    deadline = time.monotonic() + wait_seconds
    last_error = "no response"
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as resp:
                body = json.load(resp)
                if body.get("model_loaded"):
                    return body
                last_error = f"model not loaded ({body.get('status')})"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"API at {base_url} not ready after {wait_seconds:.0f}s: {last_error}")
        time.sleep(3)


def run(base_url: str, steps, post: PostFn, logger: Optional[InferenceLogger] = None,
        delay: float = 0.0, verbose: bool = True) -> Report:
    base_url = base_url.rstrip("/")
    report = Report()
    total = len(steps)
    for n, (kind, item) in enumerate(steps, 1):
        if kind == "invalid":
            name, payload = item
            report.sent += 1
            status, _ = post(f"{base_url}/predict", payload)
            if status == 422:
                report.rejected_expected += 1
            else:
                report.rejected_unexpected += 1
                if verbose:
                    print(f"  [{n}/{total}] invalid payload ({name}) returned {status}, expected 422")
        else:
            batch: List[Sample] = item
            report.sent += len(batch)
            if len(batch) == 1:
                status, body = post(f"{base_url}/predict", batch[0].payload)
                preds = [body] if status == 200 else None
            else:
                status, body = post(f"{base_url}/predict/batch", {"instances": [s.payload for s in batch]})
                preds = body["predictions"] if status == 200 else None
            if preds is None:
                report.failed += len(batch)
                if verbose:
                    detail = body.get("detail") if isinstance(body, dict) else body
                    print(f"  [{n}/{total}] batch of {len(batch)} returned {status}: {detail}")
            else:
                values = [p["predicted_appliances_wh"] for p in preds]
                report.accepted += len(batch)
                report.active += sum(s.active for s in batch)
                report.actual.extend(s.actual_wh for s in batch)
                report.predicted.extend(values)
                if logger is not None:
                    frame = build_feature_frame([PredictionRequest(**s.payload) for s in batch])
                    logger.log(frame, values, [s.payload["target_timestamp"] for s in batch],
                               preds[0]["model_version"])
        if delay:
            time.sleep(delay)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Replay dataset rows through a running energy API.")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="API base URL, local or remote (default: $ENERGY_API_URL or http://localhost:8000).")
    parser.add_argument("--split", choices=list(SPLITS), default="train")
    parser.add_argument("--rows", type=int, default=250, help="Valid rows to send.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help=f"Rows per /predict/batch request, 1-{MAX_BATCH_SIZE}; 1 uses /predict.")
    parser.add_argument("--active-share", type=float, default=None,
                        help="Force this share of active-burst rows (> 100 Wh); default keeps the split's mix.")
    parser.add_argument("--invalid-share", type=float, default=0.0,
                        help="Share of all requests that are deliberately invalid (default 0).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between requests.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries on timeouts, 429 and 5xx.")
    parser.add_argument("--wait", type=float, default=90.0,
                        help="Seconds to wait for /health before giving up (covers remote cold starts).")
    parser.add_argument("--log", default=DEFAULT_LOG, help="JSON Lines log of accepted traffic (overwritten).")
    parser.add_argument("--no-log", action="store_true", help="Do not write a traffic log.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base_url = args.url.rstrip("/")
    try:
        health = wait_for_health(base_url, args.timeout, args.wait)
    except RuntimeError as exc:
        print(f"  {exc}; is the stack running?", file=sys.stderr)
        return 1
    print(f"  Target        : {base_url} (model {health.get('model_name')} v{health.get('model_version')})")

    samples = build_samples(args.split, args.rows, args.seed, args.active_share)
    steps = plan_requests(samples, args.batch_size, args.invalid_share, args.seed)

    logger = None
    if not args.no_log:
        if os.path.isfile(args.log):
            os.remove(args.log)
        logger = InferenceLogger(args.log)

    try:
        report = run(base_url, steps, http_post(args.timeout, args.retries), logger, args.delay)
    except (urllib.error.URLError, OSError) as exc:
        print(f"  API became unreachable at {base_url}: {exc}", file=sys.stderr)
        return 1

    share = report.active / report.accepted if report.accepted else 0.0
    print(f"  Replayed      : {report.accepted}/{len(samples)} {args.split}-split rows "
          f"in batches of {args.batch_size} ({report.failed} failed)")
    print(f"  Mix           : {report.accepted - report.active} standby / {report.active} active burst "
          f"({share:.0%} active)")
    if report.rejected_expected or report.rejected_unexpected:
        print(f"  Invalid sent  : {report.rejected_expected} rejected with 422"
              f" ({report.rejected_unexpected} unexpected responses)")
    print(f"  MAE vs actual : {report.mae:.2f} Wh")
    if logger is not None:
        print(f"  Traffic log   : {args.log}")
        if report.accepted < MIN_DRIFT_ROWS:
            print(f"  Note: drift needs at least {MIN_DRIFT_ROWS} rows; rerun with --rows {MIN_DRIFT_ROWS} or more.")
        print(f"  Next          : python -m src.monitoring.drift --current {os.path.relpath(args.log, PROJECT_ROOT)}")
    return 0 if not (report.failed or report.rejected_unexpected) else 2


if __name__ == "__main__":
    sys.exit(main())
