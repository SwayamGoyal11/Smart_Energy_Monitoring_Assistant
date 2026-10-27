"""
src/monitoring/metrics.py
--------------------------
Prometheus metrics for the inference service. Each app gets its own
CollectorRegistry so several app instances (e.g. in tests) never collide.

Drift scores come from the drift_metrics.json written by
src/monitoring/drift.py; the file is re-read on scrape only when it changes,
so the API never imports or runs Evidently itself.
"""

import json
import logging
import os
from typing import Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

logger = logging.getLogger("energy_api")

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
# Wh per 10 minutes: standby sits around 40-100, bursts reach ~1000.
PREDICTION_BUCKETS = (10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500)


class ApiMetrics:
    def __init__(self, drift_metrics_path: Optional[str] = None):
        self.registry = CollectorRegistry()
        self.drift_metrics_path = drift_metrics_path
        self._drift_mtime = None
        self._stage_counts = {"active": 0, "standby": 0}

        self.requests = Counter(
            "http_requests_total", "HTTP requests by method, route and status code.",
            ["method", "path", "status"], registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds", "HTTP request latency by method and route.",
            ["method", "path"], buckets=LATENCY_BUCKETS, registry=self.registry,
        )
        self.prediction_value = Histogram(
            "model_prediction_value", "Distribution of predicted appliance energy (Wh per 10 min).",
            ["model_version"], buckets=PREDICTION_BUCKETS, registry=self.registry,
        )
        self.stage_predictions = Counter(
            "model_stage_predictions_total", "Hurdle model routing decisions by stage.",
            ["stage"], registry=self.registry,
        )
        self.active_stage_ratio = Gauge(
            "model_active_stage_ratio",
            "Share of hurdle-model predictions routed to the active-burst regressor.",
            registry=self.registry,
        )
        self.drift_score = Gauge(
            "model_drift_score", "Latest Evidently drift score per monitored column.",
            ["column"], registry=self.registry,
        )
        self.drift_share = Gauge(
            "model_drift_share", "Share of monitored features flagged as drifted.",
            registry=self.registry,
        )
        self.drift_detected = Gauge(
            "model_drift_detected", "1 if the latest drift report raised any alert, else 0.",
            registry=self.registry,
        )

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        self.requests.labels(method, path, str(status)).inc()
        self.latency.labels(method, path).observe(seconds)

    def observe_predictions(self, values, model_version: str) -> None:
        hist = self.prediction_value.labels(model_version)
        for v in values:
            hist.observe(v)

    def observe_stages(self, n_active: int, n_standby: int) -> None:
        self.stage_predictions.labels("active").inc(n_active)
        self.stage_predictions.labels("standby").inc(n_standby)
        self._stage_counts["active"] += n_active
        self._stage_counts["standby"] += n_standby
        total = sum(self._stage_counts.values())
        self.active_stage_ratio.set(self._stage_counts["active"] / total if total else 0.0)

    def refresh_drift(self) -> None:
        """Load drift_metrics.json into gauges if it changed since the last scrape."""
        path = self.drift_metrics_path
        if not path or not os.path.isfile(path):
            return
        mtime = os.path.getmtime(path)
        if mtime == self._drift_mtime:
            return
        try:
            with open(path) as f:
                report = json.load(f)
            self.drift_score.clear()
            for column, entry in report.get("features", {}).items():
                self.drift_score.labels(column).set(entry["score"])
            for name in ("prediction", "target"):
                if name in report:
                    self.drift_score.labels(report[name]["column"]).set(report[name]["score"])
            self.drift_share.set(report["dataset_drift"]["share"])
            self.drift_detected.set(1 if report.get("alerts") else 0)
            self._drift_mtime = mtime
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Could not load drift metrics from %s: %s", path, exc)
