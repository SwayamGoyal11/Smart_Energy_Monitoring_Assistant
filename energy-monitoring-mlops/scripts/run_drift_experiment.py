"""
scripts/run_drift_experiment.py
--------------------------------
Deliberate drift and model degradation experiment.

    python scripts/run_drift_experiment.py
    python scripts/run_drift_experiment.py --temp-shift 5 --rh-scale 0.9 --dry-run

Scenario: an extreme summer heatwave with a humidity sensor bias, plus a
heavier evening appliance load. Defaults:

    T1, T2, T3, T6, T_out           +8.0 degC
    RH_1, RH_2, RH_3, RH_6, RH_out  -18% (multiplied by 0.82)
    Appliances 17:00-22:59          x1.5 + 50 Wh (bigger, more frequent evening bursts)

The scenario is applied to the raw time series and features are rebuilt with
the training feature code, so lag/rolling features and the ground-truth
target stay consistent with each other. The drifted batch covers the same
timestamps as the held-out test split, so any change in RMSE/MAE/R2 against
the unshifted test split is caused by the scenario alone.

Steps:
  1. Reference = training split (what the model learned), scored by the champion.
  2. Baseline  = unshifted test split: performance + Evidently drift (control).
  3. Drifted   = shifted test split: performance + Evidently drift.
  4. Retraining decision: dataset drift alert AND a performance breach
     (release quality gate failure, or RMSE up by --max-rmse-degradation).

Outputs (reports/evidence/ by default):
    drift_report.html                   Evidently report for the drifted batch
    drift_metrics.json                  drifted-batch scores (exported by the API as model_drift_* gauges)
    retraining_trigger_evidence.json    reference vs drifted metrics, drift share, breaches, decision

Exit code: 0 when the run completes (whatever the decision), 1 if no model is available.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.load_data import load_raw_data  # noqa: E402
from src.features.feature_engineering import TARGET_COL, build_feature_matrix, chronological_split  # noqa: E402
from src.models.evaluate import compute_metrics  # noqa: E402
from src.models.quality_gates import QualityThresholds, evaluate_quality_gates  # noqa: E402
from src.monitoring.drift import (  # noqa: E402
    DRIFT_HTML_PATH, DRIFT_JSON_PATH, DriftConfig, attach_predictions, compute_drift, load_reference,
    save_reports,
)
from src.monitoring.inference_log import PREDICTION_COL  # noqa: E402
from src.models.train import REPORTS_DIR  # noqa: E402
from src.validation.input_validation import PHYSICAL_BOUNDS  # noqa: E402

EVIDENCE_PATH = os.path.join(REPORTS_DIR, "retraining_trigger_evidence.json")
TEMPERATURE_COLS = ["T1", "T2", "T3", "T6", "T_out"]
HUMIDITY_COLS = ["RH_1", "RH_2", "RH_3", "RH_6", "RH_out"]
EXPERIMENT_NAME = "summer_heatwave_sensor_bias_evening_burst"


@dataclass(frozen=True)
class DriftScenario:
    temp_shift_c: float = 8.0
    rh_scale: float = 0.82
    evening_start_hour: int = 17
    evening_end_hour: int = 23  # exclusive
    evening_load_scale: float = 1.5
    evening_load_offset_wh: float = 50.0
    temperature_cols: Tuple[str, ...] = tuple(TEMPERATURE_COLS)
    humidity_cols: Tuple[str, ...] = tuple(HUMIDITY_COLS)

    def __post_init__(self):
        if not 0.0 < self.rh_scale <= 1.5:
            raise ValueError("rh_scale must be within (0, 1.5]")
        if not 0 <= self.evening_start_hour < self.evening_end_hour <= 24:
            raise ValueError("evening hours must satisfy 0 <= start < end <= 24")
        if self.evening_load_scale < 0:
            raise ValueError("evening_load_scale must be non-negative")

    def describe(self) -> dict:
        return {
            "temperature_shift_c": {"columns": list(self.temperature_cols), "delta": self.temp_shift_c},
            "humidity_bias": {"columns": list(self.humidity_cols), "scale": self.rh_scale,
                              "relative_change": round(self.rh_scale - 1.0, 4)},
            "evening_appliance_load": {
                "hours": f"{self.evening_start_hour:02d}:00-{self.evening_end_hour - 1:02d}:59",
                "scale": self.evening_load_scale, "offset_wh": self.evening_load_offset_wh,
            },
        }


@dataclass(frozen=True)
class RetrainingPolicy:
    """Retrain when inputs drift AND the model measurably degrades."""
    max_rmse_degradation: float = 0.10  # relative RMSE increase vs the baseline window
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)


def _clip_to_bounds(frame: pd.DataFrame, col: str) -> pd.Series:
    """Keep shifted values inside the serving contract so the model still accepts them."""
    lo, hi = PHYSICAL_BOUNDS["Appliances_lag_1" if col == TARGET_COL else col]
    return frame[col].clip(lower=lo, upper=hi)


def apply_scenario(raw: pd.DataFrame, scenario: DriftScenario = DriftScenario()) -> pd.DataFrame:
    """Return a shifted copy of the raw time series (same rows, same timestamps)."""
    out = raw.copy()
    for col in scenario.temperature_cols:
        out[col] = out[col] + scenario.temp_shift_c
        out[col] = _clip_to_bounds(out, col)
    for col in scenario.humidity_cols:
        out[col] = out[col] * scenario.rh_scale
        out[col] = _clip_to_bounds(out, col)
    hours = out["date"].dt.hour
    evening = (hours >= scenario.evening_start_hour) & (hours < scenario.evening_end_hour)
    out[TARGET_COL] = out[TARGET_COL].astype(float)
    out.loc[evening, TARGET_COL] = (out.loc[evening, TARGET_COL] * scenario.evening_load_scale
                                    + scenario.evening_load_offset_wh)
    out[TARGET_COL] = _clip_to_bounds(out, TARGET_COL)
    return out


def build_windows(raw: pd.DataFrame, scenario: DriftScenario) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(baseline test split, drifted test split) over identical timestamps."""
    _, _, baseline = chronological_split(build_feature_matrix(raw))
    _, _, drifted = chronological_split(build_feature_matrix(apply_scenario(raw, scenario)))
    baseline, drifted = baseline.reset_index(drop=True), drifted.reset_index(drop=True)
    if not baseline["date"].equals(drifted["date"]):
        raise AssertionError("baseline and drifted windows must share timestamps")
    return baseline, drifted


def _drift_summary(result: dict) -> dict:
    ds = result["dataset_drift"]
    scores = [v["score"] for v in result["features"].values()]
    summary = {
        "dataset_drift_detected": ds["detected"],
        "drifted_feature_share": ds["share"],
        "drifted_count": ds["drifted_count"],
        "monitored_count": ds["monitored_count"],
        "drifted_columns": ds["drifted_columns"],
        "mean_feature_score": round(float(np.mean(scores)), 6),
        "max_feature_score": round(float(np.max(scores)), 6),
        "alerts": result["alerts"],
    }
    for name in ("prediction", "target"):
        if name in result:
            summary[f"{name}_drift"] = {"score": result[name]["score"], "drifted": result[name]["drifted"]}
    return summary


def _pct(new: float, old: float) -> Optional[float]:
    return round((new - old) / abs(old), 4) if old else None


def decide_retraining(baseline_metrics: dict, drifted_metrics: dict, drifted_drift: dict,
                      config: DriftConfig, policy: RetrainingPolicy) -> dict:
    gate = evaluate_quality_gates(drifted_metrics, policy.thresholds)
    rmse_change = _pct(drifted_metrics["rmse"], baseline_metrics["rmse"])
    breaches = {
        "dataset_drift": {
            "breached": drifted_drift["dataset_drift_detected"],
            "observed_share": drifted_drift["drifted_feature_share"],
            "threshold_share": config.drift_share,
        },
        "prediction_drift": {
            "breached": drifted_drift.get("prediction_drift", {}).get("drifted", False),
            "observed_score": drifted_drift.get("prediction_drift", {}).get("score"),
            "threshold": config.threshold,
        },
        "target_drift": {
            "breached": drifted_drift.get("target_drift", {}).get("drifted", False),
            "observed_score": drifted_drift.get("target_drift", {}).get("score"),
            "threshold": config.threshold,
        },
        "quality_gate": {"breached": not gate.passed, "failures": gate.failures,
                         "thresholds": asdict(policy.thresholds)},
        "rmse_degradation": {
            "breached": rmse_change is not None and rmse_change >= policy.max_rmse_degradation,
            "observed_relative_change": rmse_change,
            "threshold_relative_change": policy.max_rmse_degradation,
        },
    }
    performance_breach = breaches["quality_gate"]["breached"] or breaches["rmse_degradation"]["breached"]
    triggered = breaches["dataset_drift"]["breached"] and performance_breach
    reasons = []
    if breaches["dataset_drift"]["breached"]:
        reasons.append(f"dataset drift: {drifted_drift['drifted_count']}/{drifted_drift['monitored_count']} "
                       f"features drifted (share {drifted_drift['drifted_feature_share']:.2f} "
                       f">= {config.drift_share})")
    for name in ("prediction_drift", "target_drift"):
        if breaches[name]["breached"]:
            reasons.append(f"{name.replace('_', ' ')}: score {breaches[name]['observed_score']:.4f} "
                           f">= {config.threshold}")
    reasons += [f"quality gate: {f}" for f in gate.failures]
    if breaches["rmse_degradation"]["breached"]:
        reasons.append(f"RMSE up {rmse_change:.1%} vs baseline (limit {policy.max_rmse_degradation:.0%})")
    return {
        "rule": "retrain when dataset drift is detected AND performance breaches "
                "(quality gate failure or RMSE degradation over the limit)",
        "threshold_breaches": breaches,
        "retraining_triggered": bool(triggered),
        "trigger_reasons": reasons if triggered else [],
    }


def run_experiment(model, raw: Optional[pd.DataFrame] = None, reference: Optional[pd.DataFrame] = None,
                   scenario: DriftScenario = DriftScenario(), config: DriftConfig = DriftConfig(),
                   policy: RetrainingPolicy = RetrainingPolicy(), model_info: Optional[dict] = None):
    """Return (evidence dict, drifted drift result, drifted Evidently snapshot)."""
    raw = load_raw_data() if raw is None else raw
    reference = load_reference() if reference is None else reference
    baseline, drifted = build_windows(raw, scenario)

    reference = attach_predictions(reference, model)
    baseline = attach_predictions(baseline, model)
    drifted = attach_predictions(drifted, model)

    reference_metrics = compute_metrics(reference[TARGET_COL], reference[PREDICTION_COL])
    baseline_metrics = compute_metrics(baseline[TARGET_COL], baseline[PREDICTION_COL])
    drifted_metrics = compute_metrics(drifted[TARGET_COL], drifted[PREDICTION_COL])

    baseline_result, _ = compute_drift(reference, baseline, config)
    drifted_result, snapshot = compute_drift(reference, drifted, config)
    baseline_drift, drifted_drift = _drift_summary(baseline_result), _drift_summary(drifted_result)

    window = {"split": "test", "rows": len(drifted),
              "start": drifted["date"].iloc[0].isoformat(), "end": drifted["date"].iloc[-1].isoformat()}
    source = f"deliberate drift experiment: {EXPERIMENT_NAME} (test-split window)"
    drifted_result["current"].update(source=source, window=window)
    drifted_result["experiment"] = {"name": EXPERIMENT_NAME, "scenario": scenario.describe()}
    if model_info:
        drifted_result["model"] = model_info

    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "name": EXPERIMENT_NAME,
            "description": "Summer heatwave with humidity sensor bias and heavier evening appliance "
                           "bursts, applied to the raw series; features rebuilt with the training code.",
            "scenario": scenario.describe(),
            "window": window,
            "reproduce": "python scripts/run_drift_experiment.py",
        },
        "model": model_info or {},
        "drift_config": {"method": config.method, "threshold": config.threshold,
                         "drift_share": config.drift_share},
        "reference": {"split": "train", "rows": len(reference), "metrics": reference_metrics},
        "baseline": {
            "description": "unshifted test split (same timestamps as the drifted batch)",
            "rows": len(baseline), "metrics": baseline_metrics, "drift": baseline_drift,
        },
        "drifted": {"rows": len(drifted), "metrics": drifted_metrics, "drift": drifted_drift},
        "degradation_vs_baseline": {
            "rmse": {"delta": round(drifted_metrics["rmse"] - baseline_metrics["rmse"], 4),
                     "relative": _pct(drifted_metrics["rmse"], baseline_metrics["rmse"])},
            "mae": {"delta": round(drifted_metrics["mae"] - baseline_metrics["mae"], 4),
                    "relative": _pct(drifted_metrics["mae"], baseline_metrics["mae"])},
            "r2": {"delta": round(drifted_metrics["r2"] - baseline_metrics["r2"], 4)},
        },
        "baseline_decision": decide_retraining(baseline_metrics, baseline_metrics, baseline_drift, config, policy),
        **decide_retraining(baseline_metrics, drifted_metrics, drifted_drift, config, policy),
    }
    return evidence, drifted_result, snapshot


def save_evidence(evidence: dict, drift_result: dict, snapshot, html_path: str = DRIFT_HTML_PATH,
                  metrics_path: str = DRIFT_JSON_PATH, evidence_path: str = EVIDENCE_PATH) -> None:
    evidence["artifacts"] = {
        "drift_report_html": os.path.relpath(html_path, PROJECT_ROOT).replace(os.sep, "/"),
        "drift_metrics_json": os.path.relpath(metrics_path, PROJECT_ROOT).replace(os.sep, "/"),
    }
    save_reports(drift_result, snapshot, html_path, metrics_path)
    os.makedirs(os.path.dirname(os.path.abspath(evidence_path)), exist_ok=True)
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
        f.write("\n")


def parse_args(argv=None):
    defaults = DriftScenario()
    p = argparse.ArgumentParser(description="Deliberate drift and model degradation experiment.")
    p.add_argument("--temp-shift", type=float, default=defaults.temp_shift_c, help="degC added to temperatures.")
    p.add_argument("--rh-scale", type=float, default=defaults.rh_scale, help="Multiplier for humidities.")
    p.add_argument("--evening-scale", type=float, default=defaults.evening_load_scale)
    p.add_argument("--evening-offset", type=float, default=defaults.evening_load_offset_wh)
    p.add_argument("--max-rmse-degradation", type=float, default=RetrainingPolicy.max_rmse_degradation)
    p.add_argument("--html", default=DRIFT_HTML_PATH)
    p.add_argument("--metrics", default=DRIFT_JSON_PATH)
    p.add_argument("--evidence", default=EVIDENCE_PATH)
    p.add_argument("--dry-run", action="store_true", help="Print the results without writing files.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from src.api.model_loader import load_serving_model
    from src.models.registry import REGISTERED_MODEL, tracking_uri

    args = parse_args(argv)
    scenario = DriftScenario(temp_shift_c=args.temp_shift, rh_scale=args.rh_scale,
                             evening_load_scale=args.evening_scale,
                             evening_load_offset_wh=args.evening_offset)
    policy = RetrainingPolicy(max_rmse_degradation=args.max_rmse_degradation)

    loaded = load_serving_model(tracking_uri(), REGISTERED_MODEL, os.path.join(PROJECT_ROOT, "models", "champion"))
    if loaded is None:
        print("No serving model available; export or register a champion first.", file=sys.stderr)
        return 1
    info = {"name": loaded.name, "version": loaded.version, "source": loaded.source}

    evidence, result, snapshot = run_experiment(loaded.model, scenario=scenario, policy=policy, model_info=info)
    if not args.dry_run:
        save_evidence(evidence, result, snapshot, args.html, args.metrics, args.evidence)

    b, d = evidence["baseline"], evidence["drifted"]
    print(f"  Model          : {info['name']} v{info['version']} ({info['source']})")
    print(f"  Window         : {evidence['experiment']['window']['start']} -> "
          f"{evidence['experiment']['window']['end']} ({d['rows']:,} rows)")
    print(f"  {'':15s}  {'RMSE':>8s} {'MAE':>8s} {'R2':>8s}  drifted features")
    for label, block in (("Reference", evidence["reference"]), ("Baseline test", b), ("Drifted test", d)):
        m = block["metrics"]
        drift = block.get("drift")
        share = f"{drift['drifted_count']}/{drift['monitored_count']}" if drift else "-"
        print(f"  {label:15s}  {m['rmse']:8.2f} {m['mae']:8.2f} {m['r2']:8.3f}  {share}")
    print(f"  Baseline would retrain : {evidence['baseline_decision']['retraining_triggered']}")
    print(f"  Retraining triggered   : {evidence['retraining_triggered']}")
    for reason in evidence["trigger_reasons"]:
        print(f"    - {reason}")
    if not args.dry_run:
        print(f"  Saved -> {args.html}\n           {args.metrics}\n           {args.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
