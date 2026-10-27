"""
src/models/quality_gates.py
----------------------------
Release thresholds a candidate model must meet before it can be registered
or served. Thresholds sit a margin below the Phase 1 Linear Regression
results (val MAE ~29 Wh, RMSE ~61 Wh, R2 ~0.56) so normal retraining noise
passes while a real regression fails.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class QualityThresholds:
    max_mae: float = 35.0
    max_rmse: float = 65.0
    min_r2: float = 0.50
    # Candidate RMSE must not exceed this multiple of the persistence baseline's RMSE.
    max_rmse_vs_baseline: float = 1.0


@dataclass
class GateResult:
    passed: bool
    failures: list = field(default_factory=list)


def evaluate_quality_gates(
    metrics: dict,
    thresholds: QualityThresholds = QualityThresholds(),
    baseline_metrics: Optional[dict] = None,
) -> GateResult:
    """Compare a compute_metrics() result against the release thresholds."""
    missing = [k for k in ("mae", "rmse", "r2") if k not in metrics]
    if missing:
        raise KeyError(f"metrics missing keys: {missing}")

    failures = []
    if metrics["mae"] > thresholds.max_mae:
        failures.append(f"MAE {metrics['mae']:.4f} > {thresholds.max_mae}")
    if metrics["rmse"] > thresholds.max_rmse:
        failures.append(f"RMSE {metrics['rmse']:.4f} > {thresholds.max_rmse}")
    if metrics["r2"] < thresholds.min_r2:
        failures.append(f"R2 {metrics['r2']:.4f} < {thresholds.min_r2}")

    if baseline_metrics is not None:
        limit = baseline_metrics["rmse"] * thresholds.max_rmse_vs_baseline
        if metrics["rmse"] > limit:
            failures.append(
                f"RMSE {metrics['rmse']:.4f} does not beat persistence baseline RMSE "
                f"{baseline_metrics['rmse']:.4f} (limit {limit:.4f})"
            )

    return GateResult(passed=not failures, failures=failures)
