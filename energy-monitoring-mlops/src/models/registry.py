"""
src/models/registry.py
-----------------------
MLflow Model Registry integration with quality-gated promotion.

Every candidate is logged and registered as a new version of REGISTERED_MODEL
so failed candidates stay auditable. Aliases are only assigned when the
candidate passes evaluate_quality_gates() on BOTH validation and test splits:

    staging  -> latest candidate that passed the gates
    champion -> best gated candidate by validation RMSE (the serving model)

Selection uses validation RMSE so the test split is never used to choose
between models; test is only a pass/fail release check.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from xgboost import XGBModel

from src.models.evaluate import compute_metrics
from src.models.quality_gates import QualityThresholds, evaluate_quality_gates
from src.models.train import EXPERIMENT_NAME, MLFLOW_URI
from src.features.feature_engineering import get_X_y
from src.validation.input_validation import validate_feature_frame

REGISTERED_MODEL = "energy-forecaster"
STAGING_ALIAS = "staging"
CHAMPION_ALIAS = "champion"
SELECTION_METRIC = "val_rmse"

# Types we serialize ourselves; skops refuses to load them unless trusted.
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "src.models.hurdle.HurdleModel",
]


@dataclass
class RegistrationResult:
    model_type: str
    version: str
    run_id: str
    gates_passed: bool
    gate_failures: list = field(default_factory=list)
    aliases: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI overrides the project's local SQLite store."""
    return os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _split_metrics(model, frame: pd.DataFrame) -> tuple:
    """Return (candidate metrics, persistence-baseline metrics) on one split."""
    X, y = get_X_y(frame)
    X = validate_feature_frame(X)
    return compute_metrics(y, model.predict(X)), compute_metrics(y, X["Appliances_lag_1"])


def _log_model(model, example: pd.DataFrame):
    """Log with the native flavor: XGBoost boosters are not skops-serializable."""
    if isinstance(model, XGBModel):
        return mlflow.xgboost.log_model(model, name="model", input_example=example)
    return mlflow.sklearn.log_model(
        model, name="model", input_example=example, skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )


def _champion_score(client: MlflowClient, name: str) -> Optional[float]:
    try:
        champion = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException:
        return None
    return client.get_run(champion.run_id).data.metrics.get(SELECTION_METRIC)


def register_candidate(
    model,
    model_type: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Optional[dict] = None,
    thresholds: QualityThresholds = QualityThresholds(),
    registered_model_name: str = REGISTERED_MODEL,
    experiment_name: str = EXPERIMENT_NAME,
) -> RegistrationResult:
    """
    Log a fitted model, evaluate the release gates, register a new version,
    and assign staging/champion aliases only when the gates pass.
    The caller is responsible for mlflow.set_tracking_uri().
    """
    val_m, val_base = _split_metrics(model, val_df)
    test_m, test_base = _split_metrics(model, test_df)
    val_gate = evaluate_quality_gates(val_m, thresholds, baseline_metrics=val_base)
    test_gate = evaluate_quality_gates(test_m, thresholds, baseline_metrics=test_base)
    failures = [f"val: {f}" for f in val_gate.failures] + [f"test: {f}" for f in test_gate.failures]
    passed = not failures

    metrics = {f"val_{k}": v for k, v in val_m.items()}
    metrics.update({f"test_{k}": v for k, v in test_m.items()})
    metrics.update({f"baseline_val_{k}": v for k, v in val_base.items()})
    metrics.update({f"baseline_test_{k}": v for k, v in test_base.items()})

    example = validate_feature_frame(get_X_y(val_df)[0]).head(5)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"registry_{model_type}") as run:
        mlflow.log_param("model", model_type)
        if params:
            mlflow.log_params(params)
        mlflow.log_params({f"gate.{k}": v for k, v in vars(thresholds).items()})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("quality_gate_passed", int(passed))
        mlflow.set_tags({
            "model_type": model_type,
            "quality_gate": "passed" if passed else "failed",
            "git_commit": _git_commit(),
        })
        if failures:
            mlflow.log_text("\n".join(failures), "quality_gate_failures.txt")
        info = _log_model(model, example)

    client = MlflowClient()
    version = mlflow.register_model(info.model_uri, registered_model_name).version
    client.set_model_version_tag(registered_model_name, version, "model_type", model_type)
    client.set_model_version_tag(
        registered_model_name, version, "quality_gate", "passed" if passed else "failed",
    )
    client.set_model_version_tag(registered_model_name, version, SELECTION_METRIC, str(val_m["rmse"]))

    aliases = []
    if passed:
        best = _champion_score(client, registered_model_name)
        client.set_registered_model_alias(registered_model_name, STAGING_ALIAS, version)
        aliases.append(STAGING_ALIAS)
        if best is None or val_m["rmse"] < best:
            client.set_registered_model_alias(registered_model_name, CHAMPION_ALIAS, version)
            aliases.append(CHAMPION_ALIAS)

    return RegistrationResult(
        model_type=model_type, version=str(version), run_id=run.info.run_id,
        gates_passed=passed, gate_failures=failures, aliases=aliases, metrics=metrics,
    )
