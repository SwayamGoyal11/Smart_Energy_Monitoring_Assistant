"""Non-invasive orchestration of the existing Phase 2 Hurdle workflow."""

import os
import time
from dataclasses import asdict
from typing import Annotated

import mlflow
import pandas as pd
import skops.io as sio
from zenml import pipeline, step
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)

from src.data.load_data import RAW_DATA_PATH, load_raw_data
from src.features.feature_engineering import (
    build_feature_matrix, chronological_split, get_X_y,
)
from src.models.hurdle import HurdleModel
from src.models.registry import (
    REGISTERED_MODEL, SKOPS_TRUSTED_TYPES, register_candidate,
)
from src.models.train import EXPERIMENT_NAME, MLFLOW_URI
from src.validation.input_validation import validate_feature_frame


class HurdleSkopsMaterializer(BaseMaterializer):
    """Persist ZenML model artifacts with the registry's explicit trust policy."""

    ASSOCIATED_TYPES = (HurdleModel,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def save(self, data: HurdleModel) -> None:
        with fileio.open(os.path.join(self.uri, "model.skops"), "wb") as stream:
            stream.write(sio.dumps(data))

    def load(self, data_type: type[HurdleModel]) -> HurdleModel:
        with fileio.open(os.path.join(self.uri, "model.skops"), "rb") as stream:
            return sio.loads(stream.read(), trusted=SKOPS_TRUSTED_TYPES)


@step(enable_cache=False)
def load_data(raw_data_path: str) -> Annotated[pd.DataFrame, "raw_data"]:
    return load_raw_data(raw_data_path)


@step(enable_cache=False)
def preprocess(raw_data: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "val_df"],
    Annotated[pd.DataFrame, "test_df"],
]:
    train_df, val_df, test_df = chronological_split(build_feature_matrix(raw_data))
    return train_df, val_df, test_df


@step(
    experiment_tracker="mlflow_tracker",
    settings={"experiment_tracker": MLFlowExperimentTrackerSettings(
        experiment_name=EXPERIMENT_NAME,
    )},
    output_materializers={"hurdle_model": HurdleSkopsMaterializer},
    enable_cache=False,
)
def train_hurdle(
    train_df: pd.DataFrame,
    active_threshold_wh: float,
    decision_threshold: float,
) -> tuple[
    Annotated[HurdleModel, "hurdle_model"],
    Annotated[float, "fit_seconds"],
    Annotated[str, "training_run_id"],
]:
    features, target = get_X_y(train_df)
    features = validate_feature_frame(features)
    params = dict(active_threshold_wh=active_threshold_wh,
                  decision_threshold=decision_threshold)
    start = time.perf_counter()
    model = HurdleModel(**params).fit(features, target)
    elapsed = time.perf_counter() - start
    mlflow.log_params(params)
    mlflow.log_metric("fit_seconds", elapsed)
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("The mlflow_tracker did not create a training run.")
    return model, elapsed, run.info.run_id


@step(enable_cache=False)
def log_and_register(
    model: HurdleModel,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit_seconds: float,
    training_run_id: str,
    mlflow_tracking_uri: str,
) -> Annotated[dict, "registration_result"]:
    # Deliberately no experiment_tracker here: register_candidate owns its run.
    # Logging and promotion are atomic at the existing helper's API boundary.
    if mlflow.active_run() is not None:
        raise RuntimeError("Registration requires no active MLflow run.")
    # ZenML's tracker cleanup resets MLflow globals after the training step.
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    result = register_candidate(
        model, "hurdle", val_df, test_df,
        params={"active_threshold_wh": model.active_threshold_wh,
                "decision_threshold": model.decision_threshold},
        registered_model_name=REGISTERED_MODEL,
        train_df=train_df, log_artifacts=True, fit_seconds=fit_seconds,
    )
    client = mlflow.MlflowClient()
    client.set_tag(result.run_id, "mlflow.parentRunId", training_run_id)
    client.set_tag(training_run_id, "registry_run_id", result.run_id)
    client.set_tag(training_run_id, "registered_model_version", result.version)
    if not result.gates_passed:
        raise RuntimeError(f"Hurdle quality gates failed: {result.gate_failures}")
    return asdict(result)


@pipeline(enable_cache=False)
def energy_hurdle_pipeline(
    raw_data_path: str = os.path.abspath(RAW_DATA_PATH),
    active_threshold_wh: float = 100.0,
    decision_threshold: float = 0.5,
    mlflow_tracking_uri: str = MLFLOW_URI,
):
    raw = load_data(raw_data_path=raw_data_path)
    train, val, test = preprocess(raw_data=raw)
    model, elapsed, run_id = train_hurdle(
        train_df=train, active_threshold_wh=active_threshold_wh,
        decision_threshold=decision_threshold,
    )
    log_and_register(model=model, train_df=train, val_df=val, test_df=test,
                     fit_seconds=elapsed, training_run_id=run_id,
                     mlflow_tracking_uri=mlflow_tracking_uri)
