"""ZenML training pipeline for chronological appliance-energy forecasting.

The pipeline deliberately reuses the project's existing data and feature
functions.  This keeps the DVC-managed input contract and the PyTest temporal
guarantees unchanged while ZenML versions the intermediate pipeline artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from zenml import pipeline, step
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from zenml.materializers.cloudpickle_materializer import CloudpickleMaterializer

from src.data.load_data import DATE_COL, load_raw_data
from src.features.feature_engineering import (
    ALL_FEATURE_COLS,
    build_feature_matrix,
    chronological_split,
    get_X_y,
)
from src.models.evaluate import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "energydata_complete.csv"
EXPERIMENT_NAME = "energy-consumption-prediction"
MLFLOW_TRACKER_NAME = "mlflow_tracker"
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Each tracked step becomes a nested MLflow run owned by the ZenML pipeline run.
MLFLOW_SETTINGS = {
    "experiment_tracker": MLFlowExperimentTrackerSettings(
        experiment_name=EXPERIMENT_NAME,
        nested=True,
        tags={"project": "smart-energy-monitoring", "pipeline": "zenml"},
    )
}


@dataclass
class TrainedModels:
    """Fitted estimators and their reproducible training parameters."""

    models: dict[str, Any]
    hyperparameters: dict[str, dict[str, Any]]


def _assert_strictly_chronological(frame: pd.DataFrame, name: str) -> None:
    """Raise a clear error when a frame is not ordered by its timestamp."""
    if frame.empty:
        raise ValueError(f"{name} is empty.")
    if not frame[DATE_COL].is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted chronologically by '{DATE_COL}'.")
    if frame[DATE_COL].duplicated().any():
        raise ValueError(f"{name} contains duplicate '{DATE_COL}' values.")


@step(enable_cache=False)
def ingest_data_step(raw_data_path: str = str(DEFAULT_RAW_DATA_PATH)) -> pd.DataFrame:
    """Load DVC-materialized raw CSV data and validate chronological ordering.

    The step is read-only: it never calls DVC or changes the tracked CSV.
    Run ``dvc pull`` before this pipeline when the raw pointer is not materialized.
    """
    raw_data = load_raw_data(raw_data_path)
    _assert_strictly_chronological(raw_data, "raw data")
    return raw_data


@step(enable_cache=False)
def preprocess_data_step(
    raw_data: pd.DataFrame,
) -> tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "val_df"],
    Annotated[pd.DataFrame, "test_df"],
    Annotated[dict[str, str], "split_boundaries"],
]:
    """Create leak-free features and strict 70/15/15 chronological partitions."""
    _assert_strictly_chronological(raw_data, "raw data")
    feature_data = build_feature_matrix(raw_data)
    train_df, val_df, test_df = chronological_split(feature_data)

    for split_name, split_frame in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        _assert_strictly_chronological(split_frame, f"{split_name} split")
    if not (train_df[DATE_COL].max() < val_df[DATE_COL].min() < test_df[DATE_COL].min()):
        raise ValueError("Chronological split boundaries overlap or are out of order.")

    boundaries = {
        "train_min_date": train_df[DATE_COL].min().isoformat(),
        "train_max_date": train_df[DATE_COL].max().isoformat(),
        "val_min_date": val_df[DATE_COL].min().isoformat(),
        "val_max_date": val_df[DATE_COL].max().isoformat(),
        "test_min_date": test_df[DATE_COL].min().isoformat(),
        "test_max_date": test_df[DATE_COL].max().isoformat(),
        "train_rows": str(len(train_df)),
        "val_rows": str(len(val_df)),
        "test_rows": str(len(test_df)),
    }
    return train_df, val_df, test_df, boundaries


@step(
    experiment_tracker=MLFLOW_TRACKER_NAME,
    settings=MLFLOW_SETTINGS,
    enable_cache=False,
    output_materializers=CloudpickleMaterializer,
)
def train_models_step(
    train_df: pd.DataFrame,
    split_boundaries: dict[str, str],
) -> TrainedModels:
    """Fit the Phase 1 estimators and log parameters/models to MLflow.

    The persistence baseline is parameter-free and evaluated later from
    ``Appliances_lag_1``. Linear Regression is explicitly logged as the
    preferred RMSE/R² candidate from the project's Phase 1 benchmark.
    """
    _assert_strictly_chronological(train_df, "train split")
    x_train, y_train = get_X_y(train_df)
    random_forest_params: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 4,
        "random_state": 42,
        "n_jobs": -1,
    }
    xgboost_params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    models: dict[str, Any] = {
        "Persistence Baseline": None,
        "Linear Regression": LinearRegression().fit(x_train, y_train),
        "Random Forest": RandomForestRegressor(**random_forest_params).fit(x_train, y_train),
        "XGBoost": XGBRegressor(**xgboost_params).fit(x_train, y_train),
    }
    hyperparameters = {
        "Persistence Baseline": {"prediction": "Appliances_lag_1"},
        "Linear Regression": {"fit_intercept": True},
        "Random Forest": random_forest_params,
        "XGBoost": xgboost_params,
    }

    mlflow.log_params({f"split.{key}": value for key, value in split_boundaries.items()})
    mlflow.log_param("feature_count", len(ALL_FEATURE_COLS))
    for model_name, params in hyperparameters.items():
        safe_name = model_name.lower().replace(" ", "_")
        mlflow.log_params({f"{safe_name}.{key}": value for key, value in params.items()})

    # Native MLflow flavors keep artifacts browseable and deployable outside ZenML.
    mlflow.sklearn.log_model(
        models["Linear Regression"],
        name="linear_regression_candidate",
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
    mlflow.sklearn.log_model(
        models["Random Forest"],
        name="random_forest_candidate",
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
    mlflow.xgboost.log_model(models["XGBoost"], name="xgboost_candidate")
    return TrainedModels(models=models, hyperparameters=hyperparameters)


@step(
    experiment_tracker=MLFLOW_TRACKER_NAME,
    settings=MLFLOW_SETTINGS,
    enable_cache=False,
)
def evaluate_models_step(
    trained_models: TrainedModels,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_boundaries: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Calculate and log MAE, RMSE, and R² for validation and test data."""
    _assert_strictly_chronological(val_df, "validation split")
    _assert_strictly_chronological(test_df, "test split")
    if not val_df[DATE_COL].max() < test_df[DATE_COL].min():
        raise ValueError("Validation and test split boundaries overlap.")

    results: dict[str, dict[str, float]] = {}
    for split_name, split_df in (("val", val_df), ("test", test_df)):
        features, target = get_X_y(split_df)
        for model_name, model in trained_models.models.items():
            predictions = (
                features["Appliances_lag_1"].to_numpy()
                if model_name == "Persistence Baseline"
                else model.predict(features)
            )
            metrics = compute_metrics(target, predictions)
            results[f"{model_name}.{split_name}"] = metrics
            metric_prefix = f"{model_name.lower().replace(' ', '_')}.{split_name}"
            mlflow.log_metrics({f"{metric_prefix}.{name}": value for name, value in metrics.items()})

    mlflow.log_params({f"evaluated_split.{key}": value for key, value in split_boundaries.items()})
    return results


@pipeline(name="energy_monitoring_training_pipeline", enable_cache=False)
def energy_monitoring_training_pipeline(
    raw_data_path: str = str(DEFAULT_RAW_DATA_PATH),
) -> dict[str, dict[str, float]]:
    """Run ingestion, feature construction, training, and chronological evaluation."""
    raw_data = ingest_data_step(raw_data_path=raw_data_path)
    train_df, val_df, test_df, split_boundaries = preprocess_data_step(raw_data=raw_data)
    trained_models = train_models_step(train_df=train_df, split_boundaries=split_boundaries)
    return evaluate_models_step(
        trained_models=trained_models,
        val_df=val_df,
        test_df=test_df,
        split_boundaries=split_boundaries,
    )
