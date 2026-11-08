"""
tests/test_registry.py
-----------------------
Tests for quality-gated model registration against a throwaway SQLite
tracking store, so the project's mlflow.db is never touched.
"""

from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from packaging.requirements import Requirement
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.data.load_data import load_raw_data
from src.features.feature_engineering import build_feature_matrix, chronological_split, get_X_y
from src.models.hurdle import HurdleModel
from src.models.quality_gates import QualityThresholds
from src.models.registry import (
    CHAMPION_ALIAS, INFERENCE_REQUIREMENTS, STAGING_ALIAS, XGBOOST_REQUIREMENT,
    register_candidate, tracking_uri,
)
from src.validation.input_validation import validate_feature_frame

MODEL_NAME = "energy-forecaster-test"
EXPERIMENT = "registry-test"


@pytest.fixture(scope="module")
def splits():
    return chronological_split(build_feature_matrix(load_raw_data()))


@pytest.fixture(scope="module")
def fit(splits):
    """Fit each model class once per module; re-registering a fitted model is fine."""
    X, y = get_X_y(splits[0])
    X = validate_feature_frame(X)
    cache = {}

    def _fit(model):
        key = type(model).__name__
        if key not in cache:
            cache[key] = model.fit(X, y)
        return cache[key]
    return _fit


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Point MLflow at a temporary SQLite DB and artifact root for this module."""
    root = tmp_path_factory.mktemp("mlflow")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    MlflowClient().create_experiment(EXPERIMENT, artifact_location=(root / "artifacts").as_uri())
    yield MlflowClient()
    mlflow.set_tracking_uri(previous)


def register(model, model_type, splits, **kwargs):
    _, val, test = splits
    return register_candidate(
        model, model_type, val, test,
        registered_model_name=MODEL_NAME, experiment_name=EXPERIMENT, **kwargs,
    )


class TestTrackingUri:
    def test_defaults_to_project_sqlite(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert tracking_uri().startswith("sqlite:///") and tracking_uri().endswith("mlflow.db")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        assert tracking_uri() == "http://mlflow:5000"


class TestGatedRegistration:
    """Ordered scenario: each test builds on the registry state left by the previous one."""

    def test_failing_candidate_registered_without_alias(self, store, splits, fit):
        result = register(fit(DummyRegressor(strategy="mean")), "dummy_mean", splits)
        assert not result.gates_passed and result.gate_failures
        assert result.aliases == []
        version = store.get_model_version(MODEL_NAME, result.version)
        assert version.tags["quality_gate"] == "failed"
        assert store.get_registered_model(MODEL_NAME).aliases == {}

    def test_first_passing_candidate_becomes_champion(self, store, splits, fit):
        result = register(fit(LinearRegression()), "linear_regression", splits)
        assert result.gates_passed
        assert set(result.aliases) == {STAGING_ALIAS, CHAMPION_ALIAS}
        assert store.get_model_version(MODEL_NAME, result.version).tags["quality_gate"] == "passed"

    def test_better_candidate_takes_champion(self, store, splits, fit):
        result = register(fit(HurdleModel()), "hurdle", splits)
        assert set(result.aliases) == {STAGING_ALIAS, CHAMPION_ALIAS}
        assert str(store.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version) == result.version

    def test_worse_passing_candidate_only_staged(self, store, splits, fit):
        champion_before = store.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version
        result = register(fit(LinearRegression()), "linear_regression", splits)
        assert result.aliases == [STAGING_ALIAS]
        assert str(store.get_model_version_by_alias(MODEL_NAME, STAGING_ALIAS).version) == result.version
        assert store.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version == champion_before

    def test_strict_thresholds_block_promotion(self, store, splits, fit):
        aliases_before = store.get_registered_model(MODEL_NAME).aliases
        strict = QualityThresholds(max_mae=1.0)
        result = register(fit(HurdleModel()), "hurdle", splits, thresholds=strict)
        assert not result.gates_passed
        assert any("MAE" in f for f in result.gate_failures)
        assert store.get_registered_model(MODEL_NAME).aliases == aliases_before

    def test_xgboost_logged_and_blocked_by_test_gate(self, store, splits, fit):
        aliases_before = store.get_registered_model(MODEL_NAME).aliases
        params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
                  "colsample_bytree": 0.8, "random_state": 42, "tree_method": "hist"}
        result = register(fit(XGBRegressor(**params)), "xgboost", splits)
        assert not result.gates_passed
        assert any(f.startswith("test:") for f in result.gate_failures)
        assert store.get_registered_model(MODEL_NAME).aliases == aliases_before

    def test_run_records_gate_outcome(self, store, splits, fit):
        result = register(fit(LinearRegression()), "linear_regression", splits)
        run = store.get_run(result.run_id)
        assert run.data.tags["quality_gate"] == "passed"
        assert run.data.metrics["quality_gate_passed"] == 1
        for key in ("val_mae", "val_rmse", "val_r2", "test_rmse", "baseline_val_rmse"):
            assert key in run.data.metrics

    def test_logged_requirements_are_inference_only(self, store, splits, fit):
        for model, model_type, expected in [
            (LinearRegression(), "linear_regression", INFERENCE_REQUIREMENTS),
            (XGBRegressor(n_estimators=10), "xgboost", INFERENCE_REQUIREMENTS + [XGBOOST_REQUIREMENT]),
        ]:
            result = register(fit(model), model_type, splits)
            path = mlflow.artifacts.download_artifacts(
                f"models:/{MODEL_NAME}/{result.version}/requirements.txt")
            with open(path) as f:
                logged = f.read().split()
            # MLflow rewrites specifiers in canonical order, so compare parsed
            assert [Requirement(r) for r in logged] == [Requirement(r) for r in expected]

    def test_champion_loads_and_predicts(self, store, splits):
        _, _, test = splits
        X, _ = get_X_y(test)
        X = validate_feature_frame(X).head(20)
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
        preds = np.asarray(model.predict(X))
        assert preds.shape == (20,) and np.isfinite(preds).all() and (preds >= 0).all()


def _artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> set:
    paths = set()
    for item in client.list_artifacts(run_id, path):
        paths |= _artifact_paths(client, run_id, item.path) if item.is_dir else {item.path}
    return paths


class TestRunArtifacts:
    """Full artifact layout (src/models/artifact_logging.py) for the champion model type."""

    EXPECTED = {
        "data/lineage.json", "data/split_summary.json", "data/data_summary.json",
        "features/feature_names.json", "features/feature_schema.json",
        "features/feature_importance.json", "features/feature_importance.png",
        "preprocessing/feature_encoders.json", "preprocessing/model_pipeline.html",
        "preprocessing/stage1_scaler_params.json", "preprocessing/stage1_pipeline.skops",
        "config/model_config.yaml", "config/hyperparameters.yaml",
        "config/quality_gates.json", "config/hurdle_thresholds.json",
        "evaluation/metrics_summary.json", "evaluation/residuals.png",
        "evaluation/stage1_confusion_matrix.png", "evaluation/stage1_classification_report.json",
        "evaluation/stage2_residuals.png",
        "logs/training.log", "logs/cv_convergence.json", "logs/cv_convergence.png",
        "metadata/run_metadata.json", "metadata/environment.json",
        "metadata/requirements_frozen.txt",
        "deployment/openapi.json", "deployment/api_summary.json",
        "deployment/request_example.json", "deployment/batch_request_example.json",
        "deployment/response_example.json", "deployment/example_scoring.json",
        "deployment/docker_spec.yaml", "deployment/Dockerfile",
        "monitoring/reference_stats.json", "monitoring/drift_config.json",
    }

    @pytest.fixture(scope="class")
    def hurdle_run(self, store, splits, fit):
        train, _, _ = splits
        result = register(fit(HurdleModel()), "hurdle", splits, train_df=train,
                          params={"active_threshold_wh": 100.0, "decision_threshold": 0.5},
                          fit_seconds=1.0)
        return result, _artifact_paths(store, result.run_id)

    def test_every_category_logged(self, hurdle_run):
        _, paths = hurdle_run
        assert self.EXPECTED - paths == set()

    def test_logged_model_is_skops_with_mlmodel(self, hurdle_run):
        result, _ = hurdle_run
        local = mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{result.version}")
        files = {p.name for p in Path(local).iterdir()}
        assert "MLmodel" in files and any(f.endswith(".skops") for f in files)

    def test_request_example_matches_training_features(self, store, hurdle_run):
        result, _ = hurdle_run
        run = store.get_run(result.run_id)
        assert run.data.tags["serving.feature_parity"] == "true"
        assert {"cv_heldout_mae_mean", "fit_seconds"} <= set(run.data.metrics)

    def test_training_log_records_cv_and_gates(self, hurdle_run):
        result, _ = hurdle_run
        path = mlflow.artifacts.download_artifacts(run_id=result.run_id,
                                                   artifact_path="logs/training.log")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "CV fold 5/5" in text and "Quality gates passed" in text and "Model logged" in text
