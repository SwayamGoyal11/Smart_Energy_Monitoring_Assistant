"""
tests/test_registry.py
-----------------------
Tests for quality-gated model registration against a throwaway SQLite
tracking store, so the project's mlflow.db is never touched.
"""

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
