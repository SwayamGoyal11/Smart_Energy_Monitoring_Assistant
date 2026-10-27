"""
tests/test_hurdle.py
---------------------
Tests for the Two-Stage Hurdle Model: routing behaviour on synthetic data,
scikit-learn compatibility, and release-gate quality on the real splits.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from src.data.load_data import load_raw_data
from src.features.feature_engineering import build_feature_matrix, chronological_split, get_X_y
from src.models.evaluate import compute_metrics
from src.models.hurdle import PERSISTENCE_FEATURE, HurdleModel
from src.models.quality_gates import evaluate_quality_gates
from src.validation.input_validation import validate_feature_frame


@pytest.fixture
def synthetic():
    """Bursts happen exactly when `switch` > 0; burst size is 300 + 10 * load."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({
        "switch": rng.normal(size=n),
        "load": rng.uniform(0, 10, size=n),
        PERSISTENCE_FEATURE: rng.uniform(40, 60, size=n),
    })
    active = X["switch"] > 0
    y = np.where(active, 300 + 10 * X["load"], X[PERSISTENCE_FEATURE])
    return X, y, active.to_numpy()


class TestHurdleMechanics:
    def test_standby_routes_to_persistence(self, synthetic):
        X, y, active = synthetic
        model = HurdleModel(regressor=LinearRegression()).fit(X, y)
        standby = (model.predict_active_proba(X) < 0.5)
        np.testing.assert_allclose(model.predict(X)[standby], X[PERSISTENCE_FEATURE][standby])

    def test_active_routes_to_regressor(self, synthetic):
        X, y, active = synthetic
        model = HurdleModel(regressor=LinearRegression()).fit(X, y)
        routed = model.predict_active_proba(X) >= 0.5
        np.testing.assert_allclose(model.predict(X)[routed], model.regressor_.predict(X[routed]))

    def test_classifier_separates_regimes(self, synthetic):
        X, y, active = synthetic
        model = HurdleModel().fit(X, y)
        accuracy = ((model.predict_active_proba(X) >= 0.5) == active).mean()
        assert accuracy > 0.95

    def test_regressor_trained_on_active_rows_only(self, synthetic):
        X, y, active = synthetic
        model = HurdleModel(regressor=LinearRegression()).fit(X, y)
        assert model.regressor_.coef_[list(X.columns).index("load")] == pytest.approx(10.0)
        assert model.active_rate_ == pytest.approx(active.mean())

    def test_predictions_non_negative(self, synthetic):
        X, y, _ = synthetic
        model = HurdleModel(regressor=LinearRegression()).fit(X, y)
        X_neg = X.assign(load=-1000.0)  # drives the linear burst forecast below zero
        assert (model.predict(X_neg) >= 0).all()

    def test_decision_threshold_shifts_routing(self, synthetic):
        X, y, _ = synthetic
        eager = HurdleModel(decision_threshold=0.05).fit(X, y)
        strict = HurdleModel(decision_threshold=0.95).fit(X, y)
        n_eager = (eager.predict_active_proba(X) >= 0.05).sum()
        n_strict = (strict.predict_active_proba(X) >= 0.95).sum()
        assert n_eager > n_strict


class TestHurdleValidation:
    def test_requires_persistence_feature(self, synthetic):
        X, y, _ = synthetic
        with pytest.raises(ValueError, match=PERSISTENCE_FEATURE):
            HurdleModel().fit(X.drop(columns=[PERSISTENCE_FEATURE]), y)

    def test_single_class_rejected(self, synthetic):
        X, y, _ = synthetic
        with pytest.raises(ValueError, match="one class"):
            HurdleModel(active_threshold_wh=10_000).fit(X, y)

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5])
    def test_bad_decision_threshold_rejected(self, synthetic, bad):
        X, y, _ = synthetic
        with pytest.raises(ValueError, match="decision_threshold"):
            HurdleModel(decision_threshold=bad).fit(X, y)

    def test_predict_before_fit_raises(self, synthetic):
        X, _, _ = synthetic
        with pytest.raises(NotFittedError):
            HurdleModel().predict(X)

    def test_sklearn_clone_and_params(self):
        model = HurdleModel(active_threshold_wh=80.0)
        cloned = clone(model)
        assert cloned.get_params()["active_threshold_wh"] == 80.0
        assert not hasattr(cloned, "classifier_")


# Real data

@pytest.fixture(scope="module")
def fitted():
    train, val, test = chronological_split(build_feature_matrix(load_raw_data()))
    X_train, y_train = get_X_y(train)
    return HurdleModel().fit(validate_feature_frame(X_train), y_train), val, test


def _metrics(model, frame):
    X, y = get_X_y(frame)
    X = validate_feature_frame(X)
    return compute_metrics(y, model.predict(X)), compute_metrics(y, X[PERSISTENCE_FEATURE])


class TestHurdleQuality:
    @pytest.mark.parametrize("split", ["val", "test"])
    def test_passes_release_gates(self, fitted, split):
        model, val, test = fitted
        metrics, baseline = _metrics(model, val if split == "val" else test)
        result = evaluate_quality_gates(metrics, baseline_metrics=baseline)
        assert result.passed, result.failures

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_beats_persistence_mae(self, fitted, split):
        """The point of the hurdle design: beat persistence where it is strongest."""
        model, val, test = fitted
        metrics, baseline = _metrics(model, val if split == "val" else test)
        assert metrics["mae"] < baseline["mae"]

    def test_active_rate_matches_threshold(self, fitted):
        model, _, _ = fitted
        assert 0.15 < model.active_rate_ < 0.30
