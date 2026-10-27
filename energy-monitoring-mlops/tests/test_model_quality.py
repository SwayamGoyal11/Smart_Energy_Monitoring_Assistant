"""
tests/test_model_quality.py
----------------------------
Model risk tests: release thresholds, baseline comparison, determinism and
prediction sanity for the Linear Regression candidate on the chronological
validation and test splits.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.data.load_data import load_raw_data
from src.features.feature_engineering import build_feature_matrix, chronological_split, get_X_y
from src.models.evaluate import compute_metrics
from src.models.quality_gates import QualityThresholds, evaluate_quality_gates
from src.validation.input_validation import PHYSICAL_BOUNDS, validate_feature_frame

GOOD = {"mae": 28.0, "rmse": 60.0, "r2": 0.57}
BASELINE = {"mae": 26.2, "rmse": 66.4, "r2": 0.48}


# Gate logic (no data needed)

class TestQualityGateLogic:
    def test_good_metrics_pass(self):
        result = evaluate_quality_gates(GOOD, baseline_metrics=BASELINE)
        assert result.passed and result.failures == []

    @pytest.mark.parametrize("metric, value, expected", [
        ("mae", 40.0, "MAE"), ("rmse", 70.0, "RMSE"), ("r2", 0.2, "R2"),
    ])
    def test_single_threshold_failure(self, metric, value, expected):
        result = evaluate_quality_gates({**GOOD, metric: value})
        assert not result.passed
        assert len(result.failures) == 1 and result.failures[0].startswith(expected)

    def test_worse_than_baseline_fails(self):
        loose = QualityThresholds(max_rmse=100.0)
        result = evaluate_quality_gates({**GOOD, "rmse": 67.0}, loose, baseline_metrics=BASELINE)
        assert not result.passed
        assert "persistence baseline" in result.failures[0]

    def test_baseline_check_skipped_without_baseline(self):
        loose = QualityThresholds(max_rmse=100.0)
        assert evaluate_quality_gates({**GOOD, "rmse": 67.0}, loose).passed

    def test_threshold_boundary_is_inclusive(self):
        t = QualityThresholds()
        edge = {"mae": t.max_mae, "rmse": t.max_rmse, "r2": t.min_r2}
        assert evaluate_quality_gates(edge, t).passed

    def test_all_failures_collected(self):
        result = evaluate_quality_gates({"mae": 99.0, "rmse": 99.0, "r2": 0.0}, baseline_metrics=BASELINE)
        assert len(result.failures) == 4

    def test_missing_metric_key_raises(self):
        with pytest.raises(KeyError):
            evaluate_quality_gates({"mae": 1.0, "rmse": 1.0})


# Trained candidate on real data

@pytest.fixture(scope="module")
def splits():
    return chronological_split(build_feature_matrix(load_raw_data()))


@pytest.fixture(scope="module")
def candidate(splits):
    train, _, _ = splits
    X_train, y_train = get_X_y(train)
    return LinearRegression().fit(validate_feature_frame(X_train), y_train)


def _evaluate(model, frame):
    X, y = get_X_y(frame)
    preds = model.predict(validate_feature_frame(X))
    return y.to_numpy(), preds, compute_metrics(y, preds), compute_metrics(y, X["Appliances_lag_1"])


class TestCandidateModelQuality:
    @pytest.mark.parametrize("split_idx, split_name", [(1, "validation"), (2, "test")])
    def test_passes_release_gates(self, splits, candidate, split_idx, split_name):
        _, _, metrics, baseline = _evaluate(candidate, splits[split_idx])
        result = evaluate_quality_gates(metrics, baseline_metrics=baseline)
        assert result.passed, f"{split_name} gate failures: {result.failures}"

    @pytest.mark.parametrize("split_idx", [1, 2])
    def test_generalisation_gap_is_bounded(self, splits, candidate, split_idx):
        _, _, train_metrics, _ = _evaluate(candidate, splits[0])
        _, _, held_out, _ = _evaluate(candidate, splits[split_idx])
        assert held_out["rmse"] <= 1.5 * train_metrics["rmse"]

    def test_predictions_finite(self, splits, candidate):
        _, preds, _, _ = _evaluate(candidate, splits[2])
        assert np.isfinite(preds).all()

    def test_predictions_physically_plausible(self, splits, candidate):
        _, preds, _, _ = _evaluate(candidate, splits[2])
        lo, hi = PHYSICAL_BOUNDS["Appliances_lag_1"]
        assert (preds >= lo).all() and (preds <= hi).all()

    def test_mean_bias_is_small(self, splits, candidate):
        y, preds, _, _ = _evaluate(candidate, splits[2])
        assert abs(np.mean(y - preds)) < 10.0, "systematic over/under-forecasting"

    def test_training_is_deterministic(self, splits, candidate):
        train, _, test = splits
        X_train, y_train = get_X_y(train)
        X_test, _ = get_X_y(test)
        refit = LinearRegression().fit(X_train, y_train)
        np.testing.assert_allclose(refit.predict(X_test), candidate.predict(X_test))

    def test_prediction_independent_of_batch(self, splits, candidate):
        X_test, _ = get_X_y(splits[2])
        X_test = validate_feature_frame(X_test)
        batch = candidate.predict(X_test.iloc[:50])
        single = np.array([candidate.predict(X_test.iloc[[i]])[0] for i in range(50)])
        np.testing.assert_allclose(batch, single)

    def test_responds_to_recent_consumption(self, splits, candidate):
        X_test, _ = get_X_y(splits[2])
        row = validate_feature_frame(X_test.iloc[[0]])
        spike = row.copy()
        for col in ["Appliances_lag_1", "Appliances_rolling_mean_3", "Appliances_rolling_mean_6"]:
            spike[col] = row[col] + 300.0
        assert candidate.predict(spike)[0] > candidate.predict(row)[0]
