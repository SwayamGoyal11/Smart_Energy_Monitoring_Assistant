"""
tests/test_evaluation.py
-------------------------
PyTest suite for evaluation analytics, sub-group slicing, residual diagnostics,
and feature importance extraction.
Run with: python -m pytest tests/test_evaluation.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.evaluation_analytics import (
    compute_extended_metrics,
    compute_regime_breakdown,
    compute_diurnal_breakdown,
    compute_residual_diagnostics,
    extract_feature_importances,
)


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2016-01-01", periods=n, freq="10min")
    # Mixture of standby (<=60), active (60-200), and peak (>200)
    y_true = np.concatenate([
        np.random.uniform(20, 50, size=200),
        np.random.uniform(70, 180, size=70),
        np.random.uniform(250, 600, size=30),
    ])
    noise = np.random.normal(0, 15, size=n)
    y_pred = np.maximum(10.0, y_true + noise)

    df = pd.DataFrame({
        "date": dates,
        "Appliances": y_true,
    })
    return df, y_true, y_pred


class TestExtendedMetrics:
    def test_metrics_keys_present(self, dummy_data):
        _, y_true, y_pred = dummy_data
        m = compute_extended_metrics(y_true, y_pred)
        for k in ["mae", "rmse", "r2", "wape_pct", "medae", "max_error", "mean_bias"]:
            assert k in m, f"Missing key: {k}"

    def test_perfect_prediction_values(self):
        y = np.array([50.0, 100.0, 250.0])
        m = compute_extended_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0
        assert m["wape_pct"] == 0.0
        assert m["mean_bias"] == 0.0

    def test_rmse_greater_or_equal_to_mae(self, dummy_data):
        _, y_true, y_pred = dummy_data
        m = compute_extended_metrics(y_true, y_pred)
        assert m["rmse"] >= m["mae"]


class TestRegimeBreakdown:
    def test_regime_covers_all_samples(self, dummy_data):
        df, y_true, y_pred = dummy_data
        reg_df = compute_regime_breakdown(df, y_true, y_pred)
        assert reg_df["Count"].sum() == len(y_true)

    def test_regime_columns_exist(self, dummy_data):
        df, y_true, y_pred = dummy_data
        reg_df = compute_regime_breakdown(df, y_true, y_pred)
        for col in ["Regime", "Count", "Share_pct", "MAE", "RMSE", "WAPE_pct", "Mean_Bias"]:
            assert col in reg_df.columns


class TestDiurnalBreakdown:
    def test_diurnal_segments_populated(self, dummy_data):
        df, y_true, y_pred = dummy_data
        diurnal_df = compute_diurnal_breakdown(df, y_true, y_pred)
        assert len(diurnal_df) > 0
        assert "Segment" in diurnal_df.columns
        assert "MAE" in diurnal_df.columns


class TestResidualDiagnostics:
    def test_residuals_finite_and_keyed(self, dummy_data):
        _, y_true, y_pred = dummy_data
        diag = compute_residual_diagnostics(y_true, y_pred)
        for k in ["mean_residual", "std_residual", "skewness", "kurtosis", "heteroscedasticity_corr"]:
            assert k in diag
            assert np.isfinite(diag[k])


class TestFeatureImportances:
    def test_extraction_with_mock_models(self):
        class MockTreeModel:
            feature_importances_ = np.array([0.5, 0.3, 0.2])

        class MockLinearModel:
            coef_ = np.array([1.2, -0.8, 0.4])

        models = {
            "Random Forest": MockTreeModel(),
            "Linear Regression": MockLinearModel(),
        }
        feats = ["feat_A", "feat_B", "feat_C"]
        df_imp = extract_feature_importances(models, feats)

        assert len(df_imp) == 3
        assert "Feature" in df_imp.columns
        assert "Mean_Importance" in df_imp.columns
        assert df_imp["Mean_Importance"].iloc[0] >= df_imp["Mean_Importance"].iloc[-1]
