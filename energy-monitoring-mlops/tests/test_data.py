"""
tests/test_data.py
-------------------
PyTest tests for dataset loading, feature engineering, split integrity, and metrics.
Run with: python -m pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_raw_data, validate_dataset, EXPECTED_COLUMNS
from src.features.feature_engineering import (
    build_feature_matrix, chronological_split, get_X_y,
    ALL_FEATURE_COLS, TIME_FEATURES, LAG_FEATURES, ROLLING_FEATURES,
)
from src.models.evaluate import compute_metrics


# Fixtures

@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()

@pytest.fixture(scope="module")
def feature_df(raw_df):
    return build_feature_matrix(raw_df)

@pytest.fixture(scope="module")
def splits(feature_df):
    return chronological_split(feature_df)


# Dataset Loading

class TestDatasetLoading:
    def test_loads_without_error(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_not_empty(self, raw_df):
        assert raw_df.shape[0] > 0

    def test_expected_row_count(self, raw_df):
        assert raw_df.shape[0] > 19_000, f"Expected ~19735 rows, got {raw_df.shape[0]}"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data(filepath="/nonexistent/path.csv")


# Expected Columns

class TestExpectedColumns:
    def test_all_expected_columns_present(self, raw_df):
        for col in EXPECTED_COLUMNS:
            assert col in raw_df.columns, f"Missing: {col}"

    def test_target_column_exists(self, raw_df):
        assert "Appliances" in raw_df.columns

    def test_date_is_datetime(self, raw_df):
        assert pd.api.types.is_datetime64_any_dtype(raw_df["date"])


# Missing Values

class TestMissingValues:
    def test_no_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_target_not_null(self, raw_df):
        assert raw_df["Appliances"].isnull().sum() == 0


# Chronological Ordering

class TestChronologicalOrdering:
    def test_sorted_chronologically(self, raw_df):
        diffs = raw_df["date"].diff().dropna()
        assert (diffs >= pd.Timedelta(0)).all()

    def test_10min_intervals(self, raw_df):
        diffs = raw_df["date"].diff().dropna()
        assert diffs.mode()[0].total_seconds() == 600

    def test_date_range_reasonable(self, raw_df):
        days = (raw_df["date"].max() - raw_df["date"].min()).days
        assert days > 100


# Feature Engineering

class TestFeatureEngineering:
    def test_all_feature_columns_created(self, feature_df):
        for col in ALL_FEATURE_COLS:
            assert col in feature_df.columns, f"Missing feature: {col}"

    def test_no_nan_in_features(self, feature_df):
        nan_count = feature_df[ALL_FEATURE_COLS].isnull().sum().sum()
        assert nan_count == 0

    def test_hour_range(self, feature_df):
        assert feature_df["hour"].between(0, 23).all()

    def test_is_weekend_binary(self, feature_df):
        assert feature_df["is_weekend"].isin([0, 1]).all()

    def test_lag1_matches_previous_row(self, feature_df):
        for i in range(1, 10):
            expected = feature_df["Appliances"].iloc[i - 1]
            actual   = feature_df["Appliances_lag_1"].iloc[i]
            assert abs(expected - actual) < 1e-6

    def test_feature_matrix_smaller_than_raw(self, raw_df, feature_df):
        assert len(feature_df) < len(raw_df)


# Train/Val/Test Split

class TestChronologicalSplit:
    def test_no_rows_lost(self, feature_df, splits):
        train, val, test = splits
        assert len(train) + len(val) + len(test) == len(feature_df)

    def test_train_fraction_approx_70pct(self, feature_df, splits):
        train, val, test = splits
        frac = len(train) / len(feature_df)
        assert 0.68 <= frac <= 0.72

    def test_no_temporal_overlap(self, splits):
        train, val, test = splits
        assert train["date"].max() < val["date"].min()
        assert val["date"].max()   < test["date"].min()

    def test_no_date_duplicates_across_splits(self, splits):
        train, val, test = splits
        assert len(set(train["date"]) & set(val["date"]))  == 0
        assert len(set(val["date"])   & set(test["date"])) == 0


# Metric Calculation

class TestMetrics:
    def test_perfect_prediction_mae_zero(self):
        y = np.array([100.0, 200.0, 150.0])
        assert compute_metrics(y, y)["mae"] == 0.0

    def test_perfect_prediction_r2_one(self):
        y = np.array([100.0, 200.0, 150.0])
        assert compute_metrics(y, y)["r2"] == 1.0

    def test_metrics_keys_present(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y + 0.1)
        assert all(k in m for k in ["mae", "rmse", "r2"])

    def test_rmse_geq_mae(self):
        y_true = np.array([100.0, 200.0, 300.0, 50.0])
        y_pred = np.array([120.0, 180.0, 350.0, 55.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] >= m["mae"]

    def test_baseline_metrics_valid(self, splits):
        train, val, test = splits
        X_val, y_val = get_X_y(val)
        preds = X_val["Appliances_lag_1"].values
        m = compute_metrics(y_val, preds)
        assert m["mae"] > 0
        assert -1.0 < m["r2"] <= 1.0

    def test_validate_dataset_returns_expected_keys(self, raw_df):
        results = validate_dataset(raw_df)
        for key in ["shape", "n_rows", "total_missing", "duplicate_rows",
                    "is_chronological", "target_stats"]:
            assert key in results
