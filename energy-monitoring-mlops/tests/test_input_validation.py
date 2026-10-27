"""
tests/test_input_validation.py
-------------------------------
Unit tests for the model input contract. Uses synthetic rows only, so these
run without the DVC-tracked dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import ALL_FEATURE_COLS
from src.validation.input_validation import (
    PHYSICAL_BOUNDS, InputValidationError, validate_feature_frame,
)

VALID_ROW = {
    "hour": 14, "day": 15, "day_of_week": 2, "month": 3, "is_weekend": 0,
    "hour_sin": np.sin(2 * np.pi * 14 / 24), "hour_cos": np.cos(2 * np.pi * 14 / 24),
    "day_of_week_sin": np.sin(2 * np.pi * 2 / 7), "day_of_week_cos": np.cos(2 * np.pi * 2 / 7),
    "month_sin": np.sin(2 * np.pi * 2 / 12), "month_cos": np.cos(2 * np.pi * 2 / 12),
    "Appliances_lag_1": 60.0, "Appliances_lag_2": 50.0, "Appliances_lag_3": 50.0,
    "Appliances_rolling_mean_3": 53.3, "Appliances_rolling_mean_6": 55.0,
    "Appliances_rolling_std_6": 5.5,
    "T1": 21.5, "RH_1": 40.0, "T2": 20.1, "RH_2": 39.0, "T3": 22.0, "RH_3": 38.5,
    "T6": 7.0, "RH_6": 55.0, "T_out": 7.5, "Press_mm_hg": 755.0, "RH_out": 80.0,
    "Windspeed": 4.0, "Visibility": 40.0, "Tdewpoint": 3.5, "lights": 0.0,
}


@pytest.fixture
def valid_frame():
    return pd.DataFrame([VALID_ROW, VALID_ROW])


class TestContractDefinition:
    def test_valid_row_covers_every_feature(self):
        assert set(VALID_ROW) == set(ALL_FEATURE_COLS)

    def test_every_feature_has_bounds(self):
        assert set(PHYSICAL_BOUNDS) == set(ALL_FEATURE_COLS)

    def test_bounds_are_ordered(self):
        for col, (lo, hi) in PHYSICAL_BOUNDS.items():
            assert lo < hi, col


class TestValidInput:
    def test_returns_features_in_training_order(self, valid_frame):
        shuffled = valid_frame[list(reversed(ALL_FEATURE_COLS))]
        X = validate_feature_frame(shuffled)
        assert list(X.columns) == ALL_FEATURE_COLS

    def test_extra_columns_dropped(self, valid_frame):
        frame = valid_frame.assign(date=pd.Timestamp("2016-03-15"), Appliances=50.0)
        X = validate_feature_frame(frame)
        assert "date" not in X.columns and "Appliances" not in X.columns

    def test_output_is_float(self, valid_frame):
        X = validate_feature_frame(valid_frame)
        assert all(dtype == np.float64 for dtype in X.dtypes)

    def test_does_not_mutate_input(self, valid_frame):
        before = valid_frame.copy()
        validate_feature_frame(valid_frame)
        pd.testing.assert_frame_equal(valid_frame, before)

    def test_boundary_values_accepted(self, valid_frame):
        frame = valid_frame.copy()
        frame.loc[0, "RH_out"] = 100.0
        frame.loc[1, "Windspeed"] = 0.0
        validate_feature_frame(frame)


class TestRejectedInput:
    def test_not_a_dataframe(self):
        with pytest.raises(InputValidationError, match="DataFrame"):
            validate_feature_frame([VALID_ROW])

    def test_empty_frame(self):
        with pytest.raises(InputValidationError, match="no rows"):
            validate_feature_frame(pd.DataFrame(columns=ALL_FEATURE_COLS))

    def test_missing_column(self, valid_frame):
        with pytest.raises(InputValidationError, match="T_out"):
            validate_feature_frame(valid_frame.drop(columns=["T_out"]))

    def test_non_numeric_column(self, valid_frame):
        frame = valid_frame.astype({"T1": str})
        with pytest.raises(InputValidationError, match="non-numeric"):
            validate_feature_frame(frame)

    def test_boolean_column_rejected(self, valid_frame):
        frame = valid_frame.assign(is_weekend=False)
        with pytest.raises(InputValidationError, match="non-numeric"):
            validate_feature_frame(frame)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_values(self, valid_frame, bad):
        frame = valid_frame.copy()
        frame.loc[1, "RH_1"] = bad
        with pytest.raises(InputValidationError, match="NaN or infinite"):
            validate_feature_frame(frame)

    @pytest.mark.parametrize("col, value", [
        ("RH_1", 120.0),             # humidity above 100 %
        ("T_out", 300.0),            # Kelvin sent instead of Celsius
        ("Press_mm_hg", 1013.0),     # hPa sent instead of mm Hg
        ("Appliances_lag_1", -5.0),  # negative energy
        ("hour", 24),
        ("month", 13),
        ("hour_sin", 1.5),
    ])
    def test_out_of_bounds(self, valid_frame, col, value):
        frame = valid_frame.copy()
        frame[col] = frame[col].astype(float)
        frame.loc[0, col] = value
        with pytest.raises(InputValidationError, match=col):
            validate_feature_frame(frame)

    def test_fractional_hour(self, valid_frame):
        frame = valid_frame.astype({"hour": float})
        frame.loc[0, "hour"] = 14.5
        with pytest.raises(InputValidationError, match="integer-valued"):
            validate_feature_frame(frame)

    def test_weekend_flag_inconsistent(self, valid_frame):
        frame = valid_frame.assign(is_weekend=1)  # day_of_week is Wednesday
        with pytest.raises(InputValidationError, match="is_weekend inconsistent"):
            validate_feature_frame(frame)

    def test_all_issues_reported_together(self, valid_frame):
        frame = valid_frame.copy()
        frame.loc[0, "RH_1"] = 150.0
        frame.loc[0, "Windspeed"] = -1.0
        with pytest.raises(InputValidationError) as exc:
            validate_feature_frame(frame)
        assert len(exc.value.issues) == 2
