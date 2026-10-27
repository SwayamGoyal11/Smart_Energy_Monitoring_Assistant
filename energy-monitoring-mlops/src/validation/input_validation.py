"""
src/validation/input_validation.py
-----------------------------------
Guards the model input contract before any prediction is made.

Bounds are physical plausibility limits (sensor ranges, unit sanity), not the
training-data ranges: a value outside them is a broken sensor or a unit error,
not drift. Distribution drift is handled separately by monitoring.
"""

import numpy as np
import pandas as pd

from src.features.feature_engineering import ALL_FEATURE_COLS

# (min, max) inclusive. Energy in Wh per 10 min, temperatures in degC,
# humidity in %, pressure in mm Hg, wind in m/s, visibility in km.
PHYSICAL_BOUNDS = {
    # Time features
    "hour":            (0, 23),
    "day":             (1, 31),
    "day_of_week":     (0, 6),
    "month":           (1, 12),
    "is_weekend":      (0, 1),
    "hour_sin":        (-1.0, 1.0),
    "hour_cos":        (-1.0, 1.0),
    "day_of_week_sin": (-1.0, 1.0),
    "day_of_week_cos": (-1.0, 1.0),
    "month_sin":       (-1.0, 1.0),
    "month_cos":       (-1.0, 1.0),
    # Consumption history
    "Appliances_lag_1":          (0.0, 5000.0),
    "Appliances_lag_2":          (0.0, 5000.0),
    "Appliances_lag_3":          (0.0, 5000.0),
    "Appliances_rolling_mean_3": (0.0, 5000.0),
    "Appliances_rolling_mean_6": (0.0, 5000.0),
    "Appliances_rolling_std_6":  (0.0, 5000.0),
    # Environmental sensors
    "T1":          (-40.0, 60.0),
    "RH_1":        (0.0, 100.0),
    "T2":          (-40.0, 60.0),
    "RH_2":        (0.0, 100.0),
    "T3":          (-40.0, 60.0),
    "RH_3":        (0.0, 100.0),
    "T6":          (-40.0, 60.0),
    "RH_6":        (0.0, 100.0),
    "T_out":       (-40.0, 60.0),
    "Press_mm_hg": (600.0, 820.0),
    "RH_out":      (0.0, 100.0),
    "Windspeed":   (0.0, 60.0),
    "Visibility":  (0.0, 100.0),
    "Tdewpoint":   (-50.0, 40.0),
    "lights":      (0.0, 1000.0),
}

INTEGER_FEATURES = ["hour", "day", "day_of_week", "month", "is_weekend"]


class InputValidationError(ValueError):
    """Raised when a feature frame violates the model input contract."""

    def __init__(self, issues: list):
        self.issues = issues
        super().__init__("Invalid model input:\n  - " + "\n  - ".join(issues))


def validate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check a feature frame against the input contract and return it as a
    float matrix with columns in ALL_FEATURE_COLS order.

    Extra columns (e.g. date, Appliances) are dropped. All problems are
    collected and raised together so a caller sees every issue at once.
    """
    if not isinstance(df, pd.DataFrame):
        raise InputValidationError([f"expected a pandas DataFrame, got {type(df).__name__}"])
    if df.empty:
        raise InputValidationError(["input frame has no rows"])

    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        raise InputValidationError([f"missing feature columns: {missing}"])

    issues = []
    X = df[ALL_FEATURE_COLS]

    non_numeric = [c for c in ALL_FEATURE_COLS if not pd.api.types.is_numeric_dtype(X[c])
                   or pd.api.types.is_bool_dtype(X[c])]
    if non_numeric:
        raise InputValidationError([f"non-numeric feature columns: {non_numeric}"])

    X = X.astype(float)
    values = X.to_numpy()

    non_finite = [c for c, ok in zip(ALL_FEATURE_COLS, np.isfinite(values).all(axis=0)) if not ok]
    if non_finite:
        issues.append(f"NaN or infinite values in: {non_finite}")

    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        finite = X[col][np.isfinite(X[col])]
        out = finite[(finite < lo) | (finite > hi)]
        if len(out):
            issues.append(f"{col} outside [{lo}, {hi}]: {len(out)} row(s), e.g. {out.iloc[0]}")

    for col in INTEGER_FEATURES:
        finite = X[col][np.isfinite(X[col])]
        if not (finite == np.round(finite)).all():
            issues.append(f"{col} must be integer-valued")

    # is_weekend is derived from day_of_week; disagreement means a broken upstream transform.
    weekend_mismatch = (X["is_weekend"] != (X["day_of_week"] >= 5).astype(float))
    if weekend_mismatch.any():
        issues.append(f"is_weekend inconsistent with day_of_week in {int(weekend_mismatch.sum())} row(s)")

    if issues:
        raise InputValidationError(issues)

    return X.copy()
