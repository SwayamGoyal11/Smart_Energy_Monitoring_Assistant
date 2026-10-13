"""
src/features/feature_engineering.py
-------------------------------------
Creates features for predicting the NEXT 10-minute appliance energy consumption.

All features are leak-free: only information available at prediction time is used.
Rolling/lag features use shift(1) so the current target value is never used
to construct its own features.
"""

import pandas as pd
import numpy as np
from typing import Tuple

# Environmental features available at the current timestep
ENVIRONMENTAL_FEATURES = [
    "T1", "RH_1",   # Kitchen temperature & humidity
    "T2", "RH_2",   # Living room
    "T3", "RH_3",   # Laundry room
    "T6", "RH_6",   # Outside building (north)
    "T_out",         # Outside temperature
    "Press_mm_hg",   # Atmospheric pressure
    "RH_out",        # Outside humidity
    "Windspeed",     # Wind speed
    "Visibility",    # Visibility
    "Tdewpoint",     # Dew point
    "lights",        # Lighting energy at current timestep
]

TIME_FEATURES    = ["hour", "day", "day_of_week", "month", "is_weekend"]
LAG_FEATURES     = ["Appliances_lag_1", "Appliances_lag_2", "Appliances_lag_3"]
ROLLING_FEATURES = [
    "Appliances_rolling_mean_3",
    "Appliances_rolling_mean_6",
    "Appliances_rolling_std_6",
]

ALL_FEATURE_COLS = TIME_FEATURES + LAG_FEATURES + ROLLING_FEATURES + ENVIRONMENTAL_FEATURES
TARGET_COL = "Appliances"


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar/time-of-day features from the date column.
    - hour        : 0-23, captures daily usage patterns
    - day         : 1-31, day of month
    - day_of_week : 0=Mon..6=Sun
    - month       : 1-12, seasonal variation
    - is_weekend  : 1 if Sat/Sun, else 0
    """
    df = df.copy()
    df["hour"]        = df["date"].dt.hour
    df["day"]         = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Create lag features from the target column.
    - lag_1: energy 10 min ago
    - lag_2: energy 20 min ago
    - lag_3: energy 30 min ago
    Uses shift(1/2/3) - no future information leaks in.
    """
    df = df.copy()
    df["Appliances_lag_1"] = df[target_col].shift(1)
    df["Appliances_lag_2"] = df[target_col].shift(2)
    df["Appliances_lag_3"] = df[target_col].shift(3)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Create rolling window statistics.
    shift(1) is applied before rolling so the current row's target
    is excluded from its own window - no leakage.
    - rolling_mean_3: mean of last 3 readings (30 min trend)
    - rolling_mean_6: mean of last 6 readings (60 min trend)
    - rolling_std_6 : std dev of last 6 readings (volatility)
    """
    df = df.copy()
    shifted = df[target_col].shift(1)
    df["Appliances_rolling_mean_3"] = shifted.rolling(window=3, min_periods=3).mean()
    df["Appliances_rolling_mean_6"] = shifted.rolling(window=6, min_periods=6).mean()
    df["Appliances_rolling_std_6"]  = shifted.rolling(window=6, min_periods=6).std()
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps and return a clean feature matrix.
    Drops the first ~6 rows where lag/rolling features are NaN.
    """
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # Drop rows where lag/rolling are NaN (first ~6 rows)
    df = df.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).reset_index(drop=True)

    keep_cols = ["date"] + ALL_FEATURE_COLS + [TARGET_COL]
    return df[keep_cols]


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically - NO random shuffling (time-series data).
    70% train / 15% validation / 15% test
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target (y)."""
    return df[ALL_FEATURE_COLS].copy(), df[TARGET_COL].copy()


def print_split_summary(train_df, val_df, test_df) -> None:
    """Print a summary of the chronological split."""
    total = len(train_df) + len(val_df) + len(test_df)
    print("=" * 60)
    print("  CHRONOLOGICAL SPLIT SUMMARY")
    print("=" * 60)
    for name, split in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        pct = len(split) / total * 100
        print(f"\n  {name:5s} | {len(split):>6,} rows ({pct:.1f}%)")
        print(f"        | {split['date'].min()} to {split['date'].max()}")
    print("=" * 60)
