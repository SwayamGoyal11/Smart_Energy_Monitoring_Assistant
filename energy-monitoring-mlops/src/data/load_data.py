"""
src/data/load_data.py
---------------------
Loads and validates the raw energydata_complete.csv dataset.
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "energydata_complete.csv"
)

EXPECTED_COLUMNS = [
    "date", "Appliances", "lights",
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3",
    "T4", "RH_4", "T5", "RH_5", "T6", "RH_6",
    "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
    "T_out", "Press_mm_hg", "RH_out",
    "Windspeed", "Visibility", "Tdewpoint",
    "rv1", "rv2"
]

TARGET_COL = "Appliances"
DATE_COL = "date"


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV, parse the date column, sort chronologically.
    Does NOT modify the original file.
    """
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please place energydata_complete.csv in data/raw/"
        )

    df = pd.read_csv(filepath, engine="python")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run validation checks and return a results dictionary."""
    results = {}

    results["shape"] = df.shape
    results["n_rows"] = df.shape[0]
    results["n_cols"] = df.shape[1]
    results["columns"] = list(df.columns)
    results["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}

    missing = df.isnull().sum()
    results["missing_values"] = missing[missing > 0].to_dict()
    results["total_missing"] = int(df.isnull().sum().sum())
    results["has_missing"] = results["total_missing"] > 0

    results["duplicate_rows"] = int(df.duplicated().sum())

    results["date_min"] = df["date"].min()
    results["date_max"] = df["date"].max()
    results["date_range_days"] = (df["date"].max() - df["date"].min()).days

    time_diffs = df["date"].diff().dropna()
    results["time_interval_mode"] = time_diffs.mode()[0]
    results["is_regular_10min"] = (time_diffs.mode()[0].total_seconds() == 600)
    results["is_chronological"] = bool((df["date"].diff().dropna() >= pd.Timedelta(0)).all())

    target = df[TARGET_COL]
    results["target_stats"] = {
        "mean":   round(float(target.mean()),   2),
        "median": round(float(target.median()), 2),
        "std":    round(float(target.std()),    2),
        "min":    round(float(target.min()),    2),
        "max":    round(float(target.max()),    2),
        "q25":    round(float(target.quantile(0.25)), 2),
        "q75":    round(float(target.quantile(0.75)), 2),
    }

    return results


def print_validation_report(results: dict) -> None:
    """Pretty-print the validation report."""
    print("=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"\n  Shape          : {results['shape']}")
    print(f"  Rows           : {results['n_rows']:,}")
    print(f"  Columns        : {results['n_cols']}")
    print(f"\n  Date range     : {results['date_min']} to {results['date_max']}")
    print(f"  Duration       : {results['date_range_days']} days")
    print(f"  Chronological  : {results['is_chronological']}")
    print(f"  Regular 10-min : {results['is_regular_10min']}")
    print(f"\n  Missing values : {results['total_missing']}")
    if results["has_missing"]:
        for col, cnt in results["missing_values"].items():
            print(f"    {col}: {cnt}")
    else:
        print("    None")
    print(f"\n  Duplicate rows : {results['duplicate_rows']}")
    print(f"\n  Target ({TARGET_COL}) stats:")
    for k, v in results["target_stats"].items():
        print(f"    {k:8s}: {v}")
    print("=" * 60)
