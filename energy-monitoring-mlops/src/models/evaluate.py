"""
src/models/evaluate.py
-----------------------
Shared metric computation: MAE, RMSE, R2
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R2 for regression predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    return {
        "mae":  round(float(mae),  4),
        "rmse": round(float(rmse), 4),
        "r2":   round(float(r2),   4),
    }


def print_metrics(model_name: str, metrics: dict, split: str = "Validation") -> None:
    print(f"\n  [{model_name}] - {split}")
    print(f"    MAE  : {metrics['mae']:.4f} Wh")
    print(f"    RMSE : {metrics['rmse']:.4f} Wh")
    print(f"    R2   : {metrics['r2']:.4f}")
