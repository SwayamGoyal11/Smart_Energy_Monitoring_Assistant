"""
src/models/evaluation_analytics.py
-----------------------------------
Deep-dive evaluation analytics engine for the Smart Energy Monitoring Assistant.
Computes:
- Global time-series metrics (MAE, RMSE, R2, WAPE, MedAE, Bias)
- Consumption regime-stratified metrics (Standby <=60Wh, Active 60-200Wh, Peak >200Wh)
- Diurnal & temporal segment metrics (Night, Morning, Day, Evening; Weekday vs Weekend)
- Residual diagnostic statistics (Mean, Std, Skewness, Kurtosis, Heteroscedasticity)
- Feature importance attribution rankings
- Exports structured evidence to reports/evidence/
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", "evidence")
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute comprehensive regression metrics for energy forecasting.
    Includes MAE, RMSE, R2, WAPE, MedAE, Max Error, and Mean Prediction Bias.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    max_err = float(np.max(np.abs(y_true - y_pred)))

    sum_true = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / (sum_true + 1e-9) * 100.0)
    bias = float(np.mean(y_pred - y_true))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "wape_pct": round(wape, 2),
        "medae": round(medae, 4),
        "max_error": round(max_err, 4),
        "mean_bias": round(bias, 4),
    }


def compute_regime_breakdown(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_col: str = "Appliances",
) -> pd.DataFrame:
    """
    Stratifies predictions by appliance consumption regime:
    1. Standby: <= 60 Wh (~76% of observations)
    2. Active / Moderate: 60 - 200 Wh (~16% of observations)
    3. High Peak: > 200 Wh (~8% of observations)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    regimes = [
        ("Standby (<=60 Wh)", y_true <= 60),
        ("Active (60-200 Wh)", (y_true > 60) & (y_true <= 200)),
        ("High Peak (>200 Wh)", y_true > 200),
    ]

    records = []
    total_samples = len(y_true)

    for name, mask in regimes:
        count = int(np.sum(mask))
        if count == 0:
            continue
        pct = round(count / total_samples * 100.0, 1)
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]

        sub_mae = float(mean_absolute_error(sub_true, sub_pred))
        sub_rmse = float(np.sqrt(mean_squared_error(sub_true, sub_pred)))
        sub_wape = float(np.sum(np.abs(sub_true - sub_pred)) / (np.sum(sub_true) + 1e-9) * 100.0)
        sub_bias = float(np.mean(sub_pred - sub_true))

        records.append({
            "Regime": name,
            "Count": count,
            "Share_pct": pct,
            "MAE": round(sub_mae, 2),
            "RMSE": round(sub_rmse, 2),
            "WAPE_pct": round(sub_wape, 2),
            "Mean_Bias": round(sub_bias, 2),
        })

    return pd.DataFrame(records)


def compute_diurnal_breakdown(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Stratifies predictions across diurnal operational time-windows:
    - Night Standby: 00:00 - 06:00
    - Morning Peak: 06:00 - 09:00
    - Daytime Plateau: 09:00 - 17:00
    - Evening Peak: 17:00 - 22:00
    - Late Night: 22:00 - 24:00
    - Weekday vs Weekend
    """
    hours = pd.to_datetime(df["date"]).dt.hour.values
    is_weekend = (pd.to_datetime(df["date"]).dt.dayofweek >= 5).astype(int).values

    segments = [
        ("Night Standby (00-06h)", (hours >= 0) & (hours < 6)),
        ("Morning Peak (06-09h)", (hours >= 6) & (hours < 9)),
        ("Daytime (09-17h)", (hours >= 9) & (hours < 17)),
        ("Evening Peak (17-22h)", (hours >= 17) & (hours < 22)),
        ("Late Night (22-24h)", (hours >= 22) & (hours <= 23)),
        ("Weekday", is_weekend == 0),
        ("Weekend", is_weekend == 1),
    ]

    records = []
    for name, mask in segments:
        count = int(np.sum(mask))
        if count == 0:
            continue
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        sub_mae = float(mean_absolute_error(sub_true, sub_pred))
        sub_rmse = float(np.sqrt(mean_squared_error(sub_true, sub_pred)))

        records.append({
            "Segment": name,
            "Count": count,
            "MAE": round(sub_mae, 2),
            "RMSE": round(sub_rmse, 2),
        })

    return pd.DataFrame(records)


def compute_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes statistical properties of the prediction residuals (e = y_true - y_pred).
    Includes normality metrics (skewness, kurtosis) and heteroscedasticity index.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    skew_res = float(stats.skew(residuals)) if len(residuals) > 2 else 0.0
    kurt_res = float(stats.kurtosis(residuals)) if len(residuals) > 3 else 0.0
    q05 = float(np.percentile(residuals, 5))
    q95 = float(np.percentile(residuals, 95))

    # Heteroscedasticity correlation: correlation between predicted value and absolute residual
    abs_res = np.abs(residuals)
    if np.std(y_pred) > 1e-6 and np.std(abs_res) > 1e-6:
        corr, _ = stats.pearsonr(y_pred, abs_res)
    else:
        corr = 0.0

    return {
        "mean_residual": round(mean_res, 4),
        "std_residual": round(std_res, 4),
        "skewness": round(skew_res, 4),
        "kurtosis": round(kurt_res, 4),
        "quantile_05": round(q05, 4),
        "quantile_95": round(q95, 4),
        "heteroscedasticity_corr": round(float(corr), 4),
    }


def extract_feature_importances(
    models: dict,
    feature_names: list,
) -> pd.DataFrame:
    """
    Extracts and normalizes feature importance scores across models.
    """
    df_imp = pd.DataFrame({"Feature": feature_names})

    # Linear Regression (Normalized absolute coefficients)
    if "Linear Regression" in models and hasattr(models["Linear Regression"], "coef_"):
        coefs = np.abs(models["Linear Regression"].coef_)
        total = np.sum(coefs) + 1e-9
        df_imp["Linear_Reg_Importance"] = np.round(coefs / total * 100.0, 2)

    # Random Forest
    if "Random Forest" in models and hasattr(models["Random Forest"], "feature_importances_"):
        rf_imp = models["Random Forest"].feature_importances_
        df_imp["Random_Forest_Importance"] = np.round(rf_imp * 100.0, 2)

    # XGBoost
    if "XGBoost" in models and hasattr(models["XGBoost"], "feature_importances_"):
        xgb_imp = models["XGBoost"].feature_importances_
        df_imp["XGBoost_Importance"] = np.round(xgb_imp * 100.0, 2)

    # Average importance rank
    imp_cols = [c for c in df_imp.columns if c != "Feature"]
    if imp_cols:
        df_imp["Mean_Importance"] = np.round(df_imp[imp_cols].mean(axis=1), 2)
        df_imp = df_imp.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)

    return df_imp


def run_full_evaluation_audit(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: dict,
    models: dict = None,
    feature_names: list = None,
) -> dict:
    """
    Orchestrates the complete evaluation deep-dive across all models and splits.
    Saves evidence files:
    - reports/evidence/evaluation_metrics_deepdive.json
    - reports/evidence/regime_breakdown.csv
    - reports/evidence/feature_importance.csv
    """
    y_val = val_df["Appliances"].values
    y_test = test_df["Appliances"].values

    deepdive_results = {
        "dataset_summary": {
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "features_count": len(feature_names) if feature_names else 0,
        },
        "models": {},
    }

    regime_tables = []

    for model_name, preds in predictions.items():
        v_pred = preds["val"]
        t_pred = preds["test"]

        # Global metrics
        v_metrics = compute_extended_metrics(y_val, v_pred)
        t_metrics = compute_extended_metrics(y_test, t_pred)

        # Regimes
        t_regimes = compute_regime_breakdown(test_df, y_test, t_pred)
        t_regimes["Model"] = model_name
        regime_tables.append(t_regimes)

        # Diurnal
        t_diurnal = compute_diurnal_breakdown(test_df, y_test, t_pred)

        # Residuals
        t_residuals = compute_residual_diagnostics(y_test, t_pred)

        deepdive_results["models"][model_name] = {
            "validation_metrics": v_metrics,
            "test_metrics": t_metrics,
            "regime_breakdown": t_regimes.to_dict(orient="records"),
            "diurnal_breakdown": t_diurnal.to_dict(orient="records"),
            "residual_diagnostics": t_residuals,
        }

    # Save deep-dive JSON
    json_path = os.path.join(REPORTS_DIR, "evaluation_metrics_deepdive.json")
    with open(json_path, "w") as f:
        json.dump(deepdive_results, f, indent=2)

    # Save regime breakdown CSV
    if regime_tables:
        all_regimes = pd.concat(regime_tables, ignore_index=True)
        regime_csv_path = os.path.join(REPORTS_DIR, "regime_breakdown.csv")
        all_regimes.to_csv(regime_csv_path, index=False)

    # Save feature importances if provided
    if models and feature_names:
        feat_imp_df = extract_feature_importances(models, feature_names)
        feat_csv_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
        feat_imp_df.to_csv(feat_csv_path, index=False)

    print(f"  Saved deep-dive metrics -> {json_path}")
    return deepdive_results
