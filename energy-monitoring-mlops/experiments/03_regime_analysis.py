"""
experiments/03_regime_analysis.py
----------------------------------
E3 - Regime-stratified performance of the team's baseline models.

Aggregate MAE is dominated by the standby regime (most readings). This script
breaks the SAME models the team trains (same features, same hyper-parameters,
seed 42, see src/models/train.py) down by consumption regime of the ACTUAL
value, using the thresholds used in the EDA:

    Standby : Appliances <= 60 Wh
    Active  : 60 < Appliances <= 300 Wh
    Spike   : Appliances > 300 Wh

For each model and regime it reports MAE, mean signed error (prediction minus
actual; negative = under-prediction) and the regime's share of the model's
total absolute error. Nothing in the production pipeline is modified.

Caveat (stated in the output): regimes are defined from the true target, so
this is a diagnostic of WHERE errors occur, not a deployable stratification.

Outputs -> reports/evidence/regime_analysis/regime_metrics.csv
          reports/figures/E3_regime_mae.png

Run from project root:  python experiments/03_regime_analysis.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.load_data import load_raw_data                         # noqa: E402
from src.features.feature_engineering import (                       # noqa: E402
    build_feature_matrix, chronological_split, get_X_y,
)
from src.models.evaluate import compute_metrics                      # noqa: E402

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "evidence", "regime_analysis")
FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

REGIMES = {
    "Standby (<=60)": lambda y: y <= 60,
    "Active (60-300]": lambda y: (y > 60) & (y <= 300),
    "Spike (>300)": lambda y: y > 300,
}


def fit_models(X_train, y_train, X_val):
    """Same hyper-parameters as src/models/train.py (seed 42)."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=4,
            random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist"),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models


def regime_table(split_name, X, y, preds: dict) -> pd.DataFrame:
    rows = []
    y = y.values
    for model, p in preds.items():
        p = np.asarray(p)
        total_abs = np.abs(p - y).sum()
        overall = compute_metrics(y, p)["mae"]
        for regime, fn in REGIMES.items():
            mask = fn(y)
            err = p[mask] - y[mask]
            rows.append({
                "split": split_name, "model": model, "regime": regime,
                "n": int(mask.sum()), "share_of_rows_%": 100 * mask.mean(),
                "mae": np.abs(err).mean(),
                "mean_signed_error": err.mean(),
                "share_of_total_abs_error_%": 100 * np.abs(err).sum() / total_abs,
                "overall_mae": overall,
            })
    return pd.DataFrame(rows)


def main():
    df = build_feature_matrix(load_raw_data())
    train, val, test = chronological_split(df)
    X_train, y_train = get_X_y(train)
    X_val, y_val = get_X_y(val)
    X_test, y_test = get_X_y(test)

    models = fit_models(X_train, y_train, X_val)

    tables = []
    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = {"Persistence": X["Appliances_lag_1"].values}
        preds.update({k: m.predict(X) for k, m in models.items()})
        tables.append(regime_table(name, X, y, preds))
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(os.path.join(OUT_DIR, "regime_metrics.csv"), index=False)

    plot(res)

    order = ["Persistence", "Linear Regression", "Random Forest", "XGBoost"]
    print("\n" + "=" * 78)
    print("  E3 - REGIME-STRATIFIED PERFORMANCE (team's models, seed 42, Wh)")
    print("  NOTE: regimes use the ACTUAL value -> diagnostic only, not deployable.")
    print("=" * 78)
    for split in ["val", "test"]:
        sub = res[res.split == split]
        counts = sub[sub.model == "Persistence"].set_index("regime")
        print(f"\n  [{split.upper()}]  rows: " + " | ".join(
            f"{r} n={int(counts.loc[r, 'n'])} ({counts.loc[r, 'share_of_rows_%']:.1f}%)"
            for r in REGIMES))
        pv = sub.pivot(index="model", columns="regime", values="mae").loc[order][list(REGIMES)]
        pv["overall"] = sub.groupby("model")["overall_mae"].first().loc[order]
        print("\n  MAE (Wh)")
        print(pv.round(2).to_string())
        bias = sub.pivot(index="model", columns="regime",
                         values="mean_signed_error").loc[order][list(REGIMES)]
        print("\n  Mean signed error (pred - actual; negative = under-prediction)")
        print(bias.round(1).to_string())
        share = sub.pivot(index="model", columns="regime",
                          values="share_of_total_abs_error_%").loc[order][list(REGIMES)]
        print("\n  Share of each model's total absolute error (%)")
        print(share.round(1).to_string())
    print(f"\n  Outputs saved to: {OUT_DIR}")


def plot(res: pd.DataFrame):
    order = ["Persistence", "Linear Regression", "Random Forest", "XGBoost"]
    regimes = list(REGIMES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, split in zip(axes, ["val", "test"]):
        sub = res[res.split == split]
        w = 0.8 / len(order)
        for i, m in enumerate(order):
            vals = [sub[(sub.model == m) & (sub.regime == r)]["mae"].iloc[0]
                    for r in regimes]
            ax.bar(np.arange(len(regimes)) + i * w, vals, w, label=m)
        ax.set_xticks(np.arange(len(regimes)) + w * (len(order) - 1) / 2)
        ax.set_xticklabels(regimes)
        ax.set_ylabel("MAE (Wh)")
        ax.set_title(f"MAE by consumption regime - {split}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E3_regime_mae.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
