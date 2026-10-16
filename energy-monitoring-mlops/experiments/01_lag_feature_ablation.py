"""
experiments/01_lag_feature_ablation.py
---------------------------------------
Quantifies how much the autoregressive (lag / rolling) features contribute to
forecasting the NEXT 10-minute Appliances energy consumption.

Three experiments, all on the SAME chronological 70/15/15 split and the SAME
evaluation rows (first MAX_LAG rows dropped for every configuration so that
results are directly comparable):

  A. Feature-group ablation  : which groups (time / env / lag / rolling) matter?
                               (leave-one-group-out + single-group-only)
  B. Lag-depth sweep         : how many lags are useful? (0,1,2,3,6,12,144)
  C. Persistence reference   : naive y_hat = y_{t-1} for context.

Models: Linear Regression, Random Forest, XGBoost (same hyper-parameters as
src/models/train.py). Metrics: MAE, RMSE, R2 on validation and test.

Outputs -> reports/evidence/lag_ablation/
    ablation_group_results.csv, ablation_lag_depth_results.csv
         reports/figures/
    L1_group_ablation.png, L2_lag_depth_sweep.png

Run from project root:  python experiments/01_lag_feature_ablation.py
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

from src.data.load_data import load_raw_data                       # noqa: E402
from src.features.feature_engineering import (                     # noqa: E402
    TIME_FEATURES, ENVIRONMENTAL_FEATURES, TARGET_COL,
    create_time_features, chronological_split,
)
from src.models.evaluate import compute_metrics                    # noqa: E402

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "evidence", "lag_ablation")
FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MAX_LAG = 144          # 1 day of 10-min steps; rows dropped uniformly
SEED = 42
LAG_DEPTHS = [0, 1, 2, 3, 6, 12, 144]   # 144 = "same time yesterday" only (see below)

ROLLING_COLS = ["roll_mean_3", "roll_mean_6", "roll_std_6"]


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def build_full_frame() -> pd.DataFrame:
    """Build every candidate feature once; configs pick column subsets."""
    df = create_time_features(load_raw_data())
    y = df[TARGET_COL]
    for k in list(range(1, 13)) + [144]:
        df[f"lag_{k}"] = y.shift(k)
    s = y.shift(1)
    df["roll_mean_3"] = s.rolling(3).mean()
    df["roll_mean_6"] = s.rolling(6).mean()
    df["roll_std_6"] = s.rolling(6).std()
    df = df.iloc[MAX_LAG:].reset_index(drop=True)      # identical rows for all configs
    return df


def lag_cols(depth: int):
    """Contiguous lags 1..depth; depth 144 means ONLY the daily lag (lag_144)."""
    if depth == 0:
        return []
    if depth == 144:
        return ["lag_144"]
    return [f"lag_{k}" for k in range(1, depth + 1)]


def make_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=4,
            random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
            tree_method="hist"),
    }


def run_config(name, cols, splits):
    """Fit every model on `cols`; return list of result rows."""
    tr, va, te = splits
    rows = []
    for mname, model in make_models().items():
        model.fit(tr[cols], tr[TARGET_COL])
        vm = compute_metrics(va[TARGET_COL], model.predict(va[cols]))
        tm = compute_metrics(te[TARGET_COL], model.predict(te[cols]))
        rows.append({
            "config": name, "model": mname, "n_features": len(cols),
            "val_mae": vm["mae"], "val_rmse": vm["rmse"], "val_r2": vm["r2"],
            "test_mae": tm["mae"], "test_rmse": tm["rmse"], "test_r2": tm["r2"],
        })
        print(f"  {name:<28s} {mname:<18s} nF={len(cols):>2d} "
              f"val MAE={vm['mae']:7.2f}  test MAE={tm['mae']:7.2f}  "
              f"test R2={tm['r2']:6.3f}")
    return rows


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def experiment_groups(splits):
    groups = {
        "time": TIME_FEATURES,
        "env": ENVIRONMENTAL_FEATURES,
        "lag": lag_cols(3),
        "rolling": ROLLING_COLS,
    }
    full = sum(groups.values(), [])
    rows = []
    print("\n[A] Feature-group ablation")
    rows += run_config("FULL (v0 feature set)", full, splits)
    for g, gcols in groups.items():
        rows += run_config(f"- drop {g}", [c for c in full if c not in gcols], splits)
    for g, gcols in groups.items():
        rows += run_config(f"only {g}", gcols, splits)
    rows += run_config("drop lag+rolling (no AR)",
                       groups["time"] + groups["env"], splits)
    return pd.DataFrame(rows)


def experiment_depth(splits):
    base = TIME_FEATURES + ENVIRONMENTAL_FEATURES      # non-autoregressive base
    rows = []
    print("\n[B] Lag-depth sweep (time + env + lags 1..k, no rolling features)")
    for d in LAG_DEPTHS:
        rows += run_config(f"lags={d}", base + lag_cols(d), splits)
    return pd.DataFrame(rows)


def persistence_reference(splits):
    _, va, te = splits
    vm = compute_metrics(va[TARGET_COL], va["lag_1"])
    tm = compute_metrics(te[TARGET_COL], te["lag_1"])
    print(f"\n[C] Persistence baseline  val MAE={vm['mae']:.2f}  "
          f"test MAE={tm['mae']:.2f}  test R2={tm['r2']:.3f}")
    return vm, tm


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_groups(res: pd.DataFrame, pers_test_mae: float):
    configs = list(dict.fromkeys(res["config"]))
    models = list(dict.fromkeys(res["model"]))
    fig, ax = plt.subplots(figsize=(13, 6))
    w = 0.8 / len(models)
    x = np.arange(len(configs))
    for i, m in enumerate(models):
        vals = [res[(res.config == c) & (res.model == m)]["test_mae"].iloc[0]
                for c in configs]
        ax.bar(x + i * w, vals, w, label=m)
    ax.axhline(pers_test_mae, color="k", ls="--", lw=1,
               label=f"Persistence ({pers_test_mae:.1f})")
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("Test MAE (Wh)")
    ax.set_title("Feature-group ablation (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "L1_group_ablation.png"), dpi=150)
    plt.close(fig)


def plot_depth(res: pd.DataFrame, pers_test_mae: float):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [f"{d}" if d != 144 else "daily\n(lag144)" for d in LAG_DEPTHS]
    for ax, metric, title in zip(axes, ["test_mae", "test_r2"],
                                 ["Test MAE (Wh)", "Test R2"]):
        for m in dict.fromkeys(res["model"]):
            sub = res[res.model == m].set_index("config").loc[
                [f"lags={d}" for d in LAG_DEPTHS]]
            ax.plot(range(len(LAG_DEPTHS)), sub[metric], marker="o", label=m)
        ax.set_xticks(range(len(LAG_DEPTHS)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Number of lags (1..k)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    axes[0].axhline(pers_test_mae, color="k", ls="--", lw=1, label="Persistence")
    axes[0].legend()
    axes[0].set_title("Lag-depth sweep: error")
    axes[1].set_title("Lag-depth sweep: R2")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "L2_lag_depth_sweep.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    df = build_full_frame()
    splits = chronological_split(df)
    print(f"Rows: train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}")

    groups_res = experiment_groups(splits)
    depth_res = experiment_depth(splits)
    pv, pt = persistence_reference(splits)

    groups_res.to_csv(os.path.join(OUT_DIR, "ablation_group_results.csv"), index=False)
    depth_res.to_csv(os.path.join(OUT_DIR, "ablation_lag_depth_results.csv"), index=False)
    plot_groups(groups_res, pt["mae"])
    plot_depth(depth_res, pt["mae"])

    # --- Verdict summary -------------------------------------------------- #
    print("\n" + "=" * 70)
    print("  SUMMARY (Test MAE, Wh)")
    print("=" * 70)
    for m in dict.fromkeys(groups_res["model"]):
        g = groups_res[groups_res.model == m].set_index("config")["test_mae"]
        full, noar = g["FULL (v0 feature set)"], g["drop lag+rolling (no AR)"]
        print(f"  {m:<18s} full={full:6.2f}  no-AR={noar:6.2f}  "
              f"delta={noar - full:+6.2f} ({(noar / full - 1) * 100:+.1f}%)")
    print(f"  Persistence        test MAE={pt['mae']:.2f}")
    print(f"\n  Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
