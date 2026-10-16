"""
experiments/05_generalisation_gap.py
-------------------------------------
Doc 04 asks for TRAIN as well as validation/test metrics: the gap between them
is the primary indicator of overfitting. The team's pipeline reports only val
and test. This script computes MAE / RMSE / R2 (and the RMSE/MAE ratio, which
shows how concentrated errors are in large events) for all four baseline
models on train, val and test, plus the generalisation gaps.

Gaps reported per model:
    val - train   (in-sample vs out-of-sample optimism)
    test - val    (decision-set bias + shift into the held-out period)
    test / val    (ratio)

Outputs -> reports/evidence/generalisation_gap/generalisation_metrics.csv
                                              generalisation_gaps.csv
          reports/figures/E4_generalisation_gap.png

Run from project root:  python experiments/05_generalisation_gap.py
"""

import os
import numpy as np
import pandas as pd

from _common import (load_splits, get_X_y, make_model, predict_all,
                     compute_metrics, evidence_dir, FIG_DIR,
                     MODEL_NAMES, ALL_NAMES, plt)

OUT = evidence_dir("generalisation_gap")


def main():
    _, train, val, test = load_splits()
    Xt, yt = get_X_y(train)
    Xv, yv = get_X_y(val)
    Xs, ys = get_X_y(test)
    models = {n: make_model(n).fit(Xt, yt) for n in MODEL_NAMES}

    rows = []
    for split, X, y in [("train", Xt, yt), ("val", Xv, yv), ("test", Xs, ys)]:
        for name, p in predict_all(models, X).items():
            m = compute_metrics(y, p)
            rows.append({"split": split, "model": name, "mae": m["mae"],
                         "rmse": m["rmse"], "r2": m["r2"],
                         "rmse_over_mae": m["rmse"] / m["mae"]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "generalisation_metrics.csv"), index=False)

    wide = res.pivot(index="model", columns="split", values="mae").loc[ALL_NAMES]
    gaps = pd.DataFrame({
        "train_mae": wide["train"], "val_mae": wide["val"], "test_mae": wide["test"],
        "val_minus_train": wide["val"] - wide["train"],
        "test_minus_val": wide["test"] - wide["val"],
        "test_over_val": wide["test"] / wide["val"],
    })
    gaps.to_csv(os.path.join(OUT, "generalisation_gaps.csv"))

    print("=" * 78)
    print("  GENERALISATION GAP (team's models, seed 42, MAE in Wh)")
    print("=" * 78)
    print(gaps.round(2).to_string())
    print("\n  R2 by split")
    print(res.pivot(index="model", columns="split", values="r2").loc[ALL_NAMES][
        ["train", "val", "test"]].round(3).to_string())
    print("\n  RMSE/MAE ratio (higher = errors concentrated in large events)")
    print(res.pivot(index="model", columns="split", values="rmse_over_mae").loc[ALL_NAMES][
        ["train", "val", "test"]].round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    w = 0.25
    for i, (s, c) in enumerate([("train", "#2a9d8f"), ("val", "#e9c46a"), ("test", "#e76f51")]):
        ax.bar(np.arange(len(ALL_NAMES)) + i * w, wide[s].values, w, label=s, color=c)
    ax.set_xticks(np.arange(len(ALL_NAMES)) + w)
    ax.set_xticklabels(ALL_NAMES, rotation=10)
    ax.set_ylabel("MAE (Wh)")
    ax.set_title("Train / validation / test MAE per model")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E4_generalisation_gap.png"), dpi=150)
    print(f"\n  Outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
