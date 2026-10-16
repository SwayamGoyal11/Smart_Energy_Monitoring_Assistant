"""
experiments/10_bootstrap_ci.py
------------------------------
Are the MAE differences between models statistically meaningful?
Moving-block bootstrap (blocks of 144 rows = 1 day, to respect autocorrelation)
over the test set; each resample uses the same rows for every model (paired).
Reports 95% CIs for each model's MAE and for (model MAE - persistence MAE).

Outputs -> reports/evidence/bootstrap_ci/bootstrap_ci.csv
          reports/figures/E9_bootstrap_ci.png
Run from project root:  python experiments/10_bootstrap_ci.py
"""

import os
import numpy as np
import pandas as pd

from _common import (load_splits, get_X_y, make_model, predict_all, evidence_dir,
                     FIG_DIR, MODEL_NAMES, ALL_NAMES, plt)

OUT = evidence_dir("bootstrap_ci")
BLOCK, B = 144, 2000


def main():
    _, train, _, test = load_splits()
    Xt, yt = get_X_y(train)
    Xs, ys = get_X_y(test)
    models = {n: make_model(n).fit(Xt, yt) for n in MODEL_NAMES}
    ae = {n: np.abs(p - ys.values) for n, p in predict_all(models, Xs).items()}

    n = len(ys)
    nb = int(np.ceil(n / BLOCK))
    rng = np.random.RandomState(0)
    stats = {k: [] for k in ALL_NAMES}
    diffs = {k: [] for k in MODEL_NAMES}
    for _ in range(B):
        starts = rng.randint(0, n - BLOCK + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        for k in ALL_NAMES:
            stats[k].append(ae[k][idx].mean())
        for k in MODEL_NAMES:
            diffs[k].append(ae[k][idx].mean() - ae["Persistence"][idx].mean())

    rows = []
    for k in ALL_NAMES:
        lo, hi = np.percentile(stats[k], [2.5, 97.5])
        row = {"model": k, "mae": ae[k].mean(), "ci_lo": lo, "ci_hi": hi}
        if k in diffs:
            dl, dh = np.percentile(diffs[k], [2.5, 97.5])
            row.update({"diff_vs_persistence": ae[k].mean() - ae["Persistence"].mean(),
                        "diff_lo": dl, "diff_hi": dh,
                        "excludes_zero": bool(dl > 0 or dh < 0)})
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "bootstrap_ci.csv"), index=False)
    print(res.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in res.iterrows():
        ax.errorbar(r.mae, i, xerr=[[r.mae - r.ci_lo], [r.ci_hi - r.mae]], fmt="o", capsize=4)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res.model)
    ax.set_xlabel("Test MAE (Wh), 95% block-bootstrap CI")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E9_bootstrap_ci.png"), dpi=150)


if __name__ == "__main__":
    main()
