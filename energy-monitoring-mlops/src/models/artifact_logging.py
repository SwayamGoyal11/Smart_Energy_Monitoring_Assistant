"""
src/models/artifact_logging.py
-------------------------------
Run-level MLflow artifacts that make a registered model auditable and
reproducible. Layout inside the run's artifact root:

    data/            lineage.json, split_summary.json, data_summary.json, <dataset>.csv.dvc
    features/        feature_names.json, feature_schema.json, feature_importance.{json,png}
    preprocessing/   feature_encoders.json, model_pipeline.html,
                     stage1_scaler_params.json, stage1_pipeline.skops       (HurdleModel)
    model/           MLmodel, model.skops, signature, input example, requirements
                     (written by mlflow.<flavor>.log_model in registry.py)
    config/          model_config.yaml, hyperparameters.yaml, quality_gates.json,
                     hurdle_thresholds.json                                  (HurdleModel)
    evaluation/      metrics_summary.json, residuals.png, quality_gate_failures.txt,
                     stage1_confusion_matrix.png, stage1_classification_report.json,
                     stage2_residuals.png                                    (HurdleModel)
    logs/            training.log, cv_convergence.{json,png}
    metadata/        run_metadata.json, environment.json, requirements_frozen.txt
    deployment/      openapi.json, api_summary.json, request_example.json,
                     batch_request_example.json, response_example.json,
                     example_scoring.json, docker_spec.yaml, Dockerfile,
                     docker-compose.yml, requirements-serve.txt
    monitoring/      reference_stats.json, drift_config.json

Training-only module: it imports matplotlib/seaborn, which the serving image
does not install. Import it lazily (registry.py is imported by the API).
Figures are built on matplotlib.figure.Figure (no pyplot state) and passed to
mlflow.log_figure, so nothing is written to the working directory.
"""

import contextlib
import getpass
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import skops.io as sio
import yaml
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import estimator_html_repr

from src.data.load_data import RAW_DATA_PATH
from src.features.feature_engineering import (
    ALL_FEATURE_COLS, ENVIRONMENTAL_FEATURES, LAG_FEATURES, ROLLING_FEATURES, TARGET_COL,
    TIME_FEATURES,
)
from src.models.hurdle import PERSISTENCE_FEATURE, HurdleModel
from src.models.quality_gates import QualityThresholds
from src.models.train import PROJECT_ROOT
from src.validation.input_validation import PHYSICAL_BOUNDS, validate_feature_frame

logger = logging.getLogger(__name__)

DVC_FILE = os.path.abspath(RAW_DATA_PATH) + ".dvc"
DOCKERFILE = os.path.join(PROJECT_ROOT, "Dockerfile")
COMPOSE_FILE = os.path.join(PROJECT_ROOT, "docker-compose.yml")
SERVE_REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements-serve.txt")
IMAGE_NAME = "energy-forecaster-api"

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
PSI_BINS = 10
IMPORTANCE_SAMPLE = 2000
IMPORTANCE_TOP_N = 20
CV_SPLITS = 5
BATCH_EXAMPLE_SIZE = 3
RANDOM_STATE = 42

KEY_PACKAGES = (
    "mlflow", "scikit-learn", "xgboost", "pandas", "numpy", "skops",
    "evidently", "fastapi", "pydantic", "dvc", "zenml",
)

FEATURE_GROUPS = {
    **{c: "time" for c in TIME_FEATURES},
    **{c: "lag" for c in LAG_FEATURES},
    **{c: "rolling" for c in ROLLING_FEATURES},
    **{c: "environmental" for c in ENVIRONMENTAL_FEATURES},
}

# Mirrors src.features.feature_engineering.create_time_features.
CYCLICAL_ENCODINGS = {
    "hour": {"period": 24, "offset": 0, "features": ["hour_sin", "hour_cos"]},
    "day_of_week": {"period": 7, "offset": 0, "features": ["day_of_week_sin", "day_of_week_cos"]},
    "month": {"period": 12, "offset": 1, "features": ["month_sin", "month_cos"]},
}

# Chart tokens (reference dataviz palette, light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


# ---------------------------------------------------------------- helpers

def _round(value, ndigits: int = 4):
    return None if value is None or not np.isfinite(value) else round(float(value), ndigits)


def _finite(metrics: dict) -> dict:
    """mlflow.log_metrics rejects None; drop metrics that could not be computed."""
    return {k: v for k, v in metrics.items() if v is not None}


def _jsonable_params(params: dict) -> dict:
    """Keep scalar hyperparameters; drop nested estimators and callables."""
    out = {}
    for key, value in sorted(params.items()):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            out[key] = value
    return out


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL, cwd=PROJECT_ROOT,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _strip_credentials(url: Optional[str]) -> Optional[str]:
    """Remove user:token@ from remote/tracking URLs before they are stored."""
    return re.sub(r"//[^/@]+@", "//", url) if url else url


def _md5(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), dpi=120, facecolor=SURFACE, layout="constrained")


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": _round(mean_absolute_error(y_true, y_pred)),
        "rmse": _round(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": _round(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def _summary_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    finite = s.dropna()
    stats = {
        "count": int(finite.size),
        "null_rate": _round(1.0 - finite.size / max(len(s), 1), 6),
        "mean": _round(finite.mean()),
        "std": _round(finite.std()),
        "min": _round(finite.min()),
        "max": _round(finite.max()),
        "quantiles": {f"p{int(q * 100):02d}": _round(finite.quantile(q)) for q in QUANTILES},
    }
    # Decile edges + reference proportions let PSI be computed later from this
    # file alone, without the training data.
    edges = np.unique(finite.quantile(np.linspace(0, 1, PSI_BINS + 1)).to_numpy())
    if edges.size > 1:
        counts, _ = np.histogram(finite, bins=edges)
        stats["psi_bins"] = {
            "edges": [_round(e, 6) for e in edges],
            "proportions": [_round(c / finite.size, 6) for c in counts],
        }
    return stats


# ------------------------------------------------------- 0. training log

@contextlib.contextmanager
def capture_run_log(logger_name: str = "src", artifact_path: str = "logs",
                    filename: str = "training.log"):
    """
    Record INFO+ messages from the project's loggers (src.*) and Python
    warnings (e.g. sklearn ConvergenceWarning) for the duration of the block,
    then upload the file to <artifact_path>/<filename> on the active run.
    The partial log is still uploaded when the block raises.
    """
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, filename)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))

    targets = [logging.getLogger(logger_name), logging.getLogger("py.warnings")]
    previous_levels = [t.level for t in targets]
    for target in targets:
        target.addHandler(handler)
        target.setLevel(logging.INFO)
    logging.captureWarnings(True)
    try:
        yield path
    finally:
        logging.captureWarnings(False)
        for target, level in zip(targets, previous_levels):
            target.removeHandler(handler)
            target.setLevel(level)
        handler.close()
        try:
            if mlflow.active_run() is not None:
                mlflow.log_artifact(path, artifact_path=artifact_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------- 1. dataset & lineage

def _data_summary(splits: dict) -> dict:
    threshold = HurdleModel().active_threshold_wh
    summary = {"target": TARGET_COL, "active_threshold_wh": threshold, "splits": {}}
    for name, frame in splits.items():
        y = frame[TARGET_COL]
        step = frame["date"].sort_values().diff().dropna()
        summary["splits"][name] = {
            "rows": len(frame),
            "columns": int(frame.shape[1]),
            "null_cells": int(frame.isna().sum().sum()),
            "duplicate_timestamps": int(frame["date"].duplicated().sum()),
            "sampling_interval_minutes": (
                _round(step.mode().iloc[0].total_seconds() / 60, 2) if not step.empty else None
            ),
            "gaps_over_interval": int((step > step.mode().iloc[0]).sum()) if not step.empty else 0,
            "active_rate": _round((y > threshold).mean()),
            "target_stats": _summary_stats(y),
        }
    return summary


def log_data_lineage(
    train_df: Optional[pd.DataFrame],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dvc_file: str = DVC_FILE,
) -> dict:
    """DVC version of the raw data, whether the workspace matches it, and split metadata."""
    lineage = {"dvc_file": os.path.basename(dvc_file)}
    if os.path.exists(dvc_file):
        with open(dvc_file) as f:
            out = (yaml.safe_load(f) or {}).get("outs", [{}])[0]
        data_path = os.path.join(os.path.dirname(dvc_file), out.get("path", ""))
        workspace_md5 = _md5(data_path) if os.path.isfile(data_path) else None
        lineage.update({
            "dvc_md5": out.get("md5"),
            "dvc_size_bytes": out.get("size"),
            "data_path": out.get("path"),
            "workspace_md5": workspace_md5,
            "workspace_matches_dvc": workspace_md5 == out.get("md5"),
            "dvc_file_git_commit": _git("log", "-1", "--format=%H", "--", dvc_file),
        })
        mlflow.log_artifact(dvc_file, artifact_path="data")
        mlflow.set_tags({
            "data.dvc_md5": str(out.get("md5")),
            "data.workspace_matches_dvc": str(lineage["workspace_matches_dvc"]).lower(),
        })
    else:
        logger.warning("DVC file not found at %s; lineage has no data hash", dvc_file)
    lineage["git_commit"] = _git("rev-parse", "HEAD")
    lineage["git_dirty"] = bool(_git("status", "--porcelain"))
    mlflow.log_dict(lineage, "data/lineage.json")

    splits = {"train": train_df, "validation": val_df, "test": test_df}
    present = {k: v for k, v in splits.items() if v is not None}
    total = sum(len(v) for v in present.values())
    summary = {"strategy": "chronological", "shuffled": False, "target": TARGET_COL, "splits": {}}
    for name, frame in present.items():
        summary["splits"][name] = {
            "rows": len(frame),
            "fraction": _round(len(frame) / total),
            "start": str(frame["date"].min()),
            "end": str(frame["date"].max()),
            "target_mean": _round(frame[TARGET_COL].mean()),
            "target_std": _round(frame[TARGET_COL].std()),
        }
    mlflow.log_dict(summary, "data/split_summary.json")
    mlflow.log_dict(_data_summary(present), "data/data_summary.json")
    logger.info("Data lineage: dvc_md5=%s, splits=%s", lineage.get("dvc_md5"),
                {k: len(v) for k, v in present.items()})

    # Populates the run's "Datasets" panel; source points back at the DVC file.
    for name, context in (("train", "training"), ("validation", "validation"), ("test", "test")):
        frame = splits[name]
        if frame is not None:
            dataset = mlflow.data.from_pandas(
                frame, source=dvc_file, targets=TARGET_COL, name=f"energydata_{name}",
            )
            mlflow.log_input(dataset, context=context)
    return lineage


# ------------------------------- 2. preprocessing & feature engineering

def log_feature_artifacts(model, X_train: Optional[pd.DataFrame], X_val: pd.DataFrame, y_val) -> None:
    columns = list(X_val.columns)
    reference = X_train if X_train is not None else X_val
    mlflow.log_dict(
        {"n_features": len(columns), "feature_names": columns,
         "groups": {g: [c for c in columns if FEATURE_GROUPS.get(c) == g]
                    for g in ("time", "lag", "rolling", "environmental")}},
        "features/feature_names.json",
    )
    schema = {
        "n_features": len(columns),
        "feature_engineering": {
            "lags": LAG_FEATURES,
            "rolling": ROLLING_FEATURES,
            "leakage_guard": "lag/rolling built on shift(1); target never in its own window",
        },
        "features": [
            {
                "name": c,
                "position": i,
                "dtype": str(X_val[c].dtype),
                "group": FEATURE_GROUPS.get(c, "other"),
                "physical_bounds": list(PHYSICAL_BOUNDS[c]) if c in PHYSICAL_BOUNDS else None,
                "observed_range": [_round(reference[c].min()), _round(reference[c].max())],
            }
            for i, c in enumerate(columns)
        ],
    }
    mlflow.log_dict(schema, "features/feature_schema.json")

    # Permutation importance is model-agnostic, so it works for the HurdleModel
    # (no feature_importances_) and the plain regressors alike.
    sample = X_val.sample(min(IMPORTANCE_SAMPLE, len(X_val)), random_state=RANDOM_STATE)
    y_sample = pd.Series(np.asarray(y_val), index=X_val.index).loc[sample.index]
    result = permutation_importance(
        model, sample, y_sample,
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1,
    )
    importance = (
        pd.DataFrame({"feature": columns, "mean": result.importances_mean, "std": result.importances_std})
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    mlflow.log_dict(
        {
            "method": "permutation",
            "scoring": "MAE increase (Wh) when the feature is shuffled",
            "split": "validation",
            "n_samples": len(sample),
            "n_repeats": 5,
            "importances": importance.round(4).to_dict(orient="records"),
        },
        "features/feature_importance.json",
    )
    logger.info("Top features by permutation importance: %s",
                ", ".join(importance["feature"].head(5)))

    top = importance.head(IMPORTANCE_TOP_N).iloc[::-1]
    fig = _new_figure(7.5, 0.28 * len(top) + 1.2)
    ax = fig.subplots()
    ax.barh(top["feature"], top["mean"], xerr=top["std"], color=SERIES_1, height=0.6,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 0.8, "capsize": 0})
    _style_axes(ax, f"Permutation importance (top {len(top)}, validation)",
                "MAE increase when shuffled (Wh)", "")
    ax.grid(axis="y", visible=False)
    mlflow.log_figure(fig, "features/feature_importance.png")


def _find_scaler(estimator) -> Optional[StandardScaler]:
    for _, step in getattr(estimator, "steps", []):
        if isinstance(step, StandardScaler):
            return step
    return estimator if isinstance(estimator, StandardScaler) else None


def log_preprocessing_artifacts(model, feature_names: list) -> None:
    mlflow.log_text(estimator_html_repr(model), "preprocessing/model_pipeline.html")

    scaler = _find_scaler(model.classifier_) if isinstance(model, HurdleModel) else _find_scaler(model)
    mlflow.log_dict(
        {
            "categorical_encoders": [],
            "note": "All model inputs are numeric; no one-hot/ordinal encoders are fitted. "
                    "Features are rebuilt from raw readings by src.features.feature_engineering "
                    "in training and by src.serving.schemas.build_feature_frame in serving.",
            "time": {
                "source_column": "date",
                "raw": ["hour", "day", "day_of_week", "month", "is_weekend"],
                "cyclical": {
                    src: {**enc, "formula": f"sin/cos(2*pi*({src} - {enc['offset']}) / {enc['period']})"}
                    for src, enc in CYCLICAL_ENCODINGS.items()
                },
            },
            "lag": {"source_column": TARGET_COL, "features": LAG_FEATURES},
            "rolling": {"source_column": TARGET_COL, "applied_after": "shift(1)",
                        "features": ROLLING_FEATURES},
            "scalers": (
                [{"stage": "stage1_classifier" if isinstance(model, HurdleModel) else "model",
                  "type": type(scaler).__name__,
                  "artifact": "preprocessing/stage1_scaler_params.json"
                  if isinstance(model, HurdleModel) else None}]
                if scaler is not None else []
            ),
        },
        "preprocessing/feature_encoders.json",
    )

    if not isinstance(model, HurdleModel):
        return
    if scaler is not None:
        mlflow.log_dict(
            {
                "stage": "stage1_classifier",
                "transformer": type(scaler).__name__,
                "params": {
                    name: {"mean": _round(m, 6), "scale": _round(s, 6)}
                    for name, m, s in zip(feature_names, scaler.mean_, scaler.scale_)
                },
            },
            "preprocessing/stage1_scaler_params.json",
        )
    # Standalone copy of the Stage 1 pipeline for offline inspection. Serving
    # always loads the full model; this file is never used for inference.
    with tempfile.TemporaryDirectory() as tmp:
        sio.dump(model.classifier_, os.path.join(tmp, "stage1_pipeline.skops"))
        mlflow.log_artifacts(tmp, artifact_path="preprocessing")


# -------------------------------- 3. model configuration & architecture

def _stage_config(est) -> dict:
    return {
        "class": type(est).__name__,
        "steps": [type(s).__name__ for _, s in getattr(est, "steps", [])] or None,
        "params": _jsonable_params(est.get_params(deep=True)),
    }


def log_model_config(model, model_type: str, thresholds: QualityThresholds,
                     params: Optional[dict] = None) -> None:
    config = {"model_type": model_type, "class": f"{type(model).__module__}.{type(model).__name__}"}
    hyper = {"model_type": model_type, "constructor_params": _jsonable_params(params or {})}
    if isinstance(model, HurdleModel):
        hurdle = {
            "active_threshold_wh": model.active_threshold_wh,
            "decision_threshold": model.decision_threshold,
            "standby_route": f"persistence ({PERSISTENCE_FEATURE})",
            "train_active_rate": _round(model.active_rate_),
        }
        config["hurdle"] = hurdle
        stages = {"stage1_classifier": _stage_config(model.classifier_),
                  "stage2_regressor": _stage_config(model.regressor_)}
        config.update(stages)
        hyper["effective_params"] = {"hurdle": {k: hurdle[k] for k in
                                                ("active_threshold_wh", "decision_threshold")},
                                     **{k: v["params"] for k, v in stages.items()}}
        mlflow.log_dict(
            {
                **hurdle,
                "rules": {
                    "stage1_label": f"active if {TARGET_COL} > {model.active_threshold_wh} Wh",
                    "routing": f"stage2 forecast if P(active) >= {model.decision_threshold}, "
                               f"else {PERSISTENCE_FEATURE}",
                    "output_floor_wh": 0.0,
                },
            },
            "config/hurdle_thresholds.json",
        )
    else:
        config["params"] = _jsonable_params(model.get_params(deep=True))
        hyper["effective_params"] = config["params"]
    mlflow.log_dict(config, "config/model_config.yaml")
    mlflow.log_dict(hyper, "config/hyperparameters.yaml")
    mlflow.log_dict(vars(thresholds), "config/quality_gates.json")


# --------------------------------------- 4. training curves (logs/)

def log_cv_convergence(model, X: pd.DataFrame, y, n_splits: int = CV_SPLITS) -> dict:
    """
    Expanding-window TimeSeriesSplit on the training split: refit a clone of
    the model on each fold and score it on the next block. Shows how error
    converges as training history grows (a learning curve that respects time
    order), plus per-fold fit time. Per-fold values are also logged as MLflow
    step metrics (cv_*), so the UI charts them natively.
    """
    y = np.asarray(y, dtype=float)
    folds = []
    for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), start=1):
        est = clone(model)
        start = time.perf_counter()
        try:
            est.fit(X.iloc[tr], y[tr])
        except ValueError as exc:
            logger.warning("CV fold %d/%d skipped: %s", i, n_splits, exc)
            folds.append({"fold": i, "train_rows": len(tr), "error": str(exc)})
            continue
        fit_seconds = time.perf_counter() - start
        train_m = _regression_metrics(y[tr], est.predict(X.iloc[tr]))
        val_m = _regression_metrics(y[te], est.predict(X.iloc[te]))
        folds.append({
            "fold": i,
            "train_rows": len(tr),
            "heldout_rows": len(te),
            "fit_seconds": _round(fit_seconds, 3),
            "train": train_m,
            "heldout": val_m,
        })
        mlflow.log_metrics(_finite({
            "cv_train_mae": train_m["mae"], "cv_heldout_mae": val_m["mae"],
            "cv_heldout_rmse": val_m["rmse"], "cv_fit_seconds": _round(fit_seconds, 3),
        }), step=i)
        logger.info("CV fold %d/%d | train_rows=%d | fit %.2fs | train MAE %.2f | "
                    "held-out MAE %.2f RMSE %.2f", i, n_splits, len(tr), fit_seconds,
                    train_m["mae"], val_m["mae"], val_m["rmse"])

    ok = [f for f in folds if "error" not in f]
    heldout_mae = np.array([f["heldout"]["mae"] for f in ok], dtype=float)
    heldout_rmse = np.array([f["heldout"]["rmse"] for f in ok], dtype=float)
    flat = _finite({
        "cv_heldout_mae_mean": _round(heldout_mae.mean()) if ok else None,
        "cv_heldout_mae_std": _round(heldout_mae.std()) if ok else None,
        "cv_heldout_rmse_mean": _round(heldout_rmse.mean()) if ok else None,
    })
    mlflow.log_dict(
        {"method": "TimeSeriesSplit (expanding window)", "n_splits": n_splits,
         "split": "train", "folds": folds, "summary": flat},
        "logs/cv_convergence.json",
    )

    if ok:
        rows = [f["train_rows"] for f in ok]
        fig = _new_figure(10, 3.8)
        ax_mae, ax_time = fig.subplots(1, 2)
        ax_mae.plot(rows, [f["train"]["mae"] for f in ok], marker="o", color=SERIES_1,
                    label="train")
        ax_mae.plot(rows, heldout_mae, marker="o", color=SERIES_2, label="held-out")
        _style_axes(ax_mae, "CV convergence: MAE vs training history", "Training rows", "MAE (Wh)")
        ax_mae.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)
        ax_time.plot(rows, [f["fit_seconds"] for f in ok], marker="o", color=SERIES_1)
        _style_axes(ax_time, "Fit time per fold", "Training rows", "Seconds")
        mlflow.log_figure(fig, "logs/cv_convergence.png")
    return flat


# --------------------------------------------- 5. evaluation & diagnostics

def _plot_residuals(y_true, y_pred, title: str) -> Figure:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig = _new_figure(10, 3.8)
    ax_hist, ax_scatter = fig.subplots(1, 2)

    sns.histplot(residuals, bins=60, ax=ax_hist, color=SERIES_1, edgecolor=SURFACE, linewidth=0.5)
    ax_hist.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    _style_axes(ax_hist, f"{title}: residual distribution", "Actual - predicted (Wh)", "Intervals")

    ax_scatter.scatter(y_pred, residuals, s=8, alpha=0.35, color=SERIES_1, linewidths=0)
    ax_scatter.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    _style_axes(ax_scatter, f"{title}: residuals vs prediction", "Predicted (Wh)", "Residual (Wh)")
    return fig


def _plot_confusion(y_true, y_pred, title: str) -> Figure:
    labels = ["standby", "active"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    annot = np.array([[f"{cm[i, j]:,}\n{share[i, j]:.1%}" for j in range(2)] for i in range(2)])
    fig = _new_figure(4.8, 4.0)
    ax = fig.subplots()
    sns.heatmap(share, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=labels, yticklabels=labels, linewidths=2, linecolor=SURFACE,
                annot_kws={"fontsize": 10}, ax=ax)
    _style_axes(ax, title, "Predicted", "Actual")
    ax.grid(False)
    return fig


def log_evaluation_artifacts(
    model,
    splits: dict,
    metrics: dict,
    gate_failures: list,
    plot_split: str = "test",
) -> dict:
    """
    metrics_summary.json for every split, plus diagnostic plots on plot_split.
    For a HurdleModel, also scores each stage in isolation:
      stage 1 against the true active/standby label,
      stage 2 on truly active intervals (regressor quality without routing errors).
    Returns flat stage metrics so the caller can log them as run metrics.
    """
    summary = {"regression": {}, "baseline_persistence": {}, "quality_gate": {
        "passed": not gate_failures, "failures": gate_failures,
    }}
    flat = {}
    for split, (X, y) in splits.items():
        summary["regression"][split] = _regression_metrics(y, model.predict(X))
        summary["baseline_persistence"][split] = _regression_metrics(y, X[PERSISTENCE_FEATURE])

    if isinstance(model, HurdleModel):
        summary["stage1_classifier"], summary["stage2_regressor"] = {}, {}
        for split, (X, y) in splits.items():
            y = np.asarray(y, dtype=float)
            active = (y > model.active_threshold_wh).astype(int)
            proba = model.predict_active_proba(X)
            routed = (proba >= model.decision_threshold).astype(int)
            s1 = {
                "f1": _round(f1_score(active, routed, zero_division=0)),
                "precision": _round(precision_score(active, routed, zero_division=0)),
                "recall": _round(recall_score(active, routed, zero_division=0)),
                "log_loss": _round(log_loss(active, proba, labels=[0, 1])),
                "roc_auc": _round(roc_auc_score(active, proba)) if 0 < active.sum() < len(active) else None,
                "active_rate_true": _round(active.mean()),
                "active_rate_routed": _round(routed.mean()),
            }
            summary["stage1_classifier"][split] = s1
            mask = active.astype(bool)
            s2 = _regression_metrics(y[mask], model.regressor_.predict(X[mask])) if mask.any() else {}
            summary["stage2_regressor"][split] = s2
            prefix = "val" if split == "validation" else split
            flat.update({f"{prefix}_stage1_{k}": v for k, v in s1.items() if v is not None})
            flat.update({f"{prefix}_stage2_{k}": v for k, v in s2.items() if k != "n" and v is not None})

        X, y = splits[plot_split]
        y = np.asarray(y, dtype=float)
        active = (y > model.active_threshold_wh).astype(int)
        routed = (model.predict_active_proba(X) >= model.decision_threshold).astype(int)
        mlflow.log_figure(
            _plot_confusion(active, routed, f"Stage 1 hurdle ({plot_split})"),
            "evaluation/stage1_confusion_matrix.png",
        )
        mlflow.log_dict(
            classification_report(active, routed, labels=[0, 1], target_names=["standby", "active"],
                                  output_dict=True, zero_division=0),
            "evaluation/stage1_classification_report.json",
        )
        mask = active.astype(bool)
        if mask.any():
            mlflow.log_figure(
                _plot_residuals(y[mask], model.regressor_.predict(X[mask]), f"Stage 2 ({plot_split}, active)"),
                "evaluation/stage2_residuals.png",
            )

    X, y = splits[plot_split]
    mlflow.log_figure(_plot_residuals(y, model.predict(X), f"End-to-end ({plot_split})"),
                      "evaluation/residuals.png")
    summary["run_metrics"] = metrics
    mlflow.log_dict(summary, "evaluation/metrics_summary.json")
    logger.info("Evaluation: %s", {s: m["mae"] for s, m in summary["regression"].items()})
    return flat


# ------------------------------------------------- 6. model metadata

def log_run_metadata(model_type: str, fit_seconds: Optional[float] = None) -> dict:
    """Who/what/where produced this run: git state, author, environment, MLflow run info."""
    run = mlflow.active_run()
    info = run.info
    author = _git("config", "user.name") or getpass.getuser()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    run_meta = {
        "model_type": model_type,
        "author": author,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_seconds": _round(fit_seconds, 3) if fit_seconds is not None else None,
        "entry_point": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "commit_short": _git("rev-parse", "--short", "HEAD"),
            "branch": branch,
            "dirty": bool(_git("status", "--porcelain")),
            "remote": _strip_credentials(_git("remote", "get-url", "origin")),
            "commit_author": _git("log", "-1", "--format=%an"),
            "commit_date": _git("log", "-1", "--format=%cI"),
            "commit_subject": _git("log", "-1", "--format=%s"),
        },
        "mlflow": {
            "run_id": info.run_id,
            "run_name": info.run_name,
            "experiment_id": info.experiment_id,
            "experiment_name": mlflow.get_experiment(info.experiment_id).name,
            "start_time_utc": datetime.fromtimestamp(info.start_time / 1000, timezone.utc)
            .isoformat(timespec="seconds"),
            "tracking_uri": _strip_credentials(mlflow.get_tracking_uri()),
            "artifact_uri": _strip_credentials(info.artifact_uri),
            "mlflow_version": mlflow.__version__,
        },
    }
    mlflow.log_dict(run_meta, "metadata/run_metadata.json")

    environment = {
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "machine": platform.machine()},
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "packages": {name: _package_version(name) for name in KEY_PACKAGES},
    }
    mlflow.log_dict(environment, "metadata/environment.json")

    frozen = sorted(
        {f"{d.metadata['Name']}=={d.version}" for d in importlib_metadata.distributions()
         if d.metadata and d.metadata["Name"]},
        key=str.lower,
    )
    mlflow.log_text("\n".join(frozen) + "\n", "metadata/requirements_frozen.txt")

    mlflow.set_tags({"author": author, "git_branch": str(branch),
                     "python_version": platform.python_version()})
    logger.info("Run %s (%s) by %s at %s on %s", info.run_id, model_type, author,
                run_meta["git"]["commit_short"], branch)
    return run_meta


# ------------------------------------------ 7. deployment & serving

def _dockerfile_facts(path: str) -> dict:
    """Port, env, base images, CMD and HEALTHCHECK from the serving Dockerfile."""
    with open(path, encoding="utf-8") as f:
        logical = re.sub(r"\\\r?\n\s*", " ", f.read())
    facts = {"base_images": [], "expose": [], "env": {}, "cmd": None, "healthcheck": None}
    for line in (ln.strip() for ln in logical.splitlines()):
        keyword, _, rest = line.partition(" ")
        keyword = keyword.upper()
        if keyword == "FROM":
            facts["base_images"].append(rest.strip())
        elif keyword == "EXPOSE":
            facts["expose"].extend(int(p.split("/")[0]) for p in rest.split())
        elif keyword == "ENV":
            facts["env"].update({k: v.strip('"') for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)', rest)})
        elif keyword == "CMD":
            rest = rest.strip()
            facts["cmd"] = json.loads(rest) if rest.startswith("[") else rest
        elif keyword == "HEALTHCHECK":
            facts["healthcheck"] = rest.strip()
    return facts


def _request_examples(frame: pd.DataFrame, n: int) -> tuple:
    """
    Build n API requests from consecutive rows of a held-out split, using the
    raw readings a client would send. Returns (requests, row positions).
    Each request is validated by the real Pydantic model, so the example is
    guaranteed to be accepted by the API.
    """
    from pydantic import ValidationError

    from src.serving.schemas import HISTORY_LENGTH, INTERVAL, PredictionRequest

    frame = frame.reset_index(drop=True)
    dates = pd.to_datetime(frame["date"])
    for start in range(HISTORY_LENGTH, len(frame) - n + 1):
        positions = list(range(start, start + n))
        window = dates.iloc[start - HISTORY_LENGTH:start + n]
        if not (window.diff().dropna() == INTERVAL).all():
            continue
        try:
            requests = [
                PredictionRequest(
                    target_timestamp=dates.iloc[i].to_pydatetime(),
                    recent_appliances_wh=frame[TARGET_COL].iloc[i - HISTORY_LENGTH:i].astype(float).tolist(),
                    **{c: float(frame[c].iloc[i]) for c in ENVIRONMENTAL_FEATURES},
                )
                for i in positions
            ]
        except ValidationError:
            continue
        return requests, positions
    raise ValueError("no contiguous, in-bounds window found for request examples")


def log_deployment_artifacts(model, signature, example_df: pd.DataFrame,
                             registered_model_name: str) -> dict:
    """
    API contract (OpenAPI + endpoint summary), validated request/response
    examples, a training-serving feature parity check, and the container spec.
    """
    from src.api.app import app  # lazy: avoids a registry <-> app import cycle
    from src.serving.schemas import build_feature_frame

    spec = app.openapi()
    mlflow.log_dict(spec, "deployment/openapi.json")
    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {})
            ok = op.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {})
            endpoints.append({
                "method": method.upper(),
                "path": path,
                "summary": op.get("summary"),
                "request_schema": body.get("schema", {}).get("$ref", "").split("/")[-1] or None,
                "response_schema": ok.get("schema", {}).get("$ref", "").split("/")[-1] or None,
            })
    mlflow.log_dict(
        {
            "title": spec["info"]["title"],
            "api_version": spec["info"]["version"],
            "endpoints": endpoints,
            "model_signature": signature.to_dict() if signature is not None else None,
        },
        "deployment/api_summary.json",
    )

    # Request/response examples taken from real held-out rows.
    requests, positions = _request_examples(example_df, BATCH_EXAMPLE_SIZE)
    payloads = [r.model_dump(mode="json") for r in requests]
    mlflow.log_dict(payloads[0], "deployment/request_example.json")
    mlflow.log_dict({"instances": payloads}, "deployment/batch_request_example.json")

    X_served = validate_feature_frame(build_feature_frame(requests))
    preds = np.clip(np.asarray(model.predict(X_served), dtype=float), 0.0, None)
    rows = example_df.reset_index(drop=True).iloc[positions]
    X_trained = rows[ALL_FEATURE_COLS].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(X_served.to_numpy(dtype=float) - X_trained)))
    parity = bool(np.isfinite(max_abs_diff) and max_abs_diff <= 1e-6)
    if not parity:
        logger.warning("Training/serving feature skew: max |diff| = %.6g", max_abs_diff)

    version_placeholder = "<registered version>"
    mlflow.log_dict(
        {"predictions": [
            {"target_timestamp": p["target_timestamp"], "predicted_appliances_wh": round(float(v), 4),
             "model_name": registered_model_name, "model_version": version_placeholder}
            for p, v in zip(payloads, preds)
        ]},
        "deployment/response_example.json",
    )
    mlflow.log_dict(
        {
            "source_split": "test",
            "feature_parity_with_training": parity,
            "max_abs_feature_diff": _round(max_abs_diff, 9),
            "rows": [
                {"target_timestamp": p["target_timestamp"],
                 "actual_appliances_wh": _round(a), "predicted_appliances_wh": _round(v)}
                for p, a, v in zip(payloads, rows[TARGET_COL], preds)
            ],
        },
        "deployment/example_scoring.json",
    )

    # Container spec, plus the files it is built from.
    docker = _dockerfile_facts(DOCKERFILE) if os.path.isfile(DOCKERFILE) else {}
    port = (docker.get("expose") or [8000])[0]
    mlflow.log_dict(
        {
            "image": IMAGE_NAME,
            "model": {
                "registered_name": registered_model_name,
                "serving_alias_uri": f"models:/{registered_model_name}@champion",
                "this_run_uri": f"runs:/{mlflow.active_run().info.run_id}/model",
                "baked_path": docker.get("env", {}).get("MODEL_LOCAL_PATH"),
            },
            "build": [
                "python scripts/export_champion.py",
                f"docker build -t {IMAGE_NAME} .",
            ],
            "run": f"docker run --rm -p {port}:{port} {IMAGE_NAME}",
            "compose": "docker compose up --build -d",
            "container_port": port,
            "base_images": docker.get("base_images"),
            "env": docker.get("env"),
            "cmd": docker.get("cmd"),
            "healthcheck": docker.get("healthcheck"),
            "smoke_test": [
                f"curl http://localhost:{port}/health",
                f"curl -X POST http://localhost:{port}/predict "
                "-H 'Content-Type: application/json' -d @request_example.json",
            ],
            "endpoints": [f"{e['method']} {e['path']}" for e in endpoints],
        },
        "deployment/docker_spec.yaml",
    )
    for path in (DOCKERFILE, COMPOSE_FILE, SERVE_REQUIREMENTS):
        if os.path.isfile(path):
            mlflow.log_artifact(path, artifact_path="deployment")
    mlflow.set_tag("serving.feature_parity", str(parity).lower())
    logger.info("Deployment bundle: %d endpoints, feature parity=%s", len(endpoints), parity)
    return {"serving_feature_parity": int(parity)}


# ------------------------------------------------ 8. monitoring baseline

def log_monitoring_baseline(model, X_ref: pd.DataFrame, y_ref, source_split: str) -> None:
    """Reference distribution stats and drift-monitor config for post-deployment checks."""
    preds = pd.Series(model.predict(X_ref), index=X_ref.index)
    mlflow.log_dict(
        {
            "reference_split": source_split,
            "rows": len(X_ref),
            "quantiles": [f"p{int(q * 100):02d}" for q in QUANTILES],
            "psi_bins": PSI_BINS,
            "features": {c: _summary_stats(X_ref[c]) for c in X_ref.columns},
            "target": _summary_stats(pd.Series(y_ref)),
            "prediction": _summary_stats(preds),
        },
        "monitoring/reference_stats.json",
    )

    # Imported lazily: src.monitoring.drift pulls in evidently.
    try:
        from src.monitoring.drift import METHODS, MONITORED_FEATURES, DriftConfig
    except ImportError as exc:
        logger.warning("Drift config not logged (%s)", exc)
        return
    config = DriftConfig()
    mlflow.log_dict(
        {
            "tool": "evidently",
            "method": config.method,
            "score_type": METHODS[config.method],
            "threshold": config.threshold,
            "dataset_drift_share": config.drift_share,
            "min_current_rows": config.min_current_rows,
            "column_mapping": {
                "target": TARGET_COL,
                "prediction": "prediction",
                "numerical_features": MONITORED_FEATURES,
            },
            "excluded_features": {
                "columns": [c for c in X_ref.columns if c not in MONITORED_FEATURES],
                "reason": "calendar features drift by construction (serving is always later in time)",
            },
            "reference": {"artifact": "monitoring/reference_stats.json", "split": source_split},
            "run_command": "python -m src.monitoring.drift --current data/monitoring/inference_log.csv",
        },
        "monitoring/drift_config.json",
    )
    logger.info("Monitoring baseline: %d features from %s split (%d rows)",
                X_ref.shape[1], source_split, len(X_ref))
