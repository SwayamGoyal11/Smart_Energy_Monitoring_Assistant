"""CLI entry point for the ZenML energy-monitoring training pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.energy_training import (  # noqa: E402
    DEFAULT_RAW_DATA_PATH,
    energy_monitoring_training_pipeline,
)


def _start_mlflow_ui() -> None:
    """Start the existing SQLite-backed MLflow UI and open it in a browser."""
    tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    subprocess.Popen(
        [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri],
        cwd=PROJECT_ROOT,
    )
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5000")


def main() -> None:
    """Parse CLI arguments, execute the pipeline, and optionally open MLflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data-path",
        default=str(DEFAULT_RAW_DATA_PATH),
        help="DVC-materialized CSV to ingest (defaults to data/raw/energydata_complete.csv).",
    )
    parser.add_argument(
        "--open-mlflow",
        action="store_true",
        help="Start MLflow UI for the existing local mlflow.db after the run.",
    )
    args = parser.parse_args()
    energy_monitoring_training_pipeline(raw_data_path=args.raw_data_path)
    if args.open_mlflow:
        _start_mlflow_ui()


if __name__ == "__main__":
    main()
