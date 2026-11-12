"""Run from the project root using the existing MLflow store."""

import argparse
import os
from pathlib import Path

import mlflow
from zenml.client import Client

from pipelines.zenml_pipeline import energy_hurdle_pipeline
from src.data.load_data import RAW_DATA_PATH
from src.models.registry import tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", default=RAW_DATA_PATH)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    tracker = Client().active_stack.experiment_tracker
    if tracker is None or tracker.name != "mlflow_tracker":
        parser.error("Activate a stack with experiment tracker 'mlflow_tracker'.")
    uri = tracking_uri()
    if tracker.get_tracking_uri() != uri:
        parser.error(f"Tracker URI must match the project URI: {uri}")
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    energy_hurdle_pipeline(raw_data_path=str(Path(args.raw_data).resolve()),
                          mlflow_tracking_uri=uri)


if __name__ == "__main__":
    main()
