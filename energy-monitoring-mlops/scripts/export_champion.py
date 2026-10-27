"""
scripts/export_champion.py
---------------------------
Export the registry's serving model to a standalone MLflow model directory
that the Docker image mounts at MODEL_LOCAL_PATH. The registry's artifact
paths are host-absolute, so the container serves this exported copy instead
of reading mlflow.db.

    python scripts/export_champion.py                 # @champion -> models/champion
    python scripts/export_champion.py --alias staging

Writes serving_metadata.json next to the model so /health and /model-info
report the registry version, model type and quality-gate metrics.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.api.model_loader import GATE_METRIC_PREFIXES, SERVING_METADATA_FILE  # noqa: E402
from src.models.registry import CHAMPION_ALIAS, REGISTERED_MODEL, tracking_uri  # noqa: E402

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "models", "champion")


def export(alias: str, output: str, model_name: str = REGISTERED_MODEL) -> dict:
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, alias)
    run_metrics = client.get_run(version.run_id).data.metrics

    # Only replace a directory that is empty or a previous export.
    if os.path.isdir(output) and os.listdir(output):
        if not os.path.isfile(os.path.join(output, "MLmodel")):
            raise SystemExit(f"Refusing to overwrite {output}: it is not an MLflow model directory")
        shutil.rmtree(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_name}@{alias}", dst_path=output)

    metadata = {
        "model_name": model_name,
        "model_version": str(version.version),
        "alias": alias,
        "run_id": version.run_id,
        "model_type": version.tags.get("model_type"),
        "quality_gate": version.tags.get("quality_gate"),
        "metrics": {k: v for k, v in run_metrics.items() if k.startswith(GATE_METRIC_PREFIXES)},
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output, SERVING_METADATA_FILE), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export a registry alias for Docker serving.")
    parser.add_argument("--alias", default=CHAMPION_ALIAS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    meta = export(args.alias, args.output)
    print(f"  Exported {meta['model_name']} v{meta['model_version']} ({meta['model_type']}, "
          f"gate {meta['quality_gate']}) @{args.alias} -> {args.output}")


if __name__ == "__main__":
    main()
