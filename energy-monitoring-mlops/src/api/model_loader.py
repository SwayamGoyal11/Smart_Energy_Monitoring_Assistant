"""
src/api/model_loader.py
------------------------
Resolves the serving model at startup, in priority order:

    1. models:/<name>@champion   (MLflow registry)
    2. models:/<name>@staging    (MLflow registry)
    3. MODEL_LOCAL_PATH          (an MLflow model directory on disk)

The registry is skipped when its SQLite file does not exist, so a container
without mlflow.db never creates an empty one.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.models.registry import CHAMPION_ALIAS, STAGING_ALIAS

logger = logging.getLogger("energy_api")

GATE_METRIC_PREFIXES = ("val_", "test_", "baseline_val_", "baseline_test_")


@dataclass
class LoadedModel:
    model: Any
    name: str
    version: str
    source: str
    uri: str
    run_id: Optional[str] = None
    model_type: Optional[str] = None
    quality_gate: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def registry_reachable(tracking_uri: str) -> bool:
    """False when the tracking URI is a SQLite file that does not exist."""
    if not tracking_uri:
        return False
    if tracking_uri.startswith("sqlite:///"):
        return os.path.isfile(tracking_uri[len("sqlite:///"):])
    return True


def _load_from_registry(client: MlflowClient, name: str, alias: str) -> LoadedModel:
    version = client.get_model_version_by_alias(name, alias)
    uri = f"models:/{name}@{alias}"
    model = mlflow.pyfunc.load_model(uri)
    run_metrics = client.get_run(version.run_id).data.metrics
    return LoadedModel(
        model=model,
        name=name,
        version=str(version.version),
        source=f"registry:{alias}",
        uri=uri,
        run_id=version.run_id,
        model_type=version.tags.get("model_type"),
        quality_gate=version.tags.get("quality_gate"),
        metrics={k: v for k, v in run_metrics.items() if k.startswith(GATE_METRIC_PREFIXES)},
    )


def load_serving_model(tracking_uri: str, model_name: str, local_path: Optional[str]) -> Optional[LoadedModel]:
    """Return the first model that loads, or None if every source fails."""
    if registry_reachable(tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        for alias in (CHAMPION_ALIAS, STAGING_ALIAS):
            try:
                loaded = _load_from_registry(client, model_name, alias)
                logger.info("Loaded %s version %s from %s", model_name, loaded.version, loaded.source)
                return loaded
            except MlflowException as exc:
                logger.warning("Could not load %s@%s: %s", model_name, alias, exc)
    else:
        logger.warning("MLflow registry not reachable at %s; skipping registry", tracking_uri)

    if local_path and os.path.isfile(os.path.join(local_path, "MLmodel")):
        try:
            model = mlflow.pyfunc.load_model(local_path)
        except (MlflowException, OSError) as exc:
            logger.warning("Could not load local model at %s: %s", local_path, exc)
        else:
            logger.info("Loaded local model from %s", local_path)
            return LoadedModel(model=model, name=model_name, version="local", source="local", uri=local_path)

    logger.error("No serving model available (registry and local fallback both failed)")
    return None


def champion_matches(loaded: LoadedModel, tracking_uri: str) -> Optional[bool]:
    """
    True if the loaded model is still the registry's champion, False if the
    champion has moved (or a non-champion is being served), None if unknown.
    """
    if not registry_reachable(tracking_uri):
        return None
    try:
        current = MlflowClient(tracking_uri=tracking_uri).get_model_version_by_alias(
            loaded.name, CHAMPION_ALIAS,
        )
    except MlflowException:
        return False if loaded.source.startswith("registry:") else None
    return loaded.source == f"registry:{CHAMPION_ALIAS}" and str(current.version) == loaded.version
