"""
src/api/app.py
---------------
FastAPI inference service for the appliance energy forecaster.

Run from the project root:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Configuration (environment variables):
    MLFLOW_TRACKING_URI  registry location (default: project sqlite:///mlflow.db)
    MODEL_NAME           registered model name (default: energy-forecaster)
    MODEL_LOCAL_PATH     MLflow model directory used if the registry is unavailable
                         (default: models/champion)
    CORS_ORIGINS         comma-separated allowed origins (default: *)
    DRIFT_METRICS_PATH   drift_metrics.json exported as model_drift_* gauges
                         (default: reports/evidence/drift_metrics.json)
    INFERENCE_LOG_PATH   CSV to append served features + predictions to, for
                         drift monitoring (default: unset, logging disabled)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.model_loader import LoadedModel, champion_matches, load_serving_model
from src.api.schemas import (
    BatchPredictionRequest, BatchPredictionResponse, HealthResponse, ModelInfoResponse,
    PredictionRequest, PredictionResponse,
)
from src.features.feature_engineering import ALL_FEATURE_COLS
from src.serving.schemas import build_feature_frame
from src.models.quality_gates import QualityThresholds
from src.models.registry import REGISTERED_MODEL, tracking_uri
from src.models.train import PROJECT_ROOT, REPORTS_DIR
from src.monitoring.inference_log import InferenceLogger
from src.monitoring.metrics import ApiMetrics
from src.validation.input_validation import InputValidationError, validate_feature_frame

logger = logging.getLogger("energy_api")


@dataclass
class Settings:
    tracking_uri: str = field(default_factory=tracking_uri)
    model_name: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", REGISTERED_MODEL))
    local_model_path: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "MODEL_LOCAL_PATH", os.path.join(PROJECT_ROOT, "models", "champion"),
        )
    )
    cors_origins: List[str] = field(
        default_factory=lambda: [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    )
    drift_metrics_path: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "DRIFT_METRICS_PATH", os.path.join(REPORTS_DIR, "drift_metrics.json"),
        )
    )
    inference_log_path: Optional[str] = field(default_factory=lambda: os.environ.get("INFERENCE_LOG_PATH"))


def _require_model(request: Request) -> LoadedModel:
    loaded = request.app.state.loaded
    if loaded is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return loaded


def _stage_model(loaded: Optional[LoadedModel]):
    """The underlying HurdleModel, if that is what is being served."""
    if loaded is None:
        return None
    try:
        raw = loaded.model.get_raw_model()
    except Exception:
        return None
    return raw if hasattr(raw, "predict_active_proba") else None


def _predict(request: Request, requests: List[PredictionRequest]) -> List[PredictionResponse]:
    loaded = _require_model(request)
    state = request.app.state
    frame = build_feature_frame(requests)
    try:
        X = validate_feature_frame(frame)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.issues)

    try:
        preds = np.asarray(loaded.model.predict(X), dtype=float).reshape(-1)
    except Exception:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    if len(preds) != len(requests) or not np.isfinite(preds).all():
        logger.error("Model returned invalid output: %s", preds)
        raise HTTPException(status_code=500, detail="Prediction failed")

    # Energy cannot be negative; the hurdle model already clips, other flavors may not.
    preds = np.clip(preds, 0.0, None)

    state.metrics.observe_predictions(preds, loaded.version)
    if state.stage_model is not None:
        stage = state.stage_model
        n_active = int((stage.predict_active_proba(X) >= stage.decision_threshold).sum())
        state.metrics.observe_stages(n_active, len(preds) - n_active)
    if state.inference_logger is not None:
        try:
            state.inference_logger.log(X, preds, [r.target_timestamp for r in requests], loaded.version)
        except OSError:
            logger.exception("Could not write inference log")

    return [
        PredictionResponse(
            target_timestamp=r.target_timestamp,
            predicted_appliances_wh=round(float(p), 4),
            model_name=loaded.name,
            model_version=loaded.version,
        )
        for r, p in zip(requests, preds)
    ]


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at = time.monotonic()
        app.state.loaded = load_serving_model(
            settings.tracking_uri, settings.model_name, settings.local_model_path,
        )
        app.state.stage_model = _stage_model(app.state.loaded)
        yield
        logger.info("Shutting down; releasing model")
        app.state.loaded = None
        app.state.stage_model = None

    app = FastAPI(
        title="Smart Energy Monitoring API",
        description="10-minute appliance energy forecasts from the MLflow champion model.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.loaded = None
    app.state.stage_model = None
    app.state.started_at = time.monotonic()
    app.state.metrics = ApiMetrics(settings.drift_metrics_path)
    app.state.inference_logger = (
        InferenceLogger(settings.inference_log_path) if settings.inference_log_path else None
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def instrument(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        # Label by route template, not raw URL, to keep label cardinality bounded.
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        request.app.state.metrics.observe_request(request.method, path, response.status_code, elapsed)
        response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.2f}"
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        loaded = request.app.state.loaded
        body = HealthResponse(
            status="ok" if loaded else "unavailable",
            model_loaded=loaded is not None,
            model_name=loaded.name if loaded else None,
            model_version=loaded.version if loaded else None,
            model_source=loaded.source if loaded else None,
            champion_alias_verified=champion_matches(loaded, settings.tracking_uri) if loaded else None,
            uptime_seconds=round(time.monotonic() - request.app.state.started_at, 3),
        )
        return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        return _predict(request, [payload])[0]

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(payload: BatchPredictionRequest, request: Request):
        return BatchPredictionResponse(predictions=_predict(request, payload.instances))

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request):
        loaded = _require_model(request)
        return ModelInfoResponse(
            model_name=loaded.name,
            model_version=loaded.version,
            model_source=loaded.source,
            model_uri=loaded.uri,
            model_type=loaded.model_type,
            run_id=loaded.run_id,
            quality_gate=loaded.quality_gate,
            loaded_at=loaded.loaded_at,
            metrics=loaded.metrics,
            quality_thresholds=vars(QualityThresholds()),
            input_features=ALL_FEATURE_COLS,
            request_schema=PredictionRequest.model_json_schema(),
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request):
        api_metrics = request.app.state.metrics
        api_metrics.refresh_drift()
        return Response(generate_latest(api_metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
