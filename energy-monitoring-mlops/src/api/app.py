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
from fastapi.responses import JSONResponse

from src.api.model_loader import LoadedModel, champion_matches, load_serving_model
from src.api.schemas import (
    BatchPredictionRequest, BatchPredictionResponse, HealthResponse, ModelInfoResponse,
    PredictionRequest, PredictionResponse,
)
from src.features.feature_engineering import ALL_FEATURE_COLS
from src.serving.schemas import build_feature_frame
from src.models.quality_gates import QualityThresholds
from src.models.registry import REGISTERED_MODEL, tracking_uri
from src.models.train import PROJECT_ROOT
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


def _require_model(request: Request) -> LoadedModel:
    loaded = request.app.state.loaded
    if loaded is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return loaded


def _predict(loaded: LoadedModel, requests: List[PredictionRequest]) -> List[PredictionResponse]:
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
        yield
        logger.info("Shutting down; releasing model")
        app.state.loaded = None

    app = FastAPI(
        title="Smart Energy Monitoring API",
        description="10-minute appliance energy forecasts from the MLflow champion model.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.loaded = None
    app.state.started_at = time.monotonic()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
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
        return _predict(_require_model(request), [payload])[0]

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(payload: BatchPredictionRequest, request: Request):
        return BatchPredictionResponse(predictions=_predict(_require_model(request), payload.instances))

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

    return app


app = create_app()
