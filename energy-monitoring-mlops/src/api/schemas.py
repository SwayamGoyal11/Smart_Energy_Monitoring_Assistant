"""
src/api/schemas.py
-------------------
API-only response models. The prediction request/response contract lives in
src/serving/schemas.py and is re-exported here for convenience.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from src.serving.schemas import (  # noqa: F401  (re-export)
    BatchPredictionRequest, BatchPredictionResponse, PredictionRequest, PredictionResponse,
)


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str                                  # "ok" or "unavailable"
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_source: Optional[str] = None           # registry:champion | registry:staging | local
    champion_alias_verified: Optional[bool] = None  # None when the registry cannot be reached
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    model_source: str
    model_uri: str
    model_type: Optional[str] = None
    run_id: Optional[str] = None
    quality_gate: Optional[str] = None
    loaded_at: datetime
    metrics: Dict[str, float]
    quality_thresholds: Dict[str, float]
    input_features: List[str]
    request_schema: Dict[str, Any]
