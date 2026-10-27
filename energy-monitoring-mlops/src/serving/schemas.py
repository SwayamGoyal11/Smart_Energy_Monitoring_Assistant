"""
src/serving/schemas.py
-----------------------
Pydantic request/response contract for the inference service.

Clients send raw readings (timestamp, sensors, recent consumption); the
engineered features are rebuilt server-side with the same functions used in
training, so time/lag/rolling features can never drift from the training code.
"""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features.feature_engineering import (
    ALL_FEATURE_COLS, ENVIRONMENTAL_FEATURES, TARGET_COL,
    create_lag_features, create_rolling_features, create_time_features,
)
from src.validation.input_validation import PHYSICAL_BOUNDS

# Longest look-back used by any feature (Appliances_rolling_*_6).
HISTORY_LENGTH = 6
INTERVAL = timedelta(minutes=10)
ENERGY_BOUNDS = PHYSICAL_BOUNDS["Appliances_lag_1"]


def _sensor(name: str, description: str):
    lo, hi = PHYSICAL_BOUNDS[name]
    return Field(..., ge=lo, le=hi, allow_inf_nan=False, description=description)


class PredictionRequest(BaseModel):
    """One forecast request for the 10-minute interval starting at target_timestamp."""

    model_config = ConfigDict(extra="forbid")

    target_timestamp: datetime = Field(..., description="Start of the interval being forecast.")
    recent_appliances_wh: List[float] = Field(
        ...,
        min_length=HISTORY_LENGTH,
        description=(
            f"Appliance energy (Wh) for at least the last {HISTORY_LENGTH} 10-minute "
            "intervals, oldest first; the last value is the interval just before target_timestamp."
        ),
    )
    lights: float = _sensor("lights", "Lighting energy (Wh) in the current interval.")
    T1: float = _sensor("T1", "Kitchen temperature (degC).")
    RH_1: float = _sensor("RH_1", "Kitchen humidity (%).")
    T2: float = _sensor("T2", "Living room temperature (degC).")
    RH_2: float = _sensor("RH_2", "Living room humidity (%).")
    T3: float = _sensor("T3", "Laundry room temperature (degC).")
    RH_3: float = _sensor("RH_3", "Laundry room humidity (%).")
    T6: float = _sensor("T6", "Outside (north side) temperature (degC).")
    RH_6: float = _sensor("RH_6", "Outside (north side) humidity (%).")
    T_out: float = _sensor("T_out", "Weather-station temperature (degC).")
    Press_mm_hg: float = _sensor("Press_mm_hg", "Atmospheric pressure (mm Hg).")
    RH_out: float = _sensor("RH_out", "Weather-station humidity (%).")
    Windspeed: float = _sensor("Windspeed", "Wind speed (m/s).")
    Visibility: float = _sensor("Visibility", "Visibility (km).")
    Tdewpoint: float = _sensor("Tdewpoint", "Dew point (degC).")

    @field_validator("recent_appliances_wh")
    @classmethod
    def _check_history(cls, values: List[float]) -> List[float]:
        lo, hi = ENERGY_BOUNDS
        for v in values:
            if not np.isfinite(v):
                raise ValueError("recent_appliances_wh must contain only finite values")
            if not lo <= v <= hi:
                raise ValueError(f"recent_appliances_wh values must be within [{lo}, {hi}] Wh")
        return values

    def to_feature_frame(self) -> pd.DataFrame:
        """Build the single-row model input in ALL_FEATURE_COLS order."""
        history = self.recent_appliances_wh[-HISTORY_LENGTH:]
        dates = [self.target_timestamp - INTERVAL * k for k in range(HISTORY_LENGTH, 0, -1)]
        frame = pd.DataFrame({
            "date": pd.to_datetime(dates + [self.target_timestamp]),
            TARGET_COL: history + [np.nan],
        })
        for col in ENVIRONMENTAL_FEATURES:
            frame[col] = getattr(self, col)

        frame = create_time_features(frame)
        frame = create_lag_features(frame)
        frame = create_rolling_features(frame)
        return frame[ALL_FEATURE_COLS].iloc[[-1]].reset_index(drop=True)


class PredictionResponse(BaseModel):
    """Forecast returned by the inference service."""

    model_config = ConfigDict(protected_namespaces=())

    target_timestamp: datetime
    predicted_appliances_wh: float = Field(..., ge=0.0, allow_inf_nan=False)
    model_name: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)
