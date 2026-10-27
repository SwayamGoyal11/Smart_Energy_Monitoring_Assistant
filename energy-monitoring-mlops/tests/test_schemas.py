"""
tests/test_schemas.py
----------------------
Tests for the inference request/response schemas, including training/serving
parity: features rebuilt from a request must equal the training feature matrix.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.load_data import load_raw_data
from src.features.feature_engineering import (
    ALL_FEATURE_COLS, ENVIRONMENTAL_FEATURES, build_feature_matrix,
)
from src.serving.schemas import (
    HISTORY_LENGTH, PredictionRequest, PredictionResponse, build_feature_frame,
)
from src.validation.input_validation import validate_feature_frame

VALID_PAYLOAD = {
    "target_timestamp": "2016-03-15T14:00:00",
    "recent_appliances_wh": [50.0, 60.0, 50.0, 50.0, 60.0, 70.0],
    "lights": 0.0,
    "T1": 21.5, "RH_1": 40.0, "T2": 20.1, "RH_2": 39.0, "T3": 22.0, "RH_3": 38.5,
    "T6": 7.0, "RH_6": 55.0, "T_out": 7.5, "Press_mm_hg": 755.0, "RH_out": 80.0,
    "Windspeed": 4.0, "Visibility": 40.0, "Tdewpoint": 3.5,
}


def payload(**overrides):
    return {**VALID_PAYLOAD, **overrides}


class TestPredictionRequest:
    def test_valid_payload_parses(self):
        req = PredictionRequest(**VALID_PAYLOAD)
        assert req.target_timestamp == datetime(2016, 3, 15, 14, 0)

    def test_missing_sensor_rejected(self):
        body = payload()
        del body["T_out"]
        with pytest.raises(ValidationError, match="T_out"):
            PredictionRequest(**body)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            PredictionRequest(**payload(Appliances=55.0))

    def test_short_history_rejected(self):
        with pytest.raises(ValidationError, match="recent_appliances_wh"):
            PredictionRequest(**payload(recent_appliances_wh=[50.0] * (HISTORY_LENGTH - 1)))

    @pytest.mark.parametrize("bad", [-10.0, 1e6, float("nan"), float("inf")])
    def test_bad_history_value_rejected(self, bad):
        history = [50.0] * (HISTORY_LENGTH - 1) + [bad]
        with pytest.raises(ValidationError, match="recent_appliances_wh"):
            PredictionRequest(**payload(recent_appliances_wh=history))

    @pytest.mark.parametrize("field, value", [
        ("RH_1", 101.0), ("T1", -100.0), ("Press_mm_hg", 1013.0),
        ("lights", -1.0), ("Windspeed", float("nan")),
    ])
    def test_out_of_range_sensor_rejected(self, field, value):
        with pytest.raises(ValidationError, match=field):
            PredictionRequest(**payload(**{field: value}))

    def test_non_numeric_sensor_rejected(self):
        with pytest.raises(ValidationError, match="T2"):
            PredictionRequest(**payload(T2="warm"))

    def test_bad_timestamp_rejected(self):
        with pytest.raises(ValidationError, match="target_timestamp"):
            PredictionRequest(**payload(target_timestamp="not-a-date"))


class TestFeatureFrame:
    def test_shape_and_column_order(self):
        X = PredictionRequest(**VALID_PAYLOAD).to_feature_frame()
        assert X.shape == (1, len(ALL_FEATURE_COLS))
        assert list(X.columns) == ALL_FEATURE_COLS

    def test_passes_input_validation(self):
        validate_feature_frame(PredictionRequest(**VALID_PAYLOAD).to_feature_frame())

    def test_lags_and_rolling_from_history(self):
        X = PredictionRequest(**VALID_PAYLOAD).to_feature_frame().iloc[0]
        history = VALID_PAYLOAD["recent_appliances_wh"]
        assert X["Appliances_lag_1"] == history[-1]
        assert X["Appliances_lag_2"] == history[-2]
        assert X["Appliances_lag_3"] == history[-3]
        assert X["Appliances_rolling_mean_3"] == pytest.approx(np.mean(history[-3:]))
        assert X["Appliances_rolling_mean_6"] == pytest.approx(np.mean(history))
        assert X["Appliances_rolling_std_6"] == pytest.approx(np.std(history, ddof=1))

    def test_only_most_recent_history_used(self):
        long_history = [999.0] * 10 + VALID_PAYLOAD["recent_appliances_wh"]
        a = PredictionRequest(**VALID_PAYLOAD).to_feature_frame()
        b = PredictionRequest(**payload(recent_appliances_wh=long_history)).to_feature_frame()
        pd.testing.assert_frame_equal(a, b)

    def test_batch_frame_matches_individual_frames(self):
        """Stacked blocks must not leak lag/rolling values across requests."""
        requests = [
            PredictionRequest(**VALID_PAYLOAD),
            PredictionRequest(**payload(
                target_timestamp="2016-03-19T23:50:00", T_out=-2.0,
                recent_appliances_wh=[900.0, 10.0, 700.0, 20.0, 500.0, 30.0],
            )),
            PredictionRequest(**payload(recent_appliances_wh=[999.0] * 10 + [40.0] * 6)),
        ]
        batch = build_feature_frame(requests)
        individual = pd.concat([r.to_feature_frame() for r in requests], ignore_index=True)
        pd.testing.assert_frame_equal(batch, individual)
        assert batch.loc[2, "Appliances_rolling_mean_6"] == 40.0

    def test_time_features_from_target_timestamp(self):
        X = PredictionRequest(**payload(target_timestamp="2016-03-19T23:50:00")).to_feature_frame().iloc[0]
        assert X["hour"] == 23
        assert X["day_of_week"] == 5
        assert X["is_weekend"] == 1


@pytest.fixture(scope="module")
def raw_and_features():
    raw = load_raw_data()
    return raw, build_feature_matrix(raw).set_index("date")


class TestTrainingServingParity:
    """A request built from raw rows must reproduce the training feature row exactly."""

    @pytest.mark.parametrize("row_idx", [6, 500, 10_000, 19_000])
    def test_request_matches_training_features(self, raw_and_features, row_idx):
        raw, features = raw_and_features
        row = raw.iloc[row_idx]
        req = PredictionRequest(
            target_timestamp=row["date"].to_pydatetime(),
            recent_appliances_wh=raw["Appliances"].iloc[row_idx - HISTORY_LENGTH:row_idx].astype(float).tolist(),
            **{col: float(row[col]) for col in ENVIRONMENTAL_FEATURES},
        )
        served = req.to_feature_frame().iloc[0]
        trained = features.loc[row["date"], ALL_FEATURE_COLS]
        np.testing.assert_allclose(
            served.to_numpy(dtype=float), trained.to_numpy(dtype=float), rtol=1e-9, atol=1e-9,
        )


class TestPredictionResponse:
    def test_valid_response(self):
        resp = PredictionResponse(
            target_timestamp="2016-03-15T14:00:00", predicted_appliances_wh=55.2,
            model_name="energy-forecaster", model_version="1",
        )
        assert resp.model_dump()["predicted_appliances_wh"] == 55.2

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_invalid_prediction_rejected(self, bad):
        with pytest.raises(ValidationError):
            PredictionResponse(
                target_timestamp="2016-03-15T14:00:00", predicted_appliances_wh=bad,
                model_name="energy-forecaster", model_version="1",
            )

    def test_empty_model_name_rejected(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                target_timestamp="2016-03-15T14:00:00", predicted_appliances_wh=55.2,
                model_name="", model_version="1",
            )
