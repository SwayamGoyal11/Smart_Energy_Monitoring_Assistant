"""
tests/test_api.py
------------------
Integration tests for the FastAPI service. A throwaway SQLite registry is
built per module, so the project's mlflow.db is never read or written.
"""

import statistics
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression

from src.api.app import Settings, create_app
from src.data.load_data import load_raw_data
from src.features.feature_engineering import ALL_FEATURE_COLS, build_feature_matrix, chronological_split, get_X_y
from src.models.registry import CHAMPION_ALIAS, STAGING_ALIAS, register_candidate
from src.serving.schemas import MAX_BATCH_SIZE, PredictionRequest
from src.validation.input_validation import validate_feature_frame

MODEL_NAME = "energy-forecaster-api-test"
EXPERIMENT = "api-test"

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


@pytest.fixture(scope="module")
def trained():
    train, val, test = chronological_split(build_feature_matrix(load_raw_data()))
    X, y = get_X_y(train)
    return LinearRegression().fit(validate_feature_frame(X), y), val, test


@pytest.fixture(scope="module")
def registry(tmp_path_factory, trained):
    """Temporary registry holding one gated LinearRegression (champion + staging)."""
    model, val, test = trained
    root = tmp_path_factory.mktemp("registry")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    MlflowClient().create_experiment(EXPERIMENT, artifact_location=(root / "artifacts").as_uri())
    result = register_candidate(
        model, "linear_regression", val, test,
        registered_model_name=MODEL_NAME, experiment_name=EXPERIMENT,
    )
    assert result.gates_passed
    yield uri, result
    mlflow.set_tracking_uri(previous)


@pytest.fixture(scope="module")
def client(registry, tmp_path_factory):
    uri, _ = registry
    settings = Settings(
        tracking_uri=uri, model_name=MODEL_NAME,
        local_model_path=str(tmp_path_factory.mktemp("no_local_model")), cors_origins=["*"],
    )
    with TestClient(create_app(settings)) as c:
        yield c


# Health

class TestHealth:
    def test_ok_with_champion(self, client, registry):
        _, result = registry
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["model_loaded"] is True
        assert body["model_name"] == MODEL_NAME
        assert body["model_version"] == result.version
        assert body["model_source"] == f"registry:{CHAMPION_ALIAS}"
        assert body["champion_alias_verified"] is True
        assert body["uptime_seconds"] >= 0

    def test_detects_champion_moved(self, client, registry, trained):
        uri, result = registry
        model, val, test = trained
        store = MlflowClient(tracking_uri=uri)
        newer = register_candidate(
            model, "linear_regression", val, test,
            registered_model_name=MODEL_NAME, experiment_name=EXPERIMENT,
        )
        store.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, newer.version)
        try:
            assert client.get("/health").json()["champion_alias_verified"] is False
        finally:
            store.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, result.version)
            store.set_registered_model_alias(MODEL_NAME, STAGING_ALIAS, result.version)


# Single prediction

class TestPredict:
    def test_valid_request(self, client, registry):
        _, result = registry
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"target_timestamp", "predicted_appliances_wh", "model_name", "model_version"}
        assert body["target_timestamp"] == "2016-03-15T14:00:00"
        assert body["model_name"] == MODEL_NAME and body["model_version"] == result.version
        assert body["predicted_appliances_wh"] >= 0

    def test_matches_direct_model_prediction(self, client, trained):
        model, _, _ = trained
        X = validate_feature_frame(PredictionRequest(**VALID_PAYLOAD).to_feature_frame())
        expected = max(float(model.predict(X)[0]), 0.0)
        got = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_appliances_wh"]
        assert got == pytest.approx(expected, abs=1e-3)

    def test_responds_to_consumption_spike(self, client):
        idle = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_appliances_wh"]
        busy = client.post("/predict", json=payload(
            recent_appliances_wh=[50.0, 60.0, 50.0, 400.0, 500.0, 600.0],
        )).json()["predicted_appliances_wh"]
        assert busy > idle

    def test_process_time_header(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert float(r.headers["X-Process-Time-Ms"]) > 0


# Batch prediction

class TestBatchPredict:
    def test_batch_matches_single(self, client):
        instances = [
            VALID_PAYLOAD,
            payload(target_timestamp="2016-03-19T23:50:00", T_out=1.0),
            payload(recent_appliances_wh=[100.0, 200.0, 300.0, 250.0, 200.0, 150.0]),
        ]
        r = client.post("/predict/batch", json={"instances": instances})
        assert r.status_code == 200
        batch = [p["predicted_appliances_wh"] for p in r.json()["predictions"]]
        single = [client.post("/predict", json=i).json()["predicted_appliances_wh"] for i in instances]
        assert batch == pytest.approx(single, abs=1e-6)

    def test_preserves_order(self, client):
        stamps = ["2016-03-15T14:00:00", "2016-03-15T14:10:00", "2016-03-15T14:20:00"]
        r = client.post("/predict/batch", json={"instances": [payload(target_timestamp=s) for s in stamps]})
        assert [p["target_timestamp"] for p in r.json()["predictions"]] == stamps

    def test_empty_batch_rejected(self, client):
        assert client.post("/predict/batch", json={"instances": []}).status_code == 422

    def test_oversized_batch_rejected(self, client):
        r = client.post("/predict/batch", json={"instances": [VALID_PAYLOAD] * (MAX_BATCH_SIZE + 1)})
        assert r.status_code == 422

    def test_one_bad_instance_rejects_batch(self, client):
        r = client.post("/predict/batch", json={"instances": [VALID_PAYLOAD, payload(RH_1=150.0)]})
        assert r.status_code == 422
        assert any("RH_1" in str(e["loc"]) for e in r.json()["detail"])


# Malformed and out-of-bounds input

class TestInvalidInput:
    def test_invalid_json(self, client):
        r = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_empty_body(self, client):
        assert client.post("/predict", json={}).status_code == 422

    def test_missing_field(self, client):
        body = payload()
        del body["T_out"]
        r = client.post("/predict", json=body)
        assert r.status_code == 422
        assert any(e["loc"][-1] == "T_out" for e in r.json()["detail"])

    def test_unknown_field(self, client):
        assert client.post("/predict", json=payload(Appliances=55.0)).status_code == 422

    def test_wrong_type(self, client):
        assert client.post("/predict", json=payload(T2="warm")).status_code == 422

    def test_bad_timestamp(self, client):
        assert client.post("/predict", json=payload(target_timestamp="yesterday")).status_code == 422

    def test_short_history(self, client):
        assert client.post("/predict", json=payload(recent_appliances_wh=[50.0] * 5)).status_code == 422

    @pytest.mark.parametrize("field, value", [
        ("RH_1", 101.0),
        ("T_out", 300.0),           # Kelvin instead of Celsius
        ("Press_mm_hg", 1013.0),    # hPa instead of mm Hg
        ("Windspeed", -1.0),
        ("lights", 5000.0),
        ("recent_appliances_wh", [50.0, 60.0, 50.0, 50.0, 60.0, -5.0]),
        ("recent_appliances_wh", [50.0, 60.0, 50.0, 50.0, 60.0, 1e6]),
    ])
    def test_out_of_bounds_returns_422(self, client, field, value):
        r = client.post("/predict", json=payload(**{field: value}))
        assert r.status_code == 422
        assert any(e["loc"][-1] == field or field in str(e["loc"]) for e in r.json()["detail"])

    def test_wrong_method(self, client):
        assert client.get("/predict").status_code == 405


# Model info

class TestModelInfo:
    def test_metadata(self, client, registry):
        _, result = registry
        body = client.get("/model-info").json()
        assert body["model_name"] == MODEL_NAME and body["model_version"] == result.version
        assert body["model_source"] == f"registry:{CHAMPION_ALIAS}"
        assert body["model_uri"] == f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
        assert body["model_type"] == "linear_regression"
        assert body["quality_gate"] == "passed"
        assert body["run_id"] == result.run_id

    def test_quality_metrics_and_thresholds(self, client):
        body = client.get("/model-info").json()
        for key in ("val_mae", "val_rmse", "val_r2", "test_rmse", "baseline_val_rmse"):
            assert key in body["metrics"]
        assert body["metrics"]["val_rmse"] <= body["quality_thresholds"]["max_rmse"]
        assert set(body["quality_thresholds"]) == {"max_mae", "max_rmse", "min_r2", "max_rmse_vs_baseline"}

    def test_feature_schema(self, client):
        body = client.get("/model-info").json()
        assert body["input_features"] == ALL_FEATURE_COLS
        assert "recent_appliances_wh" in body["request_schema"]["properties"]


# Latency

class TestLatency:
    def test_single_prediction_latency(self, client):
        client.post("/predict", json=VALID_PAYLOAD)  # warm-up
        timings = []
        for _ in range(30):
            start = time.perf_counter()
            assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200
            timings.append(time.perf_counter() - start)
        p95 = statistics.quantiles(timings, n=20)[-1]
        assert statistics.median(timings) < 0.10, f"median {statistics.median(timings):.3f}s"
        assert p95 < 0.25, f"p95 {p95:.3f}s"

    def test_max_batch_latency(self, client):
        start = time.perf_counter()
        r = client.post("/predict/batch", json={"instances": [VALID_PAYLOAD] * MAX_BATCH_SIZE})
        elapsed = time.perf_counter() - start
        assert r.status_code == 200 and len(r.json()["predictions"]) == MAX_BATCH_SIZE
        assert elapsed < 3.0, f"{MAX_BATCH_SIZE}-instance batch took {elapsed:.3f}s"


# CORS

class TestCors:
    def test_preflight_allowed(self, client):
        r = client.options("/predict", headers={
            "Origin": "http://dashboard.local",
            "Access-Control-Request-Method": "POST",
        })
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] in ("*", "http://dashboard.local")


# Model loading fallbacks and lifecycle

class TestModelLoading:
    def test_falls_back_to_staging(self, registry, tmp_path):
        uri, result = registry
        store = MlflowClient(tracking_uri=uri)
        store.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
        try:
            settings = Settings(tracking_uri=uri, model_name=MODEL_NAME, local_model_path=str(tmp_path))
            with TestClient(create_app(settings)) as c:
                body = c.get("/health").json()
                assert body["model_source"] == f"registry:{STAGING_ALIAS}"
                assert body["champion_alias_verified"] is False
                assert c.post("/predict", json=VALID_PAYLOAD).status_code == 200
        finally:
            store.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, result.version)

    def test_falls_back_to_local_model(self, trained, tmp_path):
        model, _, _ = trained
        local = tmp_path / "champion"
        mlflow.sklearn.save_model(model, str(local))
        missing_db = f"sqlite:///{(tmp_path / 'absent.db').as_posix()}"
        settings = Settings(tracking_uri=missing_db, model_name=MODEL_NAME, local_model_path=str(local))
        with TestClient(create_app(settings)) as c:
            body = c.get("/health").json()
            assert body["model_source"] == "local" and body["model_version"] == "local"
            assert body["champion_alias_verified"] is None
            assert c.post("/predict", json=VALID_PAYLOAD).status_code == 200
        assert not (tmp_path / "absent.db").exists(), "loader must not create an empty registry"

    def test_no_model_returns_503(self, tmp_path):
        settings = Settings(
            tracking_uri=f"sqlite:///{(tmp_path / 'absent.db').as_posix()}",
            model_name=MODEL_NAME, local_model_path=str(tmp_path),
        )
        with TestClient(create_app(settings)) as c:
            health = c.get("/health")
            assert health.status_code == 503 and health.json()["status"] == "unavailable"
            assert c.post("/predict", json=VALID_PAYLOAD).status_code == 503
            assert c.post("/predict/batch", json={"instances": [VALID_PAYLOAD]}).status_code == 503
            assert c.get("/model-info").status_code == 503

    def test_shutdown_releases_model(self, registry, tmp_path):
        uri, _ = registry
        app = create_app(Settings(tracking_uri=uri, model_name=MODEL_NAME, local_model_path=str(tmp_path)))
        with TestClient(app):
            assert app.state.loaded is not None
        assert app.state.loaded is None
