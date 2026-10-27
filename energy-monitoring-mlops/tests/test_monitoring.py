"""
tests/test_monitoring.py
-------------------------
Tests for Evidently drift detection, the inference log, and the Prometheus
/metrics endpoint.
"""

import json
import os
import threading

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from src.api.app import Settings, create_app
from src.features.feature_engineering import ENVIRONMENTAL_FEATURES, get_X_y
from src.models.hurdle import HurdleModel
from src.models.registry import SKOPS_TRUSTED_TYPES
from src.monitoring.drift import (
    MONITORED_FEATURES, DriftConfig, compute_drift, load_reference, save_reports,
)
from src.monitoring.inference_log import LOG_COLUMNS, PREDICTION_COL, InferenceLogger, load_inference_log
from src.validation.input_validation import validate_feature_frame

TEMPERATURES = ["T1", "T2", "T3", "T6", "T_out", "Tdewpoint"]
HUMIDITIES = ["RH_1", "RH_2", "RH_3", "RH_6", "RH_out"]


def synthetic(n, seed, temp_shift=0.0, rh_shift=0.0, pred_scale=1.0):
    """Monitored features + prediction with plausible household distributions."""
    rng = np.random.default_rng(seed)
    energy = 10 + rng.gamma(2.0, 30.0, n)
    frame = {
        "Appliances_lag_1": energy,
        "Appliances_lag_2": 10 + rng.gamma(2.0, 30.0, n),
        "Appliances_lag_3": 10 + rng.gamma(2.0, 30.0, n),
        "Appliances_rolling_mean_3": 20 + rng.gamma(3.0, 20.0, n),
        "Appliances_rolling_mean_6": 25 + rng.gamma(4.0, 15.0, n),
        "Appliances_rolling_std_6": rng.gamma(1.5, 15.0, n),
        "Press_mm_hg": rng.normal(755, 7, n),
        "Windspeed": rng.gamma(2.0, 2.0, n),
        "Visibility": np.clip(rng.normal(40, 10, n), 1, 66),
        "lights": rng.choice([0, 10, 20, 30], n, p=[0.75, 0.15, 0.07, 0.03]).astype(float),
        PREDICTION_COL: (10 + rng.gamma(2.0, 30.0, n)) * pred_scale,
    }
    for col in TEMPERATURES:
        frame[col] = rng.normal(20, 2, n) + temp_shift
    for col in HUMIDITIES:
        frame[col] = np.clip(rng.normal(45, 5, n) + rh_shift, 0, 100)
    return pd.DataFrame(frame)


@pytest.fixture(scope="module")
def reference():
    return synthetic(3000, seed=1)


# Drift configuration

class TestDriftConfig:
    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="method"):
            DriftConfig(method="psi")

    def test_wasserstein_flags_large_distances(self):
        cfg = DriftConfig(method="wasserstein", threshold=0.1)
        assert cfg.is_drifted(0.25) and cfg.is_drifted(0.1) and not cfg.is_drifted(0.05)

    def test_ks_flags_small_p_values(self):
        cfg = DriftConfig(method="ks", threshold=0.05)
        assert cfg.is_drifted(0.001) and not cfg.is_drifted(0.3)

    def test_calendar_features_not_monitored(self):
        assert not {"hour", "day", "month", "day_of_week"} & set(MONITORED_FEATURES)
        assert set(ENVIRONMENTAL_FEATURES) <= set(MONITORED_FEATURES)


# Drift detection on synthetic data

class TestDriftDetection:
    def test_same_distribution_no_drift(self, reference):
        result, _ = compute_drift(reference, synthetic(1000, seed=2))
        assert result["dataset_drift"]["detected"] is False
        assert result["dataset_drift"]["drifted_count"] == 0
        assert result["prediction"]["drifted"] is False
        assert result["alerts"] == []

    def test_environmental_shift_triggers_alert(self, reference):
        current = synthetic(1000, seed=3, temp_shift=8.0, rh_shift=25.0)
        result, _ = compute_drift(reference, current)
        drifted = set(result["dataset_drift"]["drifted_columns"])
        assert drifted == set(TEMPERATURES + HUMIDITIES)
        assert result["dataset_drift"]["detected"] is True
        assert any(a.startswith("DATASET_DRIFT") for a in result["alerts"])
        for col in TEMPERATURES:
            assert result["features"][col]["score"] > 1.0  # ~4 reference std devs

    def test_unshifted_columns_stay_clean(self, reference):
        result, _ = compute_drift(reference, synthetic(1000, seed=3, temp_shift=8.0, rh_shift=25.0))
        for col in ("Appliances_lag_1", "Press_mm_hg", "lights"):
            assert result["features"][col]["drifted"] is False

    def test_single_feature_shift_below_dataset_share(self, reference):
        current = synthetic(1000, seed=4)
        current["T_out"] += 8.0
        result, _ = compute_drift(reference, current)
        assert result["dataset_drift"]["drifted_columns"] == ["T_out"]
        assert result["dataset_drift"]["detected"] is False

    def test_prediction_drift(self, reference):
        result, _ = compute_drift(reference, synthetic(1000, seed=5, pred_scale=3.0))
        assert result["prediction"]["drifted"] is True
        assert result["dataset_drift"]["detected"] is False
        assert any(a.startswith("PREDICTION_DRIFT") for a in result["alerts"])

    def test_target_drift_when_actuals_present(self, reference):
        ref = reference.assign(Appliances=reference["Appliances_lag_1"])
        cur = synthetic(1000, seed=6)
        cur = cur.assign(Appliances=cur["Appliances_lag_1"] * 4)
        result, _ = compute_drift(ref, cur)
        assert result["target"]["drifted"] is True
        assert any(a.startswith("TARGET_DRIFT") for a in result["alerts"])

    def test_no_prediction_section_without_predictions(self, reference):
        result, _ = compute_drift(reference.drop(columns=[PREDICTION_COL]), synthetic(500, seed=7))
        assert "prediction" not in result

    def test_ks_method(self, reference):
        cfg = DriftConfig(method="ks", threshold=0.05)
        clean, _ = compute_drift(reference, synthetic(1000, seed=8), cfg)
        shifted, _ = compute_drift(reference, synthetic(1000, seed=8, temp_shift=8.0, rh_shift=25.0), cfg)
        assert clean["dataset_drift"]["detected"] is False
        assert shifted["dataset_drift"]["detected"] is True
        assert shifted["config"]["score_type"] == "p_value"

    def test_column_subset(self, reference):
        result, _ = compute_drift(reference, synthetic(500, seed=9, temp_shift=8.0), columns=["T1", "RH_1"])
        assert set(result["features"]) == {"T1", "RH_1"}
        assert result["dataset_drift"]["share"] == 0.5

    def test_missing_columns_rejected(self, reference):
        with pytest.raises(ValueError, match="T_out"):
            compute_drift(reference, synthetic(100, seed=10).drop(columns=["T_out"]))

    def test_empty_current_rejected(self, reference):
        with pytest.raises(ValueError, match="rows"):
            compute_drift(reference, synthetic(100, seed=10).iloc[:0])

    def test_too_few_current_rows_rejected(self, reference):
        """A handful of near-identical requests would otherwise raise a false alarm."""
        with pytest.raises(ValueError, match="at least 200 rows"):
            compute_drift(reference, synthetic(8, seed=10))
        result, _ = compute_drift(reference, synthetic(8, seed=10), DriftConfig(min_current_rows=5))
        assert result["current"]["rows"] == 8


class TestReports:
    def test_html_and_json_written(self, reference, tmp_path):
        result, snapshot = compute_drift(reference, synthetic(500, seed=11, temp_shift=8.0, rh_shift=25.0))
        html, js = tmp_path / "drift_report.html", tmp_path / "drift_metrics.json"
        save_reports(result, snapshot, str(html), str(js))
        assert "<html" in html.read_text(encoding="utf-8").lower()
        loaded = json.loads(js.read_text())
        assert loaded["dataset_drift"]["detected"] is True
        assert set(loaded) >= {"generated_at", "config", "dataset_drift", "features", "prediction", "alerts"}
        assert set(loaded["features"]) == set(MONITORED_FEATURES)


class TestRealReference:
    def test_reference_is_training_split(self):
        ref = load_reference()
        assert len(ref) == 13_810
        assert ref["date"].max() < pd.Timestamp("2016-04-17")

    def test_stale_feature_cache_is_rebuilt(self, tmp_path):
        stale = load_reference().drop(columns=["hour_sin", "month_cos"])
        path = tmp_path / "features.csv"
        stale.to_csv(path, index=False)
        ref = load_reference(str(path))
        assert {"hour_sin", "month_cos"} <= set(ref.columns)
        assert len(ref) == 13_810

    def test_training_sample_does_not_drift_from_itself(self):
        ref = load_reference()
        sample = ref.sample(2000, random_state=0)
        result, _ = compute_drift(ref, sample)
        assert result["dataset_drift"]["detected"] is False
        assert result["dataset_drift"]["drifted_count"] == 0


# Inference log

class TestInferenceLog:
    def _features(self, n):
        return synthetic(n, seed=12).reindex(columns=LOG_COLUMNS[3:-1], fill_value=0.0)

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "log" / "inference.csv")
        logger = InferenceLogger(path)
        stamps = pd.date_range("2016-05-20", periods=3, freq="10min")
        logger.log(self._features(3), [50.0, 60.0, 70.0], stamps, "1")
        logger.log(self._features(2), [80.0, 90.0], stamps[:2], "1")
        frame = load_inference_log(path)
        assert list(frame.columns) == LOG_COLUMNS
        assert frame[PREDICTION_COL].tolist() == [50.0, 60.0, 70.0, 80.0, 90.0]

    def test_concurrent_writes(self, tmp_path):
        path = str(tmp_path / "inference.csv")
        logger = InferenceLogger(path)
        stamps = pd.date_range("2016-05-20", periods=5, freq="10min")
        threads = [
            threading.Thread(target=logger.log, args=(self._features(5), [1.0] * 5, stamps, "1"))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(load_inference_log(path)) == 40

    def test_missing_log_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_inference_log(str(tmp_path / "absent.csv"))


# Prometheus /metrics endpoint

IDLE = {
    "target_timestamp": "2016-05-20T03:00:00",
    "recent_appliances_wh": [50.0, 50.0, 60.0, 50.0, 50.0, 60.0],
    "lights": 0.0, "T1": 21.5, "RH_1": 40.0, "T2": 20.1, "RH_2": 39.0, "T3": 22.0, "RH_3": 38.5,
    "T6": 12.0, "RH_6": 55.0, "T_out": 12.5, "Press_mm_hg": 755.0, "RH_out": 80.0,
    "Windspeed": 4.0, "Visibility": 40.0, "Tdewpoint": 7.5,
}
BURST = {**IDLE, "target_timestamp": "2016-05-20T19:00:00", "lights": 10.0,
         "recent_appliances_wh": [60.0, 70.0, 90.0, 250.0, 420.0, 380.0]}


@pytest.fixture(scope="module")
def hurdle_model_dir(tmp_path_factory):
    train = load_reference()
    X, y = get_X_y(train)
    model = HurdleModel().fit(validate_feature_frame(X), y)
    path = tmp_path_factory.mktemp("models") / "champion"
    mlflow.sklearn.save_model(model, str(path), skops_trusted_types=SKOPS_TRUSTED_TYPES)
    return str(path)


@pytest.fixture
def monitored_app(hurdle_model_dir, tmp_path):
    settings = Settings(
        tracking_uri=f"sqlite:///{(tmp_path / 'absent.db').as_posix()}",
        local_model_path=hurdle_model_dir,
        drift_metrics_path=str(tmp_path / "drift_metrics.json"),
        inference_log_path=str(tmp_path / "inference.csv"),
    )
    with TestClient(create_app(settings)) as client:
        yield client, settings


def scrape(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    samples = {}
    for family in text_string_to_metric_families(r.text):
        for s in family.samples:
            samples[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return samples


def value(samples, name, **labels):
    return samples.get((name, tuple(sorted(labels.items()))))


class TestMetricsEndpoint:
    def test_prometheus_text_format(self, monitored_app):
        client, _ = monitored_app
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        families = {f.name: f.type for f in text_string_to_metric_families(r.text)}
        assert families["http_requests"] == "counter"
        assert families["http_request_duration_seconds"] == "histogram"
        assert families["model_prediction_value"] == "histogram"
        assert families["model_active_stage_ratio"] == "gauge"
        assert families["model_drift_score"] == "gauge"

    def test_request_counter_and_latency(self, monitored_app):
        client, _ = monitored_app
        for _ in range(3):
            assert client.post("/predict", json=IDLE).status_code == 200
        assert client.post("/predict", json={**IDLE, "RH_1": 150.0}).status_code == 422
        client.get("/does-not-exist")
        s = scrape(client)
        assert value(s, "http_requests_total", method="POST", path="/predict", status="200") == 3
        assert value(s, "http_requests_total", method="POST", path="/predict", status="422") == 1
        assert value(s, "http_requests_total", method="GET", path="unmatched", status="404") == 1
        assert value(s, "http_request_duration_seconds_count", method="POST", path="/predict") == 4

    def test_prediction_histogram(self, monitored_app):
        client, _ = monitored_app
        client.post("/predict/batch", json={"instances": [IDLE, BURST, IDLE]})
        s = scrape(client)
        assert value(s, "model_prediction_value_count", model_version="local") == 3
        assert value(s, "model_prediction_value_bucket", model_version="local", le="+Inf") == 3
        assert value(s, "model_prediction_value_sum", model_version="local") > 0

    def test_hurdle_stage_ratio(self, monitored_app):
        client, _ = monitored_app
        client.post("/predict/batch", json={"instances": [IDLE, IDLE, IDLE, BURST]})
        s = scrape(client)
        active = value(s, "model_stage_predictions_total", stage="active")
        standby = value(s, "model_stage_predictions_total", stage="standby")
        assert active + standby == 4
        assert active >= 1 and standby >= 1
        assert value(s, "model_active_stage_ratio") == pytest.approx(active / 4)

    def test_drift_gauges_follow_report_file(self, monitored_app):
        client, settings = monitored_app
        path = settings.drift_metrics_path
        assert value(scrape(client), "model_drift_score", column="T1") is None  # no report yet

        report = {
            "features": {"T1": {"score": 0.42, "drifted": True}, "RH_1": {"score": 0.03, "drifted": False}},
            "prediction": {"column": "prediction", "score": 0.07, "drifted": False},
            "dataset_drift": {"share": 0.5},
            "alerts": ["DATASET_DRIFT: 1/2 features drifted"],
        }
        with open(path, "w") as f:
            json.dump(report, f)
        s = scrape(client)
        assert value(s, "model_drift_score", column="T1") == pytest.approx(0.42)
        assert value(s, "model_drift_score", column="prediction") == pytest.approx(0.07)
        assert value(s, "model_drift_share") == pytest.approx(0.5)
        assert value(s, "model_drift_detected") == 1

        report.update(features={"T1": {"score": 0.02, "drifted": False}}, alerts=[],
                      dataset_drift={"share": 0.0})
        with open(path, "w") as f:
            json.dump(report, f)
        mtime = os.path.getmtime(path) + 5
        os.utime(path, (mtime, mtime))
        s = scrape(client)
        assert value(s, "model_drift_score", column="T1") == pytest.approx(0.02)
        assert value(s, "model_drift_score", column="RH_1") is None
        assert value(s, "model_drift_detected") == 0

    def test_corrupt_drift_file_does_not_break_scrape(self, monitored_app):
        client, settings = monitored_app
        with open(settings.drift_metrics_path, "w") as f:
            f.write("{not json")
        assert client.get("/metrics").status_code == 200

    def test_predictions_logged_for_drift(self, monitored_app):
        client, settings = monitored_app
        r = client.post("/predict/batch", json={"instances": [IDLE, BURST]})
        served = [p["predicted_appliances_wh"] for p in r.json()["predictions"]]
        log = load_inference_log(settings.inference_log_path)
        assert len(log) == 2
        assert log[PREDICTION_COL].tolist() == pytest.approx(served, abs=1e-3)
        assert log["model_version"].astype(str).tolist() == ["local", "local"]
        assert set(MONITORED_FEATURES) <= set(log.columns)

    def test_registries_isolated_between_apps(self, hurdle_model_dir, tmp_path):
        settings = Settings(tracking_uri=f"sqlite:///{(tmp_path / 'x.db').as_posix()}",
                            local_model_path=hurdle_model_dir, drift_metrics_path=None)
        with TestClient(create_app(settings)) as a, TestClient(create_app(settings)) as b:
            a.post("/predict", json=IDLE)
            assert value(scrape(b), "http_requests_total", method="POST", path="/predict", status="200") is None

    def test_metrics_hidden_from_openapi(self, monitored_app):
        client, _ = monitored_app
        assert "/metrics" not in client.get("/openapi.json").json()["paths"]
