"""
tests/test_drift_experiment.py
-------------------------------
Tests for scripts/run_drift_experiment.py: the scenario transform, the
retraining decision rule, an end-to-end run with a fast stand-in model, and
the integrity of the committed evidence files.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from scripts.run_drift_experiment import (
    HUMIDITY_COLS, TEMPERATURE_COLS, DriftScenario, RetrainingPolicy, apply_scenario, build_windows,
    decide_retraining, run_experiment, save_evidence,
)
from src.data.load_data import load_raw_data
from src.features.feature_engineering import TARGET_COL, get_X_y
from src.monitoring.drift import DriftConfig, load_reference
from src.monitoring.metrics import ApiMetrics
from src.validation.input_validation import validate_feature_frame

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVIDENCE_DIR = os.path.join(ROOT, "reports", "evidence")


@pytest.fixture(scope="module")
def raw():
    return load_raw_data()


@pytest.fixture(scope="module")
def reference():
    return load_reference()


@pytest.fixture(scope="module")
def stand_in_model(reference):
    X, y = get_X_y(reference)
    return LinearRegression().fit(validate_feature_frame(X), y)


@pytest.fixture(scope="module")
def experiment(stand_in_model, raw, reference):
    return run_experiment(stand_in_model, raw=raw, reference=reference,
                          model_info={"name": "stand-in", "version": "test", "source": "pytest"})


class TestScenario:
    def test_sensor_shifts(self, raw):
        out = apply_scenario(raw)
        for col in TEMPERATURE_COLS:
            np.testing.assert_allclose(out[col], raw[col] + 8.0)
        for col in HUMIDITY_COLS:
            np.testing.assert_allclose(out[col], raw[col] * 0.82)

    def test_only_evening_load_changes(self, raw):
        out = apply_scenario(raw)
        hours = raw["date"].dt.hour
        evening = (hours >= 17) & (hours < 23)
        np.testing.assert_allclose(out.loc[evening, TARGET_COL], raw.loc[evening, TARGET_COL] * 1.5 + 50.0)
        np.testing.assert_allclose(out.loc[~evening, TARGET_COL], raw.loc[~evening, TARGET_COL])

    def test_untouched_columns_and_rows(self, raw):
        out = apply_scenario(raw)
        assert len(out) == len(raw) and out["date"].equals(raw["date"])
        for col in ("T4", "RH_4", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint", "lights"):
            pd.testing.assert_series_equal(out[col], raw[col])
        assert raw["T1"].iloc[0] == load_raw_data()["T1"].iloc[0]  # input not mutated

    def test_shifted_values_stay_in_serving_bounds(self, raw):
        out = apply_scenario(raw, DriftScenario(temp_shift_c=100.0, evening_load_scale=50.0))
        assert out["T1"].max() == 60.0
        assert out[TARGET_COL].max() <= 5000.0

    def test_invalid_scenarios_rejected(self):
        with pytest.raises(ValueError):
            DriftScenario(rh_scale=0.0)
        with pytest.raises(ValueError):
            DriftScenario(evening_start_hour=22, evening_end_hour=18)

    def test_windows_share_timestamps(self, raw):
        baseline, drifted = build_windows(raw, DriftScenario())
        assert len(baseline) == len(drifted) > 2000
        assert baseline["date"].equals(drifted["date"])
        assert (drifted["T1"] - baseline["T1"]).round(6).eq(8.0).all()
        # Lags are rebuilt from the shifted series, so they carry the evening load shift.
        assert drifted["Appliances_lag_1"].mean() > baseline["Appliances_lag_1"].mean()


class TestRetrainingDecision:
    BASE = {"rmse": 60.0, "mae": 25.0, "r2": 0.55}

    @staticmethod
    def drift(detected):
        return {"dataset_drift_detected": detected, "drifted_feature_share": 0.9 if detected else 0.05,
                "drifted_count": 19 if detected else 1, "monitored_count": 21}

    def decide(self, metrics, detected):
        return decide_retraining(self.BASE, metrics, self.drift(detected), DriftConfig(), RetrainingPolicy())

    def test_drift_without_degradation_does_not_retrain(self):
        decision = self.decide(self.BASE, detected=True)
        assert decision["retraining_triggered"] is False
        assert decision["threshold_breaches"]["dataset_drift"]["breached"] is True
        assert decision["trigger_reasons"] == []

    def test_degradation_without_drift_does_not_retrain(self):
        decision = self.decide({"rmse": 80.0, "mae": 40.0, "r2": 0.3}, detected=False)
        assert decision["retraining_triggered"] is False
        assert decision["threshold_breaches"]["quality_gate"]["breached"] is True

    def test_drift_and_rmse_degradation_retrains(self):
        decision = self.decide({"rmse": 64.0, "mae": 26.0, "r2": 0.52}, detected=False)
        assert decision["threshold_breaches"]["rmse_degradation"]["breached"] is False  # +6.7% < 10%
        decision = self.decide({"rmse": 66.5, "mae": 26.0, "r2": 0.52}, detected=True)
        assert decision["retraining_triggered"] is True
        assert decision["threshold_breaches"]["rmse_degradation"]["breached"] is True
        assert any("RMSE up" in r for r in decision["trigger_reasons"])


class TestExperimentRun:
    def test_shifted_sensors_drift_harder_than_baseline(self, experiment):
        evidence, result, _ = experiment
        for col in TEMPERATURE_COLS + HUMIDITY_COLS:
            assert result["features"][col]["drifted"], col
        assert evidence["drifted"]["drift"]["mean_feature_score"] > evidence["baseline"]["drift"]["mean_feature_score"]
        assert evidence["drifted"]["drift"]["dataset_drift_detected"]

    def test_performance_degrades_and_triggers_retraining(self, experiment):
        evidence, _, _ = experiment
        assert evidence["drifted"]["metrics"]["rmse"] > evidence["baseline"]["metrics"]["rmse"]
        assert evidence["drifted"]["metrics"]["mae"] > evidence["baseline"]["metrics"]["mae"]
        assert evidence["retraining_triggered"] is True
        assert evidence["trigger_reasons"]
        assert evidence["baseline_decision"]["retraining_triggered"] is False

    def test_artifacts_written_and_readable_by_api(self, experiment, tmp_path):
        evidence, result, snapshot = experiment
        html, metrics, ev = tmp_path / "r.html", tmp_path / "m.json", tmp_path / "e.json"
        save_evidence(dict(evidence), dict(result), snapshot, str(html), str(metrics), str(ev))
        assert html.stat().st_size > 10_000
        saved = json.loads(ev.read_text(encoding="utf-8"))
        assert saved["retraining_triggered"] is True
        assert set(saved) >= {"reference", "baseline", "drifted", "degradation_vs_baseline",
                              "threshold_breaches", "artifacts", "experiment"}

        api = ApiMetrics(str(metrics))
        api.refresh_drift()
        assert api.drift_detected._value.get() == 1
        assert api.drift_share._value.get() == pytest.approx(result["dataset_drift"]["share"])


class TestCommittedEvidence:
    @pytest.fixture
    def evidence(self):
        path = os.path.join(EVIDENCE_DIR, "retraining_trigger_evidence.json")
        if not os.path.isfile(path):
            pytest.skip("run scripts/run_drift_experiment.py to generate evidence")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_evidence_records_triggered_retraining(self, evidence):
        assert evidence["retraining_triggered"] is True
        assert evidence["drifted"]["metrics"]["rmse"] > evidence["baseline"]["metrics"]["rmse"]
        breaches = evidence["threshold_breaches"]
        assert breaches["dataset_drift"]["breached"]
        assert breaches["dataset_drift"]["observed_share"] >= breaches["dataset_drift"]["threshold_share"]

    def test_drift_metrics_match_evidence(self, evidence):
        # The experiment's own copy; the live drift_metrics.json holds the clean baseline.
        artifacts = evidence["artifacts"]
        assert artifacts["drift_metrics_json"] != "reports/evidence/drift_metrics.json"
        assert os.path.isfile(os.path.join(ROOT, artifacts["drift_report_html"]))
        with open(os.path.join(ROOT, artifacts["drift_metrics_json"]), encoding="utf-8") as f:
            metrics = json.load(f)
        assert metrics["dataset_drift"]["share"] == evidence["drifted"]["drift"]["drifted_feature_share"]
        assert metrics["current"]["rows"] == evidence["drifted"]["rows"]
        assert metrics["experiment"]["name"] == evidence["experiment"]["name"]

    def test_live_drift_metrics_are_clean_baseline(self):
        path = os.path.join(EVIDENCE_DIR, "drift_metrics.json")
        if not os.path.isfile(path):
            pytest.skip("no live drift_metrics.json")
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
        assert "experiment" not in metrics
        assert metrics["dataset_drift"]["detected"] is False
        assert metrics["dataset_drift"]["share"] < metrics["config"]["drift_share"]
        assert metrics["alerts"] == []
