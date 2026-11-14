"""
tests/test_retraining_flow.py
------------------------------
Tests for scripts/verify_retraining_flow.py: the independent statement of the
promotion policy, the registry audit, champion-bundle verification (including
tamper detection), the sandbox replay, and report rendering.
"""

import json
import os
import shutil

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from scripts.export_champion import export
from scripts.verify_retraining_flow import (
    TRIGGER_EVIDENCE, Section, audit_registry, check_trigger, expected_outcome, render_report,
    replay_promotion_policy, verify_bundle,
)
from src.data.load_data import load_raw_data
from src.features.feature_engineering import build_feature_matrix, chronological_split, get_X_y
from src.models.registry import CHAMPION_ALIAS, STAGING_ALIAS, register_candidate
from src.models.train import EXPERIMENT_NAME
from src.validation.input_validation import validate_feature_frame

MODEL_NAME = "energy-forecaster-flow-test"


def metrics(val_rmse, val=(25.0, 0.55), test=(26.0, 61.0, 0.55), baseline_rmse=66.0):
    m = {"val_mae": val[0], "val_rmse": val_rmse, "val_r2": val[1],
         "test_mae": test[0], "test_rmse": test[1], "test_r2": test[2]}
    for split in ("val", "test"):
        m.update({f"baseline_{split}_mae": 26.0, f"baseline_{split}_rmse": baseline_rmse,
                  f"baseline_{split}_r2": 0.46})
    return m


class TestExpectedOutcome:
    def test_first_passing_candidate_becomes_champion(self):
        assert expected_outcome(metrics(60.0), None)["aliases"] == [STAGING_ALIAS, CHAMPION_ALIAS]

    def test_must_strictly_beat_incumbent(self):
        assert expected_outcome(metrics(59.0), 60.0)["aliases"] == [STAGING_ALIAS, CHAMPION_ALIAS]
        assert expected_outcome(metrics(60.0), 60.0)["aliases"] == [STAGING_ALIAS]  # tie
        assert expected_outcome(metrics(61.0), 60.0)["aliases"] == [STAGING_ALIAS]

    def test_any_gate_failure_blocks_every_alias(self):
        out = expected_outcome(metrics(55.0, test=(26.0, 70.0, 0.55)), 60.0)  # test RMSE > 65
        assert out["passed"] is False and out["aliases"] == []
        assert "test: RMSE 70.0000 > 65.0" in out["failures"]
        assert all(f.startswith("test:") for f in out["failures"])  # validation was fine
        # Better val RMSE than the incumbent is irrelevant if the baseline is not beaten.
        assert not expected_outcome(metrics(55.0, baseline_rmse=54.0), None)["passed"]


class TestTrigger:
    def test_committed_evidence_triggers_retraining(self):
        if not os.path.isfile(TRIGGER_EVIDENCE):
            pytest.skip("run scripts/run_drift_experiment.py first")
        section = check_trigger()
        assert section.passed, [c for c in section.checks if not c.passed]

    def test_untriggered_evidence_fails(self, tmp_path):
        with open(TRIGGER_EVIDENCE, encoding="utf-8") as f:
            ev = json.load(f)
        ev["retraining_triggered"] = False
        path = tmp_path / "ev.json"
        path.write_text(json.dumps(ev), encoding="utf-8")
        section = check_trigger(str(path))
        assert not section.passed
        assert [c.name for c in section.checks if not c.passed] == ["Retraining triggered"]

    def test_missing_evidence_fails(self, tmp_path):
        assert not check_trigger(str(tmp_path / "absent.json")).passed


@pytest.fixture(scope="module")
def splits():
    return chronological_split(build_feature_matrix(load_raw_data()))


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory, splits):
    """Temporary registry: v1 linear regression (@champion), v2 mean predictor (gates fail)."""
    root = tmp_path_factory.mktemp("flow-registry")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    MlflowClient().create_experiment(EXPERIMENT_NAME, artifact_location=(root / "artifacts").as_uri())
    train, val, test = splits
    X, y = get_X_y(train)
    X = validate_feature_frame(X)
    for model, kind in ((LinearRegression(), "linear_regression"), (DummyRegressor(), "mean_predictor")):
        register_candidate(model.fit(X, y), kind, val, test, registered_model_name=MODEL_NAME,
                           log_artifacts=False)
    mlflow.set_tracking_uri(previous)
    return uri, root


@pytest.fixture
def bundle(sandbox, tmp_path, monkeypatch):
    uri, _ = sandbox
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    path = tmp_path / "champion"
    previous = mlflow.get_tracking_uri()
    try:
        export(CHAMPION_ALIAS, str(path), model_name=MODEL_NAME)
    finally:
        mlflow.set_tracking_uri(previous)
    return str(path)


class TestRegistryAudit:
    def test_clean_registry_passes(self, sandbox):
        uri, _ = sandbox
        section = audit_registry(uri, MODEL_NAME)
        assert section.passed, [c for c in section.checks if not c.passed]
        assert section.data["aliases"] == {CHAMPION_ALIAS: 1, STAGING_ALIAS: 1}

    def test_alias_on_failed_version_is_flagged(self, sandbox):
        uri, _ = sandbox
        previous = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(uri)
        client = MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, STAGING_ALIAS, "2")
        mlflow.set_tracking_uri(previous)
        try:
            failing = [c.name for c in audit_registry(uri, MODEL_NAME).checks if not c.passed]
            assert "No failed candidate holds an alias" in failing
            assert "Every alias holder passed the quality gates" in failing
        finally:
            mlflow.set_tracking_uri(uri)
            MlflowClient().set_registered_model_alias(MODEL_NAME, STAGING_ALIAS, "1")
            mlflow.set_tracking_uri(previous)

    def test_unreachable_registry_fails(self, tmp_path):
        assert not audit_registry(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}", MODEL_NAME).passed


class TestBundleVerification:
    def test_fresh_export_passes(self, sandbox, bundle, splits):
        section = verify_bundle(sandbox[0], bundle, MODEL_NAME, splits=splits)
        assert section.passed, [c for c in section.checks if not c.passed]

    def test_stale_metadata_detected(self, sandbox, bundle, splits):
        path = os.path.join(bundle, "serving_metadata.json")
        with open(path) as f:
            meta = json.load(f)
        meta["model_version"] = "99"
        meta["metrics"]["test_rmse"] = 1.0
        with open(path, "w") as f:
            json.dump(meta, f)
        failing = [c.name for c in verify_bundle(sandbox[0], bundle, MODEL_NAME, splits=splits).checks
                   if not c.passed]
        assert "Metadata version/run match @champion" in failing
        assert "Metadata metrics match the registry run" in failing

    def test_swapped_model_file_detected(self, sandbox, bundle, splits, tmp_path):
        # Replace the bundle's model with another registry version's artifacts.
        other = tmp_path / "other"
        previous = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(sandbox[0])
        try:
            mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/2", dst_path=str(other))
        finally:
            mlflow.set_tracking_uri(previous)
        for name in os.listdir(other):
            if name.startswith("model."):
                shutil.copy(other / name, os.path.join(bundle, name))
        failing = [c.name for c in verify_bundle(sandbox[0], bundle, MODEL_NAME, splits=splits).checks
                   if not c.passed]
        assert "Bundle files byte-identical to registry artifacts" in failing


class TestSandboxReplay:
    def test_policy_replay_passes_and_leaves_tracking_uri(self, splits):
        before = mlflow.get_tracking_uri()
        section = replay_promotion_policy(splits)
        assert mlflow.get_tracking_uri() == before
        assert section.passed, [c for c in section.checks if not c.passed]
        steps = section.data["steps"]
        assert [CHAMPION_ALIAS in s["aliases"] for s in steps] == [True, False, False, False]
        assert [s["gates_passed"] for s in steps] == [True, True, True, False]
        assert section.data["champion_version"] == 1


class TestReport:
    def test_report_renders_all_sections(self, sandbox, bundle, splits):
        if not os.path.isfile(TRIGGER_EVIDENCE):
            pytest.skip("run scripts/run_drift_experiment.py first")
        uri = sandbox[0]
        report = render_report(check_trigger(), audit_registry(uri, MODEL_NAME), None,
                               verify_bundle(uri, bundle, MODEL_NAME, splits=splits))
        for heading in ("## 1. Drift trigger", "## 2. Retraining policy", "## 3. Live registry audit",
                        "## 4. Champion export and redeployment"):
            assert heading in report
        assert "strictly lower" in report and "**FAIL**" not in report

    def test_failures_are_visible(self):
        failed = Section()
        failed.check("Something", False, "broken")
        report = render_report(Section(), Section(), None, failed)
        assert "| Something | **FAIL** | broken |" in report
        assert "0/1 checks passed" in report


def test_committed_report_is_current():
    path = os.path.join(os.path.dirname(TRIGGER_EVIDENCE), "retraining_lifecycle.md")
    if not os.path.isfile(path):
        pytest.skip("run scripts/verify_retraining_flow.py first")
    with open(path, encoding="utf-8") as f:
        report = f.read()
    assert "**FAIL**" not in report
    assert "checks passed" in report
