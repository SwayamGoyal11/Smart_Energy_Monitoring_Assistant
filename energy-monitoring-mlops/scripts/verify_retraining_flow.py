"""
scripts/verify_retraining_flow.py
----------------------------------
Verify the closed retraining loop end to end and write a governance report:

    drift trigger  ->  gated retraining + promotion  ->  champion export  ->  redeploy

    python scripts/verify_retraining_flow.py                 # all checks, writes the report
    python scripts/verify_retraining_flow.py --no-sandbox    # skip the sandbox replay (faster)

Checks
  1. Trigger    reports/evidence/retraining_trigger_evidence.json says retraining
                was triggered, and why (drift + performance breach).
  2. Registry   the live MLflow registry (read-only): every alias holder passed
                the quality gates and no gated version beats @champion on
                validation RMSE.
  3. Sandbox    the promotion policy replayed in a throwaway registry with real
                candidates (champion-grade, passing-but-worse, exact tie, failing).
                Each outcome is compared with an independent re-derivation of the
                rule: @staging needs all gates on validation AND test; @champion
                additionally needs validation RMSE strictly below the incumbent.
                The sandbox @champion is then exported with scripts/export_champion.py
                and reloaded from disk to confirm the bundle serves identical predictions.
  4. Bundle     the committed models/champion/ matches the live @champion
                byte for byte (registry artifacts), serving_metadata.json matches
                the registry version/run/metrics, and the bundle reproduces its
                recorded test metrics.

The project's mlflow.db and mlruns/ are never written: the sandbox lives in a
temporary directory. Output: reports/evidence/retraining_lifecycle.md.
Exit code 0 when every check passes, 1 otherwise.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.api.model_loader import GATE_METRIC_PREFIXES, SERVING_METADATA_FILE  # noqa: E402
from src.models.quality_gates import QualityThresholds, evaluate_quality_gates  # noqa: E402
from src.models.registry import (  # noqa: E402
    CHAMPION_ALIAS, REGISTERED_MODEL, STAGING_ALIAS, tracking_uri,
)
from src.models.train import EXPERIMENT_NAME, REPORTS_DIR  # noqa: E402

TRIGGER_EVIDENCE = os.path.join(REPORTS_DIR, "retraining_trigger_evidence.json")
REPORT_PATH = os.path.join(REPORTS_DIR, "retraining_lifecycle.md")
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "models", "champion")
# Written by export_champion.py / the registry download, not part of the logged model.
BUNDLE_EXTRA_FILES = {SERVING_METADATA_FILE}
METRIC_TOLERANCE = 1e-3


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Section:
    checks: List[Check] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(passed), detail))
        return bool(passed)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_hashes(root: str) -> dict:
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            out[os.path.relpath(full, root).replace(os.sep, "/")] = _sha256(full)
    return out


@contextmanager
def _tracking(uri: str):
    """Point MLflow (and MLFLOW_TRACKING_URI, read by export_champion) at `uri` temporarily."""
    previous_uri, previous_env = mlflow.get_tracking_uri(), os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous_uri)
        if previous_env is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous_env


# 1. Trigger

def check_trigger(path: str = TRIGGER_EVIDENCE) -> Section:
    s = Section()
    if not s.check("Trigger evidence present", os.path.isfile(path), _rel(path)):
        return s
    with open(path, encoding="utf-8") as f:
        ev = json.load(f)
    s.data = ev
    s.check("Retraining triggered", ev.get("retraining_triggered") is True,
            "; ".join(ev.get("trigger_reasons", [])))
    breaches = ev.get("threshold_breaches", {})
    s.check("Dataset drift breached", breaches.get("dataset_drift", {}).get("breached") is True,
            f"share {breaches.get('dataset_drift', {}).get('observed_share')} >= "
            f"{breaches.get('dataset_drift', {}).get('threshold_share')}")
    perf = breaches.get("quality_gate", {}).get("breached") or breaches.get("rmse_degradation", {}).get("breached")
    s.check("Performance breached", perf is True,
            f"drifted RMSE {ev['drifted']['metrics']['rmse']} vs baseline {ev['baseline']['metrics']['rmse']}")
    s.check("Unshifted baseline does not trigger",
            ev.get("baseline_decision", {}).get("retraining_triggered") is False,
            "seasonal drift alone, without degradation, is not enough")
    return s


# 2. Live registry audit

def audit_registry(uri: str, model_name: str = REGISTERED_MODEL) -> Section:
    s = Section()
    with _tracking(uri):
        client = MlflowClient()
        try:
            registered = client.get_registered_model(model_name)
        except MlflowException as exc:
            s.check("Registry reachable", False, f"{uri}: {exc}")
            return s
        versions = []
        for mv in sorted(client.search_model_versions(f"name='{model_name}'"), key=lambda v: int(v.version)):
            metrics = client.get_run(mv.run_id).data.metrics
            versions.append({
                "version": int(mv.version), "run_id": mv.run_id,
                "model_type": mv.tags.get("model_type"), "quality_gate": mv.tags.get("quality_gate"),
                "relogged_from": mv.tags.get("relogged_from"),
                "val_rmse": metrics.get("val_rmse"), "test_rmse": metrics.get("test_rmse"),
                "val_mae": metrics.get("val_mae"), "val_r2": metrics.get("val_r2"),
            })
    aliases = {k: int(v) for k, v in registered.aliases.items()}
    by_version = {v["version"]: v for v in versions}
    for v in versions:
        v["aliases"] = sorted(a for a, ver in aliases.items() if ver == v["version"])
    s.data = {"aliases": aliases, "versions": versions}

    champ = by_version.get(aliases.get(CHAMPION_ALIAS))
    staging = by_version.get(aliases.get(STAGING_ALIAS))
    s.check("@champion assigned", champ is not None, f"v{aliases.get(CHAMPION_ALIAS)}")
    s.check("@staging assigned", staging is not None, f"v{aliases.get(STAGING_ALIAS)}")
    if champ is None:
        return s
    holders = [by_version[v] for v in aliases.values() if v in by_version]
    s.check("Every alias holder passed the quality gates",
            all(h["quality_gate"] == "passed" for h in holders),
            ", ".join(f"v{h['version']}={h['quality_gate']}" for h in holders))
    failed = [v for v in versions if v["quality_gate"] != "passed"]
    s.check("No failed candidate holds an alias", not any(v["aliases"] for v in failed),
            f"{len(failed)} failed version(s): " + ", ".join(f"v{v['version']}" for v in failed))
    better = [v for v in versions if v["quality_gate"] == "passed" and v["val_rmse"] is not None
              and v["val_rmse"] < champ["val_rmse"] - 1e-9]
    passed_not_promoted = [v for v in versions if v["quality_gate"] == "passed" and not v["aliases"]
                           and v["val_rmse"] is not None and v["val_rmse"] >= champ["val_rmse"]]
    s.check("No gated version beats @champion on validation RMSE", not better,
            f"@champion v{champ['version']} val RMSE {champ['val_rmse']:.4f}; "
            f"{len(passed_not_promoted)} later gated candidates held back")
    s.data["held_back"] = len(passed_not_promoted)
    return s


# 3. Sandbox replay of the promotion policy

def expected_outcome(metrics: dict, incumbent: Optional[float],
                     thresholds: QualityThresholds = QualityThresholds()) -> dict:
    """Independent statement of the policy, from a candidate's logged metrics."""
    failures = []
    for split in ("val", "test"):
        m = {k: metrics[f"{split}_{k}"] for k in ("mae", "rmse", "r2")}
        base = {k: metrics[f"baseline_{split}_{k}"] for k in ("mae", "rmse", "r2")}
        failures += [f"{split}: {f}" for f in evaluate_quality_gates(m, thresholds, base).failures]
    passed = not failures
    champion = passed and (incumbent is None or metrics["val_rmse"] < incumbent)
    aliases = ([STAGING_ALIAS] if passed else []) + ([CHAMPION_ALIAS] if champion else [])
    return {"passed": passed, "aliases": aliases, "failures": failures}


def _sandbox_candidates():
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression

    from src.models.hurdle import HurdleModel

    hurdle = {"active_threshold_wh": 100.0, "decision_threshold": 0.5}
    return [
        ("hurdle", "Champion-grade retrain (production config)", lambda: HurdleModel(**hurdle), hurdle),
        ("linear_regression", "Passes gates, worse validation RMSE", LinearRegression, {}),
        ("hurdle", "Identical retrain: ties the incumbent", lambda: HurdleModel(**hurdle), hurdle),
        ("mean_predictor", "Regressed model: predicts the training mean",
         lambda: DummyRegressor(strategy="mean"), {"strategy": "mean"}),
    ]


def replay_promotion_policy(splits=None) -> Section:
    from src.data.load_data import load_raw_data
    from src.features.feature_engineering import build_feature_matrix, chronological_split, get_X_y
    from src.models.registry import register_candidate
    from src.validation.input_validation import validate_feature_frame

    s = Section()
    train, val, test = splits or chronological_split(build_feature_matrix(load_raw_data()))
    X_train, y_train = get_X_y(train)
    X_train = validate_feature_frame(X_train)
    model_name = "energy-forecaster-sandbox"

    with tempfile.TemporaryDirectory(prefix="retrain-sandbox-", ignore_cleanup_errors=True) as tmp:
        uri = f"sqlite:///{os.path.join(tmp, 'mlflow.db').replace(os.sep, '/')}"
        with _tracking(uri):
            client = MlflowClient()
            client.create_experiment(EXPERIMENT_NAME,
                                     artifact_location=f"file:///{os.path.join(tmp, 'artifacts').replace(os.sep, '/')}")
            incumbent, steps, models = None, [], {}
            for model_type, scenario, factory, params in _sandbox_candidates():
                started = time.perf_counter()
                model = factory().fit(X_train, y_train)
                result = register_candidate(model, model_type, val, test, params=params,
                                            registered_model_name=model_name, log_artifacts=False,
                                            fit_seconds=time.perf_counter() - started)
                expected = expected_outcome(result.metrics, incumbent)
                matches = (expected["passed"] == result.gates_passed
                           and sorted(expected["aliases"]) == sorted(result.aliases))
                steps.append({
                    "version": int(result.version), "model_type": model_type, "scenario": scenario,
                    "val_rmse": result.metrics["val_rmse"], "test_rmse": result.metrics["test_rmse"],
                    "incumbent_val_rmse": incumbent, "gates_passed": result.gates_passed,
                    "failures": result.gate_failures, "aliases": result.aliases,
                    "expected_aliases": expected["aliases"], "matches_policy": matches,
                })
                s.check(f"Sandbox v{result.version} ({scenario}) follows the policy", matches,
                        f"aliases {result.aliases or 'none'}; expected {expected['aliases'] or 'none'}")
                models[int(result.version)] = model
                if CHAMPION_ALIAS in result.aliases:
                    incumbent = result.metrics["val_rmse"]

            s.check("Policy exercised every branch",
                    any(CHAMPION_ALIAS in st["aliases"] for st in steps)
                    and any(st["gates_passed"] and CHAMPION_ALIAS not in st["aliases"] for st in steps)
                    and any(not st["gates_passed"] for st in steps),
                    "promoted, held at @staging, and rejected outcomes all observed")
            tie = steps[2]
            s.check("A tie does not displace @champion",
                    tie["val_rmse"] != steps[0]["val_rmse"] or CHAMPION_ALIAS not in tie["aliases"],
                    f"v{tie['version']} val RMSE {tie['val_rmse']} vs incumbent {tie['incumbent_val_rmse']}")

            champion_version = int(client.get_model_version_by_alias(model_name, CHAMPION_ALIAS).version)
            staging_version = int(client.get_model_version_by_alias(model_name, STAGING_ALIAS).version)

            # Export the sandbox champion exactly as for Docker/Render, then serve it from disk.
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
            from export_champion import export

            bundle = os.path.join(tmp, "bundle", "champion")
            meta = export(CHAMPION_ALIAS, bundle, model_name=model_name)
            X_test, _ = get_X_y(test)
            X_test = validate_feature_frame(X_test)
            served = np.asarray(mlflow.pyfunc.load_model(bundle).predict(X_test), dtype=float).reshape(-1)
            direct = np.asarray(models[champion_version].predict(X_test), dtype=float).reshape(-1)
            s.check("Exported sandbox bundle carries the @champion version",
                    meta["model_version"] == str(champion_version) and meta["alias"] == CHAMPION_ALIAS,
                    f"serving_metadata.json -> v{meta['model_version']} @{meta['alias']}")
            s.check("Exported bundle serves identical predictions",
                    np.allclose(served, direct, atol=1e-6), f"{len(served):,} test rows compared")

    s.data = {"steps": steps, "champion_version": champion_version, "staging_version": staging_version}
    return s


# 4. Committed bundle vs live registry

def verify_bundle(uri: str, bundle: str = BUNDLE_DIR, model_name: str = REGISTERED_MODEL,
                  splits=None) -> Section:
    s = Section()
    meta_path = os.path.join(bundle, SERVING_METADATA_FILE)
    if not s.check("Bundle has serving_metadata.json", os.path.isfile(meta_path), _rel(meta_path)):
        return s
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    s.data["metadata"] = meta

    with _tracking(uri):
        client = MlflowClient()
        try:
            mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        except MlflowException as exc:
            s.check("Registry @champion reachable", False, str(exc))
            return s
        run_metrics = client.get_run(mv.run_id).data.metrics
        with tempfile.TemporaryDirectory(prefix="champion-artifacts-", ignore_cleanup_errors=True) as tmp:
            mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_name}@{CHAMPION_ALIAS}", dst_path=tmp)
            registry_files = _tree_hashes(tmp)

    s.check("Metadata version/run match @champion",
            meta.get("model_version") == str(mv.version) and meta.get("run_id") == mv.run_id
            and meta.get("alias") == CHAMPION_ALIAS,
            f"v{meta.get('model_version')} run {meta.get('run_id', '')[:8]} vs registry v{mv.version} "
            f"run {mv.run_id[:8]}")
    s.check("Metadata tags match registry",
            meta.get("model_type") == mv.tags.get("model_type")
            and meta.get("quality_gate") == mv.tags.get("quality_gate") == "passed",
            f"{meta.get('model_type')}, gate {meta.get('quality_gate')}")
    expected = {k: v for k, v in run_metrics.items() if k.startswith(GATE_METRIC_PREFIXES)}
    recorded = meta.get("metrics", {})
    metric_ok = set(expected) == set(recorded) and all(
        abs(expected[k] - recorded[k]) <= METRIC_TOLERANCE for k in expected)
    s.check("Metadata metrics match the registry run", metric_ok, f"{len(recorded)} gate metrics")

    bundle_files = {k: v for k, v in _tree_hashes(bundle).items() if k not in BUNDLE_EXTRA_FILES}
    mismatched = sorted(k for k in set(bundle_files) | set(registry_files)
                        if bundle_files.get(k) != registry_files.get(k))
    s.check("Bundle files byte-identical to registry artifacts", not mismatched,
            f"{len(registry_files)} files compared" + (f"; differ: {mismatched}" if mismatched else ""))
    with open(os.path.join(bundle, "MLmodel"), encoding="utf-8") as f:
        mlmodel = yaml.safe_load(f)
    s.check("MLmodel run_id matches @champion", mlmodel.get("run_id") == mv.run_id, mlmodel.get("run_id", ""))

    from src.data.load_data import load_raw_data
    from src.features.feature_engineering import build_feature_matrix, chronological_split, get_X_y
    from src.models.evaluate import compute_metrics
    from src.validation.input_validation import validate_feature_frame

    _, _, test = splits or chronological_split(build_feature_matrix(load_raw_data()))
    X_test, y_test = get_X_y(test)
    preds = np.clip(np.asarray(mlflow.pyfunc.load_model(bundle).predict(validate_feature_frame(X_test)),
                               dtype=float).reshape(-1), 0.0, None)
    reproduced = compute_metrics(y_test, preds)
    s.data["reproduced_test"] = reproduced
    s.check("Bundle reproduces recorded test RMSE",
            abs(reproduced["rmse"] - recorded.get("test_rmse", float("nan"))) <= METRIC_TOLERANCE,
            f"{reproduced['rmse']:.4f} vs recorded {recorded.get('test_rmse')}")
    return s


# Report

def _checks_table(section: Section) -> List[str]:
    lines = ["| Check | Result | Detail |", "|---|---|---|"]
    for c in section.checks:
        lines.append(f"| {c.name} | {'PASS' if c.passed else '**FAIL**'} | {c.detail.replace('|', '/')} |")
    return lines


def _fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def render_report(trigger: Section, registry: Section, sandbox: Optional[Section], bundle: Section,
                  thresholds: QualityThresholds = QualityThresholds()) -> str:
    sections = [s for s in (trigger, registry, sandbox, bundle) if s is not None]
    n_checks = sum(len(s.checks) for s in sections)
    n_passed = sum(c.passed for s in sections for c in s.checks)
    ok = n_checks == n_passed
    ev = trigger.data
    L = [
        "# Retraining Lifecycle and Champion Promotion Evidence",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`python scripts/verify_retraining_flow.py`. "
        f"**{n_passed}/{n_checks} checks passed{'' if ok else ' - see failures below'}.**",
        "",
        "```text",
        "drift trigger -> retrain candidates -> quality gates (val + test) -> @staging",
        "              -> beats @champion val RMSE (strict) -> @champion -> export_champion.py",
        "              -> models/champion/ -> Docker image / Render redeploy -> /health reports version",
        "```",
        "",
        "## 1. Drift trigger",
        "",
    ]
    if ev:
        b, d = ev["baseline"]["metrics"], ev["drifted"]["metrics"]
        deg = ev["degradation_vs_baseline"]
        gate = ev["threshold_breaches"]["quality_gate"]["thresholds"]
        L += [
            f"Source: `{_rel(TRIGGER_EVIDENCE)}` (experiment `{ev['experiment']['name']}`, "
            f"{ev['drifted']['rows']:,} rows, {ev['experiment']['window']['start'][:10]} to "
            f"{ev['experiment']['window']['end'][:10]}, model "
            f"{ev['model'].get('name')} v{ev['model'].get('version')}).",
            "",
            "| Window | RMSE (Wh) | MAE (Wh) | R² | Drifted features | Retrain |",
            "|---|---|---|---|---|---|",
            f"| Baseline test split | {b['rmse']:.2f} | {b['mae']:.2f} | {b['r2']:.3f} | "
            f"{ev['baseline']['drift']['drifted_count']}/{ev['baseline']['drift']['monitored_count']} | "
            f"{'yes' if ev['baseline_decision']['retraining_triggered'] else 'no'} |",
            f"| Heatwave scenario | {d['rmse']:.2f} | {d['mae']:.2f} | {d['r2']:.3f} | "
            f"{ev['drifted']['drift']['drifted_count']}/{ev['drifted']['drift']['monitored_count']} | "
            f"{'**yes**' if ev['retraining_triggered'] else 'no'} |",
            "",
            f"Heatwave RMSE **{d['rmse']:.2f} Wh** is **{deg['rmse']['relative']:+.1%}** against the "
            f"{b['rmse']:.2f} Wh baseline and breaches the {gate['max_rmse']:.0f} Wh quality gate "
            f"(MAE {deg['mae']['relative']:+.1%}). Mean drift score rose from "
            f"{ev['baseline']['drift']['mean_feature_score']:.2f} to {ev['drifted']['drift']['mean_feature_score']:.2f}. "
            "R² rises because the heavier evening load widens target variance; RMSE and MAE carry the "
            "degradation signal. The unshifted baseline already shows seasonal drift but passes the gates, "
            "so the rule (drift **and** a performance breach) does not retrain on it.",
            "",
            "Trigger reasons:",
            "",
            *[f"- {r}" for r in ev["trigger_reasons"]],
            "",
        ]
    L += _checks_table(trigger) + [
        "",
        "## 2. Retraining policy",
        "",
        "Retraining runs `python run_registry.py` (or the ZenML pipeline), which fits candidates and calls "
        "`register_candidate()` in `src/models/registry.py` for each:",
        "",
        "1. Every candidate is logged and registered as a new version, pass or fail, so rejected models stay auditable.",
        "2. `evaluate_quality_gates()` (`src/models/quality_gates.py`) runs on **both** validation and test: "
        f"MAE ≤ {thresholds.max_mae:.0f} Wh, RMSE ≤ {thresholds.max_rmse:.0f} Wh, R² ≥ {thresholds.min_r2:.2f}, "
        f"and RMSE ≤ {thresholds.max_rmse_vs_baseline:.1f}× the persistence baseline on the same split.",
        f"3. Passing all gates assigns `@{STAGING_ALIAS}`. Failing any gate assigns no alias.",
        f"4. `@{CHAMPION_ALIAS}` moves only if the candidate also has a validation RMSE **strictly lower** than "
        "the incumbent champion's. Ties and worse candidates stay at `@staging`, so a retrain can never "
        "silently replace production with an equal or weaker model.",
        "5. Test data is a pass/fail release check only; it is never used to choose between candidates.",
        "",
    ]
    if registry.checks:
        a = registry.data.get("aliases", {})
        L += [
            "## 3. Live registry audit",
            "",
            f"`{REGISTERED_MODEL}` in the project MLflow registry: {len(registry.data.get('versions', []))} versions, "
            f"`@{CHAMPION_ALIAS}` = v{a.get(CHAMPION_ALIAS)}, `@{STAGING_ALIAS}` = v{a.get(STAGING_ALIAS)}. "
            f"{registry.data.get('held_back', 0)} later candidates passed the gates but did not beat the champion's "
            "validation RMSE, so they were held at or below `@staging`.",
            "",
            "| Version | Model | Gate | Val RMSE | Test RMSE | Aliases | Note |",
            "|---|---|---|---|---|---|---|",
        ]
        for v in registry.data.get("versions", []):
            note = f"re-log of v{v['relogged_from']} (manual)" if v["relogged_from"] else ""
            L.append(f"| v{v['version']} | {v['model_type']} | {v['quality_gate']} | {_fmt(v['val_rmse'], 4)} | "
                     f"{_fmt(v['test_rmse'], 4)} | {', '.join('@' + x for x in v['aliases']) or '-'} | {note} |")
        L += [
            "",
            "Versions 4 and 5 re-log the fitted v1 model to refresh its recorded dependencies; they carry a "
            "`relogged_from` tag and were moved to `@champion` by hand, since a tie cannot win the strict comparison.",
            "",
        ]
        L += _checks_table(registry) + [""]
    if sandbox is not None:
        L += [
            "## 4. Promotion policy replay (sandbox registry)",
            "",
            "A temporary MLflow registry receives real candidates fitted on the training split, in this order. "
            "Each outcome is compared with an independent re-derivation of the policy.",
            "",
            "| Version | Candidate | Val RMSE | Incumbent | Gates | Aliases | Policy |",
            "|---|---|---|---|---|---|---|",
        ]
        for st in sandbox.data.get("steps", []):
            gates = "passed" if st["gates_passed"] else f"failed ({len(st['failures'])})"
            L.append(f"| v{st['version']} | {st['scenario']} | {st['val_rmse']:.4f} | "
                     f"{_fmt(st['incumbent_val_rmse'], 4)} | {gates} | "
                     f"{', '.join('@' + x for x in st['aliases']) or '-'} | "
                     f"{'as expected' if st['matches_policy'] else '**mismatch**'} |")
        rejected = [st for st in sandbox.data.get("steps", []) if not st["gates_passed"]]
        if rejected:
            L += ["", "Gate failures recorded for the rejected candidate:", ""]
            L += [f"- `{f}`" for f in rejected[0]["failures"]]
        L += ["", f"Final sandbox aliases: `@{CHAMPION_ALIAS}` = v{sandbox.data.get('champion_version')}, "
                  f"`@{STAGING_ALIAS}` = v{sandbox.data.get('staging_version')}.", ""]
        L += _checks_table(sandbox) + [""]
    meta = bundle.data.get("metadata", {})
    rt = bundle.data.get("reproduced_test")
    L += [
        f"## {5 if sandbox is not None else 4}. Champion export and redeployment",
        "",
        "`python scripts/export_champion.py` resolves `models:/energy-forecaster@champion`, refuses to overwrite "
        "anything but a previous MLflow export, downloads the model into `models/champion/`, and writes "
        "`serving_metadata.json` (version, run, model type, gate status, gate metrics) so `/health` and "
        "`/model-info` report the registry lineage without registry access.",
        "",
        "- **Docker / compose:** the Dockerfile copies `models/champion/` into the image, and compose also mounts it "
        "read-only at `MODEL_LOCAL_PATH`, so a new champion is served after `docker compose up -d --build` "
        "(or a container restart for the mount).",
        "- **CI/CD and Render:** the bundle is committed, so a push to `main` runs CI (tests and quality gates), "
        "and on success CD (`.github/workflows/cd.yml`) publishes `ghcr.io/<owner>/energy-forecaster-api` "
        "(`:latest`, `:sha-<short>`) with the new champion baked in. The Render service is redeployed from the "
        "repository/image (its configuration lives in Render, not in this repo), after which `/health` reports "
        "the new `model_version`.",
        "",
    ]
    if meta:
        L += [
            f"Committed bundle: {meta.get('model_name')} **v{meta.get('model_version')}** "
            f"({meta.get('model_type')}, gate {meta.get('quality_gate')}), run `{meta.get('run_id')}`, "
            f"exported {meta.get('exported_at')}. Recorded val RMSE {meta['metrics'].get('val_rmse')}, "
            f"test RMSE {meta['metrics'].get('test_rmse')}"
            + (f"; reproduced from the bundle on the test split: RMSE {rt['rmse']:.4f}, MAE {rt['mae']:.4f}, "
               f"R² {rt['r2']:.4f}." if rt else "."),
            "",
        ]
    L += _checks_table(bundle) + [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/run_drift_experiment.py     # drift trigger evidence",
        "python run_registry.py                     # retrain + gated registration",
        "python scripts/export_champion.py          # package @champion for Docker/Render",
        "python scripts/verify_retraining_flow.py   # this report",
        "```",
        "",
    ]
    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify the retraining loop and write the governance report.")
    parser.add_argument("--output", default=REPORT_PATH)
    parser.add_argument("--no-sandbox", action="store_true", help="Skip the sandbox promotion replay.")
    args = parser.parse_args(argv)

    uri = tracking_uri()
    splits = None
    if not args.no_sandbox:
        from src.data.load_data import load_raw_data
        from src.features.feature_engineering import build_feature_matrix, chronological_split
        splits = chronological_split(build_feature_matrix(load_raw_data()))

    trigger = check_trigger()
    registry = audit_registry(uri)
    sandbox = None if args.no_sandbox else replay_promotion_policy(splits)
    bundle = verify_bundle(uri, splits=splits)

    report = render_report(trigger, registry, sandbox, bundle)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    ok = True
    for title, section in (("Trigger", trigger), ("Registry", registry), ("Sandbox", sandbox), ("Bundle", bundle)):
        if section is None:
            continue
        ok &= section.passed
        print(f"  {title:9s}: {sum(c.passed for c in section.checks)}/{len(section.checks)} checks passed")
        for c in section.checks:
            if not c.passed:
                print(f"    FAIL {c.name}: {c.detail}")
    print(f"  Report   -> {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
