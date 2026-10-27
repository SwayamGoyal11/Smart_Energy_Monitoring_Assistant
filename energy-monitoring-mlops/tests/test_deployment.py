"""
tests/test_deployment.py
-------------------------
Static checks for the Docker / Prometheus / Grafana configuration, so config
drift is caught in CI without a Docker daemon: Dockerfile hardening rules,
build-context contents, and cross-file consistency (service names, ports,
paths, and metric names used by alerts and dashboards).
"""

import fnmatch
import json
import os
import re
import subprocess
import sys

import pytest
import yaml

from src.monitoring.metrics import ApiMetrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCKER_INSTRUCTIONS = {
    "FROM", "RUN", "CMD", "LABEL", "EXPOSE", "ENV", "ADD", "COPY", "ENTRYPOINT", "VOLUME",
    "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL",
}


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def dockerfile_instructions():
    """(INSTRUCTION, args) pairs with line continuations joined and comments dropped."""
    lines, buf = [], ""
    for raw in read("Dockerfile").splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        lines.append(buf + line)
        buf = ""
    return [(l.split(None, 1)[0].upper(), l.split(None, 1)[1] if " " in l else "") for l in lines]


def final_stage(instr):
    last_from = max(i for i, (k, _) in enumerate(instr) if k == "FROM")
    return instr[last_from:]


def dockerignore_admits(path):
    """Docker semantics: the last matching pattern wins; '!' re-includes."""
    admitted = True
    for pat in [p.strip() for p in read(".dockerignore").splitlines()]:
        if not pat or pat.startswith("#"):
            continue
        negate = pat.startswith("!")
        pat = pat.lstrip("!").rstrip("/")
        parts = path.split("/")
        candidates = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
        if any(fnmatch.fnmatch(c, pat) or fnmatch.fnmatch(c, pat.replace("**/", "")) for c in candidates):
            admitted = negate
    return admitted


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(read("docker-compose.yml"))


@pytest.fixture(scope="module")
def api_metric_names():
    names = {"up"}
    for family in ApiMetrics().registry.collect():
        names.add(family.name)
        names.update(s.name for s in family.samples)
    # Histogram/counter series that only appear once observed.
    for base in ("http_request_duration_seconds", "model_prediction_value"):
        names.update({f"{base}_bucket", f"{base}_count", f"{base}_sum"})
    names.update({"http_requests_total", "model_stage_predictions_total"})
    return names


def promql_metric_names(expr):
    functions = {"sum", "rate", "by", "histogram_quantile", "clamp_min", "sort_desc", "le", "and", "or"}
    return {m for m in re.findall(r"\b([a-z_][a-z0-9_]*)\s*(?=\{|\[|\)|$|\s*[<>=])", expr)
            if m not in functions and ("_" in m or m == "up")}


# Dockerfile

class TestDockerfile:
    def test_only_valid_instructions(self):
        assert {k for k, _ in dockerfile_instructions()} <= DOCKER_INSTRUCTIONS

    def test_multi_stage_python_312_slim(self):
        instr = dockerfile_instructions()
        froms = [a for k, a in instr if k == "FROM"]
        assert len(froms) == 2
        assert all(a.startswith("python:${PYTHON_VERSION}-slim") for a in froms)
        assert ("ARG", "PYTHON_VERSION=3.12") in instr

    def test_runs_as_non_root(self):
        stage = final_stage(dockerfile_instructions())
        users = [a for k, a in stage if k == "USER"]
        assert users and users[-1] not in ("root", "0")
        user_idx = max(i for i, (k, _) in enumerate(stage) if k == "USER")
        cmd_idx = max(i for i, (k, _) in enumerate(stage) if k == "CMD")
        assert user_idx < cmd_idx

    def test_no_pip_cache(self):
        instr = dockerfile_instructions()
        installs = [a for k, a in instr if k == "RUN" and "pip install" in a]
        assert installs and all("--no-cache-dir" in a for a in installs)
        assert any("requirements-serve.txt" in a for a in installs)

    def test_exposes_8000_and_healthchecks_health(self):
        stage = final_stage(dockerfile_instructions())
        assert ("EXPOSE", "8000") in stage
        health = [a for k, a in stage if k == "HEALTHCHECK"]
        assert len(health) == 1 and "/health" in health[0] and ":8000" in health[0]

    def test_cmd_serves_app_on_8000(self):
        cmd = json.loads([a for k, a in dockerfile_instructions() if k == "CMD"][-1])
        assert cmd[:2] == ["uvicorn", "src.api.app:app"]
        assert cmd[cmd.index("--port") + 1] == "8000"
        assert cmd[cmd.index("--host") + 1] == "0.0.0.0"

    def test_copy_sources_are_in_build_context(self):
        for k, a in dockerfile_instructions():
            if k == "COPY" and "--from=" not in a:
                sources = [s for s in a.split()[:-1] if not s.startswith("--")]
                for src in sources:
                    assert dockerignore_admits(src.rstrip("/")), f"{src} excluded by .dockerignore"
                    assert os.path.exists(os.path.join(ROOT, src))


class TestBuildContext:
    @pytest.mark.parametrize("path", ["src/api/app.py", "src/models/hurdle.py", "requirements-serve.txt"])
    def test_needed_files_admitted(self, path):
        assert dockerignore_admits(path)

    @pytest.mark.parametrize("path", [
        "venv/pyvenv.cfg", ".venv/bin/python", ".git/HEAD", ".zen/config.yaml", ".dvc/config",
        "tests/test_api.py", "notebooks/01_eda.ipynb", "data/raw/energydata_complete.csv",
        "mlflow.db", "mlruns/1/meta.yaml", "models/champion/model.skops",
        "reports/evidence/drift_report.html", "src/api/__pycache__/app.cpython-312.pyc",
    ])
    def test_heavy_or_sensitive_paths_excluded(self, path):
        assert not dockerignore_admits(path)


class TestServeRequirements:
    def test_dev_requirements_include_serve(self):
        assert "-r requirements-serve.txt" in read("requirements.txt").splitlines()

    def test_serve_requirements_are_minimal(self):
        names = {re.split(r"[<>=\[]", l)[0].strip().lower()
                 for l in read("requirements-serve.txt").splitlines() if l.strip() and not l.startswith("#")}
        assert {"fastapi", "uvicorn", "mlflow-skinny", "skops", "prometheus-client"} <= names
        assert not names & {"mlflow", "evidently", "zenml", "jupyter", "dvc", "pytest", "matplotlib"}

    def test_api_does_not_import_training_only_packages(self):
        code = ("import sys; import src.api.app; "
                "print(sorted(m for m in ('evidently', 'zenml', 'matplotlib', 'dvc') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"})
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().splitlines()[-1] == "[]"


# Compose, Prometheus, Grafana

class TestCompose:
    def test_three_services_on_shared_bridge_network(self, compose):
        assert set(compose["services"]) == {"api", "prometheus", "grafana"}
        assert compose["networks"]["monitoring"]["driver"] == "bridge"
        for svc in compose["services"].values():
            assert svc["networks"] == ["monitoring"]

    def test_ports(self, compose):
        assert "8000:8000" in compose["services"]["api"]["ports"]
        assert "9090:9090" in compose["services"]["prometheus"]["ports"]
        assert "3000:3000" in compose["services"]["grafana"]["ports"]

    def test_persistent_volumes(self, compose):
        assert {"prometheus-data", "grafana-data", "inference-logs"} <= set(compose["volumes"])
        assert "prometheus-data:/prometheus" in compose["services"]["prometheus"]["volumes"]

    def test_api_mounts_match_env_paths(self, compose):
        api = compose["services"]["api"]
        targets = {v.split(":")[1] for v in api["volumes"]}
        env = api["environment"]
        assert env["MODEL_LOCAL_PATH"] in targets
        assert os.path.dirname(env["DRIFT_METRICS_PATH"]) in targets
        assert os.path.dirname(env["INFERENCE_LOG_PATH"]) in targets
        assert "./models/champion:/app/models/champion:ro" in api["volumes"]

    def test_api_hardening(self, compose):
        api = compose["services"]["api"]
        assert "no-new-privileges:true" in api["security_opt"]
        assert api["cap_drop"] == ["ALL"]

    def test_prometheus_waits_for_healthy_api(self, compose):
        assert compose["services"]["prometheus"]["depends_on"]["api"]["condition"] == "service_healthy"

    def test_mounted_config_files_exist(self, compose):
        for svc in compose["services"].values():
            for vol in svc.get("volumes", []):
                src = vol.split(":")[0]
                if src.startswith("./") and "champion" not in src and "reports" not in src:
                    assert os.path.exists(os.path.join(ROOT, src)), src


class TestPrometheusConfig:
    def test_scrapes_api_service_every_15s(self, compose):
        cfg = yaml.safe_load(read("prometheus/prometheus.yml"))
        job = next(j for j in cfg["scrape_configs"] if j["job_name"] == "energy-api")
        assert job["metrics_path"] == "/metrics"
        assert job.get("scrape_interval", cfg["global"]["scrape_interval"]) == "15s"
        host, port = job["static_configs"][0]["targets"][0].split(":")
        assert host in compose["services"]
        assert f"{port}:{port}" in compose["services"][host]["ports"] or port == "8000"

    def test_rule_file_is_mounted(self, compose):
        cfg = yaml.safe_load(read("prometheus/prometheus.yml"))
        mounts = {v.split(":")[1] for v in compose["services"]["prometheus"]["volumes"]}
        assert set(cfg["rule_files"]) <= mounts

    def test_alert_rules_use_real_metrics(self, api_metric_names):
        rules = yaml.safe_load(read("prometheus/alerts.yml"))["groups"][0]["rules"]
        assert {r["alert"] for r in rules} >= {"EnergyApiDown", "EnergyModelDriftDetected"}
        for rule in rules:
            assert set(rule) >= {"alert", "expr", "for", "labels", "annotations"}
            unknown = promql_metric_names(rule["expr"]) - api_metric_names
            assert not unknown, f"{rule['alert']} references unknown metrics {unknown}"


class TestGrafanaProvisioning:
    def test_datasource_points_at_prometheus_service(self, compose):
        ds = yaml.safe_load(read("grafana/provisioning/datasources/prometheus.yml"))["datasources"][0]
        assert ds["type"] == "prometheus" and ds["uid"] == "prometheus" and ds["isDefault"]
        assert ds["url"] == "http://prometheus:9090"
        assert "prometheus" in compose["services"]

    def test_dashboard_provider_path_holds_dashboard(self):
        provider = yaml.safe_load(read("grafana/provisioning/dashboards/dashboards.yml"))["providers"][0]
        rel = provider["options"]["path"].replace("/etc/grafana/provisioning", "grafana/provisioning")
        assert any(f.endswith(".json") for f in os.listdir(os.path.join(ROOT, rel)))

    def test_dashboard_queries_use_real_metrics(self, api_metric_names):
        dash = json.loads(read("grafana/provisioning/dashboards/json/energy-forecaster.json"))
        exprs = [t["expr"] for p in dash["panels"] for t in p.get("targets", [])]
        assert len(exprs) >= 10
        for expr in exprs:
            unknown = promql_metric_names(expr) - api_metric_names
            assert not unknown, f"dashboard query references unknown metrics {unknown}: {expr}"
        assert all(t["datasource"]["uid"] == "prometheus"
                   for p in dash["panels"] for t in p.get("targets", []))
