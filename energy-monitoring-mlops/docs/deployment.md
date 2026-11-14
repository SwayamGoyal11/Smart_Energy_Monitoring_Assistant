# Deployment: Docker serving + monitoring stack

## Components

| Service | Image | Port | Purpose |
|---|---|---|---|
| `api` | built from `Dockerfile` | 8000 | FastAPI forecaster (`/health`, `/predict`, `/predict/batch`, `/model-info`, `/metrics`) |
| `prometheus` | `prom/prometheus:v2.53.0` | 9090 | Scrapes `api:8000/metrics` every 15s; evaluates `prometheus/alerts.yml` |
| `grafana` | `grafana/grafana:11.1.0` | 3000 | Provisioned Prometheus datasource + "Energy Forecaster - Serving & Drift" dashboard |

All three share the `monitoring` bridge network. Named volumes: `prometheus-data` (TSDB),
`grafana-data`, `inference-logs` (served inputs + predictions).

## Run

```bash
# 1. Export the registry champion to a standalone MLflow model directory.
#    The registry's artifact paths are host-absolute, so the container serves
#    this copy instead of reading mlflow.db.
python scripts/export_champion.py            # -> models/champion/

# 2. (Optional) refresh drift metrics exported as model_drift_* gauges
python -m src.monitoring.drift               # -> reports/evidence/drift_metrics.json

# 3. Build and start (PowerShell: $env:GRAFANA_ADMIN_PASSWORD = "change-me"; docker compose up --build -d)
GRAFANA_ADMIN_PASSWORD=change-me docker compose up --build -d

# 4. Check
docker compose ps                            # api should become "healthy"
curl http://localhost:8000/health
curl http://localhost:8000/metrics
open http://localhost:9090/targets           # energy-api target UP
open http://localhost:3000                   # admin / $GRAFANA_ADMIN_PASSWORD
```

Promote a new champion without rebuilding: rerun `run_registry.py`, then
`scripts/export_champion.py`, then `docker compose restart api` (compose mounts
`models/champion/` read-only over the copy baked into the image). Verify the
bundle and record the promotion with `python scripts/verify_retraining_flow.py`.
Rollback is the same path with `@champion` reassigned to an earlier gated
version; see the [model card](model_card.md#67-rollback).

## CI/CD pipeline (GitHub Actions -> GHCR)

1. **CI** (`.github/workflows/ci.yml`) runs on every push to `main`, every `v*.*.*`
   tag and every pull request: ruff, the pytest suite, a Docker build check
   (imports the app inside the image, asserts uid 10001), and the quality-gate
   job (`run_registry.py` against a throwaway MLflow store).
2. **CD** (`.github/workflows/cd.yml`) runs only after a **successful** CI run
   from a push to `main` or a `v*.*.*` tag (pull-request runs never publish).
   It checks out the exact commit CI tested and pushes
   `ghcr.io/swayamgoyal11/energy-forecaster-api`:

   | Source | Tags |
   |---|---|
   | push to `main` | `:latest`, `:sha-<short>` |
   | tag `v1.2.0` | `:1.2.0`, `:1.2`, `:sha-<short>` (never `:latest`) |

The image is self-contained: the Dockerfile copies the committed
`models/champion/` bundle into `/app/models/champion`, so it serves the champion
without a volume or registry access. To run it:

```bash
docker run -d --name energy-api -p 8000:8000 ghcr.io/swayamgoyal11/energy-forecaster-api:latest
# optional: serve a different exported bundle without rebuilding
docker run -d --name energy-api -p 8000:8000 -v "${PWD}/models/champion:/app/models/champion:ro" ghcr.io/swayamgoyal11/energy-forecaster-api:latest
```

Pin a `:sha-<short>` or release tag for reproducible deploys. If the package is
private, `docker login ghcr.io` first with a token that has `read:packages`.

## Hosted deployment (Render)

The baked champion (`COPY models/champion/` in the Dockerfile, admitted by
`.dockerignore`) exists so the API can run on a managed host such as Render with
no mounted volume. The Render service itself (whether it builds from this repo's
`Dockerfile` or pulls the GHCR image, its auto-deploy setting, environment
variables and public URL) is configured in the Render dashboard. **No Render
configuration file (e.g. `render.yaml`) is committed, so this repository cannot
verify those settings.** What the repository does guarantee for any such host:

- The container listens on port **8000** (fixed in `CMD`); point the service's
  port at 8000.
- `/health` returns 200 only once a model is loaded, so it works as the health
  check path, and reports `model_version` so a redeploy can be confirmed.
- `reports/` is excluded from the image, so `model_drift_*` gauges stay empty
  unless a `drift_metrics.json` is provided at `DRIFT_METRICS_PATH`.
- The JSONL inference log at `INFERENCE_LOG_PATH` is written to the container
  filesystem; on hosts with ephemeral disks it is lost on redeploy unless a
  persistent disk is mounted at `/app/data/monitoring`.
- Exercise a deployed instance with
  `python scripts/simulate_traffic.py --url https://<your-service>`; it waits for
  cold starts and retries transient errors.

A new champion reaches a hosted instance by committing the re-exported
`models/champion/` and pushing to `main`: CI must pass before CD publishes the
image, and the host then redeploys according to its own settings.

## Drift on served traffic

The API appends every served input and prediction to `INFERENCE_LOG_PATH`
(the `inference-logs` volume) as structured JSON Lines, one object per
predicted row with `event`, `logged_at`, `target_timestamp`, `model_version`,
the model features and `prediction`. Score it against the training distribution:

```bash
mkdir -p data/monitoring
docker cp energy-api:/app/data/monitoring/inference_log.jsonl data/monitoring/inference_log.jsonl
python -m src.monitoring.drift --current data/monitoring/inference_log.jsonl
```

This rewrites `reports/evidence/drift_report.html` and `drift_metrics.json`; the
API re-reads the JSON on the next scrape, so the `model_drift_*` gauges and the
Grafana drift panels update without a restart.

## Image design

- Multi-stage: dependencies are installed into `/opt/venv` in a builder stage and
  only the venv, `src/` and the `models/champion/` bundle are copied into the
  runtime stage.
- Runtime dependencies come from `requirements-serve.txt` (`mlflow-skinny`, not the
  full MLflow server; no Evidently, ZenML, Jupyter or DVC). `requirements.txt`
  includes it, so the API's dependencies are declared once.
- Runs as non-root `app` (uid 10001); compose adds `no-new-privileges` and drops
  all Linux capabilities.
- Dependencies installed with `uv pip install --no-cache` (uv pinned to 0.6.5 via
  `ghcr.io/astral-sh/uv:0.6.5`), `PYTHONDONTWRITEBYTECODE`; `.dockerignore` is an
  allow-list (only `src/`, `models/champion/` and `requirements-serve.txt` enter
  the build context).
- `HEALTHCHECK` calls `/health` with Python's urllib (slim images have no curl);
  `/health` returns 503 when no model loaded, which fails the check.
- One uvicorn worker: Prometheus counters are per process, so multiple workers
  would each expose a partial `/metrics`. Scale with more containers instead.

## Alerts (`prometheus/alerts.yml`)

| Alert | Condition |
|---|---|
| `EnergyApiDown` | scrape target down for 1m |
| `EnergyApiHighErrorRate` | 5xx ratio > 5% for 5m |
| `EnergyApiSlowPredictions` | `/predict` p95 > 250 ms for 10m |
| `EnergyApiInputValidationFailures` | input-validation rejections > 10% of `/predict*` requests for 10m |
| `EnergyModelDriftDetected` | latest drift report raised an alert, for 5m |
| `EnergyModelActiveRatioAnomalous` | > 50% of predictions routed to the burst regressor for 30m (training ~22%) |

## Configuration checks

`tests/test_deployment.py` validates the Dockerfile (non-root user, healthcheck,
no pip cache, build-context contents), and cross-checks compose service names,
ports and mounts against the Prometheus and Grafana configs, and every metric
name used in alerts and dashboard queries against the API's metric registry.
