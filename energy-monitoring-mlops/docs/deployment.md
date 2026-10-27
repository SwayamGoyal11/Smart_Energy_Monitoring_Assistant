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

# 3. Build and start
GRAFANA_ADMIN_PASSWORD=change-me docker compose up --build -d

# 4. Check
docker compose ps                            # api should become "healthy"
curl http://localhost:8000/health
curl http://localhost:8000/metrics
open http://localhost:9090/targets           # energy-api target UP
open http://localhost:3000                   # admin / $GRAFANA_ADMIN_PASSWORD
```

Promote a new champion without rebuilding: rerun `run_registry.py`, then
`scripts/export_champion.py`, then `docker compose restart api`.

## Image design

- Multi-stage: dependencies are installed into `/opt/venv` in a builder stage and
  only the venv + `src/` are copied into the runtime stage.
- Runtime dependencies come from `requirements-serve.txt` (`mlflow-skinny`, not the
  full MLflow server; no Evidently, ZenML, Jupyter or DVC). `requirements.txt`
  includes it, so the API's dependencies are declared once.
- Runs as non-root `app` (uid 10001); compose adds `no-new-privileges` and drops
  all Linux capabilities.
- `pip --no-cache-dir`, `PYTHONDONTWRITEBYTECODE`; `.dockerignore` is an
  allow-list (only `src/` and `requirements-serve.txt` enter the build context).
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
| `EnergyModelDriftDetected` | latest drift report raised an alert, for 5m |
| `EnergyModelActiveRatioAnomalous` | > 50% of predictions routed to the burst regressor for 30m (training ~22%) |

## Configuration checks

`tests/test_deployment.py` validates the Dockerfile (non-root user, healthcheck,
no pip cache, build-context contents), and cross-checks compose service names,
ports and mounts against the Prometheus and Grafana configs, and every metric
name used in alerts and dashboard queries against the API's metric registry.
