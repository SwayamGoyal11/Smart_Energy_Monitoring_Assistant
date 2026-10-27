# syntax=docker/dockerfile:1
# ============================================================
# Smart Energy Monitoring Assistant - inference API image
#
#   python scripts/export_champion.py      # models/champion (mounted at runtime)
#   docker build -t energy-forecaster-api .
#
# The model is not baked in: compose mounts ./models/champion read-only at
# /app/models/champion, so a new champion ships without rebuilding.
# ============================================================

ARG PYTHON_VERSION=3.12

# ---------- Stage 1: build the virtualenv ----------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN uv venv /opt/venv && \
    uv pip install --no-cache --python /opt/venv/bin/python -r /tmp/requirements-serve.txt

# ---------- Stage 2: minimal runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="energy-forecaster-api" \
      org.opencontainers.image.description="FastAPI service for 10-minute appliance energy forecasts" \
      org.opencontainers.image.source="https://github.com/SwayamGoyal11/Smart_Energy_Monitoring_Assistant"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_LOCAL_PATH=/app/models/champion \
    DRIFT_METRICS_PATH=/app/reports/evidence/drift_metrics.json \
    INFERENCE_LOG_PATH=/app/data/monitoring/inference_log.csv \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/

# Writable paths for the non-root user: the app creates models/ and
# reports/evidence/ on import, and appends the inference log under data/.
RUN mkdir -p models/champion reports/evidence data/monitoring \
    && chown -R app:app models reports data

USER app

EXPOSE 8000

# slim images ship without curl; urlopen raises on non-2xx, failing the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# One worker: Prometheus counters live in process memory, and several
# workers would each expose a partial view on /metrics.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
