# Optional ZenML Hurdle wrapper

The new entry point is `run_zenml.py`; the previous entry points remain intact.
The DAG is `load_data -> preprocess -> train_hurdle -> log_and_register`.
Named `Annotated` outputs track raw data, chronological splits, a skops model,
fit time, training run ID, and a JSON-compatible registration receipt.

`register_candidate()` already owns evaluation, artifact logging, model logging,
version creation, and gated alias assignment. Those operations intentionally stay
together in one step. Splitting them would require changing the existing helper
or duplicating side effects. A passing candidate becomes champion only if its
validation RMSE beats the current champion, following the existing policy.
Failed candidates remain registered and auditable; the wrapper fails the run.

Training uses `experiment_tracker="mlflow_tracker"`. Registration has no tracker
because its helper starts a non-nested MLflow run. Its run receives the training
run's `mlflow.parentRunId`; the training run receives registry run/version tags.
The ZenML tracker links the training run, and MLflow shows the registry child.
Historical runs are retained; each invocation creates new runs and a version.
The registry's artifact_logging.py and skops model logging are called unchanged.
ZenML's intermediate model uses an explicit skops materializer with the registry's
trust allowlist, rather than cloudpickle. DataFrames use ZenML's default materializer.
All caching is disabled because CSV input and registry side effects are mutable.

## Windows PowerShell setup

Run from `energy-monitoring-mlops`. These commands target the existing Python 3.12
`venv`; they do not replace the environment or edit requirements. Install uv first
if it is not already on PATH. Review dependency resolution for your environment.

```powershell
python -m pip install uv
$env:VIRTUAL_ENV = (Resolve-Path .\venv).Path
$env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
# This checkout's verified store lives here (including energy_mlflow_stack).
$env:ZENML_CONFIG_PATH = Join-Path $PWD '.zen/wrapper-validation'
$env:PYTHONIOENCODING = 'utf-8'
uv pip install --python .\venv\Scripts\python.exe "zenml[server,mlflow]>=0.98.0,<1.0.0"
uv run --active --no-project zenml integration install mlflow --uv -y

# Already initialized in this checkout; initialize only a fresh checkout.
if (!(Test-Path .zen)) { uv run --active --no-project zenml init }

# Use exactly the same SQLite URI representation as the existing project.
$env:MLFLOW_TRACKING_URI = (& .\venv\Scripts\python.exe -c 'from src.models.registry import tracking_uri; print(tracking_uri())').Trim()
uv run --active --no-project zenml experiment-tracker register mlflow_tracker --flavor=mlflow "--tracking_uri=$env:MLFLOW_TRACKING_URI"
uv run --active --no-project zenml stack register energy_mlflow_stack -a default -o default -e mlflow_tracker --set

# Run dvc pull first if the DVC-managed CSV is absent.
uv run --active --no-project python run_zenml.py

# Legacy UI entry point (deprecated in the installed ZenML version):
uv run --active --no-project zenml up
# Current equivalent:
uv run --active --no-project zenml login --local

# Separate terminal, same environment and project directory:
uv run --active --no-project mlflow ui --backend-store-uri "$env:MLFLOW_TRACKING_URI" --port 5000
```

Register components/stacks once. If names already exist, inspect them with
`zenml experiment-tracker describe mlflow_tracker` and `zenml stack describe
energy_mlflow_stack`; select the existing correct stack with `zenml stack set
energy_mlflow_stack`. The runner rejects a tracker pointing at a different store.
No ZenML model-registry component is needed: the existing MLflow registry remains
the authority. ZenML artifact storage is separate from existing MLflow artifacts.

ZenML officially supports Windows via WSL. Native Windows daemon startup is not
supported; the local CLI explicitly warns about it. For a supported local UI,
use a separate Python 3.12 environment inside WSL and run this whole workflow
there. Do not reuse a Windows virtualenv inside WSL. From PowerShell:

```powershell
wsl
```

Then in the WSL shell (uv must be installed there):

```bash
cd /mnt/c/Users/ASUS/Smart_Energy_Monitoring_Assistant/energy-monitoring-mlops
uv venv --python 3.12 .venv-wsl
source .venv-wsl/bin/activate
uv pip install -r requirements.txt
zenml integration install mlflow --uv -y
zenml init  # only if this repository is not already initialized
export MLFLOW_TRACKING_URI="$(python -c 'from src.models.registry import tracking_uri; print(tracking_uri())')"
zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri="$MLFLOW_TRACKING_URI"
zenml stack register energy_mlflow_stack -a default -o default -e mlflow_tracker --set
zenml login --local
python run_zenml.py
```

Use WSL-specific ZenML stack configuration and verify pre-existing MLflow artifact
URIs are accessible from WSL before switching runtimes. Windows absolute artifact
paths do not automatically become Linux paths. Avoid concurrent registry writers.

The existing project `.gitignore` already includes `.zen/`. Keep that rule; add
`.venv-wsl/` if creating the optional WSL environment in the repository.

References: https://docs.zenml.io/stacks/stack-components/experiment-trackers/mlflow,
https://docs.zenml.io/getting-started/installation,
https://docs.zenml.io/reference/faq.

## Verified Docker dashboard on this Windows checkout

The dashboard runs at http://localhost:8237 in `energy-zenml-dashboard`, using
`zenmldocker/zenml-server:0.98.0` and the existing pipeline SQLite database.
It binds only to the local machine and restarts when Docker Desktop restarts.

The requested `zenml login --local --docker --port 8237` was attempted with
`ZENML_CONFIG_PATH` set to `.zen/wrapper-validation` and `DOCKER_HOST` set to
`npipe:////./pipe/dockerDesktopLinuxEngine`. ZenML 0.98.0 constructed the Linux
entrypoint path as `/service\service.json`, causing the container to exit.
The working fallback uses the official image directly with Linux paths:

```powershell
# Run from energy-monitoring-mlops. Execute once to create the container.
$configRoot = (Resolve-Path .zen/wrapper-validation).Path
$serverConfig = Join-Path $PWD '.zen/docker-server-config'
New-Item -ItemType Directory -Force -Path $serverConfig | Out-Null
docker run -d --name energy-zenml-dashboard --restart unless-stopped `
  -p 127.0.0.1:8237:8080 `
  --mount "type=bind,source=$configRoot,target=/zenconfig" `
  --mount "type=bind,source=$serverConfig,target=/serverconfig" `
  -e ZENML_CONFIG_PATH=/serverconfig `
  -e ZENML_STORE_TYPE=sql `
  -e ZENML_STORE_URL=sqlite:////zenconfig/local_stores/default_zen_store/zenml.db `
  -e ZENML_STORE_DATABASE=/zenconfig/local_stores/default_zen_store/zenml.db `
  -e ZENML_SERVER_AUTH_SCHEME=NO_AUTH `
  -e ZENML_SERVER_ALLOW_LOCAL_FILE_ACCESS=true `
  -e ZENML_DISABLE_DATABASE_MIGRATION=true `
  -e ZENML_SERVER_AUTO_ACTIVATE=true `
  zenmldocker/zenml-server:0.98.0

# Check the existing container; start it again if it was stopped.
docker ps --filter name=energy-zenml-dashboard
docker start energy-zenml-dashboard
Invoke-WebRequest http://localhost:8237/health -UseBasicParsing
```

In the UI, select project `default`, open Pipelines, then
`energy_hurdle_pipeline`, and select completed run
`energy_hurdle_pipeline-2026_11_12-13_04_00_016944`
(`9c09347c-059b-4333-a195-14007109586b`). Its DAG API was verified to expose
all four completed steps and these output artifact nodes:

| Step | Outputs |
| --- | --- |
| load_data | raw_data |
| preprocess | train_df, val_df, test_df |
| train_hurdle | hurdle_model, fit_seconds, training_run_id |
| log_and_register | registration_result |

Click an artifact node or a step's outputs to inspect its type, version, and
lineage. Artifact records are verified through the server; rendering file previews
from their original Windows paths inside the Linux container is not verified.
The container shares the existing store; it does not migrate artifact paths or
change the Windows training environment.
