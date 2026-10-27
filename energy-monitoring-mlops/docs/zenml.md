# ZenML pipeline

The ZenML pipeline is implemented in `src/pipelines/energy_training.py`. It
uses the existing DVC-materialized CSV as a read-only input and preserves the
same feature functions and strict chronological 70/15/15 split used by tests.
ZenML stores pipeline artifacts independently; it does not replace DVC.

## One-time local setup

From `energy-monitoring-mlops`, activate the virtual environment and run:

```powershell
.\venv\Scripts\Activate.ps1
.\scripts\setup_zenml.ps1
```

The script installs ZenML and its server extras, initializes the repository,
installs the MLflow integration, and registers the active stack. It expands the
requested commands with `--tracking_uri` so `mlflow_tracker` writes to this
project's existing `mlflow.db`:

```powershell
zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri="sqlite:///C:/absolute/path/energy-monitoring-mlops/mlflow.db"
zenml stack register energy_stack -a default -o default -e mlflow_tracker --set
```

For a fresh virtual environment, `pip install -r requirements.txt` also installs
the pinned ZenML server/MLflow extras.

## Run and inspect

```powershell
dvc pull
python run_zenml_pipeline.py --open-mlflow
```

`--open-mlflow` starts the MLflow UI at `http://127.0.0.1:5000` using the
existing SQLite backend. To host the optional local ZenML dashboard, run
`zenml up` in a second terminal and use the URL it prints.

The training step logs split boundaries, all model hyperparameters, and native
MLflow model artifacts. The evaluation step logs MAE, RMSE, and R² for both
validation and test partitions. Linear Regression is logged as
`linear_regression_candidate` for the project’s RMSE/R² selection workflow.
