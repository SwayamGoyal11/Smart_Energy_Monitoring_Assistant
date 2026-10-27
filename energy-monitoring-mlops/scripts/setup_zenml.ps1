<#
.SYNOPSIS
Installs and configures the local ZenML + MLflow development stack.

.DESCRIPTION
Run from the energy-monitoring-mlops directory after activating the project's
virtual environment. The script does not alter DVC files or datasets.
#>

$ErrorActionPreference = "Stop"
$projectPath = (Resolve-Path ".").Path.Replace("\", "/")
$mlflowUri = "sqlite:///$projectPath/mlflow.db"

python -m pip install zenml "zenml[server]"
zenml init
zenml integration install mlflow -y

# The requested component/stack names make pipeline decorators portable.
zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri=$mlflowUri
zenml stack register energy_stack -a default -o default -e mlflow_tracker --set

Write-Host "ZenML stack 'energy_stack' is active. MLflow tracking URI: $mlflowUri"
Write-Host "Run: python run_zenml_pipeline.py --open-mlflow"
Write-Host "For the ZenML dashboard, run: zenml up"
