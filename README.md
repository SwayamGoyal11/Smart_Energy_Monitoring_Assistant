# AI-Based Smart Energy Monitoring Assistant

An end-to-end MLOps time-series pipeline for forecasting short-term household energy demand (Wh) at 10-minute intervals using the UCI Appliances Energy Prediction dataset.

The system enforces strict temporal evaluation guarantees: leak-free windowing, strict chronological partitioning, verifiable data versioning via DVC, automated MLflow experiment auditing, and production-grade containerized deployment.

---

## Architecture & System Design

```text
├── data/
│   ├── raw/energydata_complete.csv.dvc   # DVC-tracked raw sensor readings
│   └── processed/features.csv            # Leak-free engineered feature matrix
├── notebooks/
│   └── 01_eda.ipynb                      # Distribution, diurnal cycles, ACF/PACF
├── src/
│   ├── features/                         # Forward-only lags & rolling stats
│   └── models/                           # Baseline, OLS, RF, XGBoost runners
├── reports/
│   └── evidence/                         # Validation schemas and benchmark CSVs
├── docs/
│   └── model_card.md                     # Empirical evaluation & gap analysis
└── tests/                                # PyTest suite for schema, leaks & logic
```

---

## Documentation

Detailed architectural, mathematical, and risk analyses are maintained in the project's formal model card:

- 📄 **[Formal Model Card](energy-monitoring-mlops/docs/model_card.md)**:
  - **Temporal Validation Protocol**: Chronological 70/15/15 split mechanics (Train: 13,810 | Val: 2,959 | Test: 2,960 rows) with zero forward look-ahead leakage.
  - **Empirical Gap Analysis**: Detailed breakdown of the $r \approx 0.83$ autoregressive lag-1 correlation, 54.3% standby concentration ($\le 60\text{ Wh}$), and thermal inertia delays ($30\text{--}90$ minutes).
  - **Failure Modes & Benchmarks**: Persistence step-lag penalties, tree-based overfitting (XGBoost test $R^2 = 0.0290$), and Linear Regression metric divergence (val $R^2 = 0.5664$, test RMSE = 61.50 Wh vs. MAE lag).
  - **Production Architecture**: Recommendations for two-stage hurdle regime-switching (Standby Classification vs. Active Burst Regression) and asymmetric loss calibration.

---

## Key Pipeline Principles

1. **Strict Chronological Splitting:** Random cross-validation leaks forward-looking temporal information. All training, validation, and testing steps preserve strict temporal ordering with forward-only validation.
2. **Causal Feature Engineering:** Rolling statistics (`mean_3`, `mean_6`, `std_6`) and lag features (`lag_1`, `lag_2`, `lag_3`) are derived strictly from backward-looking windows to prevent look-ahead bias at inference time.
3. **Reproducibility Stack:**
   - **Data Versioning:** DVC manages the 19,735-record dataset with an external remote cache.
   - **Artifact & Metric Auditing:** MLflow captures parameters, loss curves, serialized `.joblib` pipelines, and SQLite run logs.
   - **Quality Testing:** PyTest suite validates schema constraints, column monotonicity, NaN bounds, and baseline integrity.

---

## Benchmark Results

*Evaluated on the UCI Appliances Energy Prediction dataset across chronological splits (Generated: 2026-10-13).*

| Model | Val MAE (Wh) | Val RMSE (Wh) | Val R2 | Test MAE (Wh) | Test RMSE (Wh) | Test R2 | Operational Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **Persistence Baseline** | **26.15** | 66.44 | 0.481 | **26.75** | 66.85 | 0.459 | y_t = y_{t-1}; exploits standby continuity |
| **Linear Regression** | 29.10 | **60.74** | **0.566** | 29.24 | **61.50** | **0.542** | Stable linear hyperplane over sensors |
| **Random Forest** | 33.34 | 64.82 | 0.506 | 45.60 | 77.03 | 0.282 | Over-smooths impulse peaks; test drift |
| **XGBoost** | 35.86 | 66.89 | 0.474 | 60.84 | 89.56 | 0.029 | Heavily penalizes rapid spike descent |

### Why Does Persistence Beat Tree Ensembles?
Nonlinear models (Random Forest and XGBoost) fail to beat a simple persistence baseline on MAE for three structural reasons:

- **Autoregressive Baseline Concentration:** Base standby consumption represents **54.3%** of all 10-minute intervals operating strictly at or below median idle power ($\le 60\text{ Wh}$), with over **75.5%** concentrated below $100\text{ Wh}$. During these dominant idle periods, the series exhibits severe autocorrelation ($r \approx 0.83$ at lag-1), allowing the trivial persistence heuristic ($y_t = y_{t-1}$) to achieve near-zero absolute error and an artificially superior MAE ($26.75\text{ Wh}$).
- **Impulse Asymmetry & Tree Averaging:** Active burst events (spikes reaching $200\text{--}1,080\text{ Wh}$) occur as sparse, short-duration step functions. Standard tree models partition continuous feature space into leaf-mean estimates, clipping peak crests while systematically over-predicting the steep decay tails—causing XGBoost test performance to collapse ($R^2 = 0.0290$). Conversely, Linear Regression balances variance better against extreme quadratic penalties, securing the lowest test RMSE ($61.50\text{ Wh}$) and highest test $R^2$ ($0.5421$).
- **Thermal Inertia & Sensor Decoupling:** Environmental signals ($T_1\text{--}T_9$, $RH_1\text{--}RH_9$) lag appliance joule heating and thermal dissipation by 30–90 minutes. Instantaneous ambient sensors cannot anticipate human-driven activation triggers without explicit autoregressive lookback windows, directly justifying a two-stage hurdle architecture (idle classification followed by burst-state regression).

*For complete mathematical and architectural derivations, read the [Formal Model Card](energy-monitoring-mlops/docs/model_card.md).*

---

## Quickstart

### 1. Environment Setup
```powershell
git clone [https://github.com/SwayamGoyal11/Smart_Energy_Monitoring_Assistant.git](https://github.com/SwayamGoyal11/Smart_Energy_Monitoring_Assistant.git)
cd Smart_Energy_Monitoring_Assistant/energy-monitoring-mlops
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify Data Tracking
```powershell
python -m dvc status
# To pull the dataset from the configured remote cache:
python -m dvc pull
```

### 3. Run Pipeline & Tests
```powershell
# Execute full feature engineering and model training lifecycle:
python run_pipeline.py

# Execute test suite:
python -m pytest tests -v
```

### 4. Review Experiments & EDA
```powershell
# Launch MLflow tracking dashboard:
python -m mlflow ui

# Interactive analysis:
jupyter notebook notebooks/01_eda.ipynb
```

---

## Project Roadmap & Phase Status

### Phase 1 – Development & Reproducibility (Complete)
- [x] **Dataset Acquisition & Approval:** Ingestion of the 19,735-record UCI Appliances Energy dataset.
- [x] **Data Exploration:** Diurnal consumption cycle analysis, ACF/PACF plots, and sensor correlations (`notebooks/01_eda.ipynb`).
- [x] **Feature Engineering:** Leak-free temporal rolling windows (`mean_3`, `mean_6`, `std_6`) and lag transformations (`lag_1`–`lag_3`).
- [x] **Baseline Models:** Persistence ($y_t = y_{t-1}$), OLS Linear Regression, Random Forest, and XGBoost runners.
- [x] **Git Repository:** Clean multi-directory project structure under Git version control.
- [x] **DVC Dataset Versioning:** Remote cache setup and tracking for raw dataset hashes (`.dvc`).
- [x] **MLflow Experiment Tracking:** Logging metrics, parameters, and model artifacts with SQLite tracking backends.
- [x] **Initial Model Evaluation & Documentation:** Comprehensive analysis documented in [energy-monitoring-mlops/docs/model_card.md](energy-monitoring-mlops/docs/model_card.md).

### Phase 2 – Productionization & Deployment (In Progress)
- [ ] **Model Testing with PyTest:** Schema boundary assertions, negative consumption guards, and chronological leak checks.
- [ ] **Model-Risk & Quality Checks:** Performance regression monitoring against the persistence baseline.
- [ ] **MLflow Model Registry:** Model versioning, staging tags (`Staging` / `Production`), and serialized artifacts.
- [ ] **FastAPI Inference Service:** REST API with Pydantic request/response contract validation.
- [ ] **Docker Containerization:** Multi-stage lightweight Docker image for portable inference.
- [ ] **GitHub Actions CI/CD:** Automated testing, linting, and build validation on push/PR.
- [ ] **Deployment:** Cloud/container orchestration and endpoint verification.