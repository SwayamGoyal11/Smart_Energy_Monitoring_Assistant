# AI-Based Smart Energy Monitoring Assistant

An end-to-end MLOps time-series pipeline for forecasting short-term household energy demand (Wh) at 10-minute intervals using the UCI Appliances Energy Prediction dataset.

The system enforces strict temporal evaluation guarantees: leak-free windowing, strict chronological partitioning, verifiable data versioning via DVC, and automated MLflow experiment auditing.

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