# Smart Energy Monitoring Assistant — MLOps Pipeline

> Short-term household appliance energy demand forecasting using strict time-series MLOps practices.

---

## Overview

This module implements an end-to-end ML pipeline for forecasting household appliance energy consumption (`Appliances`, Wh) at **10-minute intervals** using the [UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction).

**Dataset:** 19,735 readings · 137 days (Jan–May 2016) · 29 raw features

---

## Pipeline Steps

```
run_pipeline.py
│
├── [1] Load & Validate Dataset         → reports/evidence/dataset_validation.json
├── [2] Feature Engineering             → data/processed/features.csv
├── [3] Chronological Train/Val/Test Split
│         Train : 70%  (13,810 rows)
│         Val   : 15%  ( 2,959 rows)
│         Test  : 15%  ( 2,960 rows)
├── [4] MLflow Experiment Setup
├── [5] Train 4 Models
│         - Persistence Baseline
│         - Linear Regression (OLS)
│         - Random Forest
│         - XGBoost
├── [6] Model Comparison Table          → reports/evidence/model_comparison.csv
├── [7] Deep-Dive Evaluation Analytics  → reports/evidence/evaluation_metrics_deepdive.json
└── [8] Dashboard Slide Generation      → reports/evidence/slide1_key_metrics.png
                                           reports/evidence/slide2_model_diagnostics.png
```

---

## Model Results (Test Set)

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Persistence Baseline | 26.75 Wh | 66.85 Wh | 0.459 |
| **Linear Regression** ⭐ | **29.24 Wh** | **61.50 Wh** | **0.542** |
| Random Forest | 45.60 Wh | 77.03 Wh | 0.282 |
| XGBoost | 60.97 Wh | 87.79 Wh | 0.067 |

> Linear Regression achieves the best generalisation (R² = 0.542). Tree-based models overfit the training distribution and underperform on the held-out test window.

---

## Project Structure

```
energy-monitoring-mlops/
├── data/
│   └── raw/energydata_complete.csv.dvc   # DVC-tracked dataset pointer
├── src/
│   ├── data/
│   │   └── load_data.py                  # Data loading & schema validation
│   ├── features/
│   │   └── feature_engineering.py        # Lag features, rolling stats, time features
│   ├── models/
│   │   ├── evaluate.py                   # Core metric functions (MAE, RMSE, R²)
│   │   ├── evaluation_analytics.py       # Regime & diurnal stratified analytics
│   │   └── train.py                      # Training runners for all 4 models
│   └── visualization/
│       └── generate_dashboard_slides.py  # PPT-ready evaluation slide generator
├── tests/
│   ├── test_data.py                      # 28 data pipeline tests
│   └── test_evaluation.py               # 8 evaluation analytics tests
├── docs/
│   └── evaluation_guide.md              # Evaluation methodology & metric definitions
├── reports/evidence/                    # Generated outputs (gitignored)
├── run_pipeline.py                      # Main pipeline entry point
└── requirements.txt
```

---

## Quickstart

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Navigate into the MLOps workspace
cd energy-monitoring-mlops

# 3. Run the full pipeline (data → features → train → evaluate → slides)
python run_pipeline.py

# 4. Regenerate slides only (if models are already trained)
python -m src.visualization.generate_dashboard_slides

# 5. Run all tests
python -m pytest tests/ -v
```

---

## Evaluation Philosophy

- **Strict chronological splitting** — no shuffling, no future leakage
- **Persistence Baseline** as the minimum bar every model must beat
- **Regime-stratified metrics** — Standby (≤60 Wh), Active (60–200 Wh), Peak (>200 Wh)
- **Diurnal segmentation** — Night, Morning, Daytime, Evening, Late Night
- **MLflow** for experiment tracking and metric logging

---

## Key Findings

1. **Linear Regression beats tree models** on the test set because its learned linear weights generalise better than memorised tree splits across a different month.
2. **Persistence Baseline is strong** (MAE 26.75 Wh) because ~76% of readings are in "standby" state (≤60 Wh) where energy barely changes between 10-min intervals.
3. **Peak regime is hardest** — all models show MAE > 100 Wh on High Peak (>200 Wh) readings due to unpredictable appliance activation bursts.
