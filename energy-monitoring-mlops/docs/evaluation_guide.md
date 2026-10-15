# Model Training & Evaluation Guide
## AI-Based Smart Energy Monitoring Assistant

---

## 1. Overview & Evaluation Philosophy

The **Smart Energy Monitoring Assistant** forecasts short-term household energy consumption (`Appliances`, in Watt-hours) at **10-minute intervals** using the UCI Appliances Energy Prediction dataset. 

Because energy consumption exhibits strong autoregressive patterns, diurnal periodicity, and abrupt stochastic spikes from appliance activations, standard shuffled cross-validation is strictly forbidden. The system operates under a rigorous **MLOps time-series evaluation protocol**:

1. **Strict Chronological Splitting:** Forward-in-time train, validation, and test splits with zero future leakage.
2. **Causal Feature Engineering:** Rolling windows and lag features strictly derived with backward-looking shifts ($t-1, t-2, t-3$).
3. **Multi-Model Benchmarking:** Candidate machine learning models (Linear Regression, Random Forest, XGBoost) are strictly benchmarked against a **Persistence Baseline** ($y_t = y_{t-1}$).
4. **Regime-Stratified & Diurnal Slicing:** Error is evaluated not just globally, but segmented by operational consumption regimes (Standby, Active, Peak) and temporal windows (Diurnal peaks, Night standby, Weekday vs. Weekend).
5. **Auditable Artifact Generation:** Standardized metrics, residual distributions, time-series trace overlays, and executive evaluation dashboard slides logged to MLflow and saved under `reports/evidence/`.

---

## 2. Evaluation Metrics Framework

### 2.1 Primary Global Metrics

| Metric | Mathematical Definition | Target Unit | Interpretation in Energy Monitoring |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | $\frac{1}{N} \sum_{i=1}^N \|y_i - \hat{y}_i\|$ | Watt-hours ($\text{Wh}$) | Direct expected deviation per 10-minute interval. Primary KPI for household energy budgets. |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$ | Watt-hours ($\text{Wh}$) | Heavily penalizes large forecasting misses during high-draw appliance activations (e.g., ovens, laundry). |
| **Coefficient of Determination ($R^2$)** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Dimensionless ($\le 1.0$) | Percentage of energy consumption variance captured by the model compared to a naive mean predictor. |
| **Weighted Absolute Percentage Error (WAPE)** | $\frac{\sum \|y_i - \hat{y}_i\|}{\sum y_i} \times 100$ | Percentage ($\%$) | Scale-independent percentage error that avoids division-by-zero during low standby draws ($10\text{--}40\text{ Wh}$). |
| **Median Absolute Error (MedAE)** | $\text{median}(\|y_1 - \hat{y}_1\|, \dots, \|y_N - \hat{y}_N\|)$ | Watt-hours ($\text{Wh}$) | Measures typical baseline error during dominant standby states, unaffected by extreme burst outliers. |
| **Mean Prediction Bias** | $\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$ | Watt-hours ($\text{Wh}$) | Indicates whether the model systematically over-predicts ($>0$) or under-predicts ($<0$) consumption. |

---

### 2.2 Stratified Sub-Group Evaluation

Global aggregates can be deceptive in household energy data: because standby states account for $>75\%$ of all timesteps, a model can show a low overall MAE while failing completely during active appliance bursts. 

Therefore, models must be evaluated across three critical slicing dimensions:

#### A. Operational Consumption Regimes
- **Standby Regime ($\le 60\text{ Wh}$):** Continuous baseline power (refrigerators, standby electronics). Accounts for ~76% of observations.
- **Active / Moderate Regime ($60\text{--}200\text{ Wh}$):** Lighting, entertainment, moderate appliance use. Accounts for ~16% of observations.
- **High-Draw Peak Regime ($> 200\text{ Wh}$):** Heavy appliances (ovens, washing machines, microwaves, water kettles). Accounts for ~8% of observations.

#### B. Diurnal Time Windows
- **Night Standby (00:00 – 06:00):** Minimum human activity; tests baseline stability and thermal inertia decay.
- **Morning Peak (06:00 – 09:00):** Wake-up routines, breakfast cooking, rapid lighting changes.
- **Daytime Plateau (09:00 – 17:00):** Working hours; distinguishes weekday vs. weekend occupancy.
- **Evening Peak (17:00 – 22:00):** Highest volatility, dinner preparation, multiple appliances operating concurrently.
- **Late Evening (22:00 – 00:00):** Wind-down transition.

#### C. Calendar & Day-Type
- **Weekday vs. Weekend:** Tests ability to adapt to human behavioral shift without retraining.

---

## 3. Executive Evaluation Dashboard Specification

To provide clear, publication-grade visual reporting for stakeholders, researchers, and automated CI/CD audits, the evaluation suite generates two standardized dashboard visual slides (`1920x1080` high-resolution PNGs).

```
====================================================================================================
SLIDE 1: MODEL TRAINING & EVALUATION - KEY METRICS (EXECUTIVE SUMMARY)
====================================================================================================
+--------------------------------------------------------------------------------------------------+
|                            Model Training & Evaluation - Key Metrics                             |
|                           AI-Based Smart Energy Monitoring Assistant                             |
+---------------------------------+--------------------------------+---------------+---------------+
| [Test MAE]                      | [Test RMSE]                    | [Test R2]     | [Total Data]  |
| 26.75 Wh (Baseline)             | 61.50 Wh (Linear Reg)          | 0.542 (LR)    | 19,735 Steps  |
| 29.24 Wh (Linear Regression)    | 66.85 Wh (Persistence)         | 0.459 (Base)  | 137 Days      |
+---------------------------------+--------------------------------+---------------+---------------+
| Validation vs Test Results      | Chronological Dataset Split    | Pipeline Configuration        |
| - Persistence Baseline          | - Train: 13,809 (70.0%)        | - Horizon: 10-min lead time   |
| - Linear Regression             | - Val:    2,960 (15.0%)        | - Features: 37 total          |
| - Random Forest                 | - Test:   2,960 (15.0%)        |   (Lags, Rolling, Ambient)    |
| - XGBoost Regressor             | - Strict Temporal Order        | - Framework: Scikit / XGBoost |
+---------------------------------+--------------------------------+-------------------------------+
| Training & Validation Loss      | 48-Hour Actual vs Predicted    | Consumption Regime Breakdown  |
| - XGBoost RMSE convergence      | - High-resolution time-series  | - Standby (<=60W) : 76.2%     |
|   over 200 boosting rounds      |   trace overlay showing active | - Active (60-200W): 16.5%     |
|   with early-stopping point     |   peak capture & standby       | - Peak (>200W)    :  7.3%     |
+---------------------------------+--------------------------------+-------------------------------+
| Per-Regime & Diurnal Performance (Test Set MAE in Wh)                                            |
| [Standby: 12.4 Wh]   [Active: 68.2 Wh]   [High Peak: 184.6 Wh]   [Morning: 34.1 Wh] [Night: 16.2 Wh] |
+--------------------------------------------------------------------------------------------------+
```

```
====================================================================================================
SLIDE 2: INITIAL MODEL EVALUATION & DIAGNOSTICS (TECHNICAL AUDIT)
====================================================================================================
+--------------------------------------------------------------------------------------------------+
|                               INITIAL MODEL EVALUATION & DIAGNOSTICS                             |
|                               Error Distributions, Time Traces & Drivers                         |
+----------------------------------------------------+---------------------------------------------+
| Convergence & Error Diagnostics                    | Regime-Stratified Evaluation Table          |
|                                                    | (Regime, Support, Baseline MAE, LR MAE,     |
| 1. Iteration Loss Curve (Train vs Val RMSE)        |  XGBoost MAE, Bias, Peak Error Degradation) |
| 2. Residual Distribution Histogram + KDE           +---------------------------------------------+
|    (Evaluates symmetry, mean bias, fat tails)      | Feature Importance & Sensor Drivers         |
| 3. Residuals vs. Fitted Values Scatter             | - Top 10 Features (Appliances_lag_1,        |
|    (Tests for peak heteroscedasticity)             |   rolling_mean_3, lights, hour, T3, RH_1)   |
+----------------------------------------------------+---------------------------------------------+
| 72-Hour Micro-Trace (Test Set Validation)                                                        |
| - Step-by-step visual overlay: Ground Truth vs Persistence vs Linear Regression vs XGBoost       |
| - Visualizes the 1-Step Persistence Lag Penalty and the Tree Model Peak-Smoothing Effect         |
+--------------------------------------------------------------------------------------------------+
```

---

## 4. Benchmark Models & Baseline Protocol

Every evaluation run must assess candidate models side-by-side:

### 1. Persistence Baseline ($y_t = y_{t-1}$)
- **Role:** Foundational operational baseline. Predicts that the next 10 minutes will equal the current 10-minute consumption (`Appliances_lag_1`).
- **Characteristics:** Achieves very low MAE during standby ($10\text{--}15\text{ Wh}$), but incurs a systematic **step-lag penalty** during transitions (predicts spikes one step late, and stays high one step after the spike ends).

### 2. Ordinary Least Squares (OLS) Linear Regression
- **Role:** Interpretable linear benchmark.
- **Characteristics:** Provides strong regularization across environmental sensors; currently achieves the highest overall test $R^2$ (0.542) and lowest test RMSE (61.50 Wh).

### 3. Random Forest Regressor
- **Role:** Non-linear bagged ensemble (`n_estimators=100, max_depth=15, min_samples_leaf=4`).
- **Characteristics:** Robust on training splits, but prone to over-smoothing high-amplitude impulse peaks on test sets.

### 4. XGBoost Regressor
- **Role:** Gradient boosted decision trees (`n_estimators=200, lr=0.05, max_depth=6, colsample_bytree=0.8`).
- **Characteristics:** Captures non-linear feature interactions; evaluated with early stopping to prevent over-fitting.

---

## 5. Automated Evidence & Artifact Outputs

When the evaluation suite executes, it writes the following structured artifacts to `reports/evidence/`:

```text
reports/evidence/
├── dataset_validation.json              # Data schema, missingness, monotonicity, and raw statistics
├── model_comparison.csv                 # Tabular benchmark (Train/Val/Test MAE, RMSE, R2 for all models)
├── evaluation_metrics_deepdive.json     # Detailed JSON with per-regime, diurnal, and residual metrics
├── slide1_key_metrics.png               # High-res executive summary dashboard (Slide 1)
├── slide2_model_diagnostics.png         # High-res technical diagnostic deep-dive (Slide 2)
└── feature_importance.csv               # Ranked feature attribution scores
```

---

## 6. Implementation Architecture

The evaluation pipeline is modularized into two dedicated components:

```text
energy-monitoring-mlops/
├── src/
│   ├── models/
│   │   ├── evaluate.py                  # Core regression metrics calculation functions
│   │   ├── evaluation_analytics.py      # Regime stratification, diurnal slicing & residual stats
│   │   └── train.py                     # Training runners with MLflow metric logging
│   └── visualization/
│       └── generate_dashboard_slides.py # Programmatic generator for Slide 1 and Slide 2
└── run_pipeline.py                      # Orchestrates validation, feature engineering, training,
                                         # evaluation analytics, and slide generation in one run
```

### Module Responsibilities:

1. **`evaluation_analytics.py`**:
   - Takes model predictions and ground-truth time series.
   - Computes global metrics (MAE, RMSE, $R^2$, WAPE, MedAE).
   - Computes regime-sliced metrics (Standby, Active, Peak).
   - Computes diurnal metrics (Morning Peak, Evening Peak, Night Standby).
   - Computes residual diagnostics (Skewness, Kurtosis, Mean signed bias).
   - Exports `evaluation_metrics_deepdive.json`.

2. **`generate_dashboard_slides.py`**:
   - Renders **Slide 1** and **Slide 2** matching the clean dark-green theme (`#134e4a`, `#0f766e`, `#f0fdf4`) using `matplotlib` and `seaborn`.
   - Embeds KPI cards, comparison tables, chronological partition timeline, actual vs. predicted 48h/72h traces, and per-regime error bars.
   - Saves figures directly to `reports/evidence/` and logs them as artifacts in MLflow.

---

## 7. Execution Commands

Once implemented, the complete evaluation suite can be run directly:

```powershell
# Navigate to the MLOps workspace
cd energy-monitoring-mlops

# Run the complete end-to-end pipeline (data -> features -> train -> evaluate -> slides):
python run_pipeline.py

# Or run the evaluation dashboard generator standalone:
python -m src.visualization.generate_dashboard_slides

# Launch the MLflow UI to inspect logged metrics and slide artifacts:
python -m mlflow ui
```
