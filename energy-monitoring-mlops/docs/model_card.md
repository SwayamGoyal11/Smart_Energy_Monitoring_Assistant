# Model Card: Short-Term Household Appliance Energy Predictor

## 1. Model Details
- **Task:** 10-minute horizon household appliance energy consumption forecasting (Wh).
- **Target Variable:** `Appliances` (Watt-hours per 10-minute interval).
- **Dataset:** UCI Appliances Energy Prediction dataset (19,735 instances, 4.5 months, 2016-01-11 to 2016-05-27).
- **Features:** 32 leak-free features: 11 calendar/cyclical time features, 3 consumption lags, 3 rolling statistics computed on the shifted target, and 14 environmental sensor readings plus `lights`.
- **Validation Protocol:** Chronological 70/15/15 train/validation/test split with no shuffling and no overlapping timestamps (19,729 rows after dropping the 6 warm-up rows needed by lag/rolling features):

  | Split | Rows | Date range |
  |---|---|---|
  | Train | 13,810 | 2016-01-11 18:00 to 2016-04-16 15:30 |
  | Validation | 2,959 | 2016-04-16 15:40 to 2016-05-07 04:40 |
  | Test | 2,960 | 2016-05-07 04:50 to 2016-05-27 18:00 |

- **Architectures Explored:**
  - Naive Persistence Baseline ($\hat{y}_t = y_{t-1}$)
  - Ordinary Least Squares (OLS) Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- **Experiment Tracking:** MLflow (SQLite backend, `mlflow.db`) with parameter and metric versioning; ZenML orchestrates the same steps.

---

## 2. Performance Comparison

Metrics regenerated with the reference environment (Python 3.12, scikit-learn 1.9, XGBoost 3.4). Ranked by test RMSE.

| Model | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ |
|---|---|---|---|---|---|---|
| **Linear Regression** | 28.40 | **60.22** | **0.574** | 27.24 | **60.91** | **0.551** |
| **Persistence Baseline** | **26.15** | 66.44 | 0.481 | **26.75** | 66.85 | 0.459 |
| **Random Forest** | 33.43 | 65.07 | 0.503 | 45.24 | 76.85 | 0.285 |
| **XGBoost** | 37.56 | 68.64 | 0.446 | 60.15 | 90.13 | 0.017 |

MAE and RMSE in Wh per 10-minute interval.

---

## 3. Empirical Analysis

### Linear Regression vs. Persistence
Linear Regression is the only model that beats persistence on RMSE and $R^2$ on both held-out splits (about 9% lower test RMSE). Persistence keeps a slightly lower MAE, so the two models trade off typical-case error against large-error penalties.

### Why Persistence is hard to beat on MAE
1. **Strong Autoregressive Lag-1 Correlation ($r \approx 0.75$):**  
   Household loads are dominated by standby states: the median interval is 60 Wh and about 79% of intervals are at or below 100 Wh. During continuous standby $y_t \approx y_{t-1}$, so persistence makes near-zero absolute error on most intervals.
2. **Short-Lived Impulse Spikes vs. Partition Averages:**  
   Appliance activations produce stochastic, high-amplitude pulses (kettles, ovens, washing cycles). Tree models predict leaf means, over-smoothing peaks and over-predicting descents. Their generalisation gap is large: Random Forest and XGBoost test errors are far worse than their validation errors.
3. **Decoupled Environmental Lag (Thermal Inertia):**  
   Indoor temperature and humidity respond to appliance waste heat with 30 to 90 minute delays, so instantaneous sensor readings give little lead time on discretionary switching.

---

## 4. Operational Limitations
- **Step-Lag Penalty:** Persistence (and, to a lesser degree, the lag-driven Linear Regression) lags regime shifts, predicting a spike one step *after* it begins and staying high one step *after* it ends.
- **Sensor-Only Decoupling:** Environmental sensors alone cannot forecast discretionary human behaviour without consumption history.
- **Seasonal coverage:** Training data spans January to mid-April only; behaviour in summer months is untested.

---

## 5. Release Quality Gates
A candidate must pass `src/models/quality_gates.py` on held-out data before registration or serving:

| Gate | Threshold |
|---|---|
| MAE | ≤ 35.0 Wh |
| RMSE | ≤ 65.0 Wh |
| $R^2$ | ≥ 0.50 |
| RMSE vs. persistence baseline | must not be worse |

Linear Regression passes on both validation and test splits; Random Forest and XGBoost fail on test.

---

## 6. Production Recommendations (Phase 2 & 3)
1. **Two-Stage Regime Switching (Hurdle) Architecture:**
   - **Stage 1 (Classifier):** Predict operational state (Standby vs. Active Burst).
   - **Stage 2 (Forecaster):** Route to Persistence if Standby; route to a dedicated regressor if Active.
2. **Asymmetric Loss Calibration:** Train on Huber loss or asymmetric quantile loss rather than default MSE to prevent excessive smoothing of rare, high-draw spikes.
3. **Longer-horizon lags:** Add lag terms such as $y_{t-6}$ to capture hourly inertia.
