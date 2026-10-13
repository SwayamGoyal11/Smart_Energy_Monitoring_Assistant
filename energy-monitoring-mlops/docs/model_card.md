# Model Card: Short-Term Household Appliance Energy Predictor

## 1. Model Details
- **Task:** 10-minute horizon household appliance energy consumption forecasting (Wh).
- **Target Variable:** `Appliances` (Watt-hours per 10-minute interval).
- **Dataset:** UCI Appliances Energy Prediction dataset (19,735 instances, 4.5 months).
- **Validation Protocol:** Chronological temporal train/test split (80/20, zero forward leakage).
- **Architectures Explored:**
  - Naive Persistence Baseline ($y_t = y_{t-1}$)
  - Ordinary Least Squares (OLS) Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- **Experiment Tracking:** MLflow (`mlruns/`) with parameter and metric versioning.

---

## 2. Performance Comparison

| Model | Test MAE (Wh) | Test RMSE (Wh) | $R^2$ Score | Key Characteristic |
|---|---|---|---|---|
| **Persistence Baseline** | **26.75** | **58.20** | **0.68** | Direct temporal shift ($y_t = y_{t-1}$) |
| **Random Forest** | 45.60 | 82.40 | 0.36 | Ensemble leaf-mean estimation |
| **Linear Regression** | 53.12 | 89.44 | 0.24 | Hyperplane fitting over sensors |
| **XGBoost** | 60.84 | 93.15 | 0.18 | Gradient boosted trees |

---

## 3. The Performance Gap: Empirical Analysis

### Why Persistence Outperforms Complex Ensembles
1. **Strong Autoregressive Lag-1 Correlation ($r \approx 0.83$):**  
   Household loads are dominated by appliance standby states (40–60 Wh). During continuous standby, $y_t \approx y_{t-1}$, incurring an absolute error near zero across over $75\%$ of recorded intervals.
2. **Short-Lived Impulse Spikes vs. Partition Averages:**  
   Appliance activations produce stochastic, high-amplitude pulse patterns (kettles, ovens, washing cycles). Tree models split feature space on environmental sensors ($T_1$–$T_9$, $RH_1$–$RH_9$) and predict leaf means. During spike tails, trees over-smooth peaks and over-predict descents, heavily penalizing MAE and RMSE.
3. **Decoupled Environmental Lag (Thermal Inertia):**  
   Indoor temperature and humidity respond gradually to appliance waste heat with physical delays of 30 to 90 minutes. Instantaneous sensor readings at time $t$ lack the predictive lead time needed to anticipate on-demand appliance switching.

---

## 4. Operational Limitations
- **Step-Lag Penalty:** While achieving lower MAE, persistence systematically lags during regime shifts—predicting a high spike one step *after* it begins, and staying high one step *after* it concludes.
- **Sensor-Only Decoupling:** Environmental sensors alone cannot forecast discretionary human behavior without historical consumption memory.

---

## 5. Production Recommendations (Phase 2 & 3)
1. **Autoregressive Feature Ingestion:** Tree models must ingest lag terms ($y_{t-1}, y_{t-2}, y_{t-6}$) and rolling consumption windows to match baseline inertia.
2. **Two-Stage Regime Switching Architecture:**
   - **Stage 1 (Classifier):** Predict operational state (Standby vs. Active Burst).
   - **Stage 2 (Forecaster):** Route to Persistence if Standby; route to Gradient Boosting if Active.
3. **Asymmetric Loss Calibration:** Train on Huber loss or asymmetric quantile loss rather than default MSE to prevent excessive smoothing of rare, high-draw spikes.