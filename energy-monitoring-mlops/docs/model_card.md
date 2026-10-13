# Model Card: Short-Term Household Appliance Energy Predictor

## 1. Model Details
- **Task:** 10-minute horizon appliance energy consumption forecasting (Wh).
- **Architecture Explored:**
  - Naive Persistence Baseline ($y_t = y_{t-1}$)
  - Ordinary Least Squares (OLS) Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- **Experiment Tracking:** MLflow (`mlruns/`)
- **Dataset:** UCI Appliances Energy Prediction dataset (19,735 instances, 10-min frequency).

---

## 2. Performance Comparison

| Model | Test MAE (Wh) | Test RMSE (Wh) | $R^2$ Score | Key Characteristic |
|---|---|---|---|---|
| **Persistence Baseline** | **26.75** | **58.20** | **0.68** | Direct temporal copy ($y_t = y_{t-1}$) |
| **Random Forest** | 45.60 | 82.40 | 0.36 | Ensemble leaf-mean estimation |
| **XGBoost** | 60.84 | 93.15 | 0.18 | Gradient boosted trees |
| **Linear Regression** | 53.12 | 89.44 | 0.24 | Hyperplane fitting over sensors |

---

## 3. The Performance Gap: Empirical Analysis

### Why Persistence Outperforms Complex Ensembles
1. **Strong Autoregressive Lag-1 Correlation ($r \approx 0.83$):**
   The time series is dominated by appliance standby states (40–60 Wh). During continuous standby, $y_t \approx y_{t-1}$, giving persistence an error near zero for over $75\%$ of recorded intervals.
2. **Short-Lived Impulse Spikes vs. Partition Averages:**
   Appliance usage exhibits stochastic pulse patterns (e.g., microwave, kettle, washing machine cycle). Decision trees split on environmental sensor features ($T_1$-$T_9$, $RH_1$-$RH_9$) and predict the leaf mean. During the onset and recovery tails of an impulse spike, trees over-smooth the peak and over-predict the descent, inflating both MAE and RMSE.
3. **Decoupled Environmental Lag:**
   Indoor temperature and humidity respond gradually to appliance heat output with thermal inertia delays of 30 to 90 minutes. Instantaneous sensor readings at time $t$ do not contain sufficient lead information to anticipate on-demand appliance activations.

---

## 4. Production Recommendations (Phase 2 & 3)
1. **Autoregressive Integration:** Tree-based models must ingest explicit lag terms ($y_{t-1}, y_{t-2}, y_{t-6}$) and rolling standard deviations to capture state persistence.
2. **Regime Switching / Two-Stage Architecture:**
   - **Stage 1 (Classifier):** Predict binary regime (Standby vs. Active spike).
   - **Stage 2 (Forecaster):** If Standby, default to persistence; if Active, invoke gradient boosting conditioned on recent cumulative demand.
3. **Loss Function Calibration:** Optimize using asymmetric quantile loss or Huber loss rather than default MSE to penalize persistence lags during transitions without over-smoothing standby intervals.