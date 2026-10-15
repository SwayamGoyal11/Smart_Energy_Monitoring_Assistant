# Model Card: Short-Term Household Appliance Energy Predictor

## 1. Model Details
- **Task:** 10-minute horizon household appliance energy consumption forecasting (Wh).
- **Target Variable:** `Appliances` (Watt-hours per 10-minute interval).
- **Dataset:** UCI Appliances Energy Prediction dataset (19,735 instances, 137 days, Jan–May 2016).
  - Source: Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Energy and Buildings*, 140, 81–97.
- **Validation Protocol:** Strict chronological 70 / 15 / 15 split (train / validation / test).
  - Train:  13,810 rows — 2016-01-11 to 2016-04-16
  - Val:     2,959 rows — 2016-04-16 to 2016-05-07
  - Test:    2,960 rows — 2016-05-07 to 2016-05-27
  - No random shuffling. Zero forward leakage.
- **Architectures Evaluated:**
  - Naïve Persistence Baseline (ŷ_t = y_{t-1})
  - Ordinary Least Squares (OLS) Linear Regression
  - Random Forest Regressor (n_estimators=200, max_depth=10, random_state=42)
  - XGBoost Regressor (n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)
- **Experiment Tracking:** MLflow (SQLite backend, `mlflow.db`) — parameters, metrics, and model artefacts logged per run.
- **Primary Metric:** MAE (Mean Absolute Error, Wh) — chosen over RMSE due to right-skewed target distribution.

---

## 2. Performance Benchmark

All metrics verified against MLflow run logs from `run_pipeline.py`.

| Model | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² | Val→Test MAE gap |
|---|---|---|---|---|---|---|---|
| **Persistence Baseline** | 26.15 | 66.44 | 0.481 | **26.75** | 66.85 | 0.459 | +0.60 |
| **Linear Regression** | 29.10 | 60.74 | 0.566 | 29.24 | 61.50 | 0.542 | **+0.14** |
| **Random Forest** | 33.29 | 64.82 | 0.506 | 45.56 | 77.02 | 0.282 | +12.27 |
| **XGBoost** | 34.25 | 65.14 | 0.502 | 57.10 | 84.52 | 0.135 | +22.85 |

> All values in Wh. Lower MAE / RMSE is better. Higher R² is better.

---

## 3. Key Findings

### Finding 1 — Persistence outperforms all learned models on Test MAE
The naïve persistence model achieves test MAE of 26.75 Wh, outperforming all learned models. This is explained by the strong autoregressive structure (lag-1 autocorrelation r ≈ 0.83) and the dominant standby regime (54.4% of readings ≤60 Wh), where consecutive readings differ by less than 5 Wh. Persistence is near-optimal in this regime.

### Finding 2 — Linear Regression is the most stable learned model
Linear Regression achieves a val→test MAE gap of only 0.14 Wh (29.10 → 29.24), indicating near-perfect generalisation. It outperforms persistence on RMSE and R² while remaining close to persistence on MAE. Its stability is attributable to low model variance with a modest, well-specified feature set.

### Finding 3 — Tree models overfit significantly to the training period
Random Forest shows a val→test gap of +12.27 Wh (33.29 → 45.56). XGBoost shows a gap of +22.85 Wh (34.25 → 57.10). Both models compete with Linear Regression on validation but degrade substantially on test. The likely cause is seasonal distribution shift — the training period (Jan–mid-April) has higher variance and more frequent spikes than the test period (May). Tree models learn threshold combinations specific to the training period that do not transfer.

### Finding 4 — Current feature set lacks autoregressive lag features
The current feature matrix includes temporal features and environmental sensors but **no lag features of the target** (y_{t-1}, y_{t-2}, y_{t-3}). Without direct access to recent consumption history — the dominant signal at r ≈ 0.83 — learned models cannot match persistence. This is the primary performance bottleneck for Phase 1.

---

## 4. Operational Limitations
- **Feature set excludes autoregressive lags:** Tree and regression models predict from sensor state only. Without lag features, they cannot exploit the autoregressive structure. Adding lag features is the highest-priority Phase 2 task.
- **Step-lag penalty for persistence:** Persistence lags by one step at every regime transition — predicting the spike one interval after it begins and staying elevated one interval after it ends.
- **Seasonal generalisation gap:** The collection window (Jan–May) excludes summer load profiles. Models trained on this data will not generalise to July–August behaviour.
- **Single household:** Results apply to one specific low-energy house in Belgium. Cross-household generalisation is not established.

---

## 5. Production Recommendations (Phase 2)
1. **Add autoregressive lag features (highest priority):** Include y_{t-1}, y_{t-2}, y_{t-3} and rolling mean/std over 30 and 60 minute windows. Expected outcome: learned models approach or match persistence on MAE.
2. **Two-Stage Regime-Switching Architecture:**
   - **Stage 1 (Classifier):** Predict operational state (Standby ≤60 Wh vs. Active >60 Wh) using lag features.
   - **Stage 2 (Forecaster):** Route to Persistence if Standby; route to a lag-enhanced model if Active.
3. **Asymmetric Loss Calibration:** Train on Huber loss or asymmetric quantile loss rather than default MSE to reduce excessive smoothing of high-draw spikes (>300 Wh).
4. **Stratified Evaluation:** Report MAE separately per regime (Standby / Active / Spike) in addition to aggregate metrics — aggregate MAE is dominated by standby and masks spike-period failure.