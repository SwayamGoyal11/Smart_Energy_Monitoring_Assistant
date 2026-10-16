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
- **Features (26):** 5 calendar (`hour`, `day`, `day_of_week`, `month`, `is_weekend`), 3 target lags (`Appliances_lag_1..3`), 3 rolling statistics (`rolling_mean_3`, `rolling_mean_6`, `rolling_std_6`, all computed on shifted data), and 15 environmental/sensor columns.
- **Architectures Evaluated:**
  - Naïve Persistence Baseline (ŷ_t = y_{t-1})
  - Ordinary Least Squares (OLS) Linear Regression
  - Random Forest Regressor (n_estimators=100, max_depth=15, min_samples_leaf=4, random_state=42, n_jobs=-1)
  - XGBoost Regressor (n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
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
>
> **Reproducibility note:** Random Forest results are identical across thread counts (test MAE 45.56 at seed 42; 45.57 ± 1.32 across 10 seeds). XGBoost results depend on the seed and on the `n_jobs` setting: at seed 42, test MAE ranged from 55.0 to 66.3 across thread counts, and was 62.0 ± 4.7 across 10 seeds. The XGBoost figures above are one run (`n_jobs=-1`, 16 cores) and should be read as approximate (± ~5 Wh).

---

## 3. Key Findings

### Finding 1 — Persistence outperforms all learned models on Test MAE
The naïve persistence model achieves test MAE of 26.75 Wh, outperforming all learned models. This is explained by the strong autoregressive structure (lag-1 autocorrelation r ≈ 0.75 over the full series) and the dominant standby regime (54.4% of readings ≤60 Wh), where persistence's MAE is only ~8.8 Wh. A block-bootstrap (1-day blocks, 2,000 resamples) shows the gap to persistence is clearly positive for Random Forest (+18.8 Wh, 95% CI 13.3–24.3) and XGBoost (+30.4 Wh, CI 22.7–39.4), and small for Linear Regression (+2.5 Wh, CI 0.4–4.8).

### Finding 2 — Linear Regression is the most stable learned model
Linear Regression achieves a val→test MAE gap of only 0.14 Wh (29.10 → 29.24), indicating near-perfect generalisation. It outperforms persistence on RMSE and R² while remaining close to persistence on MAE (statistically only marginally worse; see Finding 1). Its stability is attributable to low model variance with a modest feature set.

### Finding 3 — Tree models overfit to the training period
Random Forest and XGBoost fit the training set well (train MAE 19.3 and 20.3 Wh; train R² 0.85) but degrade on validation and further on test (val→test gap +12.27 and +22.85 Wh; test R² 0.282 and 0.135). Linear Regression and persistence show no such degradation. On test, the tree models over-predict in the Active regime (60–300 Wh; mean signed error +62 Wh for Random Forest, +73 Wh for XGBoost). A feature-group ablation shows that removing the 15 environmental sensor columns brings both tree models to ≈26.5 Wh test MAE, comparable to persistence. A likely contributing factor is seasonal distribution shift: the target's mean is stable across splits, but sensor readings (e.g. temperatures) in the May test window sit well above the Jan–mid-April training range, so thresholds learned on sensor values do not transfer. This explanation is consistent with the evidence but has not been isolated experimentally.

### Finding 4 — Autoregressive lag features carry almost all of the predictive signal
The 26-feature matrix includes target lags (`Appliances_lag_1..3`) and rolling statistics. A feature-group ablation (first 144 rows dropped for all configurations; persistence on those rows = 26.72 Wh) shows that removing the lag and rolling features raises test MAE to 74.6 (Linear Regression), 147.9 (Random Forest) and 128.7 (XGBoost). Using only the three lag features gives 26.8, 26.5 and 25.7 Wh respectively, i.e. essentially persistence-level accuracy. Using only the sensors gives 75–201 Wh. Sensors and calendar features add little beyond the lags. Note that `rolling_mean_3` is exactly the mean of `lag_1..3`, so the three families are partly redundant.

---

## 4. Operational Limitations
- **No model reliably beats persistence on MAE:** Learned models match persistence only when restricted to lag features; the additional sensor features do not improve and, for tree models, degrade test accuracy.
- **Tree models over-predict in active periods and under-predict spikes:** Every model under-predicts spikes (>300 Wh, mean signed error −111 to −174 Wh), although the tree models have a lower spike MAE (≈146–152 Wh) than persistence (≈206 Wh).
- **Step-lag penalty for persistence:** Persistence lags by one step at every regime transition — predicting the spike one interval after it begins and staying elevated one interval after it ends.
- **Seasonal generalisation gap:** The collection window (Jan–May) excludes summer load profiles. Models trained on this data will not generalise to July–August behaviour.
- **Short test window:** The test set covers ~20 days of a single month, so metric differences of a few Wh are within sampling uncertainty.
- **XGBoost non-determinism:** Results vary with the seed and `n_jobs` (see §2).
- **Single household:** Results apply to one specific low-energy house in Belgium. Cross-household generalisation is not established.

---

## 5. Production Recommendations (Phase 2)
1. **Extend and regularise the autoregressive features:** In the lag-depth ablation, increasing the number of lags from 3 to 12 reduced tree-model test MAE (Random Forest 46.5 → 36.9, XGBoost 95.8 → 41.8 Wh in that setup). Consider deeper lags, and consider reducing or regularising the sensor features, which degrade the tree models.
2. **Two-Stage Regime-Switching Architecture:**
   - **Stage 1 (Classifier):** Predict operational state (Standby ≤60 Wh vs. Active >60 Wh) using lag features.
   - **Stage 2 (Forecaster):** Route to Persistence if Standby; route to a lag-enhanced model if Active.
3. **Asymmetric Loss Calibration:** Train on Huber loss or asymmetric quantile loss rather than default MSE to address systematic under-prediction of high-draw spikes (>300 Wh).
4. **Stratified Evaluation:** Report MAE separately per regime (Standby / Active / Spike) in addition to aggregate metrics — aggregate MAE is dominated by standby and masks spike-period failure.
5. **Deterministic training:** Fix `n_jobs` (or use a single thread) when reporting XGBoost results so metrics are reproducible across machines.