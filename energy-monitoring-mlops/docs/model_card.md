# Model Card: Short-Term Household Appliance Energy Forecaster

| | |
|---|---|
| **Model** | `energy-forecaster` **v5**, alias `@champion` (MLflow Model Registry) |
| **Type** | Two-stage hurdle: `LogisticRegression` regime classifier + `HistGradientBoostingRegressor` burst regressor (`src/models/hurdle.py`) |
| **Run** | `e78a36504a434cfa832213de713c6278`; quality gate **passed** on validation and test |
| **Serving bundle** | `models/champion/` (exported by `scripts/export_champion.py`, verified byte-identical to the registry in [`retraining_lifecycle.md`](../reports/evidence/retraining_lifecycle.md)) |
| **Release** | `v1.2.0` (Phase 3: monitoring, drift experiment and governance) |

---

## 1. Model Overview and Intended Use

**Task.** Forecast the appliance energy use (`Appliances`, Wh) of one household for the next 10-minute interval, given the six preceding 10-minute readings, the current indoor/outdoor sensor readings and the lighting load.

**Intended use.**
- Short-term load forecasting for a residential home energy dashboard, demand awareness and anomaly flagging.
- Decision support where a forecast error of tens of Wh per 10 minutes is acceptable (validation MAE 24.63 Wh, RMSE 59.66 Wh).
- Served through the FastAPI service (`/predict`, `/predict/batch`) with input validation, structured logging, Prometheus metrics and drift monitoring.

**Out of scope.**
- **Industrial, commercial or multi-dwelling loads.** The model learned the load profile of one low-energy house: a median 60 Wh standby level and bursts up to ~1,080 Wh. Larger loads have different scales and dynamics.
- **Sub-second or real-time control.** Inputs and outputs are 10-minute aggregates. The model cannot drive protection relays, inverter or HVAC control loops, or demand response that needs sub-interval actuation.
- **Uncalibrated climate zones and houses.** Training covers one Belgian house from January to mid-April. Other climates, building envelopes or occupancy patterns need retraining and validation. The heatwave experiment in §6.5 shows RMSE rising 16.8% under a hot, dry regime.
- **Billing, settlement or safety-critical decisions.** Forecasts are not metered values and carry no uncertainty bounds.
- **Inferring occupant behaviour or occupancy.** The model is not designed or validated for this, and doing so has privacy implications.

---

## 2. Data

- **Source:** UCI Appliances Energy Prediction dataset (Candanedo, Feldheim and Deramaix, 2017): 19,735 records at 10-minute resolution, 2016-01-11 17:00 to 2016-05-27 18:00, from a single low-energy house in Belgium. Indoor temperature and humidity come from a wireless sensor network. Outdoor conditions (`T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`) come from the nearest airport weather station.
- **Versioning:** the raw CSV is tracked by DVC (`data/raw/energydata_complete.csv.dvc`, MD5-pinned). CI downloads the public archive and verifies it against that MD5. Validation evidence: `reports/evidence/dataset_validation.json` (no missing values).
- **Protocol:** chronological 70/15/15 split, no shuffling, no overlapping timestamps. The first 6 rows are dropped as lag/rolling warm-up (19,729 rows remain).

  | Split | Rows | Date range |
  |---|---|---|
  | Train | 13,810 | 2016-01-11 18:00 to 2016-04-16 15:30 |
  | Validation | 2,959 | 2016-04-16 15:40 to 2016-05-07 04:40 |
  | Test | 2,960 | 2016-05-07 04:50 to 2016-05-27 18:00 |

---

## 3. Architecture, Features and Input Guardrails

### 3.1 Two-stage hurdle model
- **Stage 1, regime classifier:** `StandardScaler` + `LogisticRegression` predicts whether the interval is an active burst (`Appliances > 100 Wh`, about 22% of training intervals).
- **Stage 2, burst regressor:** `HistGradientBoostingRegressor` (`max_depth=4`, `learning_rate=0.05`, `max_iter=200`, `random_state=42`) trained on active intervals only.
- **Routing:** intervals with P(active) ≥ 0.5 use the Stage 2 forecast. Standby intervals use persistence (ŷ<sub>t</sub> = y<sub>t-1</sub>). Predictions are clipped at 0 Wh.
- The 100 Wh boundary and hard routing were chosen on the validation split only (100 Wh beat 60 Wh, and hard routing to persistence gave the lowest MAE).

### 3.2 Features (32, leak-free)

| Group | Count | Features |
|---|---|---|
| Calendar | 5 | `hour`, `day`, `day_of_week`, `month`, `is_weekend` |
| Cyclical encodings | 6 | sin/cos of hour, day of week and month |
| Consumption lags | 3 | `Appliances_lag_1..3` |
| Rolling statistics | 3 | `Appliances_rolling_mean_3`, `_mean_6`, `_rolling_std_6`, computed on the target shifted by one step |
| Sensors and lighting | 15 | `T1`, `RH_1`, `T2`, `RH_2`, `T3`, `RH_3`, `T6`, `RH_6`, `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`, `lights` |

The API builds every feature from the request (six recent readings plus the current sensors) using the same code as training (`src/serving/schemas.py` → `src/features/feature_engineering.py`), so there is no training/serving skew.

### 3.3 Input guardrails
Two layers reject bad inputs before the model runs. Each rejection is counted in `model_input_validation_failures_total{stage}`.

1. **Schema (`stage="schema"`):** Pydantic request models enforce types, required fields, at least 6 history values, no unknown fields, finite values, and per-field `PHYSICAL_BOUNDS`.
2. **Domain (`stage="domain"`):** `validate_feature_frame()` re-checks the derived 32-feature frame: bounds, finiteness, integer calendar fields, and `is_weekend` consistency with `day_of_week`. This is defense in depth, since the schema already enforces the same bounds.

`PHYSICAL_BOUNDS` (`src/validation/input_validation.py`) are **physical plausibility limits, not training ranges**. A value outside them signals a broken sensor or a unit error, not drift. Drift is detected separately (§6). The bounds were set in code as deliberately wide sanity limits and are not taken from a published standard. Every value observed in the dataset (Candanedo et al., 2017) falls well inside them:

| Variable | Bound | Observed in dataset | Unit |
|---|---|---|---|
| Indoor temperatures (`T1`…`T6`) | −40 to 60 | e.g. `T1` 16.8 to 26.3 | °C |
| `T_out` | −40 to 60 | −5.0 to 26.1 | °C |
| `Tdewpoint` | −50 to 40 | −6.6 to 15.5 | °C |
| Humidities (`RH_*`) | 0 to 100 | e.g. `RH_1` 27.0 to 63.4, `RH_out` 24 to 100 | % |
| `Press_mm_hg` | 600 to 820 | 729.3 to 772.3 | mm Hg |
| `Windspeed` | 0 to 60 | 0 to 14 | m/s |
| `Visibility` | 0 to 100 | 1 to 66 | km |
| `lights` | 0 to 1,000 | 0 to 70 | Wh |
| Appliance history and lags | 0 to 5,000 | 10 to 1,080 | Wh / 10 min |

Because the bounds are wide, out-of-distribution but physically possible inputs, such as the +8 °C heatwave in §6.5, pass validation and are caught by drift monitoring instead. This is intentional.

---

## 4. Quantitative Evaluation

All metrics are on chronological held-out splits, with MAE and RMSE in Wh per 10-minute interval. They were regenerated in the reference environment (Python 3.12, scikit-learn 1.9, XGBoost 3.4). Bold marks the best value per column.

| Model | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Gates |
|---|---|---|---|---|---|---|---|
| **Two-Stage Hurdle (champion v5)** | **24.63** | **59.66** | **0.582** | **26.21** | 62.56 | 0.526 | pass |
| Linear Regression | 28.40 | 60.22 | 0.574 | 27.24 | **60.91** | **0.551** | pass |
| Persistence Baseline | 26.15 | 66.44 | 0.481 | 26.75 | 66.85 | 0.459 | n/a |
| Random Forest | 33.43 | 65.07 | 0.503 | 45.24 | 76.85 | 0.285 | fail |
| XGBoost | 37.56 | 68.64 | 0.446 | 60.15 | 90.13 | 0.017 | fail |

- Against persistence, the champion cuts validation RMSE from 66.44 to 59.66 Wh (−10.2%) and test RMSE from 66.85 to 62.56 Wh (−6.4%). It is the only candidate that beats persistence on MAE on both held-out splits.
- The champion was selected on validation RMSE only. Test is a pass/fail gate and is never used to choose a model. On test, Linear Regression has a lower RMSE (60.91 vs 62.56 Wh). This is reported as-is rather than used to re-select.
- `scripts/verify_retraining_flow.py` reloads the committed bundle and reproduces test RMSE 62.5553, MAE 26.2072 and R² 0.5263 exactly.

### 4.1 Why persistence is hard to beat on MAE
1. **Strong lag-1 autocorrelation (r ≈ 0.75).** The median interval is 60 Wh and about 79% of intervals are at or below 100 Wh. During standby y<sub>t</sub> ≈ y<sub>t-1</sub>, so persistence makes near-zero absolute error on most intervals.
2. **Short impulse spikes vs. partition averages.** Kettles, ovens and washing cycles produce stochastic high-amplitude pulses. Tree models predict leaf means, which over-smooths peaks. Random Forest and XGBoost generalise poorly from validation to test.
3. **Thermal inertia.** Indoor temperature and humidity respond to appliance waste heat with 30 to 90 minute delays, so instantaneous sensor readings give little lead time on discretionary switching.

---

## 5. Release Quality Gates and Registry Promotion

`src/models/quality_gates.py` applies the same thresholds to **both** validation and test before a candidate can receive any alias:

| Gate | Threshold |
|---|---|
| MAE | ≤ 35.0 Wh |
| RMSE | ≤ 65.0 Wh |
| $R^2$ | ≥ 0.50 |
| RMSE vs. persistence baseline (same split) | must not be worse |

Promotion (`register_candidate()` in `src/models/registry.py`, driven by `run_registry.py` or the ZenML pipeline):
1. Every candidate is registered as a new version, pass or fail, and tagged `model_type`, `quality_gate` and `val_rmse`, so rejected models stay auditable.
2. Passing every gate on validation **and** test assigns `@staging`. Failing any gate assigns no alias, and the model is never served.
3. `@champion` moves only if the candidate's validation RMSE is **strictly lower** than the incumbent champion's. Ties and worse candidates stay at `@staging`, so a retrain cannot silently replace production with an equal or weaker model.
4. CI runs the gates on every push and pull request (quality-gate job). A change that drops every candidate below threshold fails the build.

**Registry state (audited in [`retraining_lifecycle.md`](../reports/evidence/retraining_lifecycle.md)):**
- 23 versions are registered. `@champion` is v5 (val RMSE 59.6618) and `@staging` is v23.
- The only failed candidate (XGBoost v3) holds no alias.
- 21 other versions passed the gates but have a validation RMSE no better than v5's: its own re-logs (v1, v4), the current `@staging` v23, and every later retrain. A Hurdle retrain with the current code reaches val RMSE 59.6986, which does not displace the champion.
- Versions 4 and 5 re-log the fitted v1 model to refresh its recorded dependencies (v5 records the inference-only requirements from `requirements-serve.txt`). They carry a `relogged_from` tag. Because their metrics are identical, a tie cannot win the strict comparison, so they were moved to `@champion` by hand.

---

## 6. Monitoring, Drift and Retraining Governance

### 6.1 Observability
- **Metrics (`GET /metrics`, `src/monitoring/metrics.py`):**
  - Request count (`http_requests_total`) and latency (`http_request_duration_seconds`) by route and status.
  - Prediction distribution (`model_prediction_value`).
  - Hurdle routing (`model_stage_predictions_total`, `model_active_stage_ratio`).
  - Rejected inputs (`model_input_validation_failures_total{path, stage}`).
  - Latest drift result (`model_drift_score{column}`, `model_drift_share`, `model_drift_detected`), re-read from `drift_metrics.json` whenever it changes.
- **Dashboards and alerts:** Prometheus scrapes the API every 15 s and evaluates `prometheus/alerts.yml`. Grafana provisions the "Energy Forecaster - Serving & Drift" dashboard. A test checks that every metric name used in alerts and panels exists in the API registry.

  | Alert | Condition |
  |---|---|
  | `EnergyApiDown` | scrape target down for 1 min |
  | `EnergyApiHighErrorRate` | 5xx ratio > 5% for 5 min |
  | `EnergyApiSlowPredictions` | `/predict` p95 > 250 ms for 10 min |
  | `EnergyApiInputValidationFailures` | rejections > 10% of `/predict*` requests for 10 min |
  | `EnergyModelDriftDetected` | latest drift report raised an alert, for 5 min |
  | `EnergyModelActiveRatioAnomalous` | > 50% of predictions routed to the burst regressor for 30 min (training ≈ 22%) |

### 6.2 Structured inference logging
Every served prediction is appended to `INFERENCE_LOG_PATH` as one JSON Lines record (`src/monitoring/inference_log.py`):

```json
{"event": "inference", "logged_at": "...", "target_timestamp": "...", "model_version": "5", "<32 features>": ..., "prediction": 63.2}
```

Rejected requests are not logged. The log is the "current" window for drift checks. `scripts/simulate_traffic.py` replays real dataset rows (mixed standby and burst batches, optionally with invalid payloads) against a local or remote API and writes the same format.

### 6.3 Drift detection
`src/monitoring/drift.py` runs Evidently on 21 monitored features (lags, rolling statistics, sensors) plus the prediction, and the target when actuals are available. It compares them against the training split using normed Wasserstein distance.
- A column counts as drifted when its score is ≥ **0.1**.
- **Dataset drift** is flagged when the drifted share of monitored features is ≥ **0.30**.
- Calendar features are excluded, because serving data always comes later in time than training.
- Fewer than 200 current rows are refused as too noisy to judge.

### 6.4 Retraining rule
Retraining is triggered only when **both** signals fire (`scripts/run_drift_experiment.py`, `decide_retraining()`):
1. **Dataset drift:** drifted feature share ≥ 0.30, **and**
2. **Performance breach** on labelled data: a release quality gate fails (RMSE > 65 Wh, MAE > 35 Wh or R² < 0.50), **or** RMSE rises by ≥ 10% over the baseline window.

Drift alone is not enough on purpose. The unshifted test split already drifts seasonally (20/21 features), yet the champion still passes every gate on it. A drift-only rule would retrain on every season change.

### 6.5 Deliberate drift experiment (evidence)
`scripts/run_drift_experiment.py` applies a summer heatwave scenario to the raw series and rebuilds features, so lags and ground truth stay consistent. The scenario is +8 °C on `T1`, `T2`, `T3`, `T6` and `T_out`; −18% on `RH_1`, `RH_2`, `RH_3`, `RH_6` and `RH_out`; and evening appliance load ×1.5 + 50 Wh from 17:00 to 22:59. It then scores the same timestamps as the test split.

| Window (champion v5) | RMSE | MAE | $R^2$ | Drifted features (mean score) | Retrain |
|---|---|---|---|---|---|
| Baseline test split | 62.56 | 26.21 | 0.526 | 20/21 (0.95) | no |
| Heatwave scenario | **73.06** | 30.66 | 0.633 | 20/21 (2.25) | **yes** |

- RMSE rises **+16.8%** and breaches the 65 Wh gate, and MAE rises +17.0%.
- Prediction drift (0.245) and target drift (0.241) also exceed the 0.1 threshold.
- Evidence files: [`retraining_trigger_evidence.json`](../reports/evidence/retraining_trigger_evidence.json), [`heatwave_drift_metrics.json`](../reports/evidence/heatwave_drift_metrics.json) and [`heatwave_drift_report.html`](../reports/evidence/heatwave_drift_report.html).

### 6.6 Closed loop: retrain → promote → export → redeploy
1. Retrain with `python run_registry.py` (or the ZenML pipeline). Candidates are gated and promoted under §5.
2. Package with `python scripts/export_champion.py`. It resolves `models:/energy-forecaster@champion`, refuses to overwrite anything except a previous MLflow export, and writes `models/champion/` plus `serving_metadata.json` (version, run, model type, gate status and metrics, reported by `/health` and `/model-info`).
3. Verify with `python scripts/verify_retraining_flow.py`, which writes [`retraining_lifecycle.md`](../reports/evidence/retraining_lifecycle.md). It checks the trigger, audits the registry, replays the promotion policy in a throwaway registry (promote / hold at `@staging` / tie / reject), and confirms the committed bundle is byte-identical to `@champion` and reproduces its metrics.
4. Redeploy:
   - **Compose:** `docker compose restart api` (the bundle is mounted read-only) or `up -d --build`.
   - **Hosted:** commit the bundle and push to `main`. CI runs the tests and gates, then CD publishes the GHCR image with the champion baked in.

### 6.7 Rollback
To roll back, reassign `@champion` to a previous version that passed the gates, then re-export:

```bash
python -c "import mlflow; from mlflow.tracking import MlflowClient; from src.models.registry import tracking_uri; mlflow.set_tracking_uri(tracking_uri()); MlflowClient().set_registered_model_alias('energy-forecaster', 'champion', '<version>')"
python scripts/export_champion.py
python scripts/verify_retraining_flow.py --no-sandbox
docker compose restart api        # or commit models/champion/ and push for CI/CD
```

Only roll back to a version tagged `quality_gate=passed`. If the target has a worse validation RMSE than another gated version, the registry audit reports "No gated version beats @champion" as **FAIL**. That is intended: the report then records that the rollback was a deliberate override of the selection rule.

---

## 7. Limitations and Risks

- **Single house, single season.** One low-energy Belgian house, January to May 2016. Training covers January to mid-April only, so summer behaviour, other buildings, other climates and other occupants are untested. Treat any new deployment as out of distribution until drift checks and labelled metrics confirm otherwise.
- **10-minute resolution.** Short spikes within an interval are averaged away, and the model cannot see or forecast sub-interval events.
- **Covariate-shift sensitivity.** Under the heatwave scenario, RMSE rose 16.8% (62.56 → 73.06 Wh) and broke the release gate. Extreme weather or sensor bias degrades accuracy without failing input validation, because the bounds are physical, not statistical. Drift monitoring is the control for this.
- **R² inflates under a widened target distribution.** In the same experiment R² *rose* from 0.526 to 0.633 while RMSE and MAE got worse, because the heavier evening load widened the target variance that R² normalises by. Use RMSE and MAE, not R², to judge degradation across shifted windows.
- **Step-lag on regime changes.** Persistence, which the hurdle model uses for standby intervals, and the lag-driven models predict a spike one step after it begins and stay high one step after it ends.
- **Sensor-only forecasting is weak.** Environmental sensors cannot predict discretionary human behaviour without consumption history. Requests with fewer than six history readings are rejected, not imputed.
- **Pre-existing seasonal drift.** The held-out windows already drift from training (20/21 features). The two-signal retraining rule prevents constant false triggers but delays retraining until a performance breach is measured, which needs labelled actuals.
- **No uncertainty estimates.** Point forecasts only; consumers cannot tell a confident forecast from an extrapolation.

---

## 8. Recommendations

1. **Hurdle tuning:** calibrate the Stage 1 probability and tune the burst regressor. Test RMSE still trails Linear Regression.
2. **Asymmetric loss:** train on Huber or quantile loss rather than MSE, to reduce smoothing of rare, high-draw spikes and to provide prediction intervals.
3. **Longer-horizon lags:** add terms such as y<sub>t-6</sub> to capture hourly inertia.
4. **Collect actuals in production:** join served predictions with metered values, so the performance half of the retraining rule can run on real traffic rather than on experiments.

---

## 9. References and Evidence

- Candanedo, L. M., Feldheim, V., and Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 140, 81–97. Dataset: [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).
- Evidence (`reports/evidence/`): [`dataset_validation.json`](../reports/evidence/dataset_validation.json), [`model_comparison.csv`](../reports/evidence/model_comparison.csv) (Phase 1 baselines from `run_pipeline.py`), [`drift_report.html`](../reports/evidence/drift_report.html) and [`drift_metrics.json`](../reports/evidence/drift_metrics.json) (live clean baseline), [`heatwave_drift_report.html`](../reports/evidence/heatwave_drift_report.html) and [`heatwave_drift_metrics.json`](../reports/evidence/heatwave_drift_metrics.json) (experiment), [`retraining_trigger_evidence.json`](../reports/evidence/retraining_trigger_evidence.json), [`retraining_lifecycle.md`](../reports/evidence/retraining_lifecycle.md).
- Runbooks: [`deployment.md`](deployment.md), [`zenml_wrapper.md`](zenml_wrapper.md).
