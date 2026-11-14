# Retraining Lifecycle and Champion Promotion Evidence

Generated 2026-11-14 03:02 UTC by `python scripts/verify_retraining_flow.py`. **25/25 checks passed.**

```text
drift trigger -> retrain candidates -> quality gates (val + test) -> @staging
              -> beats @champion val RMSE (strict) -> @champion -> export_champion.py
              -> models/champion/ -> Docker image / Render redeploy -> /health reports version
```

## 1. Drift trigger

Source: `reports/evidence/retraining_trigger_evidence.json` (experiment `summer_heatwave_sensor_bias_evening_burst`, 2,960 rows, 2016-05-07 to 2016-05-27, model energy-forecaster v5).

| Window | RMSE (Wh) | MAE (Wh) | R² | Drifted features | Retrain |
|---|---|---|---|---|---|
| Baseline test split | 62.56 | 26.21 | 0.526 | 20/21 | no |
| Heatwave scenario | 73.06 | 30.66 | 0.633 | 20/21 | **yes** |

Heatwave RMSE **73.06 Wh** is **+16.8%** against the 62.56 Wh baseline and breaches the 65 Wh quality gate (MAE +17.0%). Mean drift score rose from 0.95 to 2.25. R² rises because the heavier evening load widens target variance; RMSE and MAE carry the degradation signal. The unshifted baseline already shows seasonal drift but passes the gates, so the rule (drift **and** a performance breach) does not retrain on it.

Trigger reasons:

- dataset drift: 20/21 features drifted (share 0.95 >= 0.3)
- prediction drift: score 0.2451 >= 0.1
- target drift: score 0.2408 >= 0.1
- quality gate: RMSE 73.0592 > 65.0
- RMSE up 16.8% vs baseline (limit 10%)

| Check | Result | Detail |
|---|---|---|
| Trigger evidence present | PASS | reports/evidence/retraining_trigger_evidence.json |
| Retraining triggered | PASS | dataset drift: 20/21 features drifted (share 0.95 >= 0.3); prediction drift: score 0.2451 >= 0.1; target drift: score 0.2408 >= 0.1; quality gate: RMSE 73.0592 > 65.0; RMSE up 16.8% vs baseline (limit 10%) |
| Dataset drift breached | PASS | share 0.9524 >= 0.3 |
| Performance breached | PASS | drifted RMSE 73.0592 vs baseline 62.5553 |
| Unshifted baseline does not trigger | PASS | seasonal drift alone, without degradation, is not enough |

## 2. Retraining policy

Retraining runs `python run_registry.py` (or the ZenML pipeline), which fits candidates and calls `register_candidate()` in `src/models/registry.py` for each:

1. Every candidate is logged and registered as a new version, pass or fail, so rejected models stay auditable.
2. `evaluate_quality_gates()` (`src/models/quality_gates.py`) runs on **both** validation and test: MAE ≤ 35 Wh, RMSE ≤ 65 Wh, R² ≥ 0.50, and RMSE ≤ 1.0× the persistence baseline on the same split.
3. Passing all gates assigns `@staging`. Failing any gate assigns no alias.
4. `@champion` moves only if the candidate also has a validation RMSE **strictly lower** than the incumbent champion's. Ties and worse candidates stay at `@staging`, so a retrain can never silently replace production with an equal or weaker model.
5. Test data is a pass/fail release check only; it is never used to choose between candidates.

## 3. Live registry audit

`energy-forecaster` in the project MLflow registry: 23 versions, `@champion` = v5, `@staging` = v23. 21 other versions passed the gates but have a validation RMSE no better than the champion's (including its own re-logs and the current `@staging`), so none of them could take `@champion`.

| Version | Model | Gate | Val RMSE | Test RMSE | Aliases | Note |
|---|---|---|---|---|---|---|
| v1 | hurdle | passed | 59.6618 | 62.5553 | - |  |
| v2 | linear_regression | passed | 60.2179 | 60.9127 | - |  |
| v3 | xgboost | failed | 68.6365 | 90.1280 | - |  |
| v4 | hurdle | passed | 59.6618 | 62.5553 | - | re-log of v1 (manual) |
| v5 | hurdle | passed | 59.6618 | 62.5553 | @champion | re-log of v4 (manual) |
| v6 | hurdle | passed | 59.6986 | 63.1637 | - |  |
| v7 | linear_regression | passed | 60.2179 | 60.9127 | - |  |
| v8 | hurdle | passed | 59.9706 | 62.7041 | - |  |
| v9 | linear_regression | passed | 60.2179 | 60.9127 | - |  |
| v10 | hurdle | passed | 59.6986 | 63.1637 | - |  |
| v11 | linear_regression | passed | 60.2179 | 60.9127 | - |  |
| v12 | hurdle | passed | 59.9706 | 62.7041 | - |  |
| v13 | linear_regression | passed | 60.2179 | 60.9127 | - |  |
| v14 | hurdle | passed | 59.9706 | 62.7041 | - |  |
| v15 | linear_regression | passed | 60.2179 | 60.9127 | - |  |
| v16 | hurdle | passed | 59.6986 | 63.1637 | - |  |
| v17 | hurdle | passed | 59.6986 | 63.1637 | - |  |
| v18 | hurdle | passed | 59.6986 | 63.1637 | - |  |
| v19 | hurdle | passed | 59.6986 | 63.1637 | - |  |
| v20 | linear_regression | passed | 60.2179 | 60.9127 | - |  |
| v21 | hurdle | passed | 59.6986 | 63.1637 | - |  |
| v22 | linear_regression | passed | 60.2179 | 60.9127 | - |  |
| v23 | hurdle | passed | 59.6986 | 63.1637 | @staging |  |

Versions 4 and 5 re-log the fitted v1 model to refresh its recorded dependencies; they carry a `relogged_from` tag and were moved to `@champion` by hand, since a tie cannot win the strict comparison.

| Check | Result | Detail |
|---|---|---|
| @champion assigned | PASS | v5 |
| @staging assigned | PASS | v23 |
| Every alias holder passed the quality gates | PASS | v5=passed, v23=passed |
| No failed candidate holds an alias | PASS | 1 failed version(s): v3 |
| No gated version beats @champion on validation RMSE | PASS | @champion v5 val RMSE 59.6618; 21 other gated versions are no better |

## 4. Promotion policy replay (sandbox registry)

A temporary MLflow registry receives real candidates fitted on the training split, in this order. Each outcome is compared with an independent re-derivation of the policy.

| Version | Candidate | Val RMSE | Incumbent | Gates | Aliases | Policy |
|---|---|---|---|---|---|---|
| v1 | Champion-grade retrain (production config) | 59.6986 | - | passed | @staging, @champion | as expected |
| v2 | Passes gates, worse validation RMSE | 60.2179 | 59.6986 | passed | @staging | as expected |
| v3 | Identical retrain: ties the incumbent | 59.6986 | 59.6986 | passed | @staging | as expected |
| v4 | Regressed model: predicts the training mean | 92.4084 | 59.6986 | failed (8) | - | as expected |

Gate failures recorded for the rejected candidate:

- `val: MAE 53.9867 > 35.0`
- `val: RMSE 92.4084 > 65.0`
- `val: R2 -0.0034 < 0.5`
- `val: RMSE 92.4084 does not beat persistence baseline RMSE 66.4376 (limit 66.4376)`
- `test: MAE 52.8443 > 35.0`
- `test: RMSE 90.9052 > 65.0`
- `test: R2 -0.0004 < 0.5`
- `test: RMSE 90.9052 does not beat persistence baseline RMSE 66.8482 (limit 66.8482)`

Final sandbox aliases: `@champion` = v1, `@staging` = v3.

| Check | Result | Detail |
|---|---|---|
| Sandbox v1 (Champion-grade retrain (production config)) follows the policy | PASS | aliases ['staging', 'champion']; expected ['staging', 'champion'] |
| Sandbox v2 (Passes gates, worse validation RMSE) follows the policy | PASS | aliases ['staging']; expected ['staging'] |
| Sandbox v3 (Identical retrain: ties the incumbent) follows the policy | PASS | aliases ['staging']; expected ['staging'] |
| Sandbox v4 (Regressed model: predicts the training mean) follows the policy | PASS | aliases none; expected none |
| Policy exercised every branch | PASS | promoted, held at @staging, and rejected outcomes all observed |
| A tie does not displace @champion | PASS | v3 val RMSE 59.6986 vs incumbent 59.6986 |
| Exported sandbox bundle carries the @champion version | PASS | serving_metadata.json -> v1 @champion |
| Exported bundle serves identical predictions | PASS | 2,960 test rows compared |

## 5. Champion export and redeployment

`python scripts/export_champion.py` resolves `models:/energy-forecaster@champion`, refuses to overwrite anything but a previous MLflow export, downloads the model into `models/champion/`, and writes `serving_metadata.json` (version, run, model type, gate status, gate metrics) so `/health` and `/model-info` report the registry lineage without registry access.

- **Docker / compose:** the Dockerfile copies `models/champion/` into the image, and compose also mounts it read-only at `MODEL_LOCAL_PATH`, so a new champion is served after `docker compose up -d --build` (or a container restart for the mount).
- **CI/CD and Render:** the bundle is committed, so a push to `main` runs CI (tests and quality gates), and on success CD (`.github/workflows/cd.yml`) publishes `ghcr.io/<owner>/energy-forecaster-api` (`:latest`, `:sha-<short>`) with the new champion baked in. The Render service is redeployed from the repository/image (its configuration lives in Render, not in this repo), after which `/health` reports the new `model_version`.

Committed bundle: energy-forecaster **v5** (hurdle, gate passed), run `e78a36504a434cfa832213de713c6278`, exported 2026-11-13T07:38:33.421057+00:00. Recorded val RMSE 59.6618, test RMSE 62.5553; reproduced from the bundle on the test split: RMSE 62.5553, MAE 26.2072, R² 0.5263.

| Check | Result | Detail |
|---|---|---|
| Bundle has serving_metadata.json | PASS | models/champion/serving_metadata.json |
| Metadata version/run match @champion | PASS | v5 run e78a3650 vs registry v5 run e78a3650 |
| Metadata tags match registry | PASS | hurdle, gate passed |
| Metadata metrics match the registry run | PASS | 12 gate metrics |
| Bundle files byte-identical to registry artifacts | PASS | 8 files compared |
| MLmodel run_id matches @champion | PASS | e78a36504a434cfa832213de713c6278 |
| Bundle reproduces recorded test RMSE | PASS | 62.5553 vs recorded 62.5553 |

## Reproduce

```bash
python scripts/run_drift_experiment.py     # drift trigger evidence
python run_registry.py                     # retrain + gated registration
python scripts/export_champion.py          # package @champion for Docker/Render
python scripts/verify_retraining_flow.py   # this report
```
