# MVP validation — 2026-09-23

Environment: Python 3.12, pinned requirements.lock.txt, macOS arm64.

- `python -m pytest -q`: 30 passed. One upstream Starlette TestClient deprecation
  warning about httpx; no test failures.
- `python -m pip check`: no broken requirements.
- Forecast: T1, 2026-02-10, 24h, demo, all 24 hourly predictions valid.
  Synthetic predicted energy: 35721.904385 kWh. One aggregated hourly_jump flag (5 intervals).
  This number demonstrates pipeline execution, not real production or model accuracy.
- Default backtest: 2026-01-31–2026-02-28 inclusive, 29 days, 696/696 valid hours,
  zero failed days. Actuals unavailable; MAE/RMSE/bias remain null.
- T2 and 72h horizon exercised in tests.
- Streamlit AppTest: form submit → forecast → 3 metrics, no exceptions.
- Browser smoke: form, 24h forecast, power/wind plots, deterministic analysis,
  anomaly table and JSON download button visible. Axes explicitly use UTC.
- OpenAI SDK 2.54.0: responses.parse available. Structured success/refusal/invalid
  output/failure exercised using mocks. No live OpenAI request (no configured key/model).
- Production weather API and trained model not available yet; only adapter contracts,
  validation and cache behavior tested. No production accuracy or deployment claim.
- Source/config/docs credential-pattern scan and git diff whitespace check passed.

## Person B integration / real MODE A — 2026-09-23

- Integrated ML branch through `abc7060`, preserving artifact, metadata and holdout bytes.
- `verify_ml_provider` checksums and all 47006 rows reproduce raw MAE 82.90469335 kW,
  RMSE 145.19032792 kW, R² 0.9530159122. No retraining.
- Real native day: Kelmarsh 1, 2017-11-08, 144/144 samples; MAE 51.39334479 kW,
  RMSE 71.45668974 kW; rectangular energy estimate 8808.38752409 kWh.
- Native daily backtest: Kelmarsh 1, 2017-11-08–2017-12-31, all 54 days completed,
  7771/7776 paired samples (99.9357%), 5 gaps preserved. Nine days have rule flags.
  Raw MAE 97.22500016 kW, RMSE 173.70032123 kW. Exit 2 correctly signals incomplete coverage;
  it does not mean the model failed to load. Different scope from the all-turbine full holdout.
- Synthetic CLI remains functional. UI includes separate real, synthetic and configured forecast modes.
- Full integration suite: 101 passed, 1 skipped (optional LightGBM absent; selected artifact uses sklearn).
  One upstream Starlette TestClient deprecation warning remains. Native timestep integration,
  absence of target in features, missing observations, Goldwind rejection, shared context,
  API/UI and multi-day metric consistency covered.
- `pip check` passes. Required model libraries match training; runtime is Python 3.12.
- The running Streamlit server requires a restart after adding new schema modules (hot reload can retain old imports).
- Still pending: A's real weather adapter, source-direction agreement and Goldwind validation.
  No live OpenAI call made. main is not yet the final integrated release.


## Strict target forecast + B Goldwind preparation — 2026-09-23

- Final integrated suite: **212 passed, 1 skipped**, 5.91 s on Python 3.12/macOS arm64.
  The skip is optional LightGBM; selected frozen model uses sklearn. One upstream Starlette
  TestClient deprecation warning remains. `pip check` reports no broken requirements.
- PR #3 `e7e1f44` preparation loader integrated through Git. Corrected turbine identity after
  caller reorders DataFrame indices; added regression. All 29 Goldwind parser tests pass,
  including `-W error::DeprecationWarning`. No real organizer data or model is claimed.
- New strict tests cover 24/48h grids, aware timestamps and local calendar/DST, issue/actual-revision
  availability, training+selection cutoff, immutable snapshots, changed-input recomputation,
  no repeated ML/LLM for unchanged inputs, partial station sums, raw metrics, retry after model
  failure, provider failures, malformed/duplicate forecast rejection, and API/UI validation.
- Provider dependency changes invalidate numeric cache. Unused optional wind direction does
  not invalidate a model that uses only wind speed and temperature. Numeric/model/analysis
  results publish atomically with a POSIX process lock; failures remain retryable.
- Actual CLI attempts: readiness exit 2, ready=false; diagnostic origin Jan31 23:00Z exit 2,
  blocked and zero predictions; February replay exit 2, blocked and zero origins because
  organizer timezone/schedule are unknown. The origin is a diagnostic example, not confirmed
  protocol. Copied JSON, header-only CSV and manifest: `reports/target-acceptance/`.
- Real target acceptance remains blocked: organizer SCADA/definitions, actual archived weather
  provider with availability evidence, Goldwind artifact/manifest. February actuals are absent,
  so real target metrics remain null. Tests use explicit artificial contract fixtures only.
- Regression smoke: synthetic 24h forecast succeeds; Kelmarsh 1 observed 2017-11-08 returns
  144/144 pairs, MAE 51.39 kW and RMSE 71.46 kW. Frozen Kelmarsh artifact/data unchanged.
- Browser smoke after server restart: target is default, exact blockers visible, calculate
  disabled; switching to Kelmarsh and running shows real 144/144 pairs, model/fact plots,
  deterministic analysis and JSON download. Returned browser to target screen.
- No live target-weather fetch, target training/inference or OpenAI call validated.
  The strict manifest is an assertion that needs independent lineage/source review.
- Source/config/docs credential-pattern scan and Git whitespace check passed.

Requirement-level evidence: [TARGET_ACCEPTANCE.md](TARGET_ACCEPTANCE.md).
