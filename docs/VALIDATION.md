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
