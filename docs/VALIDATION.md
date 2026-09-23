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
