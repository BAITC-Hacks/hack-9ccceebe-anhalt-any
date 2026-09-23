# Team integration

The official target path is specified in [TARGET_FORECAST_CONTRACT.md](TARGET_FORECAST_CONTRACT.md).
Old `docs/CONTRACTS.md` describes legacy/demo interfaces, not historical target acceptance.

## Integrated work

- B PR #2, `ml-model-mode-a` through `abc7060`: frozen Kelmarsh model and raw-kW adapter.
  Kept as an independent observed-weather MODE A example, including native 10-minute data.
- B PR #3, `feature/goldwind-model` through `e7e1f44`: organizer-data contract and hourly audit loader.
  Reviewed and integrated; corrected turbine assignment for reordered DataFrame indices with a regression test.
  Preparation only: target provider, trained model and target metrics remain absent. A user-supplied hourly export has now been received and audited; its metadata and native source files remain missing.
- C: strict point-in-time forecasting, immutable weather/model versions, 24/48h rolling replay,
  T1/T2/farm results, local/optional OpenAI analysis, CLI/API/UI, acceptance report and tests.
- A: no actual archived forecast adapter delivered in the reviewed remote branches.

## Next handoff

1. Review `data/goldwind/incoming/received_20260923/` and `docs/GOLDWIND_DELIVERY_REVIEW.md`; obtain original inputs and documented timezone, timestamp semantics, normalized power,
   source wind height and data availability. Confirm `config/forecast_protocol.json`, including the
   first Jan 31 origin; do not guess or mark confirmed to bypass readiness.
2. A delivers the archive adapter with actual payload availability evidence, units and 80m conversion.
3. B fills the Goldwind dataset contract, prepares/audits data, trains without using unavailable
   intervals, and delivers a trusted target artifact/provider plus the strict availability manifest.
   Use `FORECAST_*` variables; do not point target settings at Kelmarsh.
4. Review A/B diffs, provenance and dependencies through Git; run the complete tests, then actual commands:

```bash
python -m pytest -q
python scripts/run_target_forecast.py --check
# FORECAST_ORIGIN must be the confirmed aware timestamp:
python scripts/run_target_forecast.py --origin "$FORECAST_ORIGIN" --horizon 48 --output results/target.json
python scripts/replay_february.py --horizon 48 --output-dir results/february
# Add --actuals only once organizer February facts are available.
```

5. Check real-source time availability independently. Fixtures do not establish archive validity.
   Record numeric coverage and real failures. Keep overlapping origins. Missing actuals leave metrics null.
6. Smoke-test the UI. Optional `--with-agent` requires configured OpenAI key/model, and must preserve numbers.
7. Commit substantive progress and push to the team repository at least hourly during the hackathon.
   Merge reviewed/tested integration into main. Never publish fabricated target forecasts or quality claims.

Current reproducible blocked attempts and empty delivery CSVs are in `reports/target-acceptance/`.
See [TARGET_ACCEPTANCE.md](TARGET_ACCEPTANCE.md) for requirement-level verification and remaining blockers.
