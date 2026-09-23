# Team integration

The official T1/T2 path is specified in [TARGET_FORECAST_CONTRACT.md](TARGET_FORECAST_CONTRACT.md).
Legacy/demo interfaces remain in [CONTRACTS.md](CONTRACTS.md).

## Integrated work

- Main `3caf01f`: final WindAgent browser UI, launcher, API and frontend tests.
- A `feature/data-weather` through `13e3edb`, PR #4: received-data audit/cleaning,
  features, ECMWF Single Runs client, integrity-checked cache and strict weather adapter.
- B `feature/goldwind-model` through `202b6ba`, PR #5: explicit mixed-hourly CSV
  contract, coverage handling, temporal training/evaluation and raw-kW Goldwind provider.
  No real Goldwind model has been trained; software fixtures are not station validation.
- Earlier B PRs #2/#3: frozen Kelmarsh model, MODE A adapter and organizer-data preparation.
- C: strict origin/availability guards, immutable results, 24/48-hour forecasts,
  both turbines and farm aggregation, February replay, deterministic/optional OpenAI analysis.

A/B branches were merged through Git. The documentation conflict retained the team-selected
schedule and B's implemented provider. Tests exercise A's real feature adapter, a B artifact
trained on artificial software data, and C's orchestrator together at 24/48 hours.

Default target providers now resolve to `src.data.weather` and `src.ml.goldwind_provider`.
Both UIs use the first origin from `config/forecast_schedule.json`:
**2026-01-31T23:00:00+05:00** (18:00 UTC), then daily 23:00 Asia/Almaty, first lead 1.
The model's confirmed source/calendar timezone may differ from this execution calendar;
all cutoff and weather comparisons use aware UTC instants.

## Remaining data handoff

1. Confirm the received CSV's source timezone, interval labels, physical turbine mapping,
   units/normalization, sample_count/aggregation, sensor height and release delay.
   Fill `config/goldwind_hourly.template.json`; unknown fields remain null.
2. Supply evidence that each weather payload/version was available at its historical origin.
   Downloading an old run today is not that evidence. A's availability policy stays unconfirmed.
3. Train B's Goldwind bundle from eligible pre-origin records; inspect coverage, splits,
   validation and manifest. Keep Kelmarsh separate. Confirm the target protocol using evidence.
4. Execute the strict forecast and February replay. Add actuals only if real February
   generation is delivered; absent actuals leave accuracy metrics null.

```bash
python scripts/run_target_forecast.py --check
python scripts/run_target_forecast.py --origin 2026-01-31T23:00:00+05:00 --horizon 48 --output results/target.json
python scripts/replay_february.py --horizon 48 --output-dir results/february
```

No guessed source metadata, fabricated production artifacts or demo substitution enables
this path. Current MVP evidence and working commands: [final verification](../reports/final-mvp/README.md).
