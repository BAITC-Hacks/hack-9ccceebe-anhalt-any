# Target acceptance attempt — 2026-09-23

Actual command outputs with repository configuration; **not real predictions**.

- `python scripts/run_target_forecast.py --check`: exit 2, ready=false.
- `python scripts/run_target_forecast.py --origin 2026-01-31T23:00:00Z --horizon 48`: exit 2, blocked, zero predictions. The timestamp is a diagnostic example, not the confirmed organizer origin.
- `python scripts/replay_february.py --horizon 48 --output-dir results/target-february`: exit 2, blocked, zero launches. Without timezone/schedule there is no honest calendar to run.
- CSVs contain headers only; MAE/RMSE are unavailable.

Blockers: organizer data/definitions and confirmed protocol, actual archived forecast provider, organizer-trained Goldwind model + availability manifest, and February facts for scoring. No fallback was used.

The copied manifest retains original output paths; CSV hashes match these copies. Re-run the commands after A/B deliver real inputs.
