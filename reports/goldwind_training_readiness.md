# T1/T2 training and provider readiness — 2026-09-23

Continued from main c43262d after merged MVP/PR #3. Working tree was clean before fetch;
fast-forward preserved all integration work. Scope: ML only, no edits to weather, agent,
UI, shared backtest, source CSV bytes, Kelmarsh artifact/metadata/provider or holdout.

## Delivered

- Mixed hourly export reader with explicit physical turbine mapping and confirmed sample-count semantics.
- Full-hour eligibility, partial observed-duration energy, raw audit preservation and no zero filling.
- Gated mean/HistGradientBoosting training, chronological validation, publication-boundary purges.
- Separate T1/T2 provider returning raw float64 kW in input order; no direction/lag requirement.
- Saved pipeline, source lineage, units, dependency versions, cutoffs and C-compatible forecast manifest.
- Frozen-artifact MODE A reproduction CLI; no training during loading/evaluation.

Commands and exact input contract: [GOLDWIND_MODEL_CONTRACT](../docs/GOLDWIND_MODEL_CONTRACT.md).

## Verification

`python -m pytest -q -W error::DeprecationWarning tests/test_goldwind_model.py tests/test_goldwind_data.py tests/test_ml.py tests/test_provider.py tests/test_cli.py`

122 passed, no skips. Includes 32 new training/provider tests and existing loader/ML regression tests.
Tests fit deterministic **artificial software fixtures only** in temporary directories; those scores
are not recorded as station metrics. No final Goldwind artifact was created or uploaded.

Verified: hourly coverage and explicit turbine remapping, unknown metadata/origin blocks fit,
chronological publication purges, source-hash mismatch rejection, loaded pipeline same-order output,
missing artifact/Kelmarsh rejection, raw negative/overcapacity/NaN predictions reaching unchanged
agent rules, and manifest validation against C's actual ModelManifest schema. Test evaluation and
inference succeed with estimator/transformer fit methods disabled.

Windows test environment: Python 3.13, pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0.
Boundary checks additionally use existing project dependencies pydantic 2.13.5 and python-dotenv 1.2.3.
No new project dependency added. UI/API/full forecast replay was not tested in this ML change.
Current C-owned forecast orchestrator imports POSIX fcntl; Windows whole-system portability remains
outside this provider's verification and is reported to C, without changing orchestrator code.

## Remaining evidence

1. Author/organizer definitions: source timezone/history, hour labels, aggregation and sample_count,
   wind/temperature/target units, normalization denominator, physical T1/T2 identity and sensor height.
2. SCADA publication availability and C's exact first forecast_origin, consistent with confirmed protocol.
3. Actual training and holdout evaluation only after these confirmations; no February tuning.
4. A's archived weather must prove actual issue/publication/valid-time coverage, model/product revision,
   wind height and step. Successful downloads alone do not prove point-in-time eligibility. NOAA/NCEI
   GFS is a candidate for A to investigate if the current archive cannot satisfy these requirements;
   this ML change neither verifies nor downloads that archive.
5. February actual power is needed only for February error metrics. Hidden organizer actuals do not
   prevent submitting forecasts once inputs/model are ready. Those metrics must remain unknown.

No final model, Goldwind accuracy, forecast-system validation or jury score is claimed. The received
CSV value range is not used to infer semantics. The 80 m deployment contract does not prove source
sensor height; a mismatch needs A/C agreement and evidence before readiness can pass.
