# Goldwind organizer-data readiness — 2026-09-23

**Updated status: hourly export received / TRAINING BLOCKED BY UNCONFIRMED SEMANTICS.**

Since the original B report below, the user supplied an aggregated T1/T2 export.
C audited 50,784 rows, copied all six supplied files byte-for-byte into
`data/goldwind/incoming/received_20260923/`, and recorded SHA256 checksums.
The user has no additional metadata. See `docs/GOLDWIND_DELIVERY_REVIEW.md`.
The following original preparation report describes the state before that delivery.

The user explicitly confirmed that organizer T1/T2 files/path have not been provided,
and timezone, statistical timestamp meaning and normalized target definition are unconfirmed.
Known nameplate capacities (2500 kW each) do not establish the normalization denominator.
No substitute station was downloaded. No Goldwind artifact, provider implementation,
validation metric or end-to-end forecast claim is produced by this PR.

## Completed

- Fetched origin; started `feature/goldwind-model` from current `codex/energy-agent-mvp` at `be26361`.
  PR #2 is integrated there; MVP PR #1 remains open against main. Working tree was clean before switching.
- Read the current DATA/ML/agent contracts; C now passes authoritative turbine ID and wind height attrs.
- Added `GoldwindContract` with explicit file/turbine identities, timezone, units, interval means,
  timestamp labels/formats, normalization reference and publication-availability provenance.
- Local CSV/Parquet reader only; no weather API or remote data substitution.
- Source rows and hashes preserved; invalid/conflicting structures fail rather than guess.
- Hourly duration-weighted preparation with full 60/60 valid-minute coverage for training eligibility;
  incomplete hours, raw out-of-range values and duplicate flags remain auditable.
- Energy uses observed duration; missing generation never becomes zero.
- March 2023–January 2026 boundaries use confirmed source timezone. February and later rows excluded.
- Actual availability and interval end must both precede/equal C's exact first forecast_origin.
  Without C confirmation, preparation may save audit but no eligible training export is produced.
- A/C notified in MVP PR #1; exact model-side proposed schema documented in GOLDWIND_MODEL_CONTRACT.md.
- Existing Kelmarsh artifact, metadata, holdout and provider were not modified.

## Tests

Command: `python -m pytest -q -W error::DeprecationWarning tests/test_goldwind_data.py tests/test_ml.py tests/test_provider.py tests/test_cli.py`.

**89 passed**: 28 Goldwind loader/contract tests plus 61 existing ML/provider/CLI tests.
Goldwind fixtures are explicitly artificial parser examples, with no training or accuracy metrics.
Checks include fraction versus percent (never inferred), W/kW/MW, explicit timezone/date format,
start/end labels, hourly mean/energy, partial coverage, no zero filling, duplicate conflicts,
negative/above-rated audit retention, file identity, availability cutoff, February exclusion,
and refusal to overwrite earlier audit output.

Test environment: Python 3.13, pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1,
joblib 1.6.0; existing ML environment/lock. No new runtime dependencies added.
The first attempt to collect the entire integrated suite failed because the local ML environment
lacks FastAPI/Streamlit. Agent/UI suites are not claimed passing in this change; ML-owned suites
were then run explicitly. Agent/UI/common-backtest code is unchanged.

## Still required

1. Actual organizer files and documentary mapping to T1/T2.
2. Confirmed timezone, timestamp labels/formats, units, interval semantics and target normalization.
3. Training measurement wind height and availability timestamps/delay.
4. C's exact first forecast_origin and evidence/source of that agreement.
5. Real EDA and any source-specific import adjustments, then baseline/sklearn time validation,
   separate artifact + provider, reload/inference tests and dependency/metric metadata.

Reserved names: `src.ml.goldwind_provider`, `models/goldwind/power_model.joblib`.
**They do not exist yet and must not be configured in the agent.** Wind direction is not planned as required.
The target integration uses `FORECAST_ML_MODULE`, `FORECAST_MODEL_PATH`,
`FORECAST_MANIFEST_PATH` and `FORECAST_PROTOCOL_PATH`; legacy `ML_MODULE`/`MODEL_PATH`
do not configure this strict path. Its required availability manifest and confirmed origin
protocol are specified in [TARGET_FORECAST_CONTRACT.md](../docs/TARGET_FORECAST_CONTRACT.md).
Preparation output is not that manifest: B still must publish the actual artifact hash,
training/selection cutoffs and interval ends, schema, units, heights and provenance.
Training measured-weather validation will be MODE A; forecast-weather deployment shift and MODE B
accuracy require separate evidence. February 2026 will not be used for model selection.

Estimate: first artifact 1–2 hours after receipt of usable documented files and C's origin agreement;
unknown file quality may require revising this estimate. No artifact completion time can be guaranteed
while the required organizer data is absent.


## Integration verification by C

This preparation branch was reviewed and merged into the integration branch. A reordered-index
identity bug was fixed and covered by a regression test (29 Goldwind tests now pass).
The full Python 3.12 integration suite passes 212 tests with one optional LightGBM skip.
This supersedes the limited ML-only environment check above; it does not remove data/training blockers.
