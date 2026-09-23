# Target T1/T2 forecast contract (Person A / Person B / Person C)

Status: integration and contract tests implemented; real organizer data, weather provider
and target model are **not supplied**. No accuracy claim for T1/T2. Kelmarsh is an independent
observed-weather example and never a fallback for this path.

## Organizer protocol — required before enabling the target scenario

`config/forecast_protocol.json` intentionally has `confirmed: false` and unknown timezone/hour.
A must obtain the dataset and written definitions; B must use the same definitions for training.
C then records evidence and confirms the protocol. Coordinates do not establish SCADA timezone.
Do not toggle confirmation merely to make readiness pass.

Required fields:

| Field | Meaning |
| --- | --- |
| `timezone` | Confirmed IANA timezone for source labels, origin calendar dates and February evaluation |
| `daily_origin_hour` | Agreed daily local launch hour, 0–23 |
| `first_lead_hour` | 0 or 1; agreed first hourly interval relative to origin |
| `interval_label` | Currently `start`: valid_time denotes [valid_time, valid_time + 1 h) |
| `normalization` | Currently `fraction_of_rated_power`: normalized_power = raw kW / rated kW |
| `evidence` | Dataset dictionary / organizer confirmation and conversion decisions |
| `confirmed` | True only after checking all of the above |

Only these hourly-start/fraction conventions are implemented. If organizers use end labels,
percent, a different normalization or energy, A/B must explicitly convert and document it;
otherwise extend the contract before enabling. Never infer this from column names alone.
Hourly forecasts are average kW; hourly energy is kW × 1 hour, not a meter reading.

Origins and all persisted timestamps carry an offset and are normalized to UTC. Naive timestamps
and non-hourly origins are rejected. The launch date and forecast valid_time are different fields.
A 48-hour run has exactly 48 times per turbine, beginning at origin + first_lead_hour.
Daily replay launches on local calendar dates Jan 31–Feb 28 inclusive. DST ambiguous/nonexistent
launch hours fail explicitly. Evaluation includes only valid_time in local Feb 1 inclusive–Mar 1 exclusive.
Forecasts outside February remain in CSV; the manifest records the exact evaluation window.

Training may not include a full Jan 31 interval ending after the first Jan 31 origin. B must choose
a cutoff whose interval completion **and** publication delay precede that origin, including every
preprocessing fit, validation, hyperparameter selection and feature selection step.

## Person A: strict archived-weather adapter

Set `FORECAST_DATA_MODULE` to an importable module implementing:

```python
get_forecast_weather(lat: float, lon: float, forecast_origin, horizon_h: int) -> pandas.DataFrame
build_features(weather: pandas.DataFrame) -> pandas.DataFrame
```

Input origin is aware UTC. Output index is aware `DatetimeIndex` of hourly **valid_time**.
Each row has `wind_speed_ms` (m/s at 80 m), `temperature_c` (°C),
`weather_issued_at` and `weather_available_at` (aware timestamps).
Optional `wind_direction_deg` is meteorological direction **from**, clockwise from north,
[0,360); required only if listed by the model. No invented direction values.
Confirm the publisher's convention and any wind-height transformation with B and record the method.

DataFrame attrs:

```python
{
    "kind": "archived_forecast",
    "source": "publisher + model/product/version + archive URL (no credentials)",
    "wind_height_m": 80.0,
    "availability_basis": "publisher_timestamp",  # or documented_delay
    "availability_evidence": "Evidence for publication of this actual payload revision",
    "turbine_id": "T1",  # optional; if supplied must match the request
}
```

Multiple issue/revision pairs may be returned. C picks the latest issue with issue ≤ origin and
availability ≤ origin, then its latest eligible revision. All selected rows come from that pair;
missing hours stay missing, never filled from another vintage. Availability must refer to the
actual payload revision, not its original issue before a later backfill.

If publication timestamp is absent, a verified documented publication delay is required.
Free-text evidence is structurally checked, **not independently certified by code**. Before real
acceptance, inspect the publisher's archive retention/revision policy and prove historical
availability. Actual weather, reanalysis, stitched historical observations and synthetic data are rejected.
No source is currently approved or configured for T1/T2.

Cache key includes coordinates, height, turbine, origin, horizon, module and protocol. Snapshots
store query identity, selected vintage, evidence, values and SHA256. `--refresh` fetches again;
failures never silently reuse old data. A known issue/availability pair cannot change its values;
refresh cannot regress to an older known vintage or silently change source. Intentional source
changes use a separate `FORECAST_CACHE_DIR`. First-seen historical claims still need external evidence.

Normalized weather retains only wind speed, temperature and optional direction. `build_features`
receives this frame; output index must remain identical and its real numeric columns must exactly
match the model manifest. Derived features are allowed, but no observed target/future SCADA is an input.
If a fitted pipeline derives its own features, return raw inputs instead of computing them twice.

## Person B: target model and manifest

Set `FORECAST_ML_MODULE` to a module with `load_model(path)` and `predict(model, features)`.
Return a one-dimensional array of **raw kW**, in unchanged row order. No clipping. If the model
predicts a fraction, the adapter must convert using the requested turbine's 2500 kW capacity.
Features always have authoritative attrs `turbine_id` and `wind_height_m=80` from the registry.
No target or actual generation is supplied to inference. Only trusted team model artifacts may be loaded.

Provide `models/goldwind/power_model.joblib` and `models/goldwind/forecast_manifest.json`.
The artifact can be a bundle of two target models if the adapter selects by turbine_id.
Manifest schema (`src/forecast/contracts.py:ModelManifest`):

| Field | Required value / evidence |
| --- | --- |
| `model_version`, `artifact_sha256` | Stable model version and SHA256 of exact artifact bytes |
| `data_source` | `organizer_scada` |
| `provenance` | Organizer file names, hashes, training preparation and split protocol |
| `turbine_ids` | Both `T1`, `T2` |
| `rated_power_kw` | `{"T1":2500,"T2":2500}` |
| `wind_height_m` | `{"T1":80,"T2":80}` |
| `feature_columns` | Ordered raw/derived names, at least wind_speed_ms and temperature_c |
| `output_unit` | `kW` |
| `normalized_target_definition` | `fraction_of_rated_power`, after confirmed conversion |
| `timezone`, `interval_minutes` | Match organizer protocol, 60 minutes |
| `training_data_available_until` | Latest actual availability among all fit data, aware ISO |
| `selection_data_available_until` | Latest availability among every validation/selection datum |
| `training_target_interval_end` | Latest training target interval end, not its start label |
| `selection_target_interval_end` | Latest selection target interval end |
| `availability_evidence` | Explanation of SCADA reporting delay and cutoff verification |

Each interval end must be ≤ its availability cutoff; both cutoffs must be ≤ origin.
Artifact hash, T1/T2 identity, capacity, wind height, hourly step and normalization are checked.
A manifest assertion alone is not proof of training lineage: B must deliver reproducible preparation,
training split and data hashes. The existing Kelmarsh artifact is rejected as target model.

## Person C: execution, cache, failures and metrics

```bash
python scripts/run_target_forecast.py --check
# Set ORIGIN to the confirmed aware historical launch timestamp before this command:
python scripts/run_target_forecast.py --origin "$ORIGIN" --horizon 48 --output results/target.json
python scripts/replay_february.py --horizon 48 --output-dir results/february
# Only once real hourly facts are received:
python scripts/replay_february.py --horizon 48 --actuals data/organizer/february_actuals.csv --output-dir results/february-evaluated
```

Set `FORECAST_DATA_MODULE`, `FORECAST_ML_MODULE`, `FORECAST_MODEL_PATH`,
`FORECAST_MANIFEST_PATH`, `FORECAST_PROTOCOL_PATH`, optional `FORECAST_CACHE_DIR` and
`FORECAST_RESULTS_DIR`. They are separate from legacy/demo provider variables.
Exit 0 means ready/complete; exit 2 means blocked/partial. Missing inputs produce explicit blocked
JSON, not a pretend successful forecast. Unconfirmed protocol produces header-only replay CSVs
and a blocked manifest because the schedule itself cannot be determined.

`--refresh` accepts only historically eligible revisions. `run_id` hashes origin, horizon,
protocol, registry, manifest/model hash, weather hashes, engine source and provider package source.
Local src and dependency lock changes invalidate calculations. External dynamic code, remote services
inside feature builders and dependency versions outside the lock are unsupported for reproducible replay.
Weather and result versions are immutable. File publication is atomic; POSIX process locks prevent
UI/API/CLI with the same result directory from duplicating inference/LLM for the same run.
Runtime is Python 3.12 on macOS/Linux; a Windows locking adapter has not been implemented.

Default replay reuses accepted cache snapshots; retain the original cache, artifacts, lock and source
revision with each delivery. Use a separate output directory per delivery. `--refresh` may produce a
new run_id but never overwrites earlier result snapshots. Failed model attempts are saved separately
and can be retried; weather/provider failures are explicit. No silent station/data/model substitution.

Only identical (forecast_origin, valid_time) pairs are summed across T1/T2. Missing or physically
invalid turbine values leave station `power_kw` / `energy_kwh` null; `raw_power_kw` retains both finite
raw values for unbiased error metrics, including negatives/overcapacity. Never treat missing as zero.

Output CSVs: `predictions.csv` and `farm.csv`; manifest includes configuration, origins, coverage,
failures, provenance, run IDs and file hashes. Actuals CSV must contain
`timestamp,turbine_id,actual_power_kw`: aware hourly interval-start labels, raw kW, only T1/T2,
unique turbine/time keys. Missing values remain missing. Actuals enter scoring only, never prediction.
No February facts currently supplied; metrics are null. If supplied, report MAE/RMSE/bias,
finite paired counts, per turbine, exact lead hour, first/second 24-hour block, and station.
Overlapping origins remain distinct predictions; do not average or pick the best after the fact.

`--with-agent` enables one compact optional OpenAI analysis per station/run, not one per hour.
Numbers are determined by local ML and rules. Missing key/model, provider failure, refusal,
invalid output or timeout uses deterministic analysis without changing numeric predictions.
No live OpenAI request has been verified. API: GET `/readiness`, POST `/target-forecast`.
Legacy `/forecast` and MODE A `/observed` remain separate examples.
