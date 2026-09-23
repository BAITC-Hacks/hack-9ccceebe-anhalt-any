# Integration contracts v1

Modules are loaded through `DATA_MODULE` and `ML_MODULE` (Python dotted paths).
`src/data/demo.py` and `src/ml/demo.py` are isolated demo adapters, not replacements for Person A/B.
Person A owns weather retrieval and feature construction. Person B owns trained model,
preprocessing, inference and output unit conversion. The agent imports neither model internals nor training code.

## Turbines

| ID | Latitude | Longitude | Rated kW | Hub m | Model |
| --- | --- | --- | --- | --- | --- |
| T1 | 43.645150 | 78.535604 | 2500 | 80 | Goldwind GW109/2500 |
| T2 | 43.643198 | 78.538828 | 2500 | 80 | Goldwind GW109/2500 |

Coordinates, nameplate power and hub height were supplied by the team. Combined nameplate capacity is 5000 kW.
Cut-in 3 m/s and cut-out 25 m/s in configuration are provisional rule thresholds,
not verified GW109 manufacturer specifications. Confirm them before operational use.
Forecasts are per turbine; a farm aggregation endpoint is not implemented.

## Person A — DATA

```python
def get_archival_weather(lat: float, lon: float, run_date: datetime.date,
                         horizon_h: int) -> pandas.DataFrame: ...
def build_features(df: pandas.DataFrame) -> pandas.DataFrame: ...
```

`run_date` means issuance at 00:00 UTC. Horizon is 1–72 hours; index is a unique,
timezone-aware hourly DatetimeIndex beginning at that midnight. Each row covers
`[timestamp, timestamp + 1h)`. No timezone guessing, duplicate timestamps, or silent interpolation.
Missing requested hours are reindexed to NaN and flagged; rows with invalid weather do not enter ML.

Required weather columns:

- `wind_speed_ms`: m/s at hub height (80 m).
- `temperature_c`: degrees Celsius.
- Additional numeric columns may be used by the feature builder.

Required DataFrame attrs:

```python
frame.attrs = {
    "source": "provider-and-model-version",
    "kind": "archived_forecast",  # or "reanalysis"; synthetic is demo-only
    "issued_at": "2026-02-09T18:00:00Z",  # required for archived_forecast
    "wind_height_m": 80.0,
}
```

`issued_at` must be timezone-aware and no later than run_date 00:00 UTC.
Do not label historical observations, reanalysis, or today's reconstructed forecast as
an archived forecast available on the run date. If 80 m wind is derived from 10/100 m data,
record the transformation in Person A's source metadata and apply it consistently in training/inference.
The local cache preserves provenance and keys provider module + coordinates + run date + horizon + schema version.
Cache is read first, so a previously cached date works without the external service.
Invalidate the cache when changing provider code/feature-relevant weather semantics.

`build_features` returns nonempty numeric features in model order with exactly the
same index as weather. No label/actual power, dropped rows, reordered index, or future leakage.
Preprocessing fitted on training data belongs in the saved ML pipeline. NaN/nonfinite
feature rows are excluded from inference and flagged via non-finite power.

## Person B — ML

```python
def load_model(path: pathlib.Path) -> object: ...
def predict(model: object, features: pandas.DataFrame) -> numpy.ndarray: ...
```

Output: 1D float array shape `(len(features),)` in **kW**, in the input row order.
Convert MW × 1000 or normalized power × rated kW in your adapter, not in the agent.
No implicit clipping: negative/above-rated/non-finite results remain visible and are flagged.
`MODEL_PATH` is a local file/directory, never an untrusted upload.
Supply model version, training cutoff, feature list/order, units, dependencies, validation scores,
and relevant turbine IDs in model documentation. Model must be trained before the backtest period
for a prospective quality claim; the orchestrator cannot infer this from a generic model artifact.
The current loader path is shared; implement turbine-specific routing in an agreed adapter if required.

## Agent / result

```python
run_forecast(turbine_id, forecast_date, horizon_h=24,
             latitude=None, longitude=None, *, settings=None, with_agent=True)
```

Coordinates default to the turbine registry. Explicit coordinates must match it.
Returns `ForecastResult` from `src/agent/schemas.py` with hourly predictions, provenance,
statistics, deterministic flags, Pydantic `Analysis`, source/reason of analysis and demo flag.
All JSON non-finite values become null. Missing/physically invalid hours make total energy null;
`valid_hours` and status `ok/partial/invalid` describe coverage. Rule flags do not all invalidate
numeric power: an hourly jump can be a valid but suspicious prediction.

Analysis is one compact request per forecast run (24h in backtest), never one call per hour.
`OPENAI_MODEL` has no guessed default. Missing configuration, refusal, invalid schema, timeout,
or unexpected anomaly codes produce deterministic fallback. SDK automatic retries are disabled.
The LLM cannot modify hourly predictions, statistics, deterministic flags, risk floor or uncertainty note.
LLM prose is advisory and may still be wrong; deterministic result fields remain authoritative.

## Actuals

CSV columns: `timestamp,turbine_id,actual_power_kw`.
Timestamps must include timezone and mark hourly interval starts; actual power is hourly mean kW.
Unique `(turbine_id,timestamp)` rows; values finite and nonnegative. Missing rows are allowed,
but never forward-filled or treated as zero. Backtest joins by both keys, reports match coverage,
and computes MAE, RMSE, bias only on valid matched points.
