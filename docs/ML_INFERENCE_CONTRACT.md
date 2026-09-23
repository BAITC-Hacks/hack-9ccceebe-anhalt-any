# ML provider contract — Kelmarsh MODE A

For A (DATA) and C (agent/integration). Target contract inspected at MVP commit
`bc67f05864a2a5e7a6ef12c060872c9ccfc427c7`; MVP PR #1 was open at implementation.
Coordination: https://github.com/BAITC-Hacks/hack-9ccceebe-anhalt-any/pull/1#issuecomment-5792929432

```text
ML_MODULE=src.ml.provider
MODEL_PATH=models/power_model.joblib
```

`load_model(path)` loads the trusted published artifact without training. `predict(model,
features)` returns a float64 ndarray of shape `(len(features),)` in **kW**, preserving input
row order. No clipping: negative, above-rated and nonfinite raw output stays visible to C's rules.
The existing `src.ml.model.predict` DataFrame API retains its previous behavior.

## Exact input — one explicitly named turbine per call

`features` is a nonempty pandas DataFrame with a unique timezone-aware DatetimeIndex;
NaT is forbidden. UTC is canonical; other aware timezones are converted to UTC.
Index is valid measurement time, never issuance time. Unsorted input is accepted and
not reordered. Cadence is not changed: validated on native 10-minute SCADA. Feeding hourly
averaged inputs is mechanically supported but has not been validated as hourly mean power.
Do not sum 10-minute predictions as if they were 1-hour energy in the agent.

Exactly these numeric columns are required, in any column order. No string/bool/complex dtypes:

| Column | Unit and meaning | Validity |
|---|---|---|
| wind_speed_ms | m/s, observed nacelle wind at turbine hub | finite, >=0 |
| temperature_c | °C, outdoor nacelle ambient temperature | finite |
| wind_direction_deg | degrees, original Kelmarsh Wind direction signal | finite, [0,360) |

No NaN/Inf input accepted in this adapter. C filters invalid rows before inference; the existing
core tree model's ability to accept some NaNs does not change this strict integration boundary.
Extra columns (including target, lag, pre-encoded features and turbine_id) are rejected.
C's broad temperature/weather sanity rules remain responsible for physical flags.

**Direction:** the downloaded signal is `Wind direction`, not `Vane position` or `Nacelle position`.
The publisher's mapping specifies degrees but does not explicitly establish from/to or north reference.
This provider preserves the native signal convention; it does not silently rotate by 180° or substitute yaw.
Agreement with an external weather provider's meteorological north/from convention remains unverified.
Missing direction is an error, not zero/north or NaN substitution. A must explicitly provide this
additional field; the two-field DATA baseline alone cannot serve the saved artifact.

Required attrs (not columns, to preserve C's numeric feature check):

```python
features.attrs['turbine_id'] = 'Kelmarsh 1'  # exact identity from the request/registry
features.attrs['wind_height_m'] = 78.5
```

| Exact turbine IDs | Hub height | Confirmed rated capacity |
|---|---:|---:|
| Kelmarsh 1, 2, 4, 5 (each prefixed `Kelmarsh `) | 78.5 m | 2050 kW |
| Kelmarsh 3, Kelmarsh 6 | 68.5 m | 2050 kW |

Heights/capacities come from `data/raw/kelmarsh/Kelmarsh_WT_static.csv`.
SCADA sensor elevation is treated as the published hub height; detailed sensor offset is not provided.
No wind-height extrapolation is performed in ML; matching the nominal height alone does not validate
forecast/reanalysis wind against nacelle SCADA. Unknown IDs, wrong heights and missing context fail closed.
Specifically **T1/T2 Goldwind GW109/2500 are rejected**, never aliased to Kelmarsh or scaled to 2500 kW.

## Saved preprocessing — A must not recompute

The artifact constructs direction sin/cos, hour/month/day-of-year cyclic encodings in UTC,
and a training-only turbine one-hot vocabulary. Wind speed and temperature are retained.
No scaler, imputer, target lags, rolling features or fitted future statistics are needed upstream.
The provider only maps the three raw weather names and the index to this saved pipeline.

Raw normalized prediction is converted to kW using the matched turbine capacity from verified
artifact metadata, cross-checked against the saved model config and Kelmarsh registry. Capacity
is never taken from request weather or a generic station nominal. `predicted_power` (clipped
DataFrame output) is deliberately not used; the adapter reads `raw_prediction`.

## Minimal handoff required from C — proposal awaiting confirmation

The current orchestrator has request.turbine_id but does not put it in features; therefore
setting ML_MODULE alone is insufficient. C should attach the authoritative request turbine ID
and validated wind height after DATA feature construction and before ML inference, preserving
attrs through `features.loc[usable]`. Suggested placement in C's code (not implemented by B):

```python
features.attrs['turbine_id'] = request.turbine_id
features.attrs['wind_height_m'] = weather.attrs['wind_height_m']
```

A returns the three raw columns and preserves the DatetimeIndex. No changes to predict's two-argument
signature are needed. A/C own the final agreement and station registry. Until that handoff and a
Kelmarsh registry are present, current MVP intentionally fails rather than guessing an ID.
The MVP weather-kind enum does not include observed SCADA; a real observed-data path must also
be agreed by A/C. Do not relabel SCADA as archived_forecast or claim the hourly demo is a
validated 10-minute MODE A backtest. B changes neither orchestrator nor UI.

## Frozen artifact and reproducible evaluation

- Published artifact: `5f7b569`, HistGradientBoostingRegressor; no retraining for this adapter.
- Training cutoff: **2017-09-14 02:40 UTC**.
- Validation/model-selection cutoff: **2017-11-07 13:00 UTC**.
- Test: **2017-11-07 13:10 through 2017-12-31 23:50 UTC**, 47006 rows.
- Training environment: Python 3.13.13; NumPy 2.5.3; pandas 2.3.3; scikit-learn 1.9.1;
  joblib 1.6.0; LightGBM 4.7.0 was compared, not selected. Full lock: requirements-trained.txt.
- Model SHA256: `e2e996fb4289384dd2b776270ce010dfdf1076bb7bf0ea94b19819e88c6e8618`.
- Holdout SHA256 (canonical CRLF): `0d25be3cf926332c2f923f16f5c6c7a6b008b105be74634c6cebafa303e51701`.

```powershell
python -m scripts.verify_ml_provider --report reports/provider_verification.json
python -m scripts.evaluate_model data/holdout.csv --mode A
python -m pytest -q -W error::DeprecationWarning
```

The first command checks hashes, loads the saved artifact, checks original MODE A metrics,
runs all holdout rows through the per-turbine adapter and compares raw predictions/metrics.
It never fits, resamples, tunes on, or overwrites holdout. Original metadata/artifact are unchanged.
Provider contract and new evidence are separate additive documents/reports.

Raw normalized MAE **0.04044131383**, RMSE **0.07082455021**, R² **0.9530159122**.
Raw kW MAE **82.90469335**, RMSE **145.19032792**. The earlier clipped MAE 82.82 kW is
not the agent/provider metric. Those clipped results remain available through the original API.
These are Kelmarsh **MODE A** results only. MODE B requires genuine archived forecasts;
neither transfer to Goldwind nor operational forecast performance has been validated.
