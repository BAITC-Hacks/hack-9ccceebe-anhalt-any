# Goldwind T1/T2 — explicit organizer dataset contract (not trained)

Status, 2026-09-23: **organizer files have not been provided**. User confirmed there is only
Kelmarsh and synthetic demo in the checked integration build. Timezone, statistical timestamp
semantics and normalized target definition are unknown. No substitute dataset is permitted.
No Goldwind model has been trained; there are no Goldwind validation/forecast metrics or artifact.
Kelmarsh files, provider and published results are unchanged and remain a separate verified example.

Work starts from integration commit `be26361`; PR #2 is merged into `codex/energy-agent-mvp`,
while MVP PR #1 is not yet merged into main. Current branch: `feature/goldwind-model`.
Coordination with A/C: https://github.com/BAITC-Hacks/hack-9ccceebe-anhalt-any/pull/1#issuecomment-5793484746

## Required confirmations before parsing/aggregation

Copy `config/goldwind_dataset.template.json` and fill it using organizer documentation.
Nulls deliberately block execution; they are not defaults to be inferred from values.
Paths are relative to the contract JSON directory (absolute paths also accepted).

| Setting | Must be confirmed |
|---|---|
| files | Separate exact file path for T1 and T2; no mapping by filename guess. If mixed files exist, request an explicit partition policy. |
| file_identity_source | Organizer evidence linking each file to the physical turbine. Optional mapped turbine_id column must agree. |
| columns | Logical → physical: timestamp, wind_speed, temperature, target; optionally available_at and turbine_id. |
| dataset_source / metadata_source | Actual organizer dataset version and document/person confirming semantics. |
| timezone | IANA zone or explicit fixed offset; never inferred from local PC or numerical timestamps. Historical timezone changes must match source semantics. |
| timestamp_format / available_at_format | Without explicit formats, only ISO year-month-day timestamps are accepted. Ambiguous day/month strings and numeric epochs are not guessed. |
| timestamp_label | interval_start or interval_end; instant samples are not automatically interval means. |
| measurement_semantics | interval_mean: applies to power and weather values used in this first loader. |
| interval_minutes | Confirmed constant integer divisor of 60 (1–60). Irregular, overlapping or off-grid intervals need a reviewed extension. |
| wind_unit / temperature_unit | m/s or km/h; C or K. Conversion is explicit. |
| target_unit | kW, W, MW, fraction, or percent; target scale is never guessed. |
| normalized_reference_kw | For fraction/percent, explicitly confirmed per-turbine denominator. Current contract requires T1=T2=2500 kW. Knowing nameplate power alone does not confirm normalization. |
| wind_height_m | Confirmed measurement heights for each source turbine; cannot be inferred from 80 m hub height. |
| availability_source | Provenance of release timestamps or publication-delay rule. |
| available_at column or release_delay_minutes | Actual availability, not just statistical timestamp. Delay zero requires explicit confirmation. |

Known from station registry: T1/T2 are Goldwind GW109/2500, **2500 kW each**, hub height **80 m**.
Source wind sensor height and normalized target denominator remain unknown until documentation arrives.
No wind-direction requirement: the future model uses wind speed, temperature, calendar features and
explicit turbine identity. A does not build calendar encodings or learned transformations.

## First hourly policy (proposal, explicit and conservative)

- Input interval means with fixed documented duration, on a nonoverlapping UTC grid after timestamp conversion.
- End labels are shifted backward by the documented duration; output timestamp is UTC interval start.
- Output interval is `[timestamp, timestamp + 1 hour)`.
- Eligible hour requires **60/60 valid minutes** for both weather features and power. Coverage policy = 1.0.
- Weather and power means are weighted by observed interval duration; no arithmetic mean across unequal undocumented durations.
- `energy_kwh_observed = sum(power_kw * observed_duration_hours)`. For a complete hour it equals mean kW ×1h.
- Incomplete hours retain observed-duration means, energy and coverage for audit only. They are not used for training
  and are not extrapolated to full-hour energy. Missing power/energy stays NaN, never zero.
- Exact duplicates are excluded once, with a flag in raw audit; conflicting duplicates stop preparation.
- Missing/nonfinite power, negative/nonfinite wind and temperature outside -90…65 °C prevent full usable coverage;
  raw values are preserved with flags. Wind >60 m/s is flagged for review, not automatically deleted.
- Negative and above-rated **finite power are retained** as observed target values, with flags. No hidden clipping.
- Zero power is marked as an observation, not proof of shutdown. Actual stoppage/curtailment cause needs organizer
  status codes and a confirmed mapping; no inferred outages or fabricated zeros.
- Training period is March 1, 2023 through January 31, 2026 **in confirmed source timezone**, converted to UTC.
  Outside-period rows, including February 2026, stay in raw audit and never enter training eligibility.
- Entirely missing hours and missing months are represented by coverage zero and NaN measurements.

Outputs: source_rows.csv, raw_audit.csv, hourly_audit.csv and preparation_report.json with source hashes,
file/turbine mapping, observed interval deltas, flags and full-period coverage per turbine.
Structural errors preserve source_rows.csv plus preparation_error.json; original files are never overwritten.
The command refuses a nonempty output directory, preventing destruction of previous audit results.

## Forecast origin / leakage gate — C confirmation still required

`first_forecast_origin` must be an exact offset-aware timestamp; `forecast_origin_source` records C's agreement.
The final allowed hour satisfies both interval_end <= origin **and available_at <= origin**.
Hour availability is the maximum availability of its constituent observations. A Jan 31 interval released after
the first run is excluded even though its statistical date is in the historical period.
No assumption that “trained through January 31” is sufficient. Until C confirms origin, preparation may produce
audits but does **not** export eligible_hourly.csv or permit training. Helper `training_rows` enforces this gate.

## Future target provider contract (reserved; not an existing artifact)

```text
ML_MODULE=src.ml.goldwind_provider
MODEL_PATH=models/goldwind/power_model.joblib
```

These are reserved integration names; **the provider/artifact will be implemented and verified after real
data semantics and training are available**. Do not switch production settings to these names yet.

- Input: nonempty numeric DataFrame with exactly `wind_speed_ms` (m/s) and `temperature_c` (°C).
- Unique timezone-aware hourly DatetimeIndex labels interval starts; UTC canonical; no sorting/dropping/reordering.
- Explicit `features.attrs['turbine_id']` = T1 or T2 and `wind_height_m` matching confirmed model training schema.
- Missing/nonfinite feature rows are filtered/flagged upstream by C; provider rejects them rather than imputing silently.
- Wind direction is not required. Calendar cycles and turbine encoding will be saved inside the ML pipeline.
- `load_model(path)` loads without fit; `predict(model, features)` returns raw 1D kW array of input length.
- No hidden clipping, default turbine, Kelmarsh mapping or unconfirmed normalized-to-kW conversion.
- A forecast-vs-measured-weather distribution shift must be recorded. MODE A validation is not MODE B forecast quality.

## Commands and next steps

After receiving actual files **and confirmed metadata**, fill a contract then run:

```powershell
python -m scripts.prepare_goldwind --contract config/goldwind_dataset.confirmed.json --output data/goldwind/prepared
python -m pytest tests/test_goldwind_data.py -q
```

`goldwind_dataset.confirmed.json` is not supplied because its values are unknown; use the template as a checklist.
The template itself must fail with a list of missing confirmations. CSV/Parquet are supported; no second weather
downloader and no automatic download of another station is added. For Parquet install an existing pandas engine
only if required by actual files.

Next, once data and origin are confirmed: inspect real EDA/coverage, agree any needed source-specific parser,
train mean baseline and sklearn HistGradientBoosting using past→future splits entirely before February,
verify availability cutoffs for every training/validation run, save a separate artifact with schema/units,
cutoffs/dependency versions and metrics, implement the reserved adapter and test reload/order/raw kW.
No training CLI or metrics are claimed in this preparation-only PR. Estimated first artifact: 1–2 hours after
receipt of usable documented files and C's origin confirmation; source cleanup may change that estimate.
