# Received T1/T2 hourly export

The user supplied six files from `Новая папка 3` on 2026-09-23. Exact byte copies are
preserved under `incoming/received_20260923/`; SHA256 hashes and independent inspection
are recorded in `reports/goldwind_delivery_audit.json`. Only this received batch is
explicitly included in Git; future generated audit/training outputs remain ignored.

The source data is already aggregated: `historical_hourly.csv` has 50,784 rows for the
labels T1/T2, 2023-03-11 through 2026-01-31. It includes wind, temperature, power and
sample_count. The file does not include February facts or timezone offsets. The native
10-minute inputs and aggregation code are not in this delivery. Power values lie within
0–1, but the denominator and any prior clipping/filtering are unconfirmed.

The user reports having no additional metadata. Do not silently confirm turbine mapping,
timezone, interval labels, normalization, measurement height or SCADA availability.
The supplied quality report explicitly assigns T1/T2 by upload order, not verified identity.

`historical_hourly_features.csv` is a received derived export, NOT the approved forecast
feature contract. Observed power/wind lags and rolling values are not automatically known
at the origin for every future hour. Start B's review with the base hourly table, preserving
sample_count and restricting any provisional fitting to fully observed hours with an
explicit temporal split. This is not authorization to label an unconfirmed model as ready.

`weather_requests_manifest.csv` is a plan of 58 URLs, not a cache of weather responses.
The two supplied Python files are preserved as source material and are NOT imported into
production or executed by the app. They do not yet implement the strict forecast contract.

Full findings and A/B handoff: `docs/GOLDWIND_DELIVERY_REVIEW.md`.
Forecast contract: `docs/TARGET_FORECAST_CONTRACT.md`.

Reproduce the read-only audit from the repository root:

```bash
python scripts/audit_goldwind_delivery.py --source data/goldwind/incoming/received_20260923 --output results/goldwind-delivery-audit.json
```
