"""Organizer-only, explicit-semantics Goldwind history preparation. No downloader."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import logging
import re

import numpy as np
import pandas as pd

from .data import timestamps

LOG = logging.getLogger(__name__)
TURBINES = {"T1": 2500.0, "T2": 2500.0}


@dataclass
class GoldwindContract:
    files: dict = field(default_factory=dict)  # exact T1/T2 -> organizer file path
    columns: dict = field(default_factory=dict)  # logical -> source names
    dataset_source: str | None = None
    metadata_source: str | None = None
    file_identity_source: str | None = None
    timezone: str | None = None
    timestamp_format: str | None = None  # None permits ISO year-month-day only, not inferred day/month
    available_at_format: str | None = None
    timestamp_label: str | None = None  # interval_start / interval_end
    interval_minutes: int | None = None
    measurement_semantics: str | None = None  # interval_mean, never instantaneous inferred as mean
    wind_unit: str | None = None
    temperature_unit: str | None = None
    target_unit: str | None = None  # kW / W / MW / fraction / percent
    normalized_reference_kw: dict | None = None
    wind_height_m: dict | None = None
    availability_source: str | None = None
    release_delay_minutes: float | None = None  # alternative: mapped available_at column
    first_forecast_origin: str | None = None
    forecast_origin_source: str | None = None
    minimum_hour_coverage: float = 1.0

    def validate(self, training=False):
        required = ["dataset_source", "metadata_source", "file_identity_source", "timezone",
                    "timestamp_label", "interval_minutes", "measurement_semantics", "wind_unit",
                    "temperature_unit", "target_unit", "wind_height_m", "availability_source"]
        missing = [name for name in required if getattr(self, name) is None or getattr(self, name) == ""]
        if missing:
            raise ValueError(f"Confirm organizer metadata; missing: {', '.join(missing)}")
        if set(self.files) != set(TURBINES) or not all(isinstance(p, str) and p for p in self.files.values()):
            raise ValueError("files must explicitly map both T1 and T2 to real organizer paths")
        if len(set(self.files.values())) != 2:
            raise ValueError("Use separate confirmed exports for T1/T2; a mixed file needs an explicit partition policy")
        if not {"timestamp", "wind_speed", "temperature", "target"}.issubset(self.columns):
            raise ValueError("columns must map timestamp, wind_speed, temperature, target")
        if len(set(self.columns.values())) != len(self.columns):
            raise ValueError("Duplicate source column mapping")
        if not all(isinstance(v, str) and v for v in self.columns.values()):
            raise ValueError("Every mapped source column needs a confirmed nonempty name")
        if self.timestamp_label not in {"interval_start", "interval_end"} or self.measurement_semantics != "interval_mean":
            raise ValueError("Confirm start/end labels and interval_mean semantics; instantaneous samples need a separate reviewed policy")
        n = self.interval_minutes
        if isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > 60 or 60 % n:
            raise ValueError("interval_minutes must be a confirmed integer divisor of 60")
        if self.wind_unit not in {"m/s", "km/h"} or self.temperature_unit not in {"C", "K"}:
            raise ValueError("Confirm wind m/s|km/h and temperature C|K")
        if self.target_unit not in {"W", "kW", "MW", "fraction", "percent"}:
            raise ValueError("Confirm target unit; scale is never inferred from values")
        if self.target_unit in {"fraction", "percent"} and self.normalized_reference_kw != TURBINES:
            raise ValueError("Normalized target requires confirmed per-turbine reference T1=T2=2500 kW")
        if (not isinstance(self.wind_height_m, dict) or set(self.wind_height_m) != set(TURBINES)
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) or v <= 0
                       for v in self.wind_height_m.values())):
            raise ValueError("Confirm training wind measurement heights for both turbines (do not infer from hub height)")
        if "available_at" not in self.columns:
            delay = self.release_delay_minutes
            if delay is None or isinstance(delay, bool) or not np.isfinite(delay) or delay < 0:
                raise ValueError("Confirm publication delay or map per-row available_at; zero delay is not assumed")
        if self.minimum_hour_coverage != 1.0:
            raise ValueError("Initial model requires 60/60 valid minutes; incomplete hours remain audit-only")
        if training:
            if not self.first_forecast_origin or not self.forecast_origin_source:
                raise ValueError("C must confirm exact first_forecast_origin and forecast_origin_source before training")
            origin = pd.Timestamp(self.first_forecast_origin)
            if pd.isna(origin) or origin.tzinfo is None:
                raise ValueError("first_forecast_origin must include UTC offset/timezone")


def read_organizer_files(contract, base_dir=Path(".")):
    contract.validate()
    frames, provenance = [], []
    resolved_paths = [(base_dir / filename).resolve() for filename in contract.files.values()]
    if len(set(resolved_paths)) != 2:
        raise ValueError("The same file cannot stand in for two separate turbine exports")
    for turbine, filename in contract.files.items():
        path = (base_dir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Organizer data missing for {turbine}: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError("Supported organizer exports: CSV/Parquet; do not infer undocumented workbook sheets")
        mapped_id = contract.columns.get("turbine_id")
        if mapped_id and (mapped_id not in frame or not frame[mapped_id].eq(turbine).all()):
            raise ValueError(f"File identity disagrees with confirmed {turbine} mapping")
        frame = frame.copy()
        frame["_organizer_turbine"] = turbine
        frame["_source_file"] = path.name
        frame["_source_row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)
        provenance.append({"turbine_id": turbine, "file": path.name, "rows": len(frame),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.concat(frames, ignore_index=True), provenance


def _source_time(values, timezone, explicit_format=None):
    if explicit_format:
        parsed = pd.to_datetime(values, format=explicit_format, errors="raise")
        return timestamps(parsed, timezone)
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        iso = values.map(lambda v: isinstance(v, str) and bool(re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", v)))
        if not iso.all():
            raise ValueError("Non-ISO timestamps require an explicit timestamp_format/available_at_format; no date-order guessing")
    return timestamps(values, timezone)


def prepare_hourly(raw, contract):
    """Return full raw audit, hourly audit, summary. No zero filling or guessed units.

    v1 supports constant nonoverlapping intervals aligned to the UTC hourly grid.
    More complex/instantaneous sampling must be reviewed once actual data exists.
    """
    contract.validate()
    if raw.empty or "_organizer_turbine" not in raw:
        raise ValueError("Nonempty data with confirmed file identities required")
    data = raw.copy().reset_index(drop=True)
    canonical = raw.reset_index(drop=True).rename(columns={v: k for k, v in contract.columns.items()})
    if canonical.columns.duplicated().any():
        raise ValueError("Source mapping creates duplicate logical columns")
    for name in ["timestamp", "wind_speed", "temperature", "target"]:
        if name not in canonical:
            raise ValueError(f"Missing mapped column: {name}")
    # Keep identity attached to the same row after resetting a caller's index.
    data["turbine_id"] = data["_organizer_turbine"]
    if not data.turbine_id.isin(TURBINES).all():
        raise ValueError("Unconfirmed turbine identity")
    duration = pd.Timedelta(np.timedelta64(contract.interval_minutes, "m"))
    stamp = _source_time(canonical.timestamp, contract.timezone, contract.timestamp_format)
    data["interval_start"] = stamp if contract.timestamp_label == "interval_start" else stamp - duration
    data["interval_end"] = data.interval_start + duration
    if not data.interval_start.eq(data.interval_start.dt.floor(f"{contract.interval_minutes}min")).all():
        raise ValueError("Source intervals must align to the documented UTC hour grid; no rounding/splitting guessed")
    for name in ["wind_speed", "temperature", "target"]:
        data[f"numeric_{name}"] = pd.to_numeric(canonical[name], errors="coerce")
    data["wind_speed_ms"] = data.numeric_wind_speed / (3.6 if contract.wind_unit == "km/h" else 1)
    data["temperature_c"] = data.numeric_temperature - (273.15 if contract.temperature_unit == "K" else 0)
    factor = {"W": .001, "kW": 1., "MW": 1000., "fraction": 2500., "percent": 25.}[contract.target_unit]
    data["power_kw"] = data.numeric_target * factor
    if "available_at" in contract.columns:
        data["available_at"] = _source_time(canonical.available_at, contract.timezone, contract.available_at_format)
    else:
        delay = pd.Timedelta(float(contract.release_delay_minutes), unit="min")
        data["available_at"] = data.interval_end + delay
    if (data.available_at < data.interval_end).any():
        raise ValueError("Interval-mean data cannot be available before the interval ends")
    keys = ["turbine_id", "interval_start"]
    relevant = keys + ["wind_speed_ms", "temperature_c", "power_kw", "available_at"]
    data["exact_duplicate"] = data.duplicated(relevant)
    unique = data.loc[~data.exact_duplicate]
    if unique.duplicated(keys).any():
        raise ValueError("Conflicting duplicate observations; review raw source before aggregation")
    start = pd.Timestamp("2023-03-01", tz=contract.timezone).tz_convert("UTC")
    end = pd.Timestamp("2026-02-01", tz=contract.timezone).tz_convert("UTC")
    data["within_training_period"] = (data.interval_start >= start) & (data.interval_end <= end)
    data["invalid_wind"] = ~np.isfinite(data.wind_speed_ms) | (data.wind_speed_ms < 0)
    data["invalid_temperature"] = ~np.isfinite(data.temperature_c) | ~data.temperature_c.between(-90, 65)
    data["missing_power"] = ~np.isfinite(data.power_kw)
    data["negative_power"] = data.power_kw < 0
    data["above_rated_power"] = data.power_kw > 2500
    data["zero_power_observation"] = data.power_kw.eq(0)  # Not proof of a shutdown; status info required.
    data["suspicious_wind_above_60"] = data.wind_speed_ms > 60
    data["valid_joint"] = ~(data.invalid_wind | data.invalid_temperature | data.missing_power)
    subset = data.loc[data.within_training_period & ~data.exact_duplicate].copy()
    subset["timestamp"] = subset.interval_start.dt.floor("h")
    minutes = contract.interval_minutes
    for source, valid, key in [("wind_speed_ms", ~subset.invalid_wind, "wind"),
                               ("temperature_c", ~subset.invalid_temperature, "temp"),
                               ("power_kw", ~subset.missing_power, "power")]:
        subset[f"{key}_minutes"] = valid.astype(int) * minutes
        subset[f"{key}_weighted"] = subset[source].where(valid, 0) * minutes
    subset["joint_minutes"] = subset.valid_joint.astype(int) * minutes
    sums = [f"{key}_{suffix}" for key in ["wind", "temp", "power"] for suffix in ["minutes", "weighted"]] + ["joint_minutes"]
    grouped = subset.groupby(["turbine_id", "timestamp"])
    hourly = grouped[sums].sum()
    hourly["available_at"] = grouped.available_at.max()
    hourly["source_rows"] = grouped.size()
    full_index = pd.MultiIndex.from_product([list(TURBINES), pd.date_range(start.ceil("h"), end.floor("h"), freq="h", inclusive="left")], names=["turbine_id", "timestamp"])
    hourly = hourly.reindex(full_index).reset_index()
    for col in [c for c in sums if c.endswith("minutes")] + ["source_rows"]:
        hourly[col] = hourly[col].fillna(0)  # Coverage/counts only, never power/weather.
    for key, name in [("wind", "wind_speed"), ("temp", "temperature"), ("power", "power")]:
        hourly[name] = hourly[f"{key}_weighted"] / hourly[f"{key}_minutes"].replace(0, np.nan)
    hourly["energy_kwh_observed"] = hourly.power_weighted / 60
    hourly.loc[hourly.power_minutes.eq(0), "energy_kwh_observed"] = np.nan
    hourly["coverage"] = hourly.joint_minutes / 60
    if (hourly.coverage > 1).any():
        raise ValueError("Overlapping intervals; hourly coverage exceeds 100%")
    hourly["interval_end"] = hourly.timestamp + pd.Timedelta(np.timedelta64(1, "h"))
    hourly["complete"] = hourly.joint_minutes.eq(60)
    hourly["rated_power"] = 2500.
    summary = {"raw_rows": len(data), "exact_duplicates_excluded": int(data.exact_duplicate.sum()),
               "rows_outside_training_period": int((~data.within_training_period).sum()),
               "complete_hours": int(hourly.complete.sum()), "incomplete_or_missing_hours": int((~hourly.complete).sum()),
               "per_turbine_complete_hours": hourly.groupby("turbine_id").complete.sum().to_dict(),
               "flags": {c: int(data[c].sum()) for c in ["invalid_wind", "invalid_temperature", "missing_power", "negative_power", "above_rated_power", "zero_power_observation", "suspicious_wind_above_60"]},
               "sampling_interval_minutes_confirmed": minutes,
               "observed_interval_deltas_seconds": {t: g.interval_start.sort_values().drop_duplicates().diff().dt.total_seconds().value_counts().head(8).to_dict() for t, g in unique.groupby("turbine_id")},
               "contract": asdict(contract), "mode": "A", "shutdown_status": "Zero power alone does not establish stoppage cause; organizer status mapping required"}
    LOG.info("Goldwind preparation: %s", {k: v for k, v in summary.items() if k != "contract"})
    return data, hourly, summary


def training_rows(hourly, contract):
    contract.validate(training=True)
    origin = pd.Timestamp(contract.first_forecast_origin).tz_convert("UTC")
    eligible = hourly.complete & (hourly.available_at <= origin) & (hourly.interval_end <= origin)
    selected = hourly.loc[eligible].copy()
    if set(selected.turbine_id) != set(TURBINES):
        raise ValueError("Both T1/T2 need complete, available organizer hours before forecast_origin")
    return selected.sort_values(["timestamp", "turbine_id"]).reset_index(drop=True)
