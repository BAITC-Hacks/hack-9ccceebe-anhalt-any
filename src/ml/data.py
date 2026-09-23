import logging
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
WEATHER = ["wind_speed", "temperature", "wind_direction"]


def timestamps(values, timezone):
    def parse(value):
        t = pd.Timestamp(value)
        if pd.isna(t):
            raise ValueError("Missing timestamp")
        if t.tzinfo is None:
            t = t.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        return t.tz_convert("UTC")
    return pd.to_datetime(values.map(parse), utc=True)


def prepare(df, config, training=False):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Expected a nonempty pandas DataFrame")
    out = df.rename(columns={v: k for k, v in config.columns.items()}).copy()
    if out.columns.duplicated().any():
        raise ValueError("Column mapping creates duplicate logical names")
    missing = {"timestamp", "wind_speed", "temperature"} - set(out)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out["timestamp"] = timestamps(out.timestamp, config.timezone)
    if "turbine_id" not in out:
        out["turbine_id"] = "__single__"
    if out.turbine_id.isna().any():
        raise ValueError("Missing turbine_id")
    out["turbine_id"] = out.turbine_id.astype(str)
    for col in WEATHER + ["power", "normalized_power", "rated_power"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="raise")
            if np.isinf(out[col]).any():
                raise ValueError(f"Infinite values in {col}")
    if config.wind_speed_unit == "km/h":
        out["wind_speed"] /= 3.6
    elif config.wind_speed_unit != "m/s":
        raise ValueError("wind_speed_unit must be m/s or km/h")
    if config.temperature_unit == "K":
        out["temperature"] -= 273.15
    elif config.temperature_unit != "C":
        raise ValueError("temperature_unit must be C or K")
    quality = {"rows_loaded": len(out), "missing": out.isna().sum().to_dict(),
               "invalid_wind_speed": int((out.wind_speed < 0).sum()),
               "suspicious_wind_above_60": int((out.wind_speed > 60).sum()),
               "suspicious_temperature": int(((out.temperature < -80) | (out.temperature > 60)).sum()),
               "negative_power": int((out.get("power", pd.Series(dtype=float)) < 0).sum())}
    LOG.info("Data quality: %s", quality)
    if quality["invalid_wind_speed"] or out.wind_speed.isna().any():
        raise ValueError("wind_speed must be present, finite and >= 0; no rows silently removed")
    if "wind_direction" in out:
        quality["directions_wrapped"] = int(((out.wind_direction < 0) | (out.wind_direction >= 360)).sum())
        out["wind_direction"] %= 360
    if config.mode == "B":
        if not config.forecast_provenance or "issued_at" not in out:
            raise ValueError("MODE B requires forecast_provenance and archived issued_at")
        out["issued_at"] = timestamps(out.issued_at, config.timezone)
        lead = (out.timestamp - out.issued_at).dt.total_seconds() / 3600
        if (lead < config.min_lead_hours).any():
            raise ValueError("Forecast issued after allowed origin; invalid lead time")
        # Multiple forecast vintages require explicit selection by upstream adapter.
    if training:
        relevant = [c for c in ["timestamp", "turbine_id", *WEATHER, "power", "normalized_power", "rated_power", "issued_at"] if c in out]
        quality["exact_duplicates_removed"] = int(out.duplicated(subset=relevant).sum())
        out = out.drop_duplicates(subset=relevant)
        if out.duplicated(["timestamp", "turbine_id"]).any():
            raise ValueError("Conflicting timestamp/turbine observations or multiple forecast vintages")
        out = out.sort_values(["timestamp", "turbine_id"]).reset_index(drop=True)
    LOG.info("Final data quality: %s", quality)
    return out, quality


def capacity(df, config):
    if not config.rated_power_verified:
        return None
    if isinstance(config.rated_power, dict):
        result = df.turbine_id.map({str(k): v for k, v in config.rated_power.items()})
    elif config.rated_power is not None:
        result = pd.Series(float(config.rated_power), index=df.index)
    elif "rated_power" in df:
        result = df.rated_power
    else:
        raise ValueError("Verified capacity is missing")
    result = pd.to_numeric(result, errors="raise")
    if result.isna().any() or (~np.isfinite(result)).any() or (result <= 0).any():
        raise ValueError("All rated capacities must be finite, known and positive")
    return result


def choose_target(df, config):
    if not config.target_source:
        raise ValueError("Training requires target_source identifying real SCADA/dataset observations")
    rated = capacity(df, config)
    if "power" in df:
        if config.power_unit not in ("W", "kW", "MW"):
            raise ValueError("Declare power_unit W/kW/MW; rated_power must use the same unit")
        target = df.power / rated if rated is not None else df.power
        kind = "normalized_power" if rated is not None else "absolute_power"
    elif "normalized_power" in df and config.normalized_target_confirmed:
        target, kind = df.normalized_power, "normalized_power"
    else:
        raise ValueError("Real power or explicitly confirmed normalized_power target required")
    if target.isna().any() or (~np.isfinite(target)).any():
        raise ValueError("Target contains missing/nonfinite values; resolve upstream explicitly")
    return target, kind


def chronological_split(df, config):
    unique = df.timestamp.sort_values().unique()
    a, b = int(len(unique) * config.train_fraction), int(len(unique) * (config.train_fraction + config.validation_fraction))
    if a < 1 or b <= a or b >= len(unique):
        raise ValueError("Too few unique timestamps for chronological split")
    train = df.index[df.timestamp < unique[a]]
    val = df.index[(df.timestamp >= unique[a]) & (df.timestamp < unique[b])]
    test = df.index[df.timestamp >= unique[b]]
    if len(train) < config.min_train_rows or min(len(val), len(test)) < config.min_eval_rows:
        raise ValueError("Insufficient observations for reliable temporal train/validation/test")
    return train, val, test
