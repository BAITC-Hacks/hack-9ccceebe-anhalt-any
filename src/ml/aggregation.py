import numpy as np
import pandas as pd


def circular_mean(values):
    radians = np.deg2rad(pd.Series(values).dropna().to_numpy())
    if not len(radians):
        return np.nan
    s, c = np.sin(radians).mean(), np.cos(radians).mean()
    if np.hypot(s, c) < 1e-10:
        return np.nan  # Antipodal or uniform directions: no defined mean.
    angle = np.rad2deg(np.arctan2(s, c)) % 360
    return 0.0 if np.isclose(angle, 360) else float(angle)


def aggregate_predictions(predictions, frequency="hourly", timezone="UTC"):
    """Sample-weighted means PER TURBINE; not energy and not fleet total.

    For irregular sampling resample upstream to a regular grid. Counts expose coverage.
    Missing absolute/normalized output remains NaN when conversion is unavailable.
    """
    rules = {"hourly": "h", "daily": "D", "monthly": "MS"}
    if frequency not in rules:
        raise ValueError("frequency must be hourly/daily/monthly")
    df = predictions.copy()
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True).dt.tz_convert(timezone)
    if "turbine_id" not in df:
        df["turbine_id"] = "__single__"
    if df.duplicated(["timestamp", "turbine_id"]).any():
        raise ValueError("Select one prediction/forecast vintage per timestamp and turbine before aggregation")
    mapping = {"wind_speed": "mean_wind_speed_ms", "wind_direction": "mean_wind_direction_deg",
               "predicted_power": "predicted_active_power", "predicted_normalized_power": "normalized_active_power",
               "temperature": "mean_ambient_temperature_c"}
    for col in mapping:
        if col not in df:
            df[col] = np.nan
    grouped = df.groupby(["turbine_id", pd.Grouper(key="timestamp", freq=rules[frequency])], observed=True)
    result = grouped.agg({c: circular_mean if c == "wind_direction" else "mean" for c in mapping})
    result["observation_count"] = grouped.size()
    return result.rename(columns=mapping).reset_index().rename(columns={"timestamp": "statistical_time"})
