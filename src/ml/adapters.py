from typing import Protocol
import pandas as pd


class WeatherProvider(Protocol):
    """Person A implements this or passes an existing DataFrame directly.

    Output logical fields: timestamp (valid time), wind_speed, temperature,
    optional wind_direction/turbine_id; forecasts additionally include issued_at.
    No power target is invented or obtained from weather endpoints.
    """
    def get_weather(self, start: str, end: str, turbine_id: str) -> pd.DataFrame: ...


def select_forecast_vintage(forecasts, origin, timezone="UTC"):
    """Latest available forecast at an explicit origin, one row per valid time/turbine.

    Supply all weather columns for each vintage; no merging of different issue times.
    origin is NOT the weather observation time. Provider must guarantee issued_at
    represents real availability (include publication latency upstream if necessary).
    """
    from .data import timestamps
    frame = forecasts.copy()
    frame["timestamp"] = timestamps(frame.timestamp, timezone)
    frame["issued_at"] = timestamps(frame.issued_at, timezone)
    cutoff = timestamps(pd.Series([origin]), timezone).iloc[0]
    keys = ["timestamp"] + (["turbine_id"] if "turbine_id" in frame else [])
    available = frame[(frame.issued_at <= cutoff) & (frame.timestamp > cutoff)]
    if available.duplicated(keys + ["issued_at"]).any():
        raise ValueError("Duplicate forecast records for same issue/valid time")
    return available.sort_values("issued_at").drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
