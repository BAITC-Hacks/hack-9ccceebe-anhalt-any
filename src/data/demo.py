"""Offline fixture adapter, not Person A's production weather implementation."""
from datetime import date

import pandas as pd

from src.config import ROOT
from src.agent.schemas import ForecastError


def get_archival_weather(lat: float, lon: float, run_date: date,
                         horizon_h: int) -> pd.DataFrame:
    if (lat, lon) not in {(43.645150, 78.535604), (43.643198, 78.538828)}:
        raise ForecastError("Demo fixture only supports configured T1/T2 coordinates")
    frame = pd.read_csv(ROOT / "data/demo/weather.csv", parse_dates=["timestamp"])
    frame = frame.set_index("timestamp")
    frame.index = pd.to_datetime(frame.index, utc=True)
    expected = pd.date_range(pd.Timestamp(run_date, tz="UTC"), periods=horizon_h, freq="h")
    if not expected.isin(frame.index).all():
        raise ForecastError("Demo dates supported: 2026-01-31 through 2026-02-28 (up to 72h)")
    frame = frame.loc[expected].copy()
    frame.attrs = {"source": "bundled-synthetic-fixture-v1", "kind": "synthetic", "wind_height_m": 80.0}
    return frame


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["wind_speed_ms", "temperature_c"]].copy()
