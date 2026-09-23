"""C-owned offline MODE A reader. This is not Person A's forecast weather API."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.schemas import ForecastError
from src.config import ROOT

HOLDOUT = ROOT / "data/holdout.csv"
COLUMNS = {"wind_speed": "wind_speed_ms", "temperature": "temperature_c",
           "wind_direction": "wind_direction_deg", "power": "actual_power_kw"}
SOURCE = "Kelmarsh SCADA 2017, observed holdout; Zenodo 16807551 (CC BY 4.0)"


def load_observations(turbine_id, observation_date, horizon_h, *, path: Path = HOLDOUT):
    try:
        source = pd.read_csv(path)
        required = {"timestamp", "turbine_id", "rated_power", *COLUMNS}
        if not required.issubset(source.columns):
            raise ValueError("missing columns")
        source = source[source.turbine_id == turbine_id].copy()
        if source.empty:
            raise ForecastError("No holdout data for this Kelmarsh turbine")
        source["timestamp"] = pd.to_datetime(source.timestamp, utc=True, errors="raise")
        source = source.set_index("timestamp").sort_index()
        if not source.index.is_unique or source.index.hasnans:
            raise ForecastError("Observed data has duplicate or invalid timestamps")
        if not (source.index == source.index.floor("10min")).all():
            raise ForecastError("Observed timestamps must follow the native 10-minute grid")
        start = pd.Timestamp(observation_date, tz="UTC")
        expected = pd.date_range(start, periods=horizon_h * 6, freq="10min")
        if start < source.index.min() or expected[-1] > source.index.max():
            raise ForecastError("Choose full holdout days 2017-11-08 through 2017-12-31; horizon must fit")
        if not source.rated_power.eq(2050).all():
            raise ForecastError("Holdout capacity differs from the published Kelmarsh registry")
        frame = source.rename(columns=COLUMNS).reindex(expected)
        for column in COLUMNS.values():
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame[list(COLUMNS.values())].replace([np.inf, -np.inf], np.nan)
    except ForecastError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise ForecastError("Cannot read valid Kelmarsh holdout data") from exc
