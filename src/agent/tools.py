from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import tempfile
from datetime import date

import numpy as np
import pandas as pd

from src.agent.schemas import ForecastError
from src.config import ROOT, Settings

log = logging.getLogger(__name__)


def provider(module_name: str, required: tuple[str, ...]):
    if not module_name:
        raise ForecastError("Provider not configured; set DATA_MODULE and ML_MODULE, or use --demo")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ForecastError(f"Cannot import provider {module_name}") from exc
    if not all(callable(getattr(module, name, None)) for name in required):
        raise ForecastError(f"Provider {module_name} does not satisfy the contract")
    return module


def normalize_weather(frame: pd.DataFrame, run_date: date, horizon: int) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ForecastError("Weather is empty")
    frame = frame.copy()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ForecastError("Weather requires a timezone-aware DatetimeIndex")
    frame.index = frame.index.tz_convert("UTC")
    if frame.index.has_duplicates:
        raise ForecastError("Weather contains duplicate timestamps")
    if (frame.index != frame.index.floor("h")).any():
        raise ForecastError("Weather timestamps must be hourly boundaries")
    required = {"wind_speed_ms", "temperature_c"}
    if not required.issubset(frame.columns):
        raise ForecastError("Weather requires wind_speed_ms and temperature_c")
    if frame.attrs.get("kind") not in {"synthetic", "archived_forecast", "reanalysis"}:
        raise ForecastError("Weather provenance kind is missing or invalid")
    if not frame.attrs.get("source"):
        raise ForecastError("Weather source is missing")
    start = pd.Timestamp(run_date, tz="UTC")
    if frame.attrs["kind"] == "archived_forecast":
        try:
            issued = pd.Timestamp(frame.attrs["issued_at"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ForecastError("Archived forecast requires issued_at") from exc
        if pd.isna(issued) or issued.tzinfo is None or issued > start:
            raise ForecastError("Weather issued_at is later than run date or has no timezone")
    expected = pd.date_range(start, periods=horizon, freq="h")
    frame = frame.reindex(expected)
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[list(required)] = frame[list(required)].replace([np.inf, -np.inf], np.nan)
    return frame


def load_weather(settings: Settings, data, lat: float, lon: float,
                 run_date: date, horizon: int) -> pd.DataFrame:
    if settings.demo:
        log.info("weather cache: bundled demo")
        return normalize_weather(data.get_archival_weather(lat, lon, run_date, horizon), run_date, horizon)
    identity = {"version": 1, "module": settings.data_module, "lat": lat, "lon": lon,
                "run_date": run_date.isoformat(), "horizon": horizon}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = settings.cache_dir / f"{digest}.json"
    if cache.is_file():
        try:
            payload = json.loads(cache.read_text())
            if payload["identity"] != identity:
                raise ValueError("cache identity mismatch")
            frame = pd.DataFrame(payload["rows"]).set_index("timestamp")
            frame.index = pd.to_datetime(frame.index, utc=True)
            frame.attrs = payload["attrs"]
            frame = normalize_weather(frame, run_date, horizon)
            if frame.attrs["kind"] == "synthetic":
                raise ValueError("synthetic production cache")
            log.info("weather cache: hit")
            return frame
        except (ValueError, KeyError, TypeError, ForecastError):
            log.warning("weather cache: invalid, fetching provider")
    log.info("weather fetch: %s", settings.data_module)
    try:
        frame = normalize_weather(data.get_archival_weather(lat, lon, run_date, horizon), run_date, horizon)
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError("Weather provider unavailable and no usable cache exists") from exc
    if frame.attrs["kind"] == "synthetic":
        raise ForecastError("Production cannot consume synthetic weather")
    # Cache only complete data. Atomic replacement prevents concurrent partial writes.
    if not frame[["wind_speed_ms", "temperature_c"]].isna().any().any():
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            serial = frame.rename_axis("timestamp").reset_index()
            serial["timestamp"] = serial["timestamp"].astype(str)
            payload = {"identity": identity, "attrs": frame.attrs,
                       "rows": json.loads(serial.to_json(orient="records"))}
            with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, delete=False) as tmp:
                json.dump(payload, tmp, allow_nan=False)
                temp_name = tmp.name
            os.replace(temp_name, cache)
        except (OSError, ValueError, TypeError):
            log.warning("weather cache: write skipped")
    return frame


def load_providers(settings: Settings):
    data_name = "src.data.demo" if settings.demo else settings.data_module
    ml_name = "src.ml.demo" if settings.demo else settings.ml_module
    return (provider(data_name, ("get_archival_weather", "build_features")),
            provider(ml_name, ("load_model", "predict")))
