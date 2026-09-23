from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
DEFAULT_HOURLY = [
    "temperature_2m",
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_speed_120m",
    "wind_direction_100m",
    "surface_pressure",
]


def _normalise_run(run_date: str | datetime, run_hour_utc: int = 0) -> str:
    ts = pd.Timestamp(run_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ts = ts.normalize() + pd.Timedelta(hours=run_hour_utc)
    if run_hour_utc not in (0, 6, 12, 18):
        raise ValueError("ECMWF IFS global run_hour_utc should be one of 0, 6, 12, 18")
    return ts.strftime("%Y-%m-%dT%H:%M")


def get_archival_weather(
    lat: float,
    lon: float,
    run_date: str | datetime,
    horizon_h: int = 48,
    *,
    run_hour_utc: int = 0,
    hourly: Iterable[str] = DEFAULT_HOURLY,
    cache_dir: str | Path | None = None,
    retries: int = 4,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Get one archived ECMWF IFS run without look-ahead leakage.

    `run_date` + `run_hour_utc` selects the exact model initialisation. Open-Meteo
    documents ECMWF IFS HRES single runs from March 2024. Forecast output becomes
    available only after model computation, so in production choose a run known to
    be available at the decision time.
    """
    if not 1 <= horizon_h <= 240:
        raise ValueError("horizon_h must be between 1 and 240")

    run = _normalise_run(run_date, run_hour_utc)
    variables = list(hourly)
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": ",".join(variables),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "run": run,
        "forecast_hours": int(horizon_h),
    }

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        cache_path = cache_dir / f"weather_{run.replace(':','')}_{lat:.5f}_{lon:.5f}_{horizon_h}h_{key}.csv"
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["datetime"])
            cached.attrs["source"] = "cache"
            cached.attrs["run"] = run
            return cached

    sess = session or requests.Session()
    last_error = None
    for attempt in range(retries):
        try:
            resp = sess.get(SINGLE_RUN_URL, params=params, timeout=timeout)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                raise requests.HTTPError(f"retryable HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            payload = resp.json()
            if "hourly" not in payload or "time" not in payload["hourly"]:
                raise ValueError(f"Unexpected weather response: {payload}")

            data = {"datetime": pd.to_datetime(payload["hourly"]["time"], utc=True)}
            for v in variables:
                vals = payload["hourly"].get(v)
                if vals is None:
                    raise ValueError(f"Variable {v!r} missing in response")
                data[v] = vals
            df = pd.DataFrame(data).head(horizon_h)
            df["forecast_run_utc"] = pd.Timestamp(run, tz="UTC")
            df["lead_hours"] = ((df["datetime"] - df["forecast_run_utc"]).dt.total_seconds() / 3600).astype(int)
            df["latitude"] = float(payload.get("latitude", lat))
            df["longitude"] = float(payload.get("longitude", lon))
            df.attrs["source"] = "open-meteo-single-runs"
            df.attrs["run"] = run
            if cache_path is not None:
                df.to_csv(cache_path, index=False)
            return df
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"Weather API failed after {retries} attempts for run={run}: {last_error}")

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def get_previous_runs_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    *,
    lead_days: tuple[int, ...] = (1, 2),
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch fixed 24h/48h lead-time forecasts for diagnostics/bias studies.

    This is not a replacement for exact Single Runs when reconstructing a particular
    initialization, but is useful for training bias correction at controlled lead times.
    """
    base_vars = ["temperature_2m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m"]
    hourly = []
    for v in base_vars:
        for d in lead_days:
            if d not in range(0, 8):
                raise ValueError("lead_days must be 0..7")
            hourly.append(f"{v}_previous_day{d}")
    params = {
        "latitude": float(lat), "longitude": float(lon),
        "hourly": ",".join(hourly),
        "wind_speed_unit": "ms", "timezone": "UTC",
        "start_date": start_date, "end_date": end_date,
    }
    sess = session or requests.Session()
    r = sess.get(PREVIOUS_RUNS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    h = payload.get("hourly", {})
    if "time" not in h:
        raise ValueError(f"Unexpected Previous Runs response: {payload}")
    out = pd.DataFrame({"datetime": pd.to_datetime(h["time"], utc=True)})
    for v in hourly:
        out[v] = h.get(v)
    return out
