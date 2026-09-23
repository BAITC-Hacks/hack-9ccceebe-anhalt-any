"""Point-in-time weather selection and immutable, versioned request snapshots.

A vintage is the pair (issue time, payload availability time). Availability must
refer to the actual returned revision, not the publication of an earlier version.
The provider is responsible for supplying evidence for that assertion.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.schemas import ForecastError
from src.forecast.contracts import expected_times, utc

log = logging.getLogger(__name__)
ISSUED = "weather_issued_at"
AVAILABLE = "weather_available_at"
REQUIRED = ("wind_speed_ms", "temperature_c")
WEATHER_COLUMNS = (*REQUIRED, "wind_direction_deg")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _timestamp(value, label):
    try:
        return utc(value)
    except (TypeError, ValueError, OverflowError, ForecastError) as exc:
        raise ForecastError(f"Weather {label} must contain timezone-aware timestamps") from exc


def _provenance(frame, turbine_id, turbine):
    attrs = frame.attrs
    if attrs.get("kind") != "archived_forecast":
        raise ForecastError("Target forecast requires archived_forecast weather; observed, reanalysis and synthetic weather are not accepted")
    if not isinstance(attrs.get("source"), str) or not attrs["source"].strip():
        raise ForecastError("Archived weather source is missing")
    if attrs.get("availability_basis") not in {"publisher_timestamp", "documented_delay"}:
        raise ForecastError("Weather availability_basis must be publisher_timestamp or documented_delay")
    if not isinstance(attrs.get("availability_evidence"), str) or not attrs["availability_evidence"].strip():
        raise ForecastError("Weather availability_evidence must document availability of the actual payload revision")
    try:
        height = float(attrs["wind_height_m"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ForecastError("Weather wind_height_m is missing or invalid") from exc
    if isinstance(attrs["wind_height_m"], bool) or not math.isfinite(height) or not math.isclose(height, turbine.hub_height_m, abs_tol=1e-6):
        raise ForecastError("Weather wind_height_m does not match turbine hub height")
    if attrs.get("turbine_id", turbine_id) != turbine_id:
        raise ForecastError("Weather turbine_id does not match requested turbine")
    return {"source": attrs["source"].strip(), "kind": "archived_forecast",
            "wind_height_m": height, "turbine_id": turbine_id,
            "availability_basis": attrs["availability_basis"],
            "availability_evidence": attrs["availability_evidence"].strip()}


def _select(frame, turbine_id, turbine, origin, expected):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ForecastError("Weather provider returned no archived weather")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ForecastError("Weather valid_time requires a timezone-aware DatetimeIndex")
    if frame.columns.has_duplicates:
        raise ForecastError("Weather contains duplicate column names")
    if not set((*REQUIRED, ISSUED, AVAILABLE)).issubset(frame.columns):
        raise ForecastError("Weather requires wind_speed_ms, temperature_c, weather_issued_at and weather_available_at columns")
    attrs = _provenance(frame, turbine_id, turbine)
    data = frame.copy(deep=True)
    data.index = data.index.tz_convert("UTC")
    if data.index.hasnans or (data.index != data.index.floor("h")).any():
        raise ForecastError("Weather valid_time must be on hourly boundaries without missing timestamps")
    # Parse individually: pd.to_datetime(..., utc=True) would silently accept naive timestamps.
    for col in (ISSUED, AVAILABLE):
        data[col] = pd.DatetimeIndex([_timestamp(value, col) for value in data[col]])
    if (data[AVAILABLE] < data[ISSUED]).any():
        raise ForecastError("Weather payload cannot be available before its issue time")
    keys = pd.DataFrame({"valid_time": data.index, ISSUED: data[ISSUED].to_numpy(),
                         AVAILABLE: data[AVAILABLE].to_numpy()})
    if keys.duplicated().any():
        raise ForecastError("Weather contains duplicate valid_time within a single vintage")
    eligible = data[(data[ISSUED] <= origin) & (data[AVAILABLE] <= origin)]
    if eligible.empty:
        raise ForecastError("No weather vintage was issued and available by forecast_origin")
    issue = eligible[ISSUED].max()
    vintage = eligible[eligible[ISSUED] == issue]
    available = vintage[AVAILABLE].max()
    vintage = vintage[vintage[AVAILABLE] == available]
    if not vintage.index.isin(expected).any():
        raise ForecastError("Latest eligible weather vintage contains no requested forecast hours")
    columns = [name for name in WEATHER_COLUMNS if name in vintage.columns]
    selected = vintage[columns].reindex(expected).copy()
    for col in columns:
        selected[col] = pd.to_numeric(selected[col], errors="coerce").astype(float)
    selected = selected.replace([np.inf, -np.inf], np.nan)
    selected.index.name = "valid_time"
    selected.attrs = {**attrs, ISSUED: issue.isoformat(), AVAILABLE: available.isoformat()}
    return selected


def _serial(frame):
    return {"attrs": frame.attrs,
            "columns": list(frame.columns),
            "rows": [[stamp.isoformat(), *[None if pd.isna(value) else float(value) for value in values]]
                     for stamp, values in zip(frame.index, frame.to_numpy())]}


def _restore(serial):
    columns = serial["columns"]
    rows = serial["rows"]
    frame = pd.DataFrame([row[1:] for row in rows], columns=columns,
                         index=pd.DatetimeIndex([_timestamp(row[0], "valid_time") for row in rows]))
    frame[ISSUED] = serial["attrs"][ISSUED]
    frame[AVAILABLE] = serial["attrs"][AVAILABLE]
    frame.attrs = serial["attrs"]
    return frame


def _atomic_pointer(path, value):
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
            temp_name = tmp.name
            tmp.write(_json(value))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def _check_known_vintages(directory, identity, serial):
    """Catch revision backdating/regression that can be disproved by local history.

    First-seen availability is still a provider assertion: evidence must be audited
    against the publisher. This check cannot independently certify an archive.
    """
    attrs = serial["attrs"]
    current = (utc(attrs[ISSUED]), utc(attrs[AVAILABLE]))
    for path in directory.glob("*.json"):
        if not _HASH.fullmatch(path.stem):
            continue
        try:
            payload = json.loads(path.read_text())
            old = payload["weather"]
            if payload["identity"] != identity or _digest(old) != path.stem:
                raise ValueError("invalid snapshot")
            old_attrs = old["attrs"]
            previous = (utc(old_attrs[ISSUED]), utc(old_attrs[AVAILABLE]))
        except (OSError, ValueError, KeyError, TypeError, ForecastError) as exc:
            raise ForecastError("Weather snapshot history is invalid; inspect cache before refreshing") from exc
        if old_attrs["source"] != attrs["source"]:
            raise ForecastError("Weather source changed within one request; use a separate cache namespace for a new source")
        if current < previous:
            raise ForecastError("Weather refresh regressed to an older already-known vintage")
        if current == previous and (old["columns"] != serial["columns"] or old["rows"] != serial["rows"]):
            raise ForecastError("An already-known weather vintage changed without a new payload availability time")


def load_weather(settings, provider, turbine_id, turbine, origin, horizon_h, protocol, refresh=False):
    """Return normalized weather and provenance; refresh never falls back to stale cache.

    Snapshots live at cache_dir/request_hash/weather_version.json. Only latest.json
    is mutable; it points to the most recently accepted revision for that request.
    A provider refresh may select a newer vintage only if it was already available
    at the historical origin. Old snapshots remain unchanged and reproducible.
    Only normalized wind speed, temperature and optional wind direction are kept;
    additional provider fields require an explicit extension of this contract.
    """
    origin = _timestamp(origin, "forecast_origin")
    expected = expected_times(origin, horizon_h, protocol)
    identity = {"schema_version": 1, "module": settings.data_module,
                "turbine_id": turbine_id, "latitude": turbine.latitude,
                "longitude": turbine.longitude, "wind_height_m": turbine.hub_height_m,
                "forecast_origin": origin.isoformat(), "horizon_h": horizon_h,
                "protocol": protocol.model_dump(mode="json")}
    directory = Path(settings.cache_dir) / _digest(identity)
    pointer = directory / "latest.json"
    if pointer.is_file() and not refresh:
        try:
            version = json.loads(pointer.read_text())["weather_version"]
            if not isinstance(version, str) or not _HASH.fullmatch(version):
                raise ValueError("invalid snapshot hash")
            snapshot = directory / f"{version}.json"
            payload = json.loads(snapshot.read_text())
            if payload["identity"] != identity or _digest(payload["weather"]) != version:
                raise ValueError("snapshot identity or checksum mismatch")
            frame = _select(_restore(payload["weather"]), turbine_id, turbine, origin, expected)
            if _digest(_serial(frame)) != version:
                raise ValueError("snapshot no longer satisfies canonical weather contract")
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ForecastError) as exc:
            raise ForecastError("Weather cache is invalid; explicitly refresh after checking provider provenance") from exc
        log.info("forecast weather: verified snapshot cache hit")
        return frame, {**frame.attrs, "weather_version": version, "snapshot_path": str(snapshot.resolve()), "cache_hit": True}
    getter = getattr(provider, "get_forecast_weather", None)
    if not callable(getter):
        raise ForecastError("Weather provider requires get_forecast_weather(lat, lon, forecast_origin, horizon_h)")
    try:
        raw = getter(turbine.latitude, turbine.longitude, origin, horizon_h)
    except Exception as exc:
        # Do not include provider exception text: HTTP errors may contain credentials.
        raise ForecastError("Weather refresh failed; no stale-cache fallback was used" if refresh
                            else "Archived weather provider unavailable") from exc
    frame = _select(raw, turbine_id, turbine, origin, expected)
    serial = _serial(frame)
    _check_known_vintages(directory, identity, serial)
    version = _digest(serial)
    snapshot = directory / f"{version}.json"
    payload = {"identity": identity, "weather": serial}
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creation makes each completed snapshot immutable. Atomic link
        # avoids exposing partially written JSON to concurrent readers/writers.
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as tmp:
                temp_name = tmp.name
                tmp.write(_json(payload))
                tmp.flush()
                os.fsync(tmp.fileno())
            try:
                os.link(temp_name, snapshot)
            except FileExistsError:
                if snapshot.read_text() != _json(payload):
                    raise ForecastError("Existing weather snapshot was modified; refusing to overwrite")
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
        _atomic_pointer(pointer, {"weather_version": version})
    except (OSError, ValueError, TypeError) as exc:
        raise ForecastError("Could not persist immutable weather snapshot") from exc
    log.info("forecast weather: validated archive snapshot saved")
    return frame, {**frame.attrs, "weather_version": version, "snapshot_path": str(snapshot.resolve()), "cache_hit": False}
