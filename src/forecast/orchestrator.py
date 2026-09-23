"""Strict historical target-station forecasting; never substitutes demo or observed data."""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import threading
import tempfile
import fcntl
from contextlib import contextmanager
from uuid import uuid4
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from src.agent.analyzer import analyze_forecast
from src.agent.rules import evaluate, finite
from src.agent.schemas import ForecastError, Statistics
from src.config import ROOT, Turbine
from src.forecast.contracts import (ForecastSettings, ModelManifest, expected_times,
                                    load_protocol, utc)

log = logging.getLogger(__name__)
LOCK = threading.RLock()
TURBINES = ("T1", "T2")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False, default=str).encode()).hexdigest()


def _registry(settings):
    raw = json.loads(settings.turbines_file.read_text())
    return {tid: Turbine.model_validate(raw[tid]) for tid in TURBINES}


def _module(name, methods):
    if not name:
        raise ForecastError("Target provider module is not configured")
    module = importlib.import_module(name)
    if not all(callable(getattr(module, method, None)) for method in methods):
        raise ForecastError("Target provider does not implement the strict forecast contract")
    return module


def _manifest(settings, protocol, registry):
    if not settings.model_path.is_file():
        raise ForecastError("Organizer-trained Goldwind artifact is missing")
    manifest = ModelManifest.model_validate_json(settings.manifest_path.read_text())
    artifact_hash = hashlib.sha256(settings.model_path.read_bytes()).hexdigest()
    if manifest.artifact_sha256 != artifact_hash:
        raise ForecastError("Model artifact SHA256 does not match its availability manifest")
    if set(manifest.turbine_ids) != set(TURBINES):
        raise ForecastError("Target manifest must explicitly support T1 and T2")
    # Model calendar encodings use its confirmed source timezone. The execution
    # calendar may differ; all forecast/cutoff comparisons use aware UTC instants.
    for tid, turbine in registry.items():
        if manifest.rated_power_kw.get(tid) != turbine.rated_power_kw:
            raise ForecastError("Model nominal capacity does not match target registry")
        if manifest.wind_height_m.get(tid) != turbine.hub_height_m:
            raise ForecastError("Model feature wind height does not match target registry")
    if not {"wind_speed_ms", "temperature_c"}.issubset(manifest.feature_columns):
        raise ForecastError("Target manifest requires raw wind_speed_ms and temperature_c features")
    return manifest


def readiness(settings: ForecastSettings | None = None) -> dict:
    settings = settings or ForecastSettings.from_env()
    blockers = []
    protocol = registry = None
    try:
        protocol = load_protocol(settings)
    except Exception:
        blockers.append("Source semantics are unconfirmed: CSV timezone, interval labels and normalized-power definition; the team execution schedule alone does not confirm them")
    try:
        registry = _registry(settings)
    except Exception:
        blockers.append("Valid T1/T2 registry is missing")
    for label, name, methods in (
        ("weather", settings.data_module, ("get_forecast_weather", "build_features")),
        ("ML", settings.ml_module, ("load_model", "predict")),
    ):
        try:
            _module(name, methods)
        except Exception:
            blockers.append(f"Target {label} provider is not configured or cannot be imported")
    if not settings.model_path.is_file() or not settings.manifest_path.is_file():
        blockers.append("Organizer-trained Goldwind model and availability manifest are missing")
    elif protocol is not None and registry is not None:
        try:
            _manifest(settings, protocol, registry)
        except Exception:
            blockers.append("Target model manifest, artifact hash or registry/protocol compatibility is invalid")
    return {"ready": not blockers, "blockers": blockers,
            "scope": "Structural readiness only; vintage availability and training cutoff are checked per origin"}


def _save_new(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete immutable file atomically, including across UI/API processes.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".pending-", delete=False) as output:
            temporary = Path(output.name)
            json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _run_lock(directory: Path):
    # POSIX advisory lock; released automatically on process exit (macOS/Linux runtime).
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _blocked(settings, origin, horizon, messages):
    result = {"run_id": "blocked-" + digest({"origin": str(origin), "horizon": horizon, "errors": messages})[:20],
              "forecast_origin": str(origin), "horizon_h": horizon, "status": "blocked",
              "rows": [], "farm_rows": [], "errors": {"configuration": "; ".join(messages)},
              "analysis": None, "analysis_source": "unavailable", "cache_hit": False,
              "note": "No target forecasts were produced; demo/Kelmarsh substitution is forbidden"}
    _save_new(settings.results_dir / "blocked" / f"{result['run_id']}.json", result)
    return result


def _engine_version():
    files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "requirements.lock.txt"]
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                   for p in files if p.is_file()})


def _provider_version(module):
    path = Path(getattr(module, "__file__", ""))
    if not path.is_file():
        return module.__name__
    # Include sibling transformers/feature builders, not just the public adapter.
    files = sorted(path.parent.rglob("*.py"))
    return digest({str(p.relative_to(path.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def run_target_forecast(forecast_origin, horizon_h=48, *, settings: ForecastSettings | None = None,
                        refresh=False, with_agent=False) -> dict:
    from src.forecast.weather import load_weather
    settings = settings or ForecastSettings.from_env()
    origin = utc(forecast_origin)
    if type(horizon_h) is not int or horizon_h not in (24, 48):
        raise ForecastError("Target forecast horizon must be exactly 24 or 48 hours")
    if origin != origin.floor("h"):
        raise ForecastError("Forecast origin must be on an hourly boundary")
    state = readiness(settings)
    if not state["ready"]:
        return _blocked(settings, origin.isoformat(), horizon_h, state["blockers"])
    protocol = load_protocol(settings)
    times = expected_times(origin, horizon_h, protocol)
    registry = _registry(settings)
    manifest = _manifest(settings, protocol, registry)
    cutoff = max(utc(manifest.training_data_available_until), utc(manifest.selection_data_available_until))
    if cutoff > origin:
        return _blocked(settings, origin.isoformat(), horizon_h,
                        ["Training/selection data was not available at forecast origin"])
    data = _module(settings.data_module, ("get_forecast_weather", "build_features"))
    ml = _module(settings.ml_module, ("load_model", "predict"))
    weather, meta, errors = {}, {}, {}
    log.info("target forecast: origin=%s horizon=%s", origin.isoformat(), horizon_h)
    for tid in TURBINES:
        try:
            weather[tid], meta[tid] = load_weather(settings, data, tid, registry[tid], origin,
                                                   horizon_h, protocol, refresh=refresh)
        except Exception:
            errors[tid] = "Archived forecast unavailable or provenance invalid for this origin; no substitute used"
    identity = {"forecast_origin": origin.isoformat(), "horizon_h": horizon_h,
                "protocol": protocol.model_dump(), "registry": {t:r.model_dump() for t,r in registry.items()},
                "model_manifest": manifest.model_dump(mode="json"),
                "weather_versions": {t:m["weather_version"] for t,m in meta.items()},
                "errors": dict(errors), "engine_version": _engine_version(),
                "data_provider": settings.data_module, "data_code_hash": _provider_version(data),
                "ml_provider": settings.ml_module, "ml_code_hash": _provider_version(ml)}
    run_id = digest(identity)
    analysis_key = digest({"enabled": with_agent, "model": settings.openai_model,
                           "configured": bool(os.getenv("OPENAI_API_KEY"))})[:16]
    result_path = settings.results_dir / run_id / f"analysis-{analysis_key}.json"
    numeric_path = settings.results_dir / run_id / "forecast.json"
    with LOCK, _run_lock(settings.results_dir / run_id):
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            result["cache_hit"] = True
            return result
        if numeric_path.is_file():
            result = json.loads(numeric_path.read_text())
        else:
            rows, statistics, flags_by_turbine = [], {}, {}
            model = None
            if weather:
                try:
                    model = ml.load_model(settings.model_path)
                except Exception:
                    errors["model"] = "Target model load failed; check trusted artifact and dependencies"
            for tid in TURBINES:
                frame = weather.get(tid)
                power = np.full(len(times), np.nan)
                if frame is not None and model is not None:
                    try:
                        features = data.build_features(frame.copy())
                        if (not isinstance(features, pd.DataFrame) or not features.index.equals(times)
                            or features.columns.has_duplicates or set(features.columns) != set(manifest.feature_columns)):
                            raise ForecastError("Feature schema/index differs from target manifest")
                        features = features[manifest.feature_columns].copy()
                        for dtype in features.dtypes:
                            if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
                                raise ForecastError("Forecast features must have real numeric dtypes")
                        features.attrs.update(turbine_id=tid, wind_height_m=registry[tid].hub_height_m)
                        usable = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
                        usable &= frame.wind_speed_ms.between(0, 100).to_numpy()
                        usable &= frame.temperature_c.between(-90, 65).to_numpy()
                        if "wind_direction_deg" in features:
                            usable &= features.wind_direction_deg.between(0, 360, inclusive="left").to_numpy()
                        if usable.any():
                            raw = np.asarray(ml.predict(model, features.loc[usable]), dtype=float)
                            if raw.shape != (int(usable.sum()),):
                                raise ForecastError("Target prediction must be 1D raw kW")
                            power[usable] = raw
                        else:
                            errors[tid] = "No finite valid feature rows for inference"
                    except Exception:
                        errors[tid] = "Target feature construction or inference failed"
                if frame is None:
                    frame = pd.DataFrame({"wind_speed_ms": np.nan, "temperature_c": np.nan}, index=times)
                rule_columns = ["wind_speed_ms", "temperature_c"]
                if "wind_direction_deg" in manifest.feature_columns:
                    rule_columns.append("wind_direction_deg")
                rule_frame = frame.reindex(columns=rule_columns)
                flags, stats, valid = evaluate(rule_frame, power, registry[tid])
                statistics[tid] = stats.model_dump()
                flags_by_turbine[tid] = [f.model_dump() for f in flags]
                for i, timestamp in enumerate(times):
                    # Rule summary codes apply to the run; row quality codes identify exact invalid points.
                    codes = []
                    if not np.isfinite(power[i]): codes.append("missing_prediction")
                    elif power[i] < 0: codes.append("negative_power")
                    elif power[i] > registry[tid].rated_power_kw: codes.append("above_rated_power")
                    if i and np.isfinite(power[i-1]) and abs(power[i]-power[i-1]) > .5*registry[tid].rated_power_kw:
                        codes.append("hourly_jump")
                    if tid in errors: codes.append("provider_error")
                    metadata = meta.get(tid, {})
                    rows.append({"run_id": run_id, "forecast_origin": origin.isoformat(), "horizon_h": horizon_h,
                        "weather_issued_at": metadata.get("weather_issued_at"),
                        "weather_available_at": metadata.get("weather_available_at"),
                        "valid_time": timestamp.isoformat(), "lead_hours": int((timestamp-origin)/pd.Timedelta(1, unit="h")),
                        "turbine_id": tid, "normalized_power": finite(power[i]/registry[tid].rated_power_kw),
                        "power_kw": finite(power[i]), "energy_kwh": finite(power[i]) if valid[i] else None,
                        "model_version": manifest.model_version, "weather_version": metadata.get("weather_version"),
                        "flags": codes, "valid": bool(valid[i]), "wind_speed_ms": finite(frame.wind_speed_ms.iloc[i])})
            farm = []
            for i, timestamp in enumerate(times):
                group = [rows[i], rows[len(times)+i]]
                missing = [r["turbine_id"] for r in group if r["power_kw"] is None]
                incomplete = [r["turbine_id"] for r in group if not r["valid"]]
                raw_sum = sum(r["power_kw"] for r in group) if not missing else None
                farm.append({"run_id": run_id, "forecast_origin": origin.isoformat(),
                    "valid_time": timestamp.isoformat(), "lead_hours": group[0]["lead_hours"],
                    "horizon_h": horizon_h, "raw_power_kw": raw_sum,
                    "power_kw": raw_sum if not incomplete else None, "energy_kwh": raw_sum if not incomplete else None,
                    "complete": not incomplete, "missing_turbines": missing, "incomplete_turbines": incomplete,
                    "model_version": manifest.model_version, "weather_versions": {r["turbine_id"]:r["weather_version"] for r in group},
                    "flags": sorted({f for r in group for f in r["flags"]})})
            result = {"run_id": run_id, "forecast_origin": origin.isoformat(), "horizon_h": horizon_h,
                      "status": "ok" if all(r["complete"] for r in farm) else
                                ("partial" if any(r["power_kw"] is not None for r in rows) else "blocked"),
                      "rows": rows, "farm_rows": farm, "errors": errors,
                      "statistics": statistics, "anomalies": flags_by_turbine,
                      "provenance": identity, "weather_snapshots": meta, "cache_hit": False}
            if not errors:
                _save_new(numeric_path, result)
        # One compact LLM request for the farm, never one per turbine/hour.
        from src.agent.schemas import Anomaly
        all_flags = [Anomaly.model_validate(flag) for flags in result["anomalies"].values() for flag in flags]
        # Same codes across turbines remain canonical; keep the most severe aggregate.
        unique = {}
        for flag in all_flags:
            if flag.code not in unique: unique[flag.code] = flag
            else: unique[flag.code].count += flag.count
        farm = result["farm_rows"]
        winds = [r["wind_speed_ms"] for r in result["rows"] if r["wind_speed_ms"] is not None]
        raw_farm = [r["raw_power_kw"] for r in farm if r["raw_power_kw"] is not None]
        stats = Statistics(avg_wind_ms=float(np.mean(winds)) if winds else None,
            max_wind_ms=max(winds) if winds else None, min_power_kw=min(raw_farm) if raw_farm else None,
            max_power_kw=max(raw_farm) if raw_farm else None,
            predicted_energy_kwh=sum(r["energy_kwh"] for r in farm) if all(r["complete"] for r in farm) else None,
            valid_hours=sum(r["complete"] for r in farm), requested_hours=horizon_h)
        summary = {"date": origin.isoformat(), "turbine_id": "T1+T2", "demo": False,
                   "weather_kind": "archived_forecast", "forecast_origin": origin.isoformat(),
                   "horizon_h": horizon_h, "model_version": manifest.model_version}
        analysis, source, reason = analyze_forecast(summary, list(unique.values()), stats,
                                                    enabled=with_agent and not result["errors"], model=settings.openai_model)
        result.update(analysis=analysis.model_dump(), analysis_source=source, analysis_reason=reason, cache_hit=False)
        # Transient inference failures should be retryable; successful numeric versions stay immutable.
        if not result["errors"]:
            _save_new(result_path, result)
        else:
            _save_new(settings.results_dir / run_id / "failed_attempts" / f"{uuid4().hex}.json", result)
        return result
