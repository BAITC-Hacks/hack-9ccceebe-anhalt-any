from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.agent.analyzer import analyze_forecast
from src.agent.rules import evaluate, finite
from src.agent.schemas import ForecastError, ForecastRequest, ForecastResult, HourlyPoint
from src.agent.tools import load_providers, load_weather
from src.config import ROOT, Settings

log = logging.getLogger(__name__)


def run_forecast(turbine_id, forecast_date, horizon_h=24, latitude=None, longitude=None,
                 *, settings: Settings | None = None, with_agent: bool = True) -> ForecastResult:
    try:
        request = ForecastRequest(turbine_id=turbine_id, forecast_date=forecast_date,
                                  horizon_h=horizon_h, latitude=latitude, longitude=longitude)
    except ValidationError as exc:
        raise ForecastError("Invalid input: use an ISO date, known turbine and integer horizon 1–72h") from exc
    settings = settings or Settings.from_env()
    try:
        turbines = settings.turbines()
    except (OSError, ValueError, TypeError) as exc:
        raise ForecastError("Cannot read valid turbine configuration") from exc
    if request.turbine_id not in turbines:
        raise ForecastError(f"Unknown turbine: {request.turbine_id}")
    turbine = turbines[request.turbine_id]
    if turbine.demo_only and not settings.demo:
        raise ForecastError("T1 is a demo turbine; use --demo or configure a real turbine")
    for provided, configured in ((latitude, turbine.latitude), (longitude, turbine.longitude)):
        if provided is not None and not np.isclose(provided, configured, rtol=0, atol=1e-6):
            raise ForecastError("Coordinates differ from registered turbine; update turbine configuration")
    data, ml = load_providers(settings)
    weather = load_weather(settings, data, turbine.latitude, turbine.longitude,
                           request.forecast_date, request.horizon_h)
    if weather.attrs.get("wind_height_m") != turbine.hub_height_m:
        raise ForecastError("Weather wind_height_m must match turbine hub height; convert in DATA provider")
    log.info("feature build")
    try:
        features = data.build_features(weather.copy())
    except Exception as exc:
        raise ForecastError("Feature builder failed") from exc
    if not isinstance(features, pd.DataFrame) or features.empty or len(features.columns) == 0:
        raise ForecastError("Feature builder returned empty features")
    if not features.index.equals(weather.index) or features.columns.has_duplicates:
        raise ForecastError("Features must preserve the weather index and have unique columns")
    features.attrs["turbine_id"] = request.turbine_id
    features.attrs["wind_height_m"] = weather.attrs["wind_height_m"]
    try:
        usable = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    except (ValueError, TypeError) as exc:
        raise ForecastError("Features must be numeric") from exc
    usable &= np.isfinite(weather[["wind_speed_ms", "temperature_c"]].to_numpy()).all(axis=1)
    usable &= weather.wind_speed_ms.between(0, 100).to_numpy()
    usable &= weather.temperature_c.between(-90, 65).to_numpy()
    if not usable.any():
        raise ForecastError("No usable features: all weather/features missing or invalid")
    model_path = ROOT / "models/demo/model.joblib" if settings.demo else settings.model_path
    if not model_path.exists():
        raise ForecastError("Model file/directory missing; configure MODEL_PATH or run prepare_demo.py")
    log.info("model inference: %d usable hours", int(usable.sum()))
    try:
        model = ml.load_model(model_path)
        values = np.asarray(ml.predict(model, features.loc[usable]), dtype=float)
    except Exception as exc:
        raise ForecastError("Model load/inference failed; check artifact, dependencies, and feature contract") from exc
    if values.shape != (int(usable.sum()),):
        raise ForecastError("ML predict must return a 1D kW array with one prediction per input row")
    power = np.full(len(weather), np.nan)
    power[usable] = values
    flags, statistics, valid = evaluate(weather, power, turbine)
    summary = {"date": request.forecast_date.isoformat(), "turbine_id": turbine_id,
               "horizon_h": horizon_h, "rated_power_kw": turbine.rated_power_kw,
               "demo": settings.demo, "weather_kind": weather.attrs["kind"]}
    analysis, source, reason = analyze_forecast(summary, flags, statistics,
                                               enabled=with_agent, model=settings.openai_model)
    result = ForecastResult(request=request, demo=settings.demo,
        weather_source=str(weather.attrs["source"]), weather_kind=weather.attrs["kind"],
        model_id=("synthetic-random-forest-v1" if settings.demo else f"{settings.ml_module}:{model_path.name}"),
        status="ok" if valid.all() else ("partial" if valid.any() else "invalid"),
        statistics=statistics, anomalies=flags, analysis=analysis, analysis_source=source,
        analysis_reason=reason,
        hourly=[HourlyPoint(timestamp=t.to_pydatetime(), wind_speed_ms=finite(w),
                            power_kw=finite(p), valid=bool(v)) for t, w, p, v in
                zip(weather.index, weather.wind_speed_ms, power, valid)])
    log.info("result: %s, %d flags, analysis=%s", result.status, len(flags), source)
    return result
