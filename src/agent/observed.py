"""Native 10-minute Kelmarsh evaluation; deliberately separate from future forecasts."""
import logging
from dataclasses import replace

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.agent.analyzer import analyze_forecast
from src.agent.rules import evaluate, finite
from src.agent.schemas import (ForecastError, ObservedMetrics, ObservedPoint,
                               ObservedRequest, ObservedResult)
from src.config import ROOT, Settings
from src.data.kelmarsh_observed import SOURCE, load_observations
from src.ml import provider

log = logging.getLogger(__name__)


def kelmarsh_turbines():
    return replace(Settings(), turbines_file=ROOT / "config/kelmarsh_turbines.json").turbines()


def run_observed_analysis(turbine_id, observation_date, horizon_h=24, *, with_agent=False):
    try:
        request = ObservedRequest(turbine_id=turbine_id, observation_date=observation_date, horizon_h=horizon_h)
    except ValidationError as exc:
        raise ForecastError("Use an ISO observation date and integer horizon 1–72h") from exc
    registry = kelmarsh_turbines()
    if turbine_id not in registry:
        raise ForecastError("MODE A supports Kelmarsh 1..6 only; Goldwind transfer is not validated")
    turbine = registry[turbine_id]
    log.info("observed SCADA: %s, native 10-minute holdout", turbine_id)
    frame = load_observations(turbine_id, request.observation_date, horizon_h)
    features = frame[list(provider.INPUT_COLUMNS)].copy()
    # Request and published registry are authoritative; targets never enter inference.
    features.attrs = {"turbine_id": turbine_id, "wind_height_m": turbine.hub_height_m}
    usable = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    usable &= features.wind_speed_ms.between(0, 100).to_numpy()
    usable &= features.temperature_c.between(-90, 65).to_numpy()
    usable &= features.wind_direction_deg.between(0, 360, inclusive="left").to_numpy()
    if not usable.any():
        raise ForecastError("No valid observed weather samples for inference")
    try:
        model = provider.load_model(ROOT / "models/power_model.joblib")
        cutoff = pd.Timestamp(model.metadata["validation_end"])
        if frame.index.min() <= cutoff:
            raise ForecastError("Requested period overlaps model fitting/selection; use held-out observations")
        log.info("observed model inference: %d samples", usable.sum())
        predicted = np.asarray(provider.predict(model, features.loc[usable]), dtype=float)
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError("Kelmarsh model load/inference failed; verify the published artifact and dependencies") from exc
    if predicted.shape != (int(usable.sum()),):
        raise ForecastError("ML provider must return one raw kW value per native sample")
    power = np.full(len(frame), np.nan)
    power[usable] = predicted
    flags, stats, valid = evaluate(frame, power, turbine, interval_minutes=10)
    actual = frame.actual_power_kw.to_numpy(dtype=float)
    # Match B's raw metric protocol: retain finite negative/above-rated predictions and targets.
    paired = np.isfinite(power) & np.isfinite(actual)
    error = power[paired] - actual[paired]
    metrics = ObservedMetrics(matched_samples=int(paired.sum()), expected_samples=len(frame),
        coverage=float(paired.mean()), mae_kw=float(np.abs(error).mean()) if len(error) else None,
        rmse_kw=float(np.sqrt((error**2).mean())) if len(error) else None,
        bias_kw=float(error.mean()) if len(error) else None)
    summary = {"date": request.observation_date.isoformat(), "turbine_id": turbine_id,
               "demo": False, "weather_kind": "observed_scada", "mode": "A",
               "sampling_interval_minutes": 10, "rated_power_kw": turbine.rated_power_kw,
               "comparison": metrics.model_dump()}
    analysis, source, reason = analyze_forecast(summary, flags, stats, enabled=with_agent,
                                                model=Settings.from_env().openai_model)
    return ObservedResult(request=request, source=SOURCE,
        model_id="Kelmarsh-HistGradientBoosting-frozen-5f7b569", hub_height_m=turbine.hub_height_m,
        status="ok" if valid.all() else ("partial" if valid.any() else "invalid"),
        statistics=stats, metrics=metrics, anomalies=flags, analysis=analysis,
        analysis_source=source, analysis_reason=reason,
        samples=[ObservedPoint(timestamp=t.to_pydatetime(), wind_speed_ms=finite(w),
            power_kw=finite(p), actual_power_kw=finite(a), valid=bool(v))
            for t, w, p, a, v in zip(frame.index, frame.wind_speed_ms, power, actual, valid)],
        limitation="MODE A, observed nacelle weather: not a weather forecast or validated Goldwind transfer. "
        "Native 10-minute samples; energy is a rectangular sum kW × 10/60 h, not a meter reading. "
        "Timestamps are grouped as labeled by the source; interval start/end convention is not verified. "
        "Raw errors retain finite negative/above-rated values. Missing coverage is not zero-filled.")
