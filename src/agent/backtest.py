from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.orchestrator import run_forecast
from src.agent.schemas import ForecastError
from src.config import Settings

log = logging.getLogger(__name__)


def load_actuals(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "turbine_id", "actual_power_kw"}
    if not required.issubset(frame.columns):
        raise ForecastError("Actuals CSV requires timestamp,turbine_id,actual_power_kw")
    parsed = []
    for value in frame.timestamp:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp != timestamp.floor("h"):
            raise ForecastError("Actual timestamps must be timezone-aware hourly boundaries")
        parsed.append(timestamp.tz_convert("UTC"))
    frame["timestamp"] = pd.DatetimeIndex(parsed)
    if frame.duplicated(["turbine_id", "timestamp"]).any():
        raise ForecastError("Actuals contain duplicate turbine/timestamp pairs")
    frame["actual_power_kw"] = pd.to_numeric(frame.actual_power_kw, errors="coerce")
    invalid = ~np.isfinite(frame.actual_power_kw) | (frame.actual_power_kw < 0)
    if invalid.any():
        raise ForecastError("Actuals contain missing, negative or non-finite power")
    return frame[list(sorted(required))]


def run_backtest(start: date, end: date, turbine: str, output_dir: Path,
                 settings: Settings, actuals: Path | None = None, with_agent=False):
    if start > end:
        raise ForecastError("Backtest start must not exceed end")
    if (end - start).days > 366:
        raise ForecastError("Backtest limited to 367 daily runs")
    actual = load_actuals(actuals) if actuals else None
    rows, failures, analyses = [], [], []
    # One forecast issuance at midnight UTC, 24 hourly intervals per day.
    for day in pd.date_range(start, end, freq="D"):
        try:
            result = run_forecast(turbine, day.date(), 24, settings=settings, with_agent=with_agent)
            for point in result.hourly:
                rows.append({"run_date": day.date().isoformat(), "turbine_id": turbine,
                    "timestamp": point.timestamp, "predicted_power_kw": point.power_kw,
                    "wind_speed_ms": point.wind_speed_ms, "valid": point.valid,
                    "weather_kind": result.weather_kind, "model_id": result.model_id,
                    "demo": result.demo})
            if with_agent:
                analyses.append({"run_date": day.date().isoformat(),
                                 "analysis_source": result.analysis_source,
                                 "analysis_reason": result.analysis_reason,
                                 "analysis": result.analysis.model_dump()})
        except ForecastError as exc:
            failures.append({"run_date": day.date().isoformat(), "error": str(exc)})
            log.warning("backtest day failed: %s", day.date())
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["run_date", "turbine_id", "timestamp", "predicted_power_kw", "wind_speed_ms",
               "valid", "weather_kind", "model_id", "demo"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    if actual is not None:
        frame = frame.merge(actual, on=["turbine_id", "timestamp"], how="left", validate="one_to_one")
    else:
        frame["actual_power_kw"] = np.nan
    finite_predictions = pd.to_numeric(frame.predicted_power_kw, errors="coerce")
    matched = frame[frame.valid.astype(bool) & np.isfinite(finite_predictions)
                    & frame.actual_power_kw.notna()]
    error = matched.predicted_power_kw - matched.actual_power_kw
    days = (end - start).days + 1
    summary = {
        "start": start.isoformat(), "end": end.isoformat(), "turbine_id": turbine,
        "demo": settings.demo, "days_requested": days, "days_completed": days - len(failures),
        "failed_days": len(failures), "expected_hours": days * 24,
        "prediction_rows": len(frame), "valid_hours": int(frame.valid.astype(bool).sum()),
        "matched_actual_hours": len(matched), "actual_coverage": len(matched) / (days * 24),
        "mae_kw": float(error.abs().mean()) if len(matched) else None,
        "rmse_kw": float(np.sqrt((error ** 2).mean())) if len(matched) else None,
        "bias_kw": float(error.mean()) if len(matched) else None,
        "weather_kinds": sorted(frame.weather_kind.unique().tolist()),
        "metrics_note": ("Synthetic predictions: not a real model quality estimate." if settings.demo else
                         "Reanalysis is hindcast; verify weather issuance and model training cutoff to avoid leakage."),
        "failures": failures,
    }
    frame.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if with_agent:
        (output_dir / "analyses.json").write_text(json.dumps(analyses, indent=2, ensure_ascii=False) + "\n")
    return summary
