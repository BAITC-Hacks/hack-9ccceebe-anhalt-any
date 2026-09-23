"""Rolling historical replay; each origin remains a separate forecast vintage.

Actual CSV timestamps label the start of an hourly interval and MUST carry a
timezone. Evaluation uses raw finite power, including physically flagged values.
No actuals are passed into the forecast runner.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.schemas import ForecastError
from src.forecast.contracts import ForecastSettings, expected_times, load_protocol, utc

TURBINES = ("T1", "T2")
PREDICTION_COLUMNS = [
    "run_id", "forecast_origin", "horizon_h", "weather_issued_at", "weather_available_at",
    "valid_time", "lead_hours", "turbine_id", "normalized_power", "power_kw",
    "energy_kwh", "model_version", "weather_version", "flags", "valid",
]
FARM_COLUMNS = [
    "run_id", "forecast_origin", "horizon_h", "valid_time", "lead_hours", "power_kw",
    "energy_kwh", "raw_power_kw", "complete", "missing_turbines",
    "incomplete_turbines", "model_version", "weather_versions", "flags",
]


def run_target_forecast(*args, **kwargs):
    # Lazy import also lets offline replay tests replace only this boundary.
    from src.forecast.orchestrator import run_target_forecast as runner
    return runner(*args, **kwargs)


def _hash(path: Path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _origins(start, end, protocol):
    """Local calendar days, not fixed 24h jumps (which break across DST)."""
    result = []
    day = start
    while day <= end:
        naive = pd.Timestamp(datetime.combine(day, time(hour=protocol.daily_origin_hour)))
        try:
            result.append(naive.tz_localize(protocol.timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC"))
        except Exception as exc:
            raise ForecastError(f"Ambiguous/nonexistent daily forecast origin: {day}") from exc
        day += timedelta(days=1)
    return result


def load_actuals(path: Path) -> pd.DataFrame:
    """Read hourly mean raw kW; timestamp is aware interval START, never inferred."""
    frame = pd.read_csv(path)
    required = {"timestamp", "turbine_id", "actual_power_kw"}
    if not required.issubset(frame.columns):
        raise ForecastError("Actuals CSV requires timestamp, turbine_id, actual_power_kw")
    if frame.empty:
        raise ForecastError("Actuals CSV is empty")
    try:
        frame["valid_time"] = pd.DatetimeIndex([utc(value) for value in frame["timestamp"]])
    except Exception as exc:
        raise ForecastError("Actual timestamps must be timezone-aware hourly interval starts") from exc
    if (frame["valid_time"] != frame["valid_time"].dt.floor("h")).any():
        raise ForecastError("Actual timestamps must be on hourly boundaries")
    if not frame["turbine_id"].isin(TURBINES).all():
        raise ForecastError("Actual turbine_id must be T1 or T2")
    try:
        frame["actual_power_kw"] = pd.to_numeric(frame["actual_power_kw"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ForecastError("Actual power must be numeric raw kW") from exc
    if not np.isfinite(frame["actual_power_kw"].to_numpy(dtype=float)).all():
        raise ForecastError("Actual power must be finite raw kW")
    if frame.duplicated(["turbine_id", "valid_time"]).any():
        raise ForecastError("Duplicate actual turbine/time pairs are not allowed")
    return frame[["valid_time", "turbine_id", "actual_power_kw"]]


def _score(frame, prediction):
    p = pd.to_numeric(frame[prediction], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(frame["actual_power_kw"], errors="coerce").to_numpy(dtype=float)
    paired = np.isfinite(p) & np.isfinite(a)
    p, a = p[paired], a[paired]
    if not len(p):
        return {"matched_forecast_rows": 0, "mae_kw": None, "rmse_kw": None, "bias_kw": None, "r2": None}
    errors = p - a
    denominator = float(np.sum((a - a.mean()) ** 2))
    return {"matched_forecast_rows": int(len(p)), "mae_kw": float(np.mean(np.abs(errors))),
            "rmse_kw": float(np.sqrt(np.mean(errors ** 2))), "bias_kw": float(errors.mean()),
            "r2": float(1 - np.sum(errors ** 2) / denominator) if len(a) > 1 and denominator > 0 else None}


def _metrics(predictions, farm, actuals, first_lead):
    if actuals is None:
        return None
    # many_to_one is deliberate: each actual can match multiple forecast origins.
    joined = predictions.merge(actuals, how="left", on=["turbine_id", "valid_time"], validate="many_to_one")
    joined["horizon_bucket"] = np.where(joined["lead_hours"] - first_lead < 24, "first_24h", "second_24h")
    paired = joined[np.isfinite(pd.to_numeric(joined["power_kw"], errors="coerce")) & joined["actual_power_kw"].notna()]
    actual_farm = actuals.groupby("valid_time").agg(actual_power_kw=("actual_power_kw", "sum"), turbines=("turbine_id", "nunique"))
    actual_farm = actual_farm.loc[actual_farm.turbines == len(TURBINES), ["actual_power_kw"]].reset_index()
    farm_joined = farm.merge(actual_farm, how="left", on="valid_time", validate="many_to_one")
    farm_joined["horizon_bucket"] = np.where(farm_joined["lead_hours"] - first_lead < 24, "first_24h", "second_24h")
    return {
        "protocol": "Every origin retained; finite raw kW scored, including physical flags; no clipping or best-vintage selection",
        "overall": _score(joined, "power_kw"),
        "unique_turbine_hours_matched": int(len(paired[["turbine_id", "valid_time"]].drop_duplicates())),
        "per_turbine": {tid: _score(joined[joined.turbine_id == tid], "power_kw") for tid in TURBINES},
        "per_lead_hour": {str(int(lead)): _score(group, "power_kw") for lead, group in joined.groupby("lead_hours")},
        "per_horizon_bucket": {key: _score(group, "power_kw") for key, group in joined.groupby("horizon_bucket")},
        "per_turbine_and_lead": {tid: {str(int(lead)): _score(group, "power_kw") for lead, group in joined[joined.turbine_id == tid].groupby("lead_hours")} for tid in TURBINES},
        "farm": _score(farm_joined, "raw_power_kw"),
        "farm_per_lead_hour": {str(int(lead)): _score(group, "raw_power_kw") for lead, group in farm_joined.groupby("lead_hours")},
        "farm_per_horizon_bucket": {key: _score(group, "raw_power_kw") for key, group in farm_joined.groupby("horizon_bucket")},
    }


def _write_csv(frame, path):
    output = frame.copy()
    for column in ("flags", "missing_turbines", "incomplete_turbines", "weather_versions"):
        if column in output:
            output[column] = output[column].map(lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value)
    output.to_csv(path, index=False)


def _frame(rows, columns):
    frame = pd.DataFrame(rows).reindex(columns=columns)
    for name in ("valid_time", "forecast_origin"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    for name in ("power_kw", "raw_power_kw", "lead_hours"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def _check_rows(result, origin, times):
    """Reject malformed provenance before writing or scoring it as a forecast."""
    for collection, farm in ((result.get("rows") or [], False), (result.get("farm_rows") or [], True)):
        seen = set()
        for row in collection:
            required = {"valid_time", "forecast_origin", "lead_hours", "run_id", "power_kw"}
            required |= {"complete", "raw_power_kw"} if farm else {"turbine_id", "valid", "model_version", "weather_version"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ForecastError("Runner returned an incomplete row schema")
            stamp = utc(row["valid_time"])
            if utc(row["forecast_origin"]) != origin or stamp not in times:
                raise ForecastError("Runner returned a row outside its origin/horizon")
            lead = (stamp - origin).total_seconds() / 3600
            if row["lead_hours"] != lead:
                raise ForecastError("Runner returned an inconsistent lead_hours")
            if not farm and row["turbine_id"] not in TURBINES:
                raise ForecastError("Runner returned an unknown target turbine")
            if row.get("run_id") != result.get("run_id"):
                raise ForecastError("Runner returned an inconsistent run_id")
            key = (stamp,) if farm else (stamp, row["turbine_id"])
            if key in seen:
                raise ForecastError("Runner returned duplicate rows within a forecast origin")
            seen.add(key)


def run_february_replay(
    output_dir: Path, *, settings: ForecastSettings | None = None, horizon_h: int = 48,
    actuals_path: Path | None = None, refresh: bool = False, with_agent: bool = False,
    start: date = date(2026, 1, 31), end: date = date(2026, 2, 28),
) -> dict:
    """Replay daily local origins and evaluate Feb 1 <= interval start < Mar 1.

    Writes all forecast vintages, including hours outside the evaluation window.
    Returns manifest. ``status == 'ok'`` requires complete physical coverage;
    missing actuals leave metrics null and do not prevent forecast production.
    """
    if type(horizon_h) is not int or horizon_h not in (24, 48):
        raise ForecastError("Replay horizon must be 24 or 48 hours")
    if start > end:
        raise ForecastError("Replay start must not be later than end")
    settings = settings or ForecastSettings.from_env()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actuals = load_actuals(Path(actuals_path)) if actuals_path is not None else None
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked", "readiness": {"protocol_confirmed": False},
        "configuration": {"origin_start": start.isoformat(), "origin_end": end.isoformat(),
            "horizon_h": horizon_h, "refresh": refresh, "with_agent": with_agent,
            "data_module": settings.data_module, "ml_module": settings.ml_module,
            "actuals_interval_label": "start", "actuals_timestamp_policy": "timezone-aware; converted to UTC",
            "actuals_power_unit": "raw kW; negatives retained"},
        "provenance": {key: {"path": str(path), "sha256": _hash(path)} for key, path in {
            "model_artifact": settings.model_path, "model_manifest": settings.manifest_path,
            "forecast_protocol": settings.protocol_path, "turbines": settings.turbines_file}.items()},
        "runs": [], "run_ids": [], "failures": [], "metrics": None,
        "coverage": {"expected_forecast_rows": None, "received_forecast_rows": 0,
                     "finite_raw_forecast_rows": 0, "missing_forecast_rows": None},
    }
    if actuals_path is not None:
        manifest["provenance"]["actuals"] = {"path": str(actuals_path), "sha256": _hash(Path(actuals_path))}
    predictions, farm_rows, expected = [], [], []
    policy = None
    try:
        policy = load_protocol(settings)
        origins = _origins(start, end, policy)
        manifest["readiness"]["protocol_confirmed"] = True
        manifest["configuration"]["protocol"] = policy.model_dump(mode="json")
        for origin in origins:
            times = expected_times(origin, horizon_h, policy)
            expected.extend({"forecast_origin": origin, "valid_time": stamp, "turbine_id": tid} for stamp in times for tid in TURBINES)
            try:
                result = run_target_forecast(origin, horizon_h, settings=settings, refresh=refresh, with_agent=with_agent)
                _check_rows(result, origin, times)
                run_id = result.get("run_id")
                if run_id:
                    manifest["run_ids"].append(run_id)
                errors = dict(result.get("errors") or {})
                if len(result.get("rows") or []) != horizon_h * len(TURBINES) or len(result.get("farm_rows") or []) != horizon_h:
                    errors["coverage"] = "Incomplete 24/48h forecast rows"
                manifest["runs"].append({"run_id": run_id, "forecast_origin": origin.isoformat(),
                                         "status": result.get("status", "blocked"), "errors": errors})
                if result.get("status") != "ok" or errors:
                    manifest["failures"].append({"forecast_origin": origin.isoformat(), "errors": errors or {"run": "Incomplete forecast"}})
                predictions.extend({**row, "horizon_h": horizon_h} for row in (result.get("rows") or []))
                farm_rows.extend({**row, "horizon_h": horizon_h} for row in (result.get("farm_rows") or []))
            except Exception as exc:
                # Provider exception text may include an authenticated URL or a key.
                error = f"{type(exc).__name__}: Forecast run failed or violated its row contract"
                manifest["failures"].append({"forecast_origin": origin.isoformat(), "errors": {"run": error}})
                manifest["runs"].append({"run_id": None, "forecast_origin": origin.isoformat(), "status": "blocked", "errors": {"run": error}})
    except ForecastError as exc:
        manifest["failures"].append({"forecast_origin": None, "errors": {"protocol": str(exc)}})

    prediction_frame = _frame(predictions, PREDICTION_COLUMNS)
    farm_frame = _frame(farm_rows, FARM_COLUMNS)
    if policy is not None and expected:
        lower = pd.Timestamp("2026-02-01", tz=policy.timezone).tz_convert("UTC")
        upper = pd.Timestamp("2026-03-01", tz=policy.timezone).tz_convert("UTC")
        manifest["configuration"]["evaluation_window"] = {"start_inclusive": lower.isoformat(), "end_exclusive": upper.isoformat()}
        def in_window(frame):
            return frame[(frame.valid_time >= lower) & (frame.valid_time < upper)]
        eval_pred, eval_farm = in_window(prediction_frame), in_window(farm_frame)
        expected_frame = in_window(pd.DataFrame(expected))
        keys = ["forecast_origin", "valid_time", "turbine_id"]
        # Duplicate predictions would distort scores and are a contract failure.
        duplicate = bool(prediction_frame.duplicated(keys).any() or farm_frame.duplicated(keys[:2]).any())
        if duplicate:
            manifest["failures"].append({"forecast_origin": None, "errors": {"contract": "Duplicate rows within one forecast origin"}})
        finite_pred = eval_pred[np.isfinite(eval_pred.power_kw)]
        matched = expected_frame.merge(finite_pred[keys].drop_duplicates(), how="inner", on=keys, validate="one_to_one")
        expected_farm = expected_frame[keys[:2]].drop_duplicates()
        finite_farm = eval_farm[np.isfinite(eval_farm.raw_power_kw)]
        matched_farm = expected_farm.merge(finite_farm[keys[:2]].drop_duplicates(), how="inner", on=keys[:2], validate="one_to_one")
        valid_pred = eval_pred[eval_pred["valid"].eq(True)]
        complete_farm = eval_farm[eval_farm["complete"].eq(True)]
        complete_matches = expected_farm.merge(complete_farm[keys[:2]].drop_duplicates(), on=keys[:2], validate="one_to_one")
        manifest["coverage"] = {
            "expected_forecast_rows": int(len(expected_frame)), "received_forecast_rows": int(len(eval_pred)),
            "finite_raw_forecast_rows": int(len(matched)), "physically_valid_forecast_rows": int(len(valid_pred)),
            "missing_forecast_rows": int(len(expected_frame) - len(matched)),
            "expected_unique_turbine_hours": int(len(expected_frame[["turbine_id", "valid_time"]].drop_duplicates())),
            "unique_turbine_hours_with_finite_prediction": int(len(matched[["turbine_id", "valid_time"]].drop_duplicates())),
            "expected_unique_hours": int(expected_frame.valid_time.nunique()),
            "unique_hours_with_finite_prediction": int(matched.valid_time.nunique()),
            "expected_farm_forecast_rows": int(len(expected_farm)), "finite_raw_farm_forecast_rows": int(len(matched_farm)),
            "physically_complete_farm_rows": int(len(complete_matches)),
            "all_emitted_forecast_rows": int(len(prediction_frame)), "all_emitted_farm_rows": int(len(farm_frame)),
        }
        if not duplicate:
            manifest["metrics"] = _metrics(eval_pred, eval_farm, actuals, policy.first_lead_hour)
        full = (len(matched) == len(expected_frame) and len(complete_matches) == len(expected_farm))
        manifest["status"] = "ok" if full and not manifest["failures"] else ("partial" if len(matched) else "blocked")
        manifest["readiness"]["forecast_complete"] = manifest["status"] == "ok"
    manifest["readiness"]["actuals_provided"] = actuals is not None
    manifest["readiness"]["model_artifact_present"] = settings.model_path.is_file()
    manifest["readiness"]["model_manifest_present"] = settings.manifest_path.is_file()
    manifest["model_versions"] = sorted(set(prediction_frame.model_version.dropna().astype(str)))
    manifest["weather_versions"] = sorted(set(prediction_frame.weather_version.dropna().astype(str)))
    _write_csv(prediction_frame, output_dir / "predictions.csv")
    _write_csv(farm_frame, output_dir / "farm.csv")
    manifest["files"] = {name: {"path": str(output_dir / name), "sha256": _hash(output_dir / name)} for name in ("predictions.csv", "farm.csv")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return manifest
