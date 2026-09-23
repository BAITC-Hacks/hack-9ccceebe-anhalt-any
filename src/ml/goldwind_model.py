"""Gated Goldwind training using existing sklearn preprocessing; no weather downloader."""
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from .config import MLConfig
from .data import chronological_split
from .goldwind_data import GoldwindContract, TURBINES, training_rows
from .model import PowerModel, _fit, scores

INPUT_COLUMNS = ("wind_speed_ms", "temperature_c")


def partition(hourly, contract):
    """Chronological 70/15/15, purging late publication at each next period start."""
    data = training_rows(hourly, contract)
    lower = pd.Timestamp("2023-03-01", tz=contract.timezone).tz_convert("UTC")
    upper = pd.Timestamp("2026-02-01", tz=contract.timezone).tz_convert("UTC")
    data = data.loc[(data.timestamp >= lower) & (data.interval_end <= upper)].reset_index(drop=True)
    if (data.duplicated(["timestamp", "turbine_id"]).any()
            or not data.interval_end.eq(data.timestamp + pd.Timedelta(1, unit="h")).all()
            or not np.isfinite(data[["wind_speed", "temperature", "power"]].to_numpy(dtype=float)).all()
            or not data.coverage.eq(1).all() or (data.available_at < data.interval_end).any()):
        raise ValueError("Hourly training data must have unique full finite intervals with valid availability")
    config = MLConfig(reports_dir=None)
    train, validation, test = chronological_split(data, config)
    val_start, test_start = data.loc[validation].timestamp.min(), data.loc[test].timestamp.min()
    train = train[(data.loc[train].available_at <= val_start).to_numpy()
                  & (data.loc[train].interval_end <= val_start).to_numpy()]
    validation = validation[(data.loc[validation].available_at <= test_start).to_numpy()
                            & (data.loc[validation].interval_end <= test_start).to_numpy()]
    indices = {"training": train, "validation": validation, "test": test}
    for label, rows in indices.items():
        minimum = config.min_train_rows if label == "training" else config.min_eval_rows
        if len(rows) < minimum or set(data.loc[rows].turbine_id) != set(TURBINES):
            raise ValueError(f"Insufficient available T1/T2 observations in {label} after temporal purge")
    return data, indices


def _metrics(frame, raw):
    result = scores(frame.power.to_numpy(), raw)
    result["normalized_MAE"] = result["MAE"] / 2500.
    result["NRMSE"] = result["RMSE"] / 2500.
    return result


def _lineage(files):
    if not files or any(not item.get("file") or len(item.get("sha256", "")) != 64 for item in files):
        raise ValueError("Source file hashes from read_organizer_files are required")


def train_model(hourly, contract: GoldwindContract, files):
    """All units/origin must be confirmed before fit. Target is converted raw kW.

    The frozen artifact remains fitted on the training partition only. Validation
    selects mean versus HistGradientBoosting; test is assessed once, never selected on.
    """
    contract.validate(training=True)
    _lineage(files)
    data, splits = partition(hourly, contract)
    train, val = splits["training"], splits["validation"]
    config = MLConfig(timezone=contract.timezone, power_unit="kW", backend="hist",
                      target_source=contract.dataset_source, weather_source="organizer measured weather",
                      rated_power=TURBINES.copy(), rated_power_verified=True,
                      rated_power_source=contract.metadata_source, clip_normalized=False,
                      compare_turbines=False, reports_dir=None)
    # Explicit feature allowlist: exclude every source lag, rolling statistic and direction.
    columns = ["timestamp", "turbine_id", "wind_speed", "temperature"]
    features = data[columns]
    candidates, comparison = {}, {}
    for name, backend in [("mean", "mean"), ("hist", "hist")]:
        pipe = _fit(features.loc[train], data.loc[train].power,
                    features.loc[val], data.loc[val].power, config, backend)
        candidates[name] = pipe
        comparison[name] = _metrics(data.loc[val], pipe.predict(features.loc[val]))
    selected = min(comparison, key=lambda name: comparison[name]["RMSE"])
    pipe = candidates[selected]
    metadata = {
        "artifact_version": 1, "goldwind_artifact_version": 1, "station": "Goldwind GW109/2500",
        "data_source": "organizer_scada", "target_type": "absolute_power", "power_unit": "kW",
        "mode": "A", "clipping": False, "rated_power": TURBINES.copy(),
        "wind_height_m": contract.wind_height_m, "input_columns": list(INPUT_COLUMNS),
        "feature_names": pipe[0].feature_names_, "timezone": contract.timezone,
        "timestamp_semantics": "UTC hourly interval start", "interval_minutes": 60,
        "target_conversion": {"source_unit": contract.target_unit,
                              "normalized_reference_kw": contract.normalized_reference_kw},
        "contract": asdict(contract), "files": files, "selected_candidate": selected,
        "baseline_comparison": comparison, "turbine_strategy": "global with train-fitted T1/T2 encoding",
        "model_type": type(pipe[-1]).__name__, "model_parameters": pipe[-1].get_params(),
        "random_state": 42, "test_used_for_selection": False, "refit_after_selection": False,
        "versions": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "joblib"]},
        "python_version": platform.python_version(),
        "split_policy": "70/15/15 unique timestamps; purge availability beyond next partition start",
        "distribution_shift": "Measured weather MODE A; forecast-weather MODE B quality remains unknown",
        "metrics": {}, "splits": {},
    }
    for label, rows in splits.items():
        subset = data.loc[rows]
        metadata["splits"][label] = {
            "start": subset.timestamp.min().isoformat(), "end": subset.timestamp.max().isoformat(),
            "interval_end": subset.interval_end.max().isoformat(),
            "available_until": subset.available_at.max().isoformat(), "rows": len(subset),
        }
        if label != "training":
            metadata["metrics"][label] = _metrics(subset, pipe.predict(features.loc[rows]))
    metadata["training_end"] = metadata["splits"]["training"]["end"]
    return PowerModel(pipe, config, metadata, {})


def evaluate_model(model, hourly, contract, files):
    """Reproduce the unchanged test partition using exact source hashes and contract."""
    from .goldwind_provider import validate_model
    validate_model(model)
    if asdict(contract) != model.metadata["contract"] or files != model.metadata["files"]:
        raise ValueError("Contract or source hashes differ from frozen training lineage")
    data, splits = partition(hourly, contract)
    subset = data.loc[splits["test"]].copy()
    raw = model.pipeline.predict(subset[["timestamp", "turbine_id", "wind_speed", "temperature"]])
    subset["raw_prediction_kw"] = raw
    report = {"mode": "A", "output_unit": "kW", "metrics": _metrics(subset, raw),
              "test_rows": len(subset), "february_metrics": None, "groups": {}}
    subset["wind_bin"] = pd.cut(subset.wind_speed, [0, 3, 8, 12, 20, np.inf], right=False)
    subset["month"] = subset.timestamp.dt.tz_convert(contract.timezone).dt.month
    for group in ["turbine_id", "wind_bin", "month"]:
        report["groups"][group] = {
            str(key): {"rows": len(part), **_metrics(part, part.raw_prediction_kw.to_numpy())}
            for key, part in subset.groupby(group, observed=True)
        }
    return report, subset


def save_model(model, directory):
    """Write a new separate bundle; never overwrite Kelmarsh or an earlier delivery."""
    from .goldwind_provider import validate_model
    validate_model(model)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Model directory must be empty; preserve previous artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "power_model.joblib"
    joblib.dump(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = model.metadata
    train = meta["splits"]["training"]
    # Conservative cutoff includes ALL evaluation data, even the untouched test.
    assessed = [meta["splits"][name] for name in ["validation", "test"]]
    manifest = {
        "model_version": "goldwind-" + digest[:12], "artifact_sha256": digest,
        "data_source": "organizer_scada", "provenance": json.dumps({"files": meta["files"],
            "contract": meta["contract"], "splits": meta["splits"], "policy": meta["split_policy"]}),
        "turbine_ids": list(TURBINES), "rated_power_kw": TURBINES,
        "wind_height_m": meta["wind_height_m"], "feature_columns": list(INPUT_COLUMNS),
        "output_unit": "kW", "normalized_target_definition": "fraction_of_rated_power",
        "timezone": meta["timezone"], "interval_minutes": 60,
        "training_data_available_until": train["available_until"],
        "training_target_interval_end": train["interval_end"],
        "selection_data_available_until": max(s["available_until"] for s in assessed),
        "selection_target_interval_end": max(s["interval_end"] for s in assessed),
        "availability_evidence": meta["contract"]["availability_source"] + "; includes untouched test cutoff",
    }
    for filename, content in [("power_model.metadata.json", meta), ("forecast_manifest.json", manifest)]:
        (directory / filename).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path
