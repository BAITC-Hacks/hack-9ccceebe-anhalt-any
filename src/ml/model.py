from dataclasses import asdict, dataclass
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import inspect
import logging
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .config import MLConfig
from .data import prepare, capacity, choose_target, chronological_split
from .features import WeatherFeatures

LOG = logging.getLogger(__name__)


@dataclass
class PowerModel:
    pipeline: Pipeline
    config: MLConfig
    metadata: dict
    turbine_models: dict


def scores(y, prediction, normalized=False):
    values = {"MAE": float(mean_absolute_error(y, prediction)),
              "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
              "R2": float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 0 else None}
    if normalized:
        values["normalized_MAE"] = values["MAE"]
    return values


def _fit(train, y, val, yval, config, backend):
    features = WeatherFeatures(config.timezone)
    X = features.fit_transform(train)
    XV = features.transform(val)
    if backend == "mean":
        estimator = DummyRegressor(strategy="mean").fit(X, y)
    elif backend == "hist":
        # Disable sklearn's random internal validation split for time series.
        estimator = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
            max_leaf_nodes=31, l2_regularization=1, early_stopping=False,
            random_state=config.random_state).fit(X, y)
    else:
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            LOG.warning("LightGBM unavailable, using HistGradientBoosting: %s", exc)
            return _fit(train, y, val, yval, config, "hist")
        estimator = lgb.LGBMRegressor(objective="regression", n_estimators=1000,
            learning_rate=0.03, num_leaves=31, max_depth=-1, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=config.random_state,
            n_jobs=1, verbosity=-1)
        try:
            validation = {"eval_X": XV, "eval_y": yval} if "eval_X" in inspect.signature(estimator.fit).parameters else {"eval_set": [(XV, yval)]}
            estimator.fit(X, y, **validation, eval_metric="rmse",
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        except (OSError, lgb.basic.LightGBMError) as exc:
            LOG.warning("LightGBM native failure, using HistGradientBoosting: %s", exc)
            return _fit(train, y, val, yval, config, "hist")
    return Pipeline([("features", features), ("regressor", estimator)])


def _raw(model, df):
    result = np.asarray(model.pipeline.predict(df), dtype=float)
    for turbine, pipeline in model.turbine_models.items():
        mask = (df.turbine_id == turbine).to_numpy()
        if mask.any():
            result[mask] = pipeline.predict(df.loc[mask])
    return result


def _postprocess(raw, config, target_type):
    return np.clip(raw, 0, 1) if target_type == "normalized_power" and config.clip_normalized else raw.copy()


def train_model(df, config=None):
    config = MLConfig(**config) if isinstance(config, dict) else (config or MLConfig())
    data, quality = prepare(df, config, training=True)
    y, kind = choose_target(data, config)
    normalized = kind == "normalized_power"
    quality["target_below_zero"] = int((y < 0).sum())
    quality["target_above_nominal"] = int((y > 1).sum()) if normalized else None
    train, val, test = chronological_split(data, config)
    if config.mode == "B":
        if ((data.loc[val].issued_at <= data.loc[train].timestamp.max()).any()
                or (data.loc[test].issued_at <= data.loc[val].timestamp.max()).any()):
            raise ValueError("Forecast origins overlap preceding fit/selection period; add a temporal gap upstream")
    candidates, comparison = {}, {}
    for name, backend in [("mean", "mean"), ("hist_baseline", "hist"), ("primary", config.backend)]:
        if name == "primary" and config.backend == "hist":
            pipeline = candidates["hist_baseline"]
        else:
            pipeline = _fit(data.loc[train], y.loc[train], data.loc[val], y.loc[val], config, backend)
        raw = pipeline.predict(data.loc[val])
        comparison[name] = {"model_type": type(pipeline[-1]).__name__,
                            "raw": scores(y.loc[val], raw, normalized),
                            "served": scores(y.loc[val], _postprocess(raw, config, kind), normalized)}
        candidates[name] = pipeline
    # Validation RMSE chooses the deployed candidate. Test is never used for selection.
    selected = min(candidates, key=lambda key: comparison[key]["served"]["RMSE"])
    model = PowerModel(candidates[selected], config, {}, {})
    turbine_comparison = {"global_validation_RMSE": comparison[selected]["served"]["RMSE"],
                          "eligible": [], "ineligible": [], "selected": "global"}
    if config.compare_turbines and data.loc[train].turbine_id.nunique() > 1:
        separate = {}
        for turbine in sorted(data.loc[train].turbine_id.unique()):
            ti = train[data.loc[train].turbine_id.to_numpy() == turbine]
            vi = val[data.loc[val].turbine_id.to_numpy() == turbine]
            if len(ti) < config.min_train_rows or len(vi) < config.min_eval_rows:
                turbine_comparison["ineligible"].append(turbine)
                continue
            separate[turbine] = _fit(data.loc[ti], y.loc[ti], data.loc[vi], y.loc[vi], config, config.backend)
            turbine_comparison["eligible"].append(turbine)
        if separate:
            model.turbine_models = separate
            pred = _postprocess(_raw(model, data.loc[val]), config, kind)
            error = scores(y.loc[val], pred, normalized)["RMSE"]
            turbine_comparison["per_turbine_validation_RMSE"] = error
            if error < turbine_comparison["global_validation_RMSE"] * (1 - config.per_turbine_min_improvement):
                turbine_comparison["selected"] = "per_turbine_with_global_fallback"
            else:
                model.turbine_models = {}
    metrics = {}
    for label, indices in [("validation", val), ("test", test)]:
        raw = _raw(model, data.loc[indices])
        metrics[label] = {"raw": scores(y.loc[indices], raw, normalized),
                          "served": scores(y.loc[indices], _postprocess(raw, config, kind), normalized)}
    libraries = {}
    for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]:
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    model.metadata = {
        "artifact_version": 1, "model_type": type(model.pipeline[-1]).__name__,
        "target_type": kind, "target_source": config.target_source,
        "feature_names": model.pipeline[0].feature_names_, "rated_power": config.rated_power,
        "rated_power_verified": config.rated_power_verified, "rated_power_source": config.rated_power_source,
        "rated_power_requires_input": config.rated_power_verified and config.rated_power is None,
        "power_unit": config.power_unit, "timezone": config.timezone,
        "model_parameters": model.pipeline[-1].get_params(), "metrics": metrics,
        "mode": config.mode, "weather_source": config.weather_source,
        "nasa_parameters": config.nasa_parameters, "random_state": config.random_state,
        "forecast_provenance": config.forecast_provenance, "quality": quality,
        "baseline_comparison": comparison, "selected_candidate": selected,
        "turbine_strategy": turbine_comparison, "versions": libraries,
        "python_version": platform.python_version(), "config": asdict(config),
        "clipping": config.clip_normalized if normalized else False,
        "test_used_for_selection": False,
        "per_turbine_model_parameters": {k: p[-1].get_params() for k, p in model.turbine_models.items()}}
    for label, indices in [("training", train), ("validation", val), ("test", test)]:
        model.metadata[f"{label}_start"] = data.loc[indices].timestamp.min().isoformat()
        model.metadata[f"{label}_end"] = data.loc[indices].timestamp.max().isoformat()
        model.metadata[f"{label}_rows"] = len(indices)
    if config.reports_dir:
        from .reports import write_reports
        write_reports(model, data, y, test, Path(config.reports_dir))
    return model


def predict(model, features):
    data, _ = prepare(features, model.config)
    known = model.pipeline[0].turbines_
    if (~data.turbine_id.isin(known)).any():
        LOG.warning("Unknown turbines use global model; validate transfer quality before deployment")
    raw = _raw(model, data)
    value = _postprocess(raw, model.config, model.metadata["target_type"])
    out = data[[c for c in ["timestamp", "turbine_id", "wind_speed", "wind_direction", "temperature", "issued_at"] if c in data]].copy()
    out["raw_prediction"] = raw
    if model.metadata["target_type"] == "normalized_power":
        out["predicted_normalized_power"] = value
        rated = capacity(data, model.config)
        if rated is not None:
            out["predicted_power"] = value * rated.to_numpy()
    else:
        out["predicted_power"] = value
    return out


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    path.with_suffix(".metadata.json").write_text(json.dumps(model.metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_model(path):
    """Load trusted joblib files only: pickle-based artifacts can execute code."""
    model = joblib.load(path)
    if not isinstance(model, PowerModel) or model.metadata.get("artifact_version") != 1:
        raise ValueError("Unsupported ML artifact")
    return model
