import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from .model import _raw, _postprocess


def write_reports(model, data, y, test_indices, directory):
    directory = Path(directory)
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    test = data.loc[test_indices].copy()
    raw = _raw(model, test)
    prediction = _postprocess(raw, model.config, model.metadata["target_type"])
    actual = y.loc[test_indices].to_numpy()
    test["actual"] = actual
    test["raw_prediction"] = raw
    test["prediction"] = prediction
    test["absolute_error"] = np.abs(actual - prediction)
    test["squared_error"] = (actual - prediction) ** 2
    test["wind_bin"] = pd.cut(test.wind_speed, [0, 3, 8, 12, 20, np.inf], right=False,
                              labels=["0-3", "3-8", "8-12", "12-20", "20+"])
    month = test.timestamp.dt.tz_convert(model.config.timezone).dt.month
    test["season"] = month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                               6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    errors = {}
    for group in ["wind_bin", "turbine_id", "season"]:
        stats = test.groupby(group, observed=True).agg(count=("actual", "size"), MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"))
        stats["RMSE"] = np.sqrt(stats.pop("MSE"))
        stats.to_csv(directory / f"errors_by_{group}.csv")
        errors[group] = stats.reset_index().to_dict(orient="records")
    test.to_csv(directory / "test_predictions.csv", index=False)
    distributions = data[[c for c in ["wind_speed", "temperature", "power", "normalized_power"] if c in data]].describe(percentiles=[.01, .05, .5, .95, .99]).to_dict()
    intervals = {}
    for turbine, group in data.groupby("turbine_id"):
        diff = group.timestamp.sort_values().diff().dropna().dt.total_seconds()
        intervals[turbine] = {"observations": len(group), "median_interval_seconds": float(diff.median()) if len(diff) else None,
                             "interval_counts": {str(k): int(v) for k, v in diff.value_counts().head(10).items()}}
    eda = {"observations": len(data), "start": data.timestamp.min().isoformat(), "end": data.timestamp.max().isoformat(),
           "timestamp_storage": "UTC", "feature_timezone": model.config.timezone,
           "turbine_count": data.turbine_id.nunique(), "sampling": intervals,
           "missing": data.isna().sum().to_dict(), "distributions": distributions,
           "quality": model.metadata["quality"],
           "interpretation": "Descriptive full-dataset EDA, not used for model selection. No automatic outlier removal. SCADA operating status and turbine specifications needed to confirm cut-in/rated/cut-out regions."}
    (directory / "eda.json").write_text(json.dumps(eda, indent=2, default=str), encoding="utf-8")
    ylabel = "Normalized power" if model.metadata["target_type"] == "normalized_power" else f"Active power ({model.config.power_unit})"
    title = f"MODE {model.config.mode} | chronological test"

    def finish(name, xlabel, ylabel_, title_=title):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_)
        plt.title(title_)
        plt.grid(alpha=.2)
        plt.tight_layout()
        plt.savefig(figures / name, dpi=140)
        plt.close()

    plt.figure(figsize=(9, 6))
    colors = plt.get_cmap("tab20")
    for i, (turbine, group) in enumerate(test.groupby("turbine_id")):
        color = colors(i % 20)
        plt.scatter(group.wind_speed, group.actual, s=9, alpha=.25, color=color, label=f"{turbine}: actual")
        binned = group.groupby(pd.cut(group.wind_speed, np.linspace(0, max(1, test.wind_speed.max()) + .001, 25)), observed=True)
        curve = binned[["wind_speed", "prediction"]].mean()
        plt.plot(curve.wind_speed, curve.prediction, color=color, label=f"{turbine}: predicted bin mean")
    if test.turbine_id.nunique() <= 10:
        plt.legend(fontsize=7)
    finish("power_curve.png", "Wind speed (m/s)", ylabel)
    # Tabulated empirical curve aids inspection without pretending to know thresholds.
    curve_data = data.assign(target=y, wind_bin=np.floor(data.wind_speed))
    curve_data.groupby(["turbine_id", "wind_bin"]).target.agg(["count", "mean", "median", "min", "max"]).to_csv(directory / "empirical_power_curve.csv")
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, prediction, s=10, alpha=.4)
    lo, hi = min(actual.min(), prediction.min()), max(actual.max(), prediction.max())
    plt.plot([lo, hi], [lo, hi], "k--")
    finish("actual_vs_predicted.png", f"Actual {ylabel}", f"Predicted {ylabel}")
    plt.figure(figsize=(9, 5))
    plt.scatter(test.wind_speed, prediction - actual, s=9, alpha=.3)
    plt.axhline(0, color="black", linestyle="--")
    finish("error_vs_wind_speed.png", "Wind speed (m/s)", "Prediction - actual")
    plt.figure(figsize=(11, 5))
    for turbine, group in test.groupby("turbine_id"):
        plt.plot(group.timestamp, group.actual, alpha=.7, label=f"{turbine} actual")
        plt.plot(group.timestamp, group.prediction, alpha=.7, linestyle="--", label=f"{turbine} prediction")
    if test.turbine_id.nunique() <= 6:
        plt.legend(fontsize=7)
    plt.xticks(rotation=20)
    finish("time_series_prediction.png", "Timestamp (UTC)", ylabel)
    importances = []
    for scope, pipeline in {"global": model.pipeline, **model.turbine_models}.items():
        subset = test if scope == "global" else test[test.turbine_id == scope]
        if subset.empty:
            continue
        estimator = pipeline[-1]
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
            method = "LightGBM split importance"
        else:
            sample = subset.iloc[np.linspace(0, len(subset) - 1, min(500, len(subset))).astype(int)]
            values = permutation_importance(estimator, pipeline[0].transform(sample), sample.actual,
                scoring="neg_mean_absolute_error", n_repeats=3, random_state=model.config.random_state,
                n_jobs=1).importances_mean
            method = "test permutation MAE increase (diagnostic only; correlated features caution)"
        importances.extend({"scope": scope, "feature": f, "importance": float(v), "method": method}
                           for f, v in zip(pipeline[0].feature_names_, values))
    importance = pd.DataFrame(importances).sort_values("importance", ascending=False)
    importance.to_csv(directory / "feature_importance.csv", index=False)
    worst = max(errors["wind_bin"], key=lambda item: item["MAE"])
    metrics = model.metadata["metrics"]["test"]
    summary = f"""# Evaluation: MODE {model.config.mode}

{'Observed weather -> power; does not measure weather forecast skill.' if model.config.mode == 'A' else 'Archived forecast -> power; includes forecast error.'}

Target: {model.metadata['target_type']}; units: {ylabel}.
Test: {model.metadata['test_start']} through {model.metadata['test_end']}.
Selected candidate: {model.metadata['selected_candidate']}; turbine strategy: {model.metadata['turbine_strategy']['selected']}.
Model fitted on training only; validation used for selection/early stopping; test held out.

Served metrics: {json.dumps(metrics['served'])}
Raw metrics before clipping: {json.dumps(metrics['raw'])}
Targets below zero: {model.metadata['quality']['target_below_zero']}; above nominal: {model.metadata['quality']['target_above_nominal']} (full dataset).
Largest observed wind-bin MAE: {worst['wind_bin']} m/s, MAE={worst['MAE']:.6g}, n={worst['count']}.
Importance: see feature_importance.csv; top entries: {', '.join(importance.head(5).feature)}.

Wind bins are diagnostic groups, not inferred cut-in/rated/cut-out thresholds.
Use empirical_power_curve.csv and turbine specifications/operating status to assess these regions.
Missing regimes/seasons cannot be evaluated. Irregular sampling and temporal correlation limit metric interpretation.
No uncertainty intervals or forecast-skill claims are produced from mock data.
"""
    (directory / "evaluation.md").write_text(summary, encoding="utf-8")
    (directory / "metrics.json").write_text(json.dumps(model.metadata, indent=2, default=str), encoding="utf-8")
