from dataclasses import replace
import pandas as pd
from .data import prepare, choose_target
from .model import PowerModel, _raw, _postprocess, scores


def evaluate_model(model, df, mode="A", forecast_provenance=None):
    """Evaluate future holdout data. MODE B requires archived forecasts available at origin.

    Caller joins forecast valid timestamp/turbine with real observed target upstream.
    Selection of forecasts must not depend on future observations.
    """
    config = replace(model.config, mode=mode, forecast_provenance=forecast_provenance or model.config.forecast_provenance)
    data, quality = prepare(df, config, training=True)
    if data.timestamp.min() <= pd.Timestamp(model.metadata["validation_end"]):
        raise ValueError("Evaluation overlaps training/validation; supply later holdout observations")
    if mode == "B" and (data.issued_at <= pd.Timestamp(model.metadata["validation_end"])).any():
        raise ValueError("Forecast origin precedes model selection cutoff; use a later origin")
    target, kind = choose_target(data, config)
    if kind != model.metadata["target_type"]:
        raise ValueError("Evaluation target differs from trained target")
    view = PowerModel(model.pipeline, config, model.metadata, model.turbine_models)
    raw = _raw(view, data)
    served = _postprocess(raw, config, kind)
    result = {"mode": mode, "rows": len(data), "raw": scores(target, raw, kind == "normalized_power"),
              "served": scores(target, served, kind == "normalized_power"), "quality": quality}
    if mode == "B":
        lead = (data.timestamp - data.issued_at).dt.total_seconds() / 3600
        result["forecast_provenance"] = config.forecast_provenance
        result["lead_hours"] = {"min": float(lead.min()), "max": float(lead.max())}
    return result
