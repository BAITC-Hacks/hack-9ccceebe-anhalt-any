"""Strict Kelmarsh adapter for the agent's numeric DATA boundary.

Keep src.ml.model.predict's DataFrame API unchanged. This provider returns RAW kW.
Context comes from explicit DataFrame attrs, never an inferred turbine or nominal.
See docs/ML_INFERENCE_CONTRACT.md for the integration handoff still required from C.
"""
from pathlib import Path
import numbers

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from . import model as core

INPUT_COLUMNS = ("wind_speed_ms", "temperature_c", "wind_direction_deg")
# Published Kelmarsh_WT_static.csv (Zenodo 16807551). This is NOT the Goldwind registry.
HUB_HEIGHTS_M = {"Kelmarsh 1": 78.5, "Kelmarsh 2": 78.5, "Kelmarsh 3": 68.5,
                 "Kelmarsh 4": 78.5, "Kelmarsh 5": 78.5, "Kelmarsh 6": 68.5}


def _validate_model(model):
    if not isinstance(model, core.PowerModel):
        raise ValueError("Expected the published Kelmarsh PowerModel artifact")
    metadata, config = model.metadata, model.config
    if metadata.get("target_type") != "normalized_power" or metadata.get("power_unit") != "kW":
        raise ValueError("Provider requires normalized Kelmarsh target with capacity in kW")
    if (not metadata.get("rated_power_verified") or not config.rated_power_verified
            or not metadata.get("rated_power_source") or config.power_unit != "kW"):
        raise ValueError("Verified rated capacity in kW is required")
    capacities = metadata.get("rated_power")
    if not isinstance(capacities, dict) or capacities != config.rated_power:
        raise ValueError("Artifact capacity metadata and saved configuration disagree")
    if set(capacities) != set(HUB_HEIGHTS_M) or any(value != 2050 for value in capacities.values()):
        raise ValueError("This provider supports the verified Kelmarsh MM92 registry only")
    if set(model.pipeline[0].turbines_) != set(capacities):
        raise ValueError("Artifact turbine encoder and registry disagree")
    return capacities


def load_model(path):
    """Load the trusted local published artifact; do not accept uploaded pickle files."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ML artifact missing: {path.resolve()}; configure MODEL_PATH")
    model = core.load_model(path)
    _validate_model(model)
    return model


def predict(model, features: pd.DataFrame) -> np.ndarray:
    """One explicitly identified Kelmarsh turbine per batch; raw float64 kW, same order.

    Required attrs: turbine_id (exact string), wind_height_m (numeric registered hub
    height). All three weather columns must be numeric, finite and not pre-encoded.
    NaN/Inf output is deliberately preserved for the agent's deterministic rules.
    """
    capacities = _validate_model(model)
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a nonempty pandas DataFrame")
    if features.columns.has_duplicates or set(features.columns) != set(INPUT_COLUMNS):
        raise ValueError(f"Expected exactly numeric columns {INPUT_COLUMNS}; no encoded or target features")
    index = features.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans or not index.is_unique:
        raise ValueError("A unique, timezone-aware DatetimeIndex without NaT is required")
    turbine = features.attrs.get("turbine_id")
    if not isinstance(turbine, str) or turbine not in capacities:
        raise ValueError("Explicit attrs['turbine_id'] must name Kelmarsh 1..6; Goldwind T1/T2 transfer is not validated")
    height = features.attrs.get("wind_height_m")
    if (not isinstance(height, numbers.Real) or isinstance(height, (bool, np.bool_))
            or not np.isfinite(height) or not np.isclose(height, HUB_HEIGHTS_M[turbine], rtol=0, atol=1e-6)):
        raise ValueError(f"attrs['wind_height_m'] must equal {HUB_HEIGHTS_M[turbine]} for {turbine}; no height extrapolation in ML")
    for column in INPUT_COLUMNS:
        dtype = features[column].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise ValueError(f"{column} must have a real numeric dtype")
    values = features.loc[:, INPUT_COLUMNS].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite weather must be filtered by the orchestrator before ML inference")
    if (values[:, 0] < 0).any():
        raise ValueError("wind_speed_ms must be nonnegative")
    if ((values[:, 2] < 0) | (values[:, 2] >= 360)).any():
        raise ValueError("wind_direction_deg must be in [0, 360); normalize upstream explicitly")
    canonical = pd.DataFrame({"timestamp": index.tz_convert("UTC"), "turbine_id": turbine,
                              "wind_speed": values[:, 0], "temperature": values[:, 1],
                              "wind_direction": values[:, 2]}, index=index)
    # core.predict keeps raw_prediction before optional clipping; never use predicted_power here.
    result = core.predict(model, canonical)
    if len(result) != len(features) or not result.index.equals(index):
        raise ValueError("ML pipeline did not preserve input row order")
    raw = result["raw_prediction"].to_numpy(dtype=np.float64, copy=True)
    if raw.shape != (len(features),):
        raise ValueError("ML pipeline must return exactly one raw prediction per row")
    return raw * float(capacities[turbine])
