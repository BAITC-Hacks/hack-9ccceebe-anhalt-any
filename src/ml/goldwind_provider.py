"""Goldwind-only agent adapter. A configured module is not proof an artifact exists."""
from pathlib import Path
import numbers

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from .goldwind_data import GoldwindContract, TURBINES
from .goldwind_model import INPUT_COLUMNS
from .model import PowerModel, load_model as load_core


def validate_model(model):
    if not isinstance(model, PowerModel) or model.metadata.get("goldwind_artifact_version") != 1:
        raise ValueError("A separately trained Goldwind artifact is required; Kelmarsh is not supported")
    meta = model.metadata
    contract = GoldwindContract(**meta["contract"])
    contract.validate(training=True)
    if (meta.get("data_source") != "organizer_scada" or meta.get("target_type") != "absolute_power"
            or meta.get("power_unit") != "kW" or meta.get("clipping") is not False
            or meta.get("rated_power") != TURBINES or meta.get("input_columns") != list(INPUT_COLUMNS)
            or meta.get("wind_height_m") != contract.wind_height_m
            or meta.get("timezone") != contract.timezone or meta.get("interval_minutes") != 60
            or model.config.power_unit != "kW" or model.config.clip_normalized
            or model.config.rated_power != TURBINES or model.turbine_models):
        raise ValueError("Incompatible Goldwind model schema, units or capacity metadata")
    transformer = model.pipeline[0]
    if (set(transformer.turbines_) != set(TURBINES) or transformer.direction_
            or transformer.timezone != contract.timezone):
        raise ValueError("Saved feature transformer disagrees with the Goldwind input contract")
    origin = pd.Timestamp(contract.first_forecast_origin)
    for split in meta["splits"].values():
        end, available = pd.Timestamp(split["interval_end"]), pd.Timestamp(split["available_until"])
        if end.tzinfo is None or available.tzinfo is None or not end <= available <= origin:
            raise ValueError("Model fit/evaluation exceeds confirmed availability cutoff")
    return meta


def load_model(path):
    """Load trusted team joblib only (pickle executes code); no fit or fallback."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Goldwind artifact missing: {path.resolve()}; training requires confirmed organizer semantics")
    model = load_core(path)
    validate_model(model)
    return model


def predict(model, features: pd.DataFrame) -> np.ndarray:
    meta = validate_model(model)
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Nonempty pandas DataFrame required")
    if features.columns.has_duplicates or set(features.columns) != set(INPUT_COLUMNS):
        raise ValueError(f"Expected exactly {INPUT_COLUMNS}; no target, direction or lag columns")
    index = features.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans or not index.is_unique:
        raise ValueError("Unique timezone-aware DatetimeIndex without NaT required")
    utc = index.tz_convert("UTC")
    if not utc.equals(utc.floor("h")):
        raise ValueError("Index must mark hourly interval starts")
    turbine = features.attrs.get("turbine_id")
    if not isinstance(turbine, str) or turbine not in TURBINES:
        raise ValueError("Explicit attrs['turbine_id'] T1 or T2 is required")
    height = features.attrs.get("wind_height_m")
    if (not isinstance(height, numbers.Real) or isinstance(height, (bool, np.bool_))
            or not np.isfinite(height) or not np.isclose(height, meta["wind_height_m"][turbine], rtol=0, atol=1e-6)):
        raise ValueError("Explicit wind_height_m must match confirmed training measurement height")
    for column in INPUT_COLUMNS:
        dtype = features[column].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise ValueError(f"{column} must have real numeric dtype")
    values = features.loc[:, INPUT_COLUMNS].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or (values[:, 0] < 0).any():
        raise ValueError("Missing/nonfinite weather or negative wind is not accepted")
    canonical = pd.DataFrame({"timestamp": utc, "turbine_id": turbine,
                              "wind_speed": values[:, 0], "temperature": values[:, 1]}, index=index)
    raw = np.asarray(model.pipeline.predict(canonical), dtype=np.float64)
    if raw.shape != (len(features),):
        raise ValueError("Expected one raw kW prediction per input row")
    # Target is already kW. Preserve negative/above-rated/nonfinite outputs for agent rules.
    return raw
