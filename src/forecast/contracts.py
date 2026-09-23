"""Strict target-station forecast contract, separate from demo and observed MODE A."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.agent.schemas import ForecastError
from src.config import ROOT


def utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ForecastError("Timezone-aware timestamps are required; naive time is not accepted")
    return stamp.tz_convert("UTC")


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ForecastProtocol(Contract):
    confirmed: bool = False
    timezone: str
    daily_origin_hour: int = Field(ge=0, le=23)
    # Forecast valid_time labels the START of the hourly interval, not issuance.
    first_lead_hour: Literal[0, 1] = 1
    interval_label: Literal["start"] = "start"
    normalization: Literal["fraction_of_rated_power"] = "fraction_of_rated_power"
    evidence: str = Field(min_length=1)

    @field_validator("timezone")
    @classmethod
    def timezone_exists(cls, value):
        ZoneInfo(value)
        return value

    def require_confirmed(self):
        if not self.confirmed:
            raise ForecastError("Organizer timezone, hourly labels and normalized-power definition are not confirmed")


class ModelManifest(Contract):
    model_version: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_source: Literal["organizer_scada"]
    provenance: str = Field(min_length=1)
    turbine_ids: list[str]
    rated_power_kw: dict[str, float]
    wind_height_m: dict[str, float]
    feature_columns: list[str]
    output_unit: Literal["kW"]
    normalized_target_definition: Literal["fraction_of_rated_power"]
    timezone: str
    interval_minutes: Literal[60]
    # Includes target interval completion, publication latency and ALL fit/selection preprocessing.
    training_data_available_until: datetime
    selection_data_available_until: datetime
    training_target_interval_end: datetime
    selection_target_interval_end: datetime
    availability_evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_times_and_features(self):
        for field in ("training_data_available_until", "selection_data_available_until",
                      "training_target_interval_end", "selection_target_interval_end"):
            utc(getattr(self, field))
        if utc(self.training_target_interval_end) > utc(self.training_data_available_until):
            raise ValueError("Training target is not available before its interval ends")
        if utc(self.selection_target_interval_end) > utc(self.selection_data_available_until):
            raise ValueError("Selection target is not available before its interval ends")
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be nonempty and unique")
        forbidden = {"power", "actual_power_kw", "power_kw", "normalized_power", "target"}
        if forbidden.intersection(self.feature_columns):
            raise ValueError("Targets cannot enter forecast features")
        return self


class ForecastSettings(Contract):
    data_module: str = ""
    ml_module: str = ""
    model_path: Path = ROOT / "models/goldwind/power_model.joblib"
    manifest_path: Path = ROOT / "models/goldwind/forecast_manifest.json"
    protocol_path: Path = ROOT / "config/forecast_protocol.json"
    turbines_file: Path = ROOT / "config/turbines.json"
    cache_dir: Path = ROOT / "data/cache/forecast_v2"
    results_dir: Path = ROOT / "results/forecast_v2"
    openai_model: str = ""

    @classmethod
    def from_env(cls):
        import os
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        def path(name, default):
            p = Path(os.getenv(name) or default)
            return p if p.is_absolute() else ROOT / p
        return cls(data_module=os.getenv("FORECAST_DATA_MODULE", ""),
                   ml_module=os.getenv("FORECAST_ML_MODULE", ""),
                   model_path=path("FORECAST_MODEL_PATH", "models/goldwind/power_model.joblib"),
                   manifest_path=path("FORECAST_MANIFEST_PATH", "models/goldwind/forecast_manifest.json"),
                   protocol_path=path("FORECAST_PROTOCOL_PATH", "config/forecast_protocol.json"),
                   turbines_file=path("TURBINES_FILE", "config/turbines.json"),
                   cache_dir=path("FORECAST_CACHE_DIR", "data/cache/forecast_v2"),
                   results_dir=path("FORECAST_RESULTS_DIR", "results/forecast_v2"),
                   openai_model=os.getenv("OPENAI_MODEL", ""))


def load_protocol(settings: ForecastSettings) -> ForecastProtocol:
    try:
        policy = ForecastProtocol.model_validate_json(settings.protocol_path.read_text())
        policy.require_confirmed()
        return policy
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError("Provide confirmed forecast protocol: timezone, daily_origin_hour, first_lead_hour and organizer evidence") from exc


def expected_times(origin, horizon_h: int, protocol: ForecastProtocol) -> pd.DatetimeIndex:
    if type(horizon_h) is not int or horizon_h not in (24, 48):
        raise ForecastError("Target forecast horizon must be exactly 24 or 48 hours")
    origin = utc(origin)
    if origin != origin.floor("h"):
        raise ForecastError("Forecast origin must be on an hourly boundary")
    return pd.date_range(origin + pd.Timedelta(protocol.first_lead_hour, unit="h"), periods=horizon_h, freq="h")
