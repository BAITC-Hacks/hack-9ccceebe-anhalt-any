from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ForecastRequest(Schema):
    turbine_id: str = Field(min_length=1, max_length=80)
    forecast_date: date
    horizon_h: int = Field(default=24, ge=1, le=72, strict=True)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class Anomaly(Schema):
    code: str
    severity: Literal["low", "medium", "high"]
    count: int
    timestamps: list[str]
    detail: str


class Analysis(Schema):
    summary: str = Field(max_length=2000)
    risk_level: Literal["low", "medium", "high"]
    anomalies: list[str] = Field(max_length=20)
    recommendation: str = Field(max_length=2000)
    confidence_note: str = Field(max_length=1000)


class HourlyPoint(Schema):
    timestamp: datetime
    wind_speed_ms: float | None
    power_kw: float | None
    valid: bool


class Statistics(Schema):
    avg_wind_ms: float | None
    max_wind_ms: float | None
    min_power_kw: float | None
    max_power_kw: float | None
    predicted_energy_kwh: float | None
    valid_hours: float
    requested_hours: int


class ForecastResult(Schema):
    request: ForecastRequest
    demo: bool
    weather_source: str
    weather_kind: Literal["synthetic", "archived_forecast", "reanalysis"]
    model_id: str
    status: Literal["ok", "partial", "invalid"]
    statistics: Statistics
    hourly: list[HourlyPoint]
    anomalies: list[Anomaly]
    analysis: Analysis
    analysis_source: Literal["openai", "deterministic"]
    analysis_reason: str


class ForecastError(RuntimeError):
    """Expected, user-facing pipeline failure without raw provider payloads."""


class ObservedRequest(Schema):
    turbine_id: str = Field(min_length=1, max_length=80)
    observation_date: date
    horizon_h: int = Field(default=24, ge=1, le=72, strict=True)


class ObservedPoint(HourlyPoint):
    actual_power_kw: float | None


class ObservedMetrics(Schema):
    matched_samples: int
    expected_samples: int
    coverage: float
    mae_kw: float | None
    rmse_kw: float | None
    bias_kw: float | None


class ObservedResult(Schema):
    request: ObservedRequest
    mode: Literal["observed_scada"] = "observed_scada"
    demo: Literal[False] = False
    source: str
    model_id: str
    sampling_interval_minutes: Literal[10] = 10
    hub_height_m: float
    status: Literal["ok", "partial", "invalid"]
    statistics: Statistics
    metrics: ObservedMetrics
    samples: list[ObservedPoint]
    anomalies: list[Anomaly]
    analysis: Analysis
    analysis_source: Literal["openai", "deterministic"]
    analysis_reason: str
    limitation: str
