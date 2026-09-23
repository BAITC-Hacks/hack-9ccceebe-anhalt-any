from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


def team_forecast_origin() -> str:
    """UI execution default, independent of unconfirmed source CSV semantics."""
    schedule = json.loads((ROOT / "config/forecast_schedule.json").read_text())
    value = schedule["first_forecast_origin"]
    if datetime.fromisoformat(value).tzinfo is None:
        raise ValueError("Team forecast origin requires a timezone")
    return value


class Turbine(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    rated_power_kw: float = Field(gt=0)
    hub_height_m: float = Field(default=80, gt=0)
    model_name: str = "unspecified"
    cut_in_ms: float = Field(default=3, ge=0)
    cut_out_ms: float = Field(default=25, gt=0)
    demo_only: bool = False

    @model_validator(mode="after")
    def valid_limits(self):
        if self.cut_out_ms <= self.cut_in_ms:
            raise ValueError("cut_out_ms must exceed cut_in_ms")
        return self


@dataclass(frozen=True)
class Settings:
    demo: bool = False
    data_module: str = ""
    ml_module: str = ""
    model_path: Path = ROOT / "models/production"
    turbines_file: Path = ROOT / "config/turbines.json"
    cache_dir: Path = ROOT / "data/cache"
    openai_model: str = ""

    @classmethod
    def from_env(cls, demo: bool | None = None):
        load_dotenv(ROOT / ".env", override=False)
        def path(name, default):
            value = Path(os.getenv(name) or default)
            return value if value.is_absolute() else ROOT / value
        return cls(
            demo=(os.getenv("DEMO_MODE", "false").lower() == "true") if demo is None else demo,
            data_module=os.getenv("DATA_MODULE", ""),
            ml_module=os.getenv("ML_MODULE", ""),
            model_path=path("MODEL_PATH", "models/production"),
            turbines_file=path("TURBINES_FILE", "config/turbines.json"),
            cache_dir=path("WEATHER_CACHE_DIR", "data/cache"),
            openai_model=os.getenv("OPENAI_MODEL", ""),
        )

    def turbines(self) -> dict[str, Turbine]:
        return {key: Turbine.model_validate(value) for key, value in
                json.loads(self.turbines_file.read_text()).items()}
