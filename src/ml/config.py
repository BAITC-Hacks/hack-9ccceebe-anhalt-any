from dataclasses import dataclass, field


@dataclass
class MLConfig:
    # Mapping: logical name -> source column. Units must be declared, never inferred.
    columns: dict = field(default_factory=dict)
    timezone: str = "UTC"
    wind_speed_unit: str = "m/s"
    temperature_unit: str = "C"
    power_unit: str | None = None
    target_source: str | None = None
    normalized_target_confirmed: bool = False
    rated_power: float | dict | None = None
    rated_power_verified: bool = False
    rated_power_source: str | None = None
    weather_source: str = "unspecified"
    nasa_parameters: list = field(default_factory=list)
    mode: str = "A"
    forecast_provenance: str | None = None
    min_lead_hours: float = 1.0
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    min_train_rows: int = 30
    min_eval_rows: int = 5
    random_state: int = 42
    backend: str = "auto"
    compare_turbines: bool = True
    per_turbine_min_improvement: float = 0.01
    clip_normalized: bool = True
    reports_dir: str | None = "reports"

    def __post_init__(self):
        if self.mode not in ("A", "B") or self.backend not in ("auto", "hist"):
            raise ValueError("mode must be A/B; backend must be auto/hist")
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("Invalid chronological split fractions")
        if self.rated_power_verified and not self.rated_power_source:
            raise ValueError("Verified rated power requires rated_power_source")
        if self.min_lead_hours < 0:
            raise ValueError("min_lead_hours must be nonnegative")
