from __future__ import annotations

import numpy as np
import pandas as pd

from src.agent.schemas import Anomaly, Statistics
from src.config import Turbine


def finite(value) -> float | None:
    return float(value) if np.isfinite(value) else None


def evaluate(weather: pd.DataFrame, power: np.ndarray, turbine: Turbine):
    wind = weather.wind_speed_ms.to_numpy(dtype=float)
    temp = weather.temperature_c.to_numpy(dtype=float)
    missing = ~np.isfinite(wind) | ~np.isfinite(temp)
    bad_weather = (wind < 0) | (wind > 100) | (temp < -90) | (temp > 65)
    bad_power = ~np.isfinite(power)
    negative = power < 0
    over_rated = power > turbine.rated_power_kw
    valid = ~(missing | bad_weather | bad_power | negative | over_rated)
    jumps = np.r_[False, np.abs(np.diff(power)) > 0.5 * turbine.rated_power_kw]
    strong_zero = ((wind >= max(10, turbine.cut_in_ms)) & (wind < turbine.cut_out_ms)
                   & (power < 0.05 * turbine.rated_power_kw) & (power >= 0))
    flags = []
    def flag(code, severity, mask, detail):
        if mask.any():
            flags.append(Anomaly(code=code, severity=severity, count=int(mask.sum()),
                                 timestamps=[t.isoformat() for t in weather.index[mask][:3]], detail=detail))
    flag("missing_weather", "high", missing, "Missing or non-finite weather; inference skipped for affected hours.")
    flag("invalid_weather", "high", bad_weather, "Weather outside broad physical sanity limits.")
    flag("non_finite_power", "high", bad_power, "Missing or non-finite model prediction.")
    flag("negative_power", "high", negative, "Model predicted negative power; not silently clipped.")
    flag("above_rated_power", "high", over_rated, "Normalized power exceeds 1; check units and model calibration.")
    flag("hourly_jump", "medium", jumps, "Hour-to-hour change exceeds 50% of rated power.")
    flag("strong_wind_low_power", "medium", strong_zero, "Wind >=10 m/s below cut-out with power below 5% rated; review conditions.")
    finite_wind = wind[np.isfinite(wind)]
    finite_power = power[np.isfinite(power)]
    statistics = Statistics(
        avg_wind_ms=finite(finite_wind.mean()) if len(finite_wind) else None,
        max_wind_ms=finite(finite_wind.max()) if len(finite_wind) else None,
        min_power_kw=finite(finite_power.min()) if len(finite_power) else None,
        max_power_kw=finite(finite_power.max()) if len(finite_power) else None,
        # Inputs represent hourly average kW; each interval is exactly 1h.
        predicted_energy_kwh=float(power.sum()) if valid.all() else None,
        valid_hours=int(valid.sum()), requested_hours=len(power),
    )
    return flags, statistics, valid
