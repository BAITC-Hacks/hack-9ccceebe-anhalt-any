import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WeatherFeatures(TransformerMixin, BaseEstimator):
    """Allowlist prevents accidental target, lag, forecast-vintage or timestamp-ID leakage."""
    def __init__(self, timezone="UTC"):
        self.timezone = timezone

    def fit(self, X, y=None):
        self.direction_ = "wind_direction" in X and X.wind_direction.notna().any()
        self.turbines_ = sorted(X.turbine_id.unique())
        self.feature_names_ = list(self.transform(X).columns)
        return self

    def transform(self, X):
        t = X.timestamp.dt.tz_convert(self.timezone)
        out = pd.DataFrame({"wind_speed": X.wind_speed, "temperature": X.temperature}, index=X.index)
        if self.direction_:
            if "wind_direction" not in X:
                raise ValueError("Training used wind_direction; inference must supply that column")
            angle = np.deg2rad(X.wind_direction)
            out["wind_dir_sin"], out["wind_dir_cos"] = np.sin(angle), np.cos(angle)
        for name, value, period in [("hour", t.dt.hour + t.dt.minute / 60, 24),
                                    ("month", t.dt.month - 1, 12),
                                    ("day_of_year", t.dt.dayofyear - 1, np.where(t.dt.is_leap_year, 366, 365))]:
            out[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
            out[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
        # Explicit train-only one-hot vocabulary; unseen turbine -> all zeros.
        for i, turbine in enumerate(self.turbines_):
            out[f"turbine_{i}"] = (X.turbine_id == turbine).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)
