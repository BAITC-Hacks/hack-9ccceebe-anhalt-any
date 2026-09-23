"""Numeric calendar/weather features; power and other target-derived columns never enter."""
import numpy as np
import pandas as pd


def build_features(df, *, calendar=True, include_lags=False, forecast_origin=None):
    time = pd.DatetimeIndex(df['datetime']) if 'datetime' in df else df.index
    if not isinstance(time, pd.DatetimeIndex) or time.hasnans:
        raise ValueError('Valid datetime column or DatetimeIndex is required')
    wind = 'wind_speed' if 'wind_speed' in df else 'wind_speed_ms'
    temp = 'temperature' if 'temperature' in df else 'temperature_c'
    if wind not in df or temp not in df:
        raise ValueError('wind_speed and temperature are required')
    result = pd.DataFrame({'wind_speed': pd.to_numeric(df[wind], errors='coerce'),
                           'temperature': pd.to_numeric(df[temp], errors='coerce')}, index=df.index)
    result = result.replace([np.inf, -np.inf], np.nan)
    if calendar:
        result['hour'], result['month'], result['day_of_year'] = time.hour, time.month, time.dayofyear
        result['sin_hour'], result['cos_hour'] = np.sin(2*np.pi*time.hour/24), np.cos(2*np.pi*time.hour/24)
        result['sin_month'], result['cos_month'] = np.sin(2*np.pi*(time.month-1)/12), np.cos(2*np.pi*(time.month-1)/12)
    if include_lags:
        # One eligible weather vintage is known as a whole at origin, unlike future SCADA observations.
        origin = pd.Timestamp(forecast_origin) if forecast_origin is not None else pd.NaT
        available = pd.Timestamp(df.attrs.get('weather_available_at'))
        if (df.attrs.get('kind') != 'archived_forecast' or pd.isna(origin) or origin.tzinfo is None
            or pd.isna(available) or available.tzinfo is None or available > origin):
            raise ValueError('Lags require one archived weather vintage available by an explicit forecast_origin; observed future lags are forbidden')
        if time.tz is None or time.has_duplicates or not time.is_monotonic_increasing:
            raise ValueError('Lag features require one ordered timezone-aware vintage/turbine')
        if 'turbine_id' in df and df.turbine_id.nunique() != 1:
            raise ValueError('Build optional weather lags separately for each turbine')
        for col in ('weather_issued_at', 'weather_available_at'):
            if col in df:
                stamps = [pd.Timestamp(value) for value in df[col]]
                if any(pd.isna(x) or x.tzinfo is None or x > origin for x in stamps) or len(set(stamps)) != 1:
                    raise ValueError('Lag features cannot mix vintages or use weather unavailable at origin')
        series = pd.Series(result.wind_speed.to_numpy(), index=time)
        grid = pd.date_range(time.min(), time.max(), freq='h')
        if not time.isin(grid).all():
            raise ValueError('Lag features require an hourly grid')
        series = series.reindex(grid)
        for lag in (1, 3):
            result[f'wind_lag_{lag}h'] = series.shift(lag).reindex(time).to_numpy()
        result['wind_roll_mean_3h'] = series.shift(1).rolling(3, min_periods=3).mean().reindex(time).to_numpy()
        result['wind_roll_std_3h'] = series.shift(1).rolling(3, min_periods=3).std(ddof=0).reindex(time).to_numpy()
    result.attrs = dict(df.attrs)
    result.attrs['feature_columns'] = list(result.columns)
    return result
