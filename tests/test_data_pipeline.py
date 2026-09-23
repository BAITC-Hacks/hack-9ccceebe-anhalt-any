import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_historical_data
from src.data.features import build_features
from src.data.loader import load_historical_data


def history():
    return pd.DataFrame({'datetime': ['2026-01-01 01:00', '2026-01-01 00:00'],
                         'turbine_id': ['T1', 'T1'], 'wind_speed': [4, 3],
                         'temperature': [-2, -1], 'power': [.2, .1], 'sample_count': [3, 6]})


def test_loader_aliases_preserves_units_and_unknown_time(tmp_path):
    source = tmp_path / 'source.csv'
    history().rename(columns={'temperature': 'temp'}).to_csv(source, index=False)
    result = load_historical_data(source)
    assert set(['datetime', 'turbine_id', 'wind_speed', 'temperature', 'power']) <= set(result)
    assert result.datetime.is_monotonic_increasing and not result.datetime.isna().any()
    assert result.datetime.dt.tz is None
    assert result.power.tolist() == [.1, .2]
    assert result.complete_samples.tolist() == [True, False]
    assert len(result.attrs['sources'][0]['sha256']) == 64


def test_explicit_file_mapping_and_timezone(tmp_path):
    files = {}
    for tid in ('T1', 'T2'):
        path = tmp_path / f'{tid}.csv'
        history().drop(columns='turbine_id').to_csv(path, index=False)
        files[tid] = path
    result = load_historical_data(files, timezone='Asia/Almaty')
    assert set(result.turbine_id) == {'T1', 'T2'}
    assert str(result.datetime.dt.tz) == 'Asia/Almaty'
    with pytest.raises(ValueError, match='explicit file-to-turbine'):
        load_historical_data(files['T1'])


@pytest.mark.parametrize('value', [None, 'not a date'])
def test_bad_timestamps_rejected(value):
    frame = history()
    frame.loc[0, 'datetime'] = value
    with pytest.raises(ValueError):
        clean_historical_data(frame)


def test_mixed_timestamps_and_conflicting_duplicates_rejected():
    frame = history()
    frame.loc[0, 'datetime'] = '2026-01-01T01:00:00Z'
    with pytest.raises(ValueError, match='Mixed'):
        clean_historical_data(frame)
    frame = history()
    frame.loc[0, 'datetime'] = frame.loc[1, 'datetime']
    with pytest.raises(ValueError, match='Conflicting'):
        clean_historical_data(frame)


def test_dedup_numeric_flags_no_silent_clipping():
    frame = history().astype({'wind_speed': object})
    frame.loc[0, 'wind_speed'] = 'broken'
    frame.loc[1, 'power'] = -.1
    frame.loc[1, 'temperature'] = 100
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = clean_historical_data(frame)
    assert len(result) == 2 and result.attrs['quality_report']['exact_duplicates_removed'] == 1
    assert result.wind_speed.isna().sum() == 1
    assert result.invalid_wind_speed.sum() == result.negative_power.sum() == result.suspect_temperature.sum() == 1
    assert result.power.min() == -.1
    assert result.wind_speed_original.iloc[1] == 'broken'


def test_calendar_features_target_never_enters():
    frame = clean_historical_data(history())
    frame['power_lag_1'] = 999
    result = build_features(frame)
    assert list(result) == ['wind_speed', 'temperature', 'hour', 'month', 'day_of_year',
                            'sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    assert all(pd.api.types.is_numeric_dtype(result[col]) for col in result)
    changed = frame.copy()
    changed['power'] = 100000
    pd.testing.assert_frame_equal(result, build_features(changed))
    with pytest.raises(ValueError, match='observed future lags'):
        build_features(frame, include_lags=True, forecast_origin='2026-01-01T00:00:00Z')


def test_boolean_measurement_is_not_numeric_power():
    frame = history().astype({'power': object})
    frame.loc[0, 'power'] = True
    result = clean_historical_data(frame)
    assert result.invalid_power.sum() == 1 and result.power.isna().sum() == 1


def test_weather_lags_use_clock_hours_and_one_eligible_vintage():
    frame = pd.DataFrame({'wind_speed': [10, 20, 40], 'temperature': [0, 0, 0]},
                         index=pd.to_datetime(['2026-01-01T01:00Z', '2026-01-01T02:00Z', '2026-01-01T04:00Z']))
    frame.attrs.update(kind='archived_forecast', weather_available_at='2026-01-01T00:00Z')
    result = build_features(frame, include_lags=True, forecast_origin='2026-01-01T00:00Z')
    assert result.wind_lag_1h.iloc[1] == 10 and np.isnan(result.wind_lag_1h.iloc[2])
    assert result.wind_lag_3h.iloc[2] == 10
    frame['weather_available_at'] = ['2026-01-01T00:00Z', '2026-01-01T00:00Z', '2026-01-01T01:00Z']
    with pytest.raises(ValueError, match='mix vintages'):
        build_features(frame, include_lags=True, forecast_origin='2026-01-01T00:00Z')
