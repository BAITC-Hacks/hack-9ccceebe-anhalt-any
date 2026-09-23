from copy import deepcopy
import json

import pandas as pd
import pytest
import requests

from src.data.weather import AvailabilityPolicy, WeatherClient, build_features, get_archival_weather
from src.data import weather as weather_module
from scripts import cache_weather

RUN = '2026-01-31T00:00:00Z'


def payload(start=RUN, hours=48, height=80):
    return {'latitude': 43.62, 'longitude': 78.48, 'elevation': 555, 'utc_offset_seconds': 0,
            'hourly_units': {'time': 'iso8601', f'wind_speed_{height}m': 'm/s', 'temperature_2m': '°C', f'wind_direction_{height}m': '°'},
            'hourly': {'time': [x.strftime('%Y-%m-%dT%H:%M') for x in pd.date_range(start, periods=hours, freq='h')],
                       f'wind_speed_{height}m': [5.0]*hours, 'temperature_2m': [-3.0]*hours,
                       f'wind_direction_{height}m': [270.0]*hours}}


class Response:
    def __init__(self, data=None, status=200, headers=None):
        self.data, self.status_code, self.headers = data, status, headers or {}

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


class Session:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def client(tmp_path, *responses, **kwargs):
    return WeatherClient(tmp_path, session=Session(*responses), sleep=lambda _: None, **kwargs)


def policy(**updates):
    document = {'confirmed': True, 'model': 'ecmwf_ifs', 'basis': 'documented_delay',
                'covers_archived_revision': True, 'evidence': 'MOCK TEST EVIDENCE ONLY; not publisher evidence',
                'valid_from': '2026-01-31T00:00Z', 'valid_until': '2026-02-28T23:00Z', 'delay_hours': 6}
    return AvailabilityPolicy({**document, **updates})


def test_success_cache_no_second_http_and_stable_attrs(tmp_path):
    api = client(tmp_path, Response(payload()))
    first = api.fetch_run(43.64515, 78.535604, RUN)
    second = api.fetch_run(43.64515, 78.535604, RUN)
    assert len(first) == 48 and first.datetime.dt.tz is not None
    assert first.attrs['kind'] == 'archive_run_unverified'
    assert 'weather_available_at' not in first
    assert second.attrs['cache_hit'] and len(api.session.calls) == 1
    assert first.attrs['payload_sha256'] == second.attrs['payload_sha256']
    params = api.session.calls[0][1]['params']
    assert params['models'] == 'ecmwf_ifs' and params['forecast_hours'] == 48
    assert 'start_hour' not in params and 'wind_speed_10m' not in params['hourly']


def test_origin_selects_available_run_and_correct_window(tmp_path):
    origin = pd.Timestamp('2026-01-31T13:00Z')
    api = client(tmp_path, Response(payload('2026-01-31T06:00Z', 56)))
    frame = get_archival_weather(43.6, 78.5, origin, 48, policy=policy(), client=api)
    assert frame.datetime.iloc[0] == origin + pd.Timedelta(1, unit='h')
    assert frame.datetime.iloc[-1] == origin + pd.Timedelta(48, unit='h')
    assert frame.weather_issued_at.iloc[0] == pd.Timestamp('2026-01-31T06:00Z')
    assert frame.weather_available_at.iloc[0] <= origin
    assert api.session.calls[0][1]['params']['forecast_hours'] == 56
    assert frame.attrs['kind'] == 'archived_forecast'
    features = build_features(frame.rename(columns={'wind_speed': 'wind_speed_ms', 'temperature': 'temperature_c'}).set_index('datetime'))
    assert list(features) == ['wind_speed_ms', 'temperature_c'] and len(features) == 48


def test_unconfirmed_policy_and_naive_origin_fail_before_http(tmp_path):
    api = client(tmp_path)
    with pytest.raises(ValueError, match='Confirmed weather'):
        get_archival_weather(43, 78, RUN, 48, client=api)
    with pytest.raises(ValueError, match='timezone-aware'):
        get_archival_weather(43, 78, '2026-01-31', 48, client=api)
    assert not api.session.calls


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(hourly={}),
    lambda d: d.update(utc_offset_seconds=3600),
    lambda d: d['hourly_units'].update(wind_speed_80m='km/h'),
    lambda d: d['hourly']['time'].__setitem__(1, d['hourly']['time'][0]),
    lambda d: d['hourly']['time'].__setitem__(1, '2026-01-31T01:30'),
    lambda d: d['hourly']['wind_speed_80m'].pop(),
    lambda d: d['hourly']['wind_speed_80m'].__setitem__(0, 'invalid'),
    lambda d: d['hourly'].update(wind_speed_80m=[None]*48),
])
def test_invalid_payload_never_cached(tmp_path, mutate):
    data = payload()
    mutate(data)
    api = client(tmp_path, Response(data))
    with pytest.raises(ValueError):
        api.fetch_run(43, 78, RUN)
    assert not list(tmp_path.glob('*.json'))


def test_missing_hours_remain_nan(tmp_path):
    data = payload()
    for values in data['hourly'].values():
        values.pop(5)
    frame = client(tmp_path, Response(data)).fetch_run(43, 78, RUN)
    assert len(frame) == 48 and frame.missing_weather.sum() == 1
    assert pd.isna(frame.wind_speed.iloc[5])


def test_retry_success_and_bounded_failure(tmp_path):
    api = client(tmp_path, Response(status=429, headers={'Retry-After': '1'}), requests.Timeout(), Response(payload()))
    assert len(api.fetch_run(43, 78, RUN)) == 48 and len(api.session.calls) == 3
    api = client(tmp_path / 'fail', Response(status=500), requests.Timeout(), Response(status=503))
    with pytest.raises(ValueError, match='bounded retries'):
        api.fetch_run(43, 78, RUN)
    assert len(api.session.calls) == 3


@pytest.mark.parametrize('response', [Response(status=400), Response(ValueError('bad json')),
                                     Response(status=429, headers={'Retry-After': '90'}),
                                     Response(status=429, headers={'Retry-After': 'Tue, 01 Jan 2030 00:00:00 GMT'}),
                                     Response(status=429, headers={'Retry-After': 'bad'})])
def test_permanent_errors_do_not_retry(tmp_path, response):
    api = client(tmp_path, response)
    with pytest.raises(ValueError):
        api.fetch_run(43, 78, RUN)
    assert len(api.session.calls) == 1


def test_refresh_cannot_overwrite_known_revision_and_corruption_rejected(tmp_path):
    changed = payload()
    changed['hourly']['wind_speed_80m'][0] = 9
    api = client(tmp_path, Response(payload()), Response(changed))
    api.fetch_run(43, 78, RUN)
    with pytest.raises(ValueError, match='changed'):
        api.fetch_run(43, 78, RUN, refresh=True)
    path = next(tmp_path.glob('*.json'))
    path.write_text('{}')
    with pytest.raises(ValueError, match='corrupt'):
        api.fetch_run(43, 78, RUN)


def test_cache_identity_changes_by_location_height_horizon(tmp_path):
    api = client(tmp_path, Response(payload()), Response(payload()), Response(payload(hours=24)))
    api.fetch_run(43, 78, RUN)
    api.fetch_run(44, 78, RUN)
    api.fetch_run(43, 78, RUN, 24)
    api100 = client(tmp_path, Response(payload(height=100)), wind_height_m=100)
    api100.fetch_run(43, 78, RUN)
    assert len(list(tmp_path.glob('*.json'))) == 4


def test_publisher_revision_hash_and_request_are_checked(tmp_path):
    origin = pd.Timestamp('2026-01-31T06:00Z')
    api = client(tmp_path, Response(payload(hours=31)))
    inspection = api.fetch_run(43, 78, RUN, 24, start=origin + pd.Timedelta(1, unit='h'))
    record = {'issued_at': RUN, 'available_at': '2026-01-31T05:30Z', 'payload_sha256': inspection.attrs['payload_sha256'],
              'request': {'latitude': 43.0, 'longitude': 78.0, 'forecast_origin': origin.isoformat(),
                          'horizon_h': 24, 'first_lead_hour': 1, 'wind_height_m': 80}}
    proof = policy(basis='publisher_timestamp', runs=[record])
    frame = get_archival_weather(43, 78, origin, 24, policy=proof, client=api)
    assert frame.attrs['availability_basis'] == 'publisher_timestamp'
    bad = deepcopy(record)
    bad['payload_sha256'] = '0'*64
    with pytest.raises(ValueError, match='differs'):
        get_archival_weather(43, 78, origin, 24, policy=policy(basis='publisher_timestamp', runs=[bad]), client=api)
    with pytest.raises(ValueError, match='No documented'):
        get_archival_weather(44, 78, origin, 24, policy=proof, client=api)


def test_cli_all_fail_writes_manifest(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('mock unavailable')
    monkeypatch.setattr(WeatherClient, 'fetch_run', fail)
    output = tmp_path / 'not-yet-created'
    code = cache_weather.main(['--start', '2026-01-31', '--end', '2026-01-31', '--output-dir', str(output)])
    manifest = json.loads(next(output.glob('manifest*')).read_text())
    assert code == 1 and len(manifest['requests']) == 2
    assert all(row['status'] == 'error' for row in manifest['requests'])


def test_strict_adapter_matches_existing_weather_boundary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.forecast.weather import _select
    origin = pd.Timestamp('2026-01-31T13:00Z')
    api = client(tmp_path / 'cache', Response(payload('2026-01-31T06:00Z', 56)))
    monkeypatch.setattr(weather_module, 'WeatherClient', lambda: api)
    protocol_path, policy_path = tmp_path / 'protocol.json', tmp_path / 'policy.json'
    protocol_path.write_text(json.dumps({'confirmed': True, 'timezone': 'UTC', 'daily_origin_hour': 13,
                                         'first_lead_hour': 1, 'interval_label': 'start',
                                         'normalization': 'fraction_of_rated_power', 'evidence': 'MOCK TEST ONLY'}))
    policy_path.write_text(json.dumps(policy().doc))
    monkeypatch.setenv('FORECAST_PROTOCOL_PATH', str(protocol_path))
    monkeypatch.setenv('WEATHER_AVAILABILITY_PATH', str(policy_path))
    frame = weather_module.get_forecast_weather(43, 78, origin, 48)
    expected = pd.date_range('2026-01-31T14:00Z', periods=48, freq='h')
    selected = _select(frame, 'T1', SimpleNamespace(hub_height_m=80), origin, expected)
    features = build_features(selected)
    assert len(features) == 48 and features.index.equals(expected)
    assert features.attrs['turbine_id'] == 'T1' and features.attrs['wind_height_m'] == 80
    assert list(features) == ['wind_speed_ms', 'temperature_c']
