"""Open-Meteo Single Runs, with explicit issue/availability separation.

fetch_weather_run is an archive inspection tool. get_archival_weather is the
point-in-time interface and requires documented availability of the revision.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import time
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests

from src.config import ROOT
from .cache import WeatherCache
from .features import build_features as calendar_features

log = logging.getLogger(__name__)
ENDPOINT = 'https://single-runs-api.open-meteo.com/v1/forecast'
MODEL = 'ecmwf_ifs'


def _utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None or stamp != stamp.floor('h'):
        raise ValueError('Explicit timezone-aware, hourly timestamp required')
    return stamp.tz_convert('UTC')


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class AvailabilityPolicy:
    """Explicit external evidence; never infer historical availability from run time.

    A documented delay must cover the ACTUAL returned archive revision, not just
    initial NWP publication. Per-run publisher records additionally pin its hash.
    """
    def __init__(self, document):
        self.doc = document
        if (document.get('confirmed') is not True or document.get('model') != MODEL
            or document.get('basis') not in ('publisher_timestamp', 'documented_delay')
            or not isinstance(document.get('evidence'), str) or not document['evidence'].strip()
            or document.get('covers_archived_revision') is not True):
            raise ValueError('Confirmed weather availability evidence for the actual archived revision is required; use fetch_weather_run for inspection')
        self.start, self.end = _utc(document['valid_from']), _utc(document['valid_until'])
        if self.end < self.start:
            raise ValueError('Invalid availability policy validity interval')
        if document['basis'] == 'documented_delay':
            delay = document.get('delay_hours')
            if type(delay) is not int or not 1 <= delay <= 24:
                raise ValueError('Documented delay_hours must be an integer from 1 to 24')

    @classmethod
    def load(cls, path=None):
        return cls(json.loads(_path(path or os.getenv('WEATHER_AVAILABILITY_PATH') or
                                    'config/weather_availability.template.json').read_text()))

    def select(self, origin, request=None):
        if not self.start <= origin <= self.end:
            raise ValueError('Forecast origin is outside availability evidence coverage')
        if self.doc['basis'] == 'documented_delay':
            delay = pd.Timedelta(self.doc['delay_hours'], unit='h')
            issue = (origin - delay).floor('6h')
            return issue, issue + delay, None
        eligible = []
        for row in self.doc.get('runs', []):
            if row.get('request') != request:
                continue
            issue, available = _utc(row['issued_at']), pd.Timestamp(row['available_at'])
            if pd.isna(available) or available.tzinfo is None or available < issue:
                raise ValueError('Invalid publisher availability timestamp')
            checksum = row.get('payload_sha256', '')
            if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
                raise ValueError('Publisher records must pin the returned request payload_sha256')
            if available <= origin:
                eligible.append((issue, available.tz_convert('UTC'), checksum))
        if not eligible:
            raise ValueError('No documented weather run available by forecast_origin')
        return max(eligible)


class WeatherClient:
    def __init__(self, cache_dir=None, *, wind_height_m=None, attempts=3,
                 timeout_s=30, min_interval_s=1, session=None, sleep=time.sleep, clock=time.monotonic):
        height = wind_height_m if wind_height_m is not None else int(os.getenv('WEATHER_WIND_HEIGHT_M') or 80)
        if type(height) is not int or height not in (80, 100):
            raise ValueError('Wind height must be explicitly 80 or 100 m; no 10 m fallback')
        if type(attempts) is not int or not 1 <= attempts <= 5:
            raise ValueError('attempts must be 1..5')
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 120 or not math.isfinite(min_interval_s) or min_interval_s < 0:
            raise ValueError('Finite positive timeout <=120s and nonnegative request interval required')
        self.height, self.attempts, self.timeout, self.interval = height, attempts, timeout_s, min_interval_s
        self.cache = WeatherCache(_path(cache_dir or os.getenv('ARCHIVAL_WEATHER_CACHE_DIR') or 'data/weather_cache'))
        self.session = session or requests.Session()
        self.sleep, self.clock, self.last_request = sleep, clock, None

    def _request(self, params):
        for attempt in range(self.attempts):
            if self.last_request is not None:
                self.sleep(max(0, self.interval - (self.clock() - self.last_request)))
            self.last_request = self.clock()
            retry_after = 0
            try:
                response = self.session.get(ENDPOINT, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError):
                reason = 'connection/timeout'
            else:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ValueError('Weather API returned invalid JSON') from exc
                    if not isinstance(payload, dict) or payload.get('error'):
                        raise ValueError('Weather API returned an invalid/error payload')
                    return payload
                if response.status_code != 429 and not 500 <= response.status_code <= 599:
                    raise ValueError(f'Weather API HTTP {response.status_code}; request was not cached')
                reason = f'HTTP {response.status_code}'
                try:
                    retry_after = float(response.headers.get('Retry-After', 0))
                except (ValueError, TypeError):
                    try:
                        retry_at = pd.Timestamp(parsedate_to_datetime(response.headers['Retry-After']))
                        retry_after = (retry_at - pd.Timestamp.now(tz='UTC')).total_seconds()
                    except (ValueError, TypeError, KeyError, OverflowError) as exc:
                        raise ValueError('Unrecognized Retry-After; stop rather than retry before the rate limit resets') from exc
                if not math.isfinite(retry_after) or retry_after > 30:
                    raise ValueError('Weather rate limit requires a later retry; stopping bounded request')
            log.warning('Weather attempt %s/%s: %s', attempt + 1, self.attempts, reason)
            if attempt + 1 < self.attempts:
                self.sleep(max(2**attempt, retry_after))
        raise ValueError('Weather API unavailable after bounded retries; no cached fallback used')

    def _frame(self, payload, identity):
        wind, direction = f'wind_speed_{self.height}m', f'wind_direction_{self.height}m'
        hourly, units = payload.get('hourly'), payload.get('hourly_units', {})
        if not isinstance(hourly, dict) or not isinstance(units, dict) or payload.get('utc_offset_seconds') != 0:
            raise ValueError('Weather JSON requires hourly data and explicit UTC offset 0')
        if units.get(wind) != 'm/s' or units.get('temperature_2m') != '°C' or units.get(direction) != '°':
            raise ValueError('Weather units/height differ from requested m/s, Celsius and degrees')
        names = ['time', wind, 'temperature_2m', direction]
        if any(not isinstance(hourly.get(k), list) for k in names):
            raise ValueError('Weather JSON missing hourly arrays')
        if not hourly['time'] or any(len(hourly[k]) != len(hourly['time']) for k in names):
            raise ValueError('Empty or misaligned weather arrays')
        try:
            # API timezone=UTC and utc_offset_seconds=0 explicitly define naive JSON times.
            index = pd.DatetimeIndex(pd.to_datetime(hourly['time'], utc=True, errors='raise'))
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid weather valid_time') from exc
        start = pd.Timestamp(identity['start'])
        end = start + pd.Timedelta(identity['horizon_h'] - 1, unit='h')
        # Single Runs rejects start_hour/end_hour. forecast_hours starts at run
        # initialization; validate that full prefix, then select the requested window.
        response_grid = pd.date_range(identity['issued_at'], end, freq='h')
        expected = pd.date_range(start, end, freq='h')
        if index.hasnans or index.has_duplicates or not index.isin(response_grid).all():
            raise ValueError('Weather contains duplicate, off-grid or out-of-request timestamps')
        frame = pd.DataFrame({'wind_speed': hourly[wind], 'temperature': hourly['temperature_2m'],
                              'wind_direction': hourly[direction]}, index=index)
        for column in frame:
            converted = pd.to_numeric(frame[column], errors='coerce')
            if (frame[column].notna() & (converted.isna() | ~np.isfinite(converted))).any():
                raise ValueError('Weather contains nonnumeric or nonfinite measurements')
            frame[column] = converted.astype(float)
        if frame.wind_speed.lt(0).any() or (frame.wind_direction.notna() & ~frame.wind_direction.between(0, 360)).any():
            raise ValueError('Weather contains invalid wind speed/direction')
        frame = frame.reindex(expected)
        if frame[['wind_speed', 'temperature']].notna().all(axis=1).sum() == 0:
            raise ValueError('Weather contains no usable wind/temperature pair in the requested window')
        frame['missing_weather'] = frame[['wind_speed', 'temperature']].isna().any(axis=1)
        if frame.missing_weather.any():
            log.warning('Weather retains %s missing hours; no interpolation', int(frame.missing_weather.sum()))
        frame.index.name = 'datetime'
        return frame.reset_index()

    def fetch_run(self, lat, lon, issued_at, horizon_h=48, *, start=None, refresh=False):
        if (isinstance(lat, bool) or isinstance(lon, bool) or not math.isfinite(lat) or not math.isfinite(lon)
            or not -90 <= lat <= 90 or not -180 <= lon <= 180):
            raise ValueError('Invalid weather coordinates')
        issue = _utc(issued_at)
        if issue.hour % 6:
            raise ValueError('ECMWF run must be on a 00/06/12/18 UTC cycle')
        if type(horizon_h) is not int or not 1 <= horizon_h <= 90:
            raise ValueError('horizon_h must be 1..90 hours')
        start = issue if start is None else _utc(start)
        end = start + pd.Timedelta(horizon_h - 1, unit='h')
        if start < issue or end > issue + pd.Timedelta(90, unit='h'):
            raise ValueError('Request must stay within the native hourly ECMWF run window (0..90h)')
        identity = {'schema': 1, 'endpoint': ENDPOINT, 'latitude': float(lat), 'longitude': float(lon),
                    'issued_at': issue.isoformat(), 'start': start.isoformat(), 'horizon_h': horizon_h,
                    'model': MODEL, 'wind_height_m': self.height, 'wind_unit': 'm/s', 'temperature_unit': 'C'}
        stored = self.cache.load(identity)
        hit = stored is not None and not refresh
        if not hit:
            params = {'latitude': lat, 'longitude': lon, 'models': MODEL,
                      'run': issue.strftime('%Y-%m-%dT%H:%M'),
                      'forecast_hours': int((end - issue) / pd.Timedelta(1, unit='h')) + 1,
                      'hourly': f'wind_speed_{self.height}m,temperature_2m,wind_direction_{self.height}m',
                      'wind_speed_unit': 'ms', 'temperature_unit': 'celsius', 'timezone': 'UTC'}
            payload = self._request(params)
            self._frame(payload, identity)  # Validate before writing any successful cache entry.
            stored = self.cache.save(identity, payload, pd.Timestamp.now(tz='UTC').isoformat())
        frame = self._frame(stored['payload'], identity)
        frame['weather_issued_at'] = issue
        frame.attrs = {'kind': 'archive_run_unverified', 'source': ENDPOINT, 'model': MODEL,
                       'wind_height_m': self.height, 'weather_issued_at': issue.isoformat(),
                       'grid_latitude': stored['payload'].get('latitude'),
                       'grid_longitude': stored['payload'].get('longitude'),
                       'wind_height_source': 'Open-Meteo output variable; provider height processing, not SCADA measurement',
                       'payload_sha256': stored['payload_sha256'], 'retrieved_at': stored['retrieved_at'],
                       'cache_hit': hit, 'cache_path': str(self.cache.path(identity))}
        log.info('Weather %s: run=%s, hours=%s, height=%sm', 'cache hit' if hit else 'downloaded', issue, horizon_h, self.height)
        return frame


def fetch_weather_run(lat, lon, issued_at, horizon_h=48, *, client=None, **kwargs):
    """Download a named run for inspection; does NOT claim historical availability."""
    return (client or WeatherClient()).fetch_run(lat, lon, issued_at, horizon_h, **kwargs)


def get_archival_weather(lat: float, lon: float, run_date, horizon_h: int, *,
                         policy=None, client=None, first_lead_hour=1, refresh=False) -> pd.DataFrame:
    """run_date is forecast_origin, not NWP initialization; returns exactly 24/48 hours."""
    origin = _utc(run_date)
    if type(horizon_h) is not int or horizon_h not in (24, 48) or type(first_lead_hour) is not int or first_lead_hour not in (0, 1):
        raise ValueError('Forecast requires 24/48 hours and first_lead_hour 0/1')
    policy = policy or AvailabilityPolicy.load()
    client = client or WeatherClient()
    request = {'latitude': float(lat), 'longitude': float(lon), 'forecast_origin': origin.isoformat(),
               'horizon_h': horizon_h, 'first_lead_hour': first_lead_hour, 'wind_height_m': client.height}
    issue, available, checksum = policy.select(origin, request)
    frame = fetch_weather_run(lat, lon, issue, horizon_h, start=origin + pd.Timedelta(first_lead_hour, unit='h'),
                              client=client, refresh=refresh)
    if checksum is not None and frame.attrs['payload_sha256'] != checksum:
        raise ValueError('Archived payload differs from the documented publisher revision')
    frame['weather_available_at'] = available
    frame.attrs.update(kind='archived_forecast', forecast_origin=origin.isoformat(),
                       weather_available_at=available.isoformat(), availability_basis=policy.doc['basis'],
                       availability_evidence=policy.doc['evidence'])
    return frame


def get_forecast_weather(lat, lon, forecast_origin, horizon_h):
    """Existing strict orchestrator adapter; keeps the actual requested wind height."""
    from src.forecast.contracts import ForecastSettings, load_protocol
    protocol = load_protocol(ForecastSettings.from_env())
    frame = get_archival_weather(lat, lon, forecast_origin, horizon_h, first_lead_hour=protocol.first_lead_hour)
    return frame.rename(columns={'wind_speed': 'wind_speed_ms', 'temperature': 'temperature_c',
                                  'wind_direction': 'wind_direction_deg'}).set_index('datetime')


def build_features(frame):
    """Two-column target adapter; the future T1/T2 model must declare this schema.

    Not compatible with the existing Kelmarsh-only artifact. Full calendar
    features are separately available from src.data.features.build_features.
    """
    result = calendar_features(frame, calendar=False).rename(columns={'wind_speed': 'wind_speed_ms',
                                                                     'temperature': 'temperature_c'})
    result.attrs['feature_columns'] = list(result.columns)
    return result
