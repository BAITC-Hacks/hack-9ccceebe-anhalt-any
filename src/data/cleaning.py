"""Auditable cleaning; missing measurements and unconfirmed units remain visible."""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
REQUIRED = ['datetime', 'turbine_id', 'wind_speed', 'temperature', 'power']


def clean_historical_data(frame, *, timezone=None, expected_samples_per_hour=6):
    missing = set(REQUIRED) - set(frame.columns)
    if missing or frame.empty or frame.columns.has_duplicates:
        raise ValueError(f'Nonempty data with unique required columns needed; missing={sorted(missing)}')
    out = frame.copy(deep=True)
    stamps = [pd.Timestamp(x) for x in out.datetime]
    if any(pd.isna(x) for x in stamps):
        raise ValueError('Invalid or missing datetime; source rows were not silently dropped')
    aware = [x.tzinfo is not None for x in stamps]
    if any(aware) and not all(aware):
        raise ValueError('Mixed naive and timezone-aware source timestamps')
    if all(aware):
        times = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))
        if timezone:
            times = times.tz_convert(timezone)
    else:
        times = pd.DatetimeIndex(stamps)
        if timezone:
            times = times.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
        else:
            log.warning('Source timezone unconfirmed: preserving naive timestamps, not assuming UTC')
    out['datetime'] = times
    if out.turbine_id.isna().any() or not out.turbine_id.isin(['T1', 'T2']).all():
        raise ValueError('Explicit T1/T2 identities are required')
    for col in ['wind_speed', 'temperature', 'power', *(['sample_count'] if 'sample_count' in out else [])]:
        original = out[col].copy()
        numeric = pd.to_numeric(original, errors='coerce')
        invalid = (original.notna() & numeric.isna()) | ~np.isfinite(numeric.fillna(0))
        invalid |= original.map(lambda value: isinstance(value, (bool, np.bool_)))
        if invalid.any():
            out[col + '_original'] = original.astype('string')
            log.warning('%s: %s invalid values retained in original column and flagged', col, int(invalid.sum()))
        out[col] = numeric.mask(invalid).astype(float)
        out['invalid_' + col] = invalid
    # Source row references do not make two otherwise identical records distinct.
    semantic = [c for c in out if c not in {'source_file', 'source_row'}]
    duplicate = out.duplicated(semantic)
    removed = int(duplicate.sum())
    if removed:
        log.warning('Removed %s exact duplicate measurements; count retained in quality_report', removed)
        out = out.loc[~duplicate].copy()
    if out.duplicated(['turbine_id', 'datetime']).any():
        raise ValueError('Conflicting duplicate turbine/datetime rows; review sources before choosing a value')
    out['missing_measurement'] = out[['wind_speed', 'temperature', 'power']].isna().any(axis=1)
    out['negative_wind'] = out.wind_speed.lt(0)
    out['suspect_temperature'] = out.temperature.notna() & ~out.temperature.between(-90, 65)
    out['negative_power'] = out.power.lt(0)  # Retain raw value, never clip or guess a rated scale.
    if 'sample_count' in out:
        if type(expected_samples_per_hour) is not int or expected_samples_per_hour < 1:
            raise ValueError('expected_samples_per_hour must be a positive integer')
        out['invalid_sample_count'] |= out.sample_count.lt(0) | out.sample_count.gt(expected_samples_per_hour) | out.sample_count.mod(1).ne(0)
        out['complete_samples'] = out.sample_count.eq(expected_samples_per_hour) & ~out.invalid_sample_count
    else:
        out['complete_samples'] = pd.Series(pd.NA, index=out.index, dtype='boolean')
    out = out.sort_values(['datetime', 'turbine_id']).reset_index(drop=True)
    flags = ['missing_measurement', 'negative_wind', 'suspect_temperature', 'negative_power']
    counts = {c: int(out[c].sum()) for c in flags}
    if any(counts.values()):
        log.warning('Historical quality flags (values retained): %s', counts)
    out.attrs = dict(frame.attrs)
    out.attrs.update(timezone=str(out.datetime.dt.tz) if out.datetime.dt.tz is not None else None,
                     power_unit='unconfirmed_source_units', kind='observed',
                     quality_report={'input_rows': len(frame), 'output_rows': len(out),
                                     'exact_duplicates_removed': removed, **counts})
    return out
