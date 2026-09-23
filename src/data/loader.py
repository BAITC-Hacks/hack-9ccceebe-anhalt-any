"""Load the existing combined export or explicitly mapped per-turbine CSVs."""
import hashlib
from pathlib import Path
from collections.abc import Mapping

import pandas as pd

from src.config import ROOT
from .cleaning import clean_historical_data

DEFAULT_HISTORY = ROOT / 'data/goldwind/incoming/received_20260923/historical_hourly.csv'
ALIASES = {'timestamp': 'datetime', 'temp': 'temperature',
           'wind_speed_ms': 'wind_speed', 'temperature_c': 'temperature'}


def load_historical_data(paths=None, *, columns=None, timezone=None, expected_samples_per_hour=6):
    """No resampling, interpolation or power conversion; audit details in DataFrame.attrs.

    paths: combined CSV path/list, or explicit {'T1': path, 'T2': path}.
    columns: source-name -> canonical-name mapping, e.g. {'Статистическое время':'datetime'}.
    """
    paths = DEFAULT_HISTORY if paths is None else paths
    items = list(paths.items()) if isinstance(paths, Mapping) else [(None, p) for p in
            ([paths] if isinstance(paths, (str, Path)) else paths)]
    if not items:
        raise ValueError('At least one historical input file is required')
    frames, provenance = [], []
    for tid, filename in items:
        path = Path(filename)
        if path.suffix.lower() != '.csv':
            raise ValueError('This loader accepts the supplied CSV format; export other formats explicitly')
        raw = pd.read_csv(path)
        mapping = {**ALIASES, **(columns or {})}
        raw = raw.rename(columns=mapping)
        if raw.columns.has_duplicates:
            raise ValueError('Column aliases collide; supply an unambiguous columns mapping')
        if tid is not None:
            if tid not in ('T1', 'T2') or ('turbine_id' in raw and not raw.turbine_id.eq(tid).all()):
                raise ValueError('Explicit file/turbine mapping conflicts with file contents')
            raw['turbine_id'] = tid
        if 'turbine_id' not in raw:
            raise ValueError('File has no turbine_id; provide an explicit file-to-turbine mapping')
        raw['source_file'] = path.name
        raw['source_row'] = range(2, len(raw) + 2)
        frames.append(raw)
        provenance.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(raw)})
    combined = pd.concat(frames, ignore_index=True)
    combined.attrs['sources'] = provenance
    return clean_historical_data(combined, timezone=timezone,
                                 expected_samples_per_hour=expected_samples_per_hour)
