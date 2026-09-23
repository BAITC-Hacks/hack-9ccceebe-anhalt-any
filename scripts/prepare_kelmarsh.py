"""Prepare the official 2017 Kelmarsh archive; no generated targets or weather download."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    raw = root / 'data/raw/kelmarsh'
    archive_path = raw / 'Kelmarsh_SCADA_2017_3083.zip'
    expected = 'c78263ee52ee0e48e2cb4bbaa1ba211a'
    if hashlib.md5(archive_path.read_bytes()).hexdigest() != expected:
        raise ValueError('Official archive checksum mismatch')
    static = pd.read_csv(raw / 'Kelmarsh_WT_static.csv')
    capacity = dict(zip(static.Title, static['Rated power (kW)']))
    frames, sources = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith('Turbine_Data_') or not name.endswith('.csv'):
                continue
            with archive.open(name) as stream:
                preamble = [stream.readline().decode('utf-8-sig').strip() for _ in range(9)]
                header = next(csv.reader([stream.readline().decode('utf-8-sig').strip()]))
            if '# Time zone: UTC' not in preamble:
                raise ValueError(f'Unconfirmed timezone in {name}')
            turbine = next(line.removeprefix('# Turbine: ') for line in preamble if line.startswith('# Turbine: '))
            mapping = {'# Date and time': 'timestamp', 'Wind speed (m/s)': 'wind_speed',
                       'Power (kW)': 'power'}
            # Unit symbols are corrupted in some source headers; retain exact source spelling in provenance.
            mapping[next(col for col in header if col.startswith('Wind direction ('))] = 'wind_direction'
            mapping[next(col for col in header if col.startswith('Nacelle ambient temperature ('))] = 'temperature'
            with archive.open(name) as stream:
                frame = pd.read_csv(stream, skiprows=9, usecols=list(mapping)).rename(columns=mapping)
            frame['turbine_id'] = turbine
            frame['rated_power'] = capacity[turbine]
            frames.append(frame)
            sources.append({'member': name, 'rows': len(frame), 'mapping': mapping, 'preamble': preamble})
            print(turbine, len(frame), 'rows read', flush=True)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows['timestamp'] = pd.to_datetime(all_rows.timestamp, utc=True, errors='raise')
    missing_before = all_rows.isna().sum().to_dict()
    reasons = pd.Series('', index=all_rows.index)
    for column in ['wind_speed', 'power']:
        invalid = ~np.isfinite(all_rows[column])
        reasons.loc[invalid] += f'{column}_missing_or_nonfinite;'
    reasons.loc[all_rows.wind_speed < 0] += 'negative_wind_speed;'
    excluded = all_rows.loc[reasons.ne('')].copy()
    excluded['exclusion_reason'] = reasons[reasons.ne('')]
    clean = all_rows.loc[reasons.eq('')].sort_values(['timestamp', 'turbine_id']).reset_index(drop=True)
    # Observed negative/above-rated power retained; no curtailment/outlier filtering or target filling.
    if clean.duplicated(['timestamp', 'turbine_id']).any():
        raise ValueError('Unexpected duplicate timestamp/turbine records')
    data_dir = root / 'data'
    clean.to_csv(data_dir / 'scada.csv', index=False)
    excluded.to_csv(data_dir / 'excluded_rows.csv', index=False)
    unique = clean.timestamp.unique()
    cutoff = unique[int(len(unique) * .85)]
    clean.loc[clean.timestamp >= cutoff].to_csv(data_dir / 'holdout.csv', index=False)
    summary = {
        'source': 'https://zenodo.org/records/16807551', 'doi': '10.5281/zenodo.16807551',
        'creators': ['Charlie Plumley', 'Roberta Takeuchi'], 'publisher': 'Cubico Sustainable Investments Ltd / Zenodo',
        'license': 'CC-BY-4.0', 'archive_md5': expected,
        'rows_raw': len(all_rows), 'rows_retained': len(clean), 'rows_excluded': len(excluded),
        'missing_before': missing_before, 'missing_after': clean.isna().sum().to_dict(),
        'negative_power_retained': int((clean.power < 0).sum()),
        'above_rated_power_retained': int((clean.power > clean.rated_power).sum()),
        'start': clean.timestamp.min().isoformat(), 'end': clean.timestamp.max().isoformat(),
        'turbines': clean.turbine_id.unique().tolist(), 'rated_power_kw': capacity,
        'interval_seconds_mode': clean.groupby('turbine_id').timestamp.diff().dt.total_seconds().mode().tolist(),
        'holdout_start': cutoff.isoformat(), 'columns': list(clean), 'files': sources,
        'unit_note': 'Measurement header explicitly specifies Power (kW). Separate signal mapping lists kWh for Power; this inconsistency is retained in raw metadata. We use actual measurement header units, not Energy Export or Potential power.',
        'processing': 'Select observed weather and actual Power only; label turbine from file metadata; UTC timestamps. Exclude missing/nonfinite power or wind and negative wind; save every excluded row. No interpolation, resampling, outlier/curtailment or negative-power removal. Holdout is a copy of chronological last 15%, not a second independent dataset.'}
    (data_dir / 'preparation_report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    config = {'columns': {}, 'timezone': 'UTC', 'wind_speed_unit': 'm/s', 'temperature_unit': 'C',
              'power_unit': 'kW', 'target_source': 'Kelmarsh SCADA 2017 actual Power (kW); doi:10.5281/zenodo.16807551',
              'weather_source': 'Kelmarsh measured nacelle wind and outdoor nacelle temperature (SCADA)',
              'rated_power': capacity, 'rated_power_verified': True,
              'rated_power_source': 'Kelmarsh_WT_static.csv, Zenodo record 16807551, Rated power (kW)',
              'mode': 'A', 'backend': 'auto', 'reports_dir': 'reports'}
    (root / 'config.kelmarsh.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    existing = root / 'config.json'
    if not existing.exists() or json.loads(existing.read_text(encoding='utf-8-sig')).get('target_source') is None:
        existing.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
