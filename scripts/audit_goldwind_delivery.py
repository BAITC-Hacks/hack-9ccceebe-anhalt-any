"""Read-only audit of user-supplied hourly station exports; never infer semantics."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source, out = args.source, args.output
if out.resolve() == source.resolve() or source.resolve() in out.resolve().parents:
    parser.error('--output must be outside the supplied source directory')
report = {'files': {}, 'assumptions_confirmed': False}
for path in sorted(source.iterdir()):
    if path.is_file():
        report['files'][path.name] = {'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

frame = pd.read_csv(source/'historical_hourly.csv')
stamp = pd.to_datetime(frame.datetime, errors='raise')
frame['parsed_time'] = stamp
report['hourly'] = {'rows': len(frame), 'columns': list(frame.columns[:-1]),
    'timezone_in_file': str(stamp.dt.tz) if stamp.dt.tz is not None else None,
    'duplicate_turbine_time': int(frame.duplicated(['datetime','turbine_id']).sum()),
    'off_hour_rows': int((stamp != stamp.dt.floor('h')).sum()),
    'missing_by_column': frame.drop(columns='parsed_time').isna().sum().to_dict(),
    'sample_count_distribution': frame.sample_count.value_counts().sort_index().to_dict(),
    'per_turbine': {}, 'february_2026_rows': int(((stamp>='2026-02-01') & (stamp<'2026-03-01')).sum())}
for tid, group in frame.groupby('turbine_id'):
    grid = pd.date_range(group.parsed_time.min(),group.parsed_time.max(),freq='h')
    complete = group.sample_count.eq(6) & np.isfinite(group[['wind_speed','temp','power']]).all(axis=1)
    report['hourly']['per_turbine'][tid] = {'rows':len(group), 'start':str(group.parsed_time.min()),
        'end':str(group.parsed_time.max()), 'grid_hours':len(grid),
        'missing_grid_hours':len(grid.difference(group.parsed_time)),
        'complete_six_sample_hours':int(complete.sum()),
        'hours_with_finite_values':int(np.isfinite(group[['wind_speed','temp','power']]).all(axis=1).sum()),
        'sample_count_total':int(group.sample_count.sum()),
        'range':{c:{'min':float(group[c].min()), 'max':float(group[c].max())} for c in ['wind_speed','temp','power']}}
report['hourly']['complete_six_sample_hours'] = int(frame.sample_count.eq(6).sum())
report['hourly']['nonempty_incomplete_hours'] = int(frame.sample_count.between(1,5).sum())

features = pd.read_csv(source/'historical_hourly_features.csv')
keys=['datetime','turbine_id']
merged = features.merge(frame.drop(columns='parsed_time'),on=keys,how='outer',suffixes=('_features','_base'),indicator=True,validate='one_to_one')
checks = {c:bool(np.allclose(merged[c+'_features'],merged[c+'_base'],equal_nan=True)) for c in ['wind_speed','temp','power','sample_count']}
features=features.sort_values(['turbine_id','datetime'])
report['features'] = {'rows':len(features), 'columns':list(features.columns),
    'join_counts':merged['_merge'].value_counts().to_dict(),'base_columns_equal':checks,
    'observed_power_derived_columns':[c for c in features if c.startswith('power_lag_') or c.startswith('power_roll_')],
    'observed_wind_derived_columns':[c for c in features if c.startswith('wind_lag_') or c.startswith('wind_roll_')],
    'rolling_checks':{}}
for col in ['power_roll_6h','power_roll_24h','wind_roll_6h']:
    window = int(col.split('_')[-1][:-1])
    value_col = 'power' if col.startswith('power') else 'wind_speed'
    candidates={}
    for shift in [0,1]:
        calc=features.groupby('turbine_id')[value_col].transform(lambda s:s.shift(shift).rolling(window,min_periods=window).mean())
        mask=features[col].notna() & calc.notna()
        candidates[str(shift)]={'compared_rows':int(mask.sum()),'matching_values':int(np.isclose(features.loc[mask,col],calc[mask],rtol=1e-10,atol=1e-10).sum())}
    report['features']['rolling_checks'][col]=candidates

manifest=pd.read_csv(source/'weather_requests_manifest.csv')
queries=[parse_qs(urlsplit(url).query) for url in manifest.request_url]
weather_response_files = []
availability_columns = []
for path in sorted(source.glob('*.csv')):
    columns = set(pd.read_csv(path, nrows=0).columns)
    if 'weather_available_at' in columns:
        availability_columns.append(path.name)
    if columns.intersection({'wind_speed_80m','wind_speed_ms','temperature_2m'}) and columns.intersection({'valid_time','datetime','timestamp'}):
        weather_response_files.append(path.name)
report['weather_requests']={'rows':len(manifest), 'first_date':str(manifest.run_date.min()),
    'last_date':str(manifest.run_date.max()), 'turbines':sorted(manifest.turbine_id.unique().tolist()),
    'horizons':sorted(manifest.horizon_h.unique().tolist()),
    'endpoint_hosts':sorted({urlsplit(url).netloc for url in manifest.request_url}),
    'requests_with_explicit_model':sum('models' in q for q in queries),
    'columns':list(manifest.columns),
    'weather_response_csv_files':weather_response_files, 'response_payloads_present':bool(weather_response_files),
    'availability_column_csv_files':availability_columns, 'weather_available_at_present':bool(availability_columns)}
claimed=json.loads((source/'data_quality_report.json').read_text())
report['provided_quality_report_matches_hourly']={
    'row_count':claimed.get('hourly_rows')==len(frame),
    'missing_columns':claimed.get('hourly_missing_by_column')==report['hourly']['missing_by_column'],
    'counts_sum_matches_claimed_native_rows':int(frame.sample_count.sum())==claimed['combined']['rows'],
    'native_rows_can_be_independently_audited':False}
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':str(out),'rows':len(frame),'complete_six_sample_hours':report['hourly']['complete_six_sample_hours'],'weather_request_rows':len(manifest),'semantics_confirmed':False}))
