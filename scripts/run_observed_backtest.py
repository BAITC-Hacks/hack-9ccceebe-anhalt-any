"""Daily Kelmarsh MODE A evaluation on native observations, without retraining."""
import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.agent.observed import run_observed_analysis
from src.agent.schemas import ForecastError
from src.config import ROOT


def run(start, end, turbine, output_dir, with_agent=False):
    if start > end or start < date(2017, 11, 8) or end > date(2017, 12, 31):
        raise ForecastError('Use full holdout days 2017-11-08 through 2017-12-31')
    rows, days, failures = [], [], []
    for day in pd.date_range(start, end, freq='D'):
        try:
            result = run_observed_analysis(turbine, day.date(), with_agent=with_agent)
            for sample in result.samples:
                rows.append(dict(sample.model_dump(mode='json'), turbine_id=turbine, run_date=str(day.date())))
            day_result = {'date': str(day.date()), 'status': result.status,
                          'statistics': result.statistics.model_dump(), 'metrics': result.metrics.model_dump(),
                          'anomalies': [flag.model_dump() for flag in result.anomalies]}
            if with_agent:
                day_result.update(analysis=result.analysis.model_dump(), analysis_source=result.analysis_source,
                                  analysis_reason=result.analysis_reason)
            days.append(day_result)
        except ForecastError as exc:
            failures.append({'date': str(day.date()), 'error': str(exc)})
    frame = pd.DataFrame(rows, columns=['timestamp', 'wind_speed_ms', 'power_kw', 'actual_power_kw',
                                       'valid', 'turbine_id', 'run_date'])
    prediction = pd.to_numeric(frame.power_kw, errors='coerce')
    actual = pd.to_numeric(frame.actual_power_kw, errors='coerce')
    paired = np.isfinite(prediction) & np.isfinite(actual)
    error = prediction[paired] - actual[paired]
    expected = ((end-start).days+1)*144
    summary = {'mode': 'A', 'weather_kind': 'observed_scada', 'turbine_id': turbine,
        'start': str(start), 'end': str(end), 'sampling_interval_minutes': 10,
        'days_completed': len(days), 'days_with_flags': sum(bool(d['anomalies']) for d in days),
        'expected_samples': expected, 'matched_samples': int(paired.sum()),
        'coverage': float(paired.sum()/expected), 'failed_days': failures,
        'raw_mae_kw': float(error.abs().mean()) if len(error) else None,
        'raw_rmse_kw': float(np.sqrt((error**2).mean())) if len(error) else None,
        'raw_bias_kw': float(error.mean()) if len(error) else None,
        'note': 'Observed native SCADA, not operational forecast. Finite negative/above-rated raw values retained in metrics. '
                'Missing observations are not zero-filled. Selected date range differs from full published holdout.'}
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir/'results.csv', index=False)
    (output_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output_dir/'daily.json').write_text(json.dumps(days, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=date.fromisoformat, default=date(2017, 11, 8))
    parser.add_argument('--end', type=date.fromisoformat, default=date(2017, 12, 31))
    parser.add_argument('--turbine', default='Kelmarsh 1')
    parser.add_argument('--with-agent', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/observed-backtest')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
    try:
        summary = run(args.start, args.end, args.turbine, args.output_dir, args.with_agent)
        print(json.dumps(summary, indent=2))
        return 2 if summary['failed_days'] or summary['coverage'] < 1 else 0
    except (ForecastError, OSError) as exc:
        print(f'Observed backtest failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
