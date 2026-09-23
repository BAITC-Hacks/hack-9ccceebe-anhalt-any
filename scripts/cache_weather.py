#!/usr/bin/env python3
"""Cache explicit daily archive runs or evidence-backed forecast-origin windows."""
import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import ROOT
from src.data.weather import AvailabilityPolicy, WeatherClient, get_archival_weather

TURBINES = {'T1': (43.645150, 78.535604), 'T2': (43.643198, 78.538828)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2026-01-31')
    parser.add_argument('--end', default='2026-02-28')
    parser.add_argument('--mode', choices=['runs', 'origins'], default='runs',
                        help='runs inspects named UTC initializations; origins requires availability evidence')
    parser.add_argument('--run-hour', type=int, choices=[0, 6, 12, 18], default=0)
    parser.add_argument('--timezone', help='Required for origins, after organizer confirmation')
    parser.add_argument('--origin-hour', type=int, choices=range(24))
    parser.add_argument('--first-lead-hour', type=int, choices=[0, 1], default=1)
    parser.add_argument('--horizon', type=int, choices=[24, 48], default=48)
    parser.add_argument('--height', type=int, choices=[80, 100], default=80)
    parser.add_argument('--turbine', action='append', choices=list(TURBINES), help='Repeat or omit for both')
    parser.add_argument('--availability-policy', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/weather_cache')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    policy = None
    try:
        dates = pd.date_range(args.start, args.end, freq='D')
        if dates.empty or len(dates) > 366 or dates.tz is not None or any(dates != dates.normalize()):
            raise ValueError('Provide 1..366 calendar dates YYYY-MM-DD')
        if args.mode == 'origins':
            if args.timezone is None or args.origin_hour is None:
                raise ValueError('origins requires --timezone and --origin-hour')
            policy = AvailabilityPolicy.load(args.availability_policy)
        client = WeatherClient(args.output_dir, wind_height_m=args.height)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Weather configuration failed: {exc}\n')
    turbines = list(dict.fromkeys(args.turbine or TURBINES))
    total, records = len(dates) * len(turbines), []
    # Even if every API request fails, the failure manifest can always be written.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for day in dates:
        for turbine in turbines:
            record = {'day': str(day.date()), 'turbine_id': turbine, 'mode': args.mode,
                      'horizon_h': args.horizon, 'wind_height_m': args.height}
            try:
                lat, lon = TURBINES[turbine]
                if args.mode == 'runs':
                    stamp = day.replace(hour=args.run_hour).tz_localize('UTC')
                    frame = client.fetch_run(lat, lon, stamp, args.horizon, refresh=args.refresh)
                else:
                    stamp = day.replace(hour=args.origin_hour).tz_localize(args.timezone, ambiguous='raise', nonexistent='raise')
                    frame = get_archival_weather(lat, lon, stamp, args.horizon, policy=policy, client=client,
                                                 first_lead_hour=args.first_lead_hour, refresh=args.refresh)
                record.update(status='ok', rows=len(frame), missing_hours=int(frame.missing_weather.sum()), **frame.attrs)
            except Exception as exc:
                # HTTP library details can contain request URLs. Only our validation messages are exposed.
                record.update(status='error', error=str(exc) if isinstance(exc, (ValueError, KeyError)) else type(exc).__name__)
            records.append(record)
            print(f'[{len(records)}/{total}] {day.date()} {turbine}: {record["status"]}', flush=True)
    name = f'manifest_{args.mode}_{dates[0].date()}_{dates[-1].date()}_{args.height}m.json'
    manifest = args.output_dir / name
    manifest.write_text(json.dumps({'mode': args.mode, 'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
                                    'note': 'runs mode does not certify historical availability', 'requests': records},
                                   indent=2, ensure_ascii=False) + '\n')
    errors = sum(row['status'] != 'ok' for row in records)
    print(f'{len(records)-errors}/{total} successful; manifest: {manifest.resolve()}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
