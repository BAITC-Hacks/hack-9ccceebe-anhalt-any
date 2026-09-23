#!/usr/bin/env python3
"""Explicit historical export; keeps raw units, gaps and quality flags."""
import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT
from src.data.loader import load_historical_data
from src.data.features import build_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Combined T1/T2 CSV; defaults to received organizer export')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--timezone', help='Only set after confirming source timezone; omitted preserves naive time')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        frame = load_historical_data(args.input, timezone=args.timezone)
        features = build_features(frame)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_dir / 'historical_clean.csv', index=False)
        # Keys are provided for alignment, but must not enter numeric ML input.
        export = features.copy()
        export.insert(0, 'turbine_id', frame.turbine_id)
        export.insert(0, 'datetime', frame.datetime)
        export.to_csv(args.output_dir / 'historical_features.csv', index=False)
        manifest = {**frame.attrs, 'feature_columns': list(features.columns),
                    'first_time': frame.datetime.min().isoformat(), 'last_time': frame.datetime.max().isoformat(),
                    'rows_per_turbine': {str(k): int(v) for k, v in frame.groupby('turbine_id').size().items()},
                    'unconfirmed': ['power normalization', 'interval labeling', 'source wind height',
                                    'T1/T2 source-file mapping', 'SCADA publication latency'] +
                                   ([] if args.timezone else ['source timezone']),
                    'notes': 'Preparation only; no training, interpolation, power scaling or forecast-quality claim.'}
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Data preparation failed: {exc}\n')
    print(f'Prepared {len(frame):,} rows; outputs: {args.output_dir.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
