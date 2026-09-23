"""Official target-station entry point; missing real inputs are explicit blockers."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.schemas import ForecastError
from src.forecast.contracts import ForecastSettings
from src.forecast.orchestrator import readiness, run_target_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', help='Timezone-aware ISO forecast origin, e.g. 2026-01-31T23:00:00Z')
    parser.add_argument('--horizon', choices=[24, 48], type=int, default=48)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--with-agent', action='store_true')
    parser.add_argument('--check', action='store_true', help='Report configuration readiness without forecasting')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if not args.check and not args.origin:
        parser.error('--origin is required unless --check is used')
    try:
        result = readiness() if args.check else run_target_forecast(args.origin, args.horizon,
                                 refresh=args.refresh, with_agent=args.with_agent)
        encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded+'\n')
        print(encoded)
        return 0 if result.get('ready') or result.get('status') == 'ok' else 2
    except (ForecastError, OSError, ValueError) as exc:
        print(f'Target forecast blocked: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
