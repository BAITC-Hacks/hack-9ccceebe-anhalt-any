import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.backtest import run_backtest
from src.agent.schemas import ForecastError
from src.config import ROOT, Settings


def main():
    parser = argparse.ArgumentParser(description="Daily UTC backtest (Jan 31–Feb 28, 2026 by default)")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 31))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--turbine", default="T1")
    parser.add_argument("--demo", action="store_true", default=None)
    parser.add_argument("--with-agent", action="store_true")
    parser.add_argument("--actuals", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtest")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = run_backtest(args.start, args.end, args.turbine, args.output_dir,
                               Settings.from_env(args.demo), args.actuals, args.with_agent)
        print(json.dumps(summary, indent=2))
        return 2 if summary["failed_days"] or summary["valid_hours"] < summary["expected_hours"] else 0
    except (ForecastError, OSError, ValueError) as exc:
        print(f"Backtest failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
