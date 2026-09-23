import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.orchestrator import run_forecast
from src.agent.schemas import ForecastError
from src.config import Settings


def main():
    parser = argparse.ArgumentParser(description="AI Energy Agent — UTC hourly forecast")
    parser.add_argument("--date", required=True, help="Forecast date YYYY-MM-DD, midnight UTC")
    parser.add_argument("--turbine", default="T1")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--demo", action="store_true", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--with-agent", dest="agent", action="store_true")
    group.add_argument("--no-agent", dest="agent", action="store_false")
    parser.set_defaults(agent=None)
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.from_env(args.demo)
    try:
        result = run_forecast(args.turbine, args.date, args.horizon, settings=settings,
                              with_agent=(not settings.demo) if args.agent is None else args.agent)
        text = result.model_dump_json(indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        print(text)
        return 0 if result.status == "ok" else 2
    except (ForecastError, OSError) as exc:
        print(f"Forecast failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
