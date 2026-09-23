"""Real Kelmarsh MODE A at native cadence, no weather API and no retraining."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.observed import run_observed_analysis
from src.agent.schemas import ForecastError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2017-11-08")
    parser.add_argument("--turbine", default="Kelmarsh 1")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--with-agent", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = run_observed_analysis(args.turbine, args.date, args.horizon, with_agent=args.with_agent)
        encoded = result.model_dump_json(indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")
        print(encoded)
        return 0 if result.status == "ok" else 2
    except (ForecastError, OSError) as exc:
        print(f"Observed analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
