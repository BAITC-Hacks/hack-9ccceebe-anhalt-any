"""Daily target-VES replay, retaining each origin and all forecast vintages."""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.schemas import ForecastError
from src.config import ROOT
from src.forecast.contracts import ForecastSettings
from src.forecast.replay import run_february_replay


def main():
    parser = argparse.ArgumentParser(description="Replay Jan31–Feb28 daily origins; evaluate only February 2026")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 31), help="First local calendar origin date")
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 28), help="Last local calendar origin date, inclusive")
    parser.add_argument("--horizon", type=int, choices=(24, 48), default=48)
    parser.add_argument("--actuals", type=Path, help="CSV: aware hourly interval-start timestamp, turbine_id, actual_power_kw (raw kW)")
    parser.add_argument("--refresh", action="store_true", help="Refresh eligible weather inputs without admitting later weather vintages")
    parser.add_argument("--with-agent", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/forecast_v2/february")
    args = parser.parse_args()
    try:
        result = run_february_replay(args.output_dir, settings=ForecastSettings.from_env(), horizon_h=args.horizon,
            start=args.start, end=args.end, actuals_path=args.actuals, refresh=args.refresh, with_agent=args.with_agent)
        print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "failed_runs": len(result["failures"]),
                          "coverage": result["coverage"], "manifest": str(args.output_dir / "manifest.json")}, indent=2))
        return 0 if result["status"] == "ok" else 2
    except (ForecastError, OSError, ValueError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
