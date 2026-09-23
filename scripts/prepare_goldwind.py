"""Prepare explicitly documented organizer exports; no network or model training."""
import argparse
from dataclasses import fields
import json
import logging
from pathlib import Path

from src.ml.goldwind_data import GoldwindContract, prepare_hourly, read_organizer_files, training_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/goldwind/prepared"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = json.loads(args.contract.read_text(encoding="utf-8-sig"))
        allowed = {f.name for f in fields(GoldwindContract)}
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError("Unexpected organizer contract fields")
        contract = GoldwindContract(**settings)
        contract.validate()
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output directory is not empty; choose a new path to preserve prior audit")
        raw, provenance = read_organizer_files(contract, args.contract.resolve().parent)
        args.output.mkdir(parents=True, exist_ok=True)
        raw.to_csv(args.output / "source_rows.csv", index=False)  # Save before further validation, never destroy raw.
        try:
            audit, hourly, summary = prepare_hourly(raw, contract)
        except ValueError as exc:
            (args.output / "preparation_error.json").write_text(json.dumps({"status": "blocked", "reason": str(exc), "files": provenance}, indent=2), encoding="utf-8")
            raise
        audit.to_csv(args.output / "raw_audit.csv", index=False)
        hourly.to_csv(args.output / "hourly_audit.csv", index=False)
        summary["files"] = provenance
        summary["model_trained"] = False
        if contract.first_forecast_origin and contract.forecast_origin_source:
            selected = training_rows(hourly, contract)
            selected.to_csv(args.output / "eligible_hourly.csv", index=False)
            summary["eligible_hours"] = len(selected)
            summary["max_available_at"] = selected.available_at.max().isoformat()
        else:
            summary["training_blocked"] = "C must confirm exact first forecast_origin and its source"
        (args.output / "preparation_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({"status": "prepared_not_trained", "output": str(args.output),
                          "complete_hours": summary["complete_hours"], "model_trained": False}, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
