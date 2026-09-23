"""Train only after authoritative metadata and C's forecast origin are confirmed."""
import argparse
import json
from pathlib import Path

from src.ml.goldwind_data import GoldwindContract, read_organizer_files, prepare_hourly
from src.ml.goldwind_model import train_model, save_model, evaluate_model
from src.ml.goldwind_provider import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/goldwind"))
    args = parser.parse_args()
    try:
        contract = GoldwindContract(**json.loads(args.contract.read_text(encoding="utf-8-sig")))
        contract.validate(training=True)  # BEFORE reading/fitting or creating any output.
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output directory must be empty; do not overwrite earlier artifacts")
        raw, files = read_organizer_files(contract, args.contract.resolve().parent)
        audit, hourly, summary = prepare_hourly(raw, contract)
        model = train_model(hourly, contract, files)
        path = save_model(model, args.output)
        # Reload the actual saved artifact and reproduce MODE A test metrics without fit.
        report, predictions = evaluate_model(load_model(path), hourly, contract, files)
        audit.to_csv(args.output / "source_audit.csv", index=False)
        hourly.to_csv(args.output / "hourly_audit.csv", index=False)
        predictions.to_csv(args.output / "test_predictions.csv", index=False)
        for filename, content in [("preparation_report.json", summary), ("evaluation.json", report)]:
            (args.output / filename).write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"artifact": str(path), "mode": "A", "metrics": report["metrics"]}, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
