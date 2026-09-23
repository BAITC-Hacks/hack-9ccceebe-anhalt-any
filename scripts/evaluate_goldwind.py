"""Reload Goldwind artifact and reproduce frozen source-hashed MODE A holdout."""
import argparse
import json
from pathlib import Path

from src.ml.goldwind_data import GoldwindContract, read_organizer_files, prepare_hourly
from src.ml.goldwind_model import evaluate_model
from src.ml.goldwind_provider import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/goldwind/power_model.joblib"))
    args = parser.parse_args()
    try:
        model = load_model(args.model)
        contract = GoldwindContract(**json.loads(args.contract.read_text(encoding="utf-8-sig")))
        raw, files = read_organizer_files(contract, args.contract.resolve().parent)
        _, hourly, _ = prepare_hourly(raw, contract)
        report, _ = evaluate_model(model, hourly, contract, files)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
