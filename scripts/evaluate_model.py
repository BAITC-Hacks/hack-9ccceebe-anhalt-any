import argparse
import json
from pathlib import Path
from src.ml import load_model
from src.ml.evaluation import evaluate_model
from scripts.train_model import read_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate a chronological holdout; A and B reported separately")
    parser.add_argument("data")
    parser.add_argument("--model", default="models/power_model.joblib")
    parser.add_argument("--mode", choices=["A", "B"], required=True)
    parser.add_argument("--forecast-provenance")
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error(f"Trained model not found: {Path(args.model).resolve()}. "
                     "First successfully train on real power observations using scripts.train_model, "
                     "or pass --model with an existing trusted artifact. Running tests does not create "
                     "models/power_model.joblib. Paths are resolved from the current directory.")
    if not Path(args.data).is_file():
        guidance = ("MODE B requires a real archived forecast dataset with issued_at and observed power. "
                    "Kelmarsh SCADA contains observed weather, not archived forecasts. "
                    if args.mode == "B" else "Supply real observations after the model's validation period. ")
        parser.error(f"Holdout dataset not found: {Path(args.data).resolve()}. " + guidance +
                     "For the downloaded Kelmarsh data, run from the project root: "
                     "python -m scripts.evaluate_model data/holdout.csv --mode A")
    result = evaluate_model(load_model(args.model), read_data(args.data), args.mode, args.forecast_provenance)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
