import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from src.ml import MLConfig, train_model, save_model


def read_data(path):
    return pd.read_parquet(path) if Path(path).suffix.lower() == ".parquet" else pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Train on real historical power targets only")
    parser.add_argument("data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="models/power_model.joblib")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f"Config not found: {config_path.resolve()}. Copy examples/config.json to this path, "
                     "then set target_source, power_unit, timezone and column mapping for your real dataset.")
    if not Path(args.data).is_file():
        parser.error(f"Dataset not found: {Path(args.data).resolve()}. Supply a real SCADA CSV/Parquet file. "
                     "data/scada.csv is an example path; no dataset is bundled.")
    try:
        settings = json.loads(config_path.read_text(encoding="utf-8-sig"))
        config = MLConfig(**settings)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"Invalid config {config_path}: {exc}")
    if not config.target_source:
        parser.error("Set target_source in the config to identify the real SCADA/dataset observations.")
    model = train_model(read_data(args.data), config)
    save_model(model, args.output)
    print(json.dumps(model.metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
