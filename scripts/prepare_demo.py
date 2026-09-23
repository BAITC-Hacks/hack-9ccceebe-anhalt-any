"""Reproducible synthetic fixtures; no real weather or measured generation."""
import json
import os
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from src.config import ROOT


def main():
    rng = np.random.default_rng(2026)
    wind = rng.uniform(0, 30, 3000)
    temp = rng.uniform(-25, 30, 3000)
    target = 2500 * np.clip((wind**3 - 3**3) / (12**3 - 3**3), 0, 1)
    target[wind >= 25] = 0
    features = pd.DataFrame({"wind_speed_ms": wind, "temperature_c": temp})
    model = RandomForestRegressor(n_estimators=12, max_depth=8, random_state=2026, n_jobs=1)
    model.fit(features, target)
    model_dir = ROOT / "models/demo"
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=model_dir, delete=False) as tmp:
        temporary_path = Path(tmp.name)
    joblib.dump(model, temporary_path, compress=3)
    os.replace(temporary_path, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps({
        "id": "synthetic-random-forest-v1", "synthetic": True,
        "training": "3000 synthetic samples, idealized power curve, seed 2026",
        "features": ["wind_speed_ms", "temperature_c"], "output_unit": "kW",
        "rated_power_kw": 2500, "sklearn_version": sklearn.__version__,
        "limitations": "Not validated on real turbines; no calibrated uncertainty"
    }, indent=2) + "\n")
    index = pd.date_range("2026-01-31", "2026-03-03", inclusive="left", freq="h", tz="UTC")
    t = np.arange(len(index))
    wind = np.clip(8 + 4 * np.sin(t / 19) + 2 * np.sin(t / 4.5), 0, 30)
    wind[250:254] = [18, 4, 20, 5]  # Exercise deterministic jump flags in backtest.
    frame = pd.DataFrame({"timestamp": index, "wind_speed_ms": wind.round(4),
                          "temperature_c": (-4 + 7 * np.sin(t / 24)).round(4)})
    folder = ROOT / "data/demo"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "weather.csv", index=False)
    print("Prepared SYNTHETIC weather cache and synthetic-trained demo model; no actuals generated.")


if __name__ == "__main__":
    main()
