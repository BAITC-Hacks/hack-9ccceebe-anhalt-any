"""Synthetic-trained demo model. Never selected implicitly in production."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.agent.schemas import ForecastError


def load_model(path: Path):
    if not path.is_file():
        raise ForecastError("Demo model missing; run python scripts/prepare_demo.py")
    # Only load the bundled trusted artifact; never accept uploads or remote pickle files.
    return joblib.load(path)


def predict(model, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)
