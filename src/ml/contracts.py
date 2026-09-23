"""Person B owns model loading, preprocessing, and conversion to kW."""
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class ModelProvider(Protocol):
    def load_model(self, path: Path) -> Any: ...
    def predict(self, model: Any, features: pd.DataFrame) -> np.ndarray: ...
