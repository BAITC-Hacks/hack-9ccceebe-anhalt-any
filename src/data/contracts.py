"""Person A owns implementation; the agent only consumes this protocol."""
from datetime import date
from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    def get_archival_weather(self, lat: float, lon: float, run_date: date,
                             horizon_h: int) -> pd.DataFrame: ...
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame: ...
