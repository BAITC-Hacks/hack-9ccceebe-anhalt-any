"""Optional MVP-boundary tests; run in an overlay of the unmodified integration branch.

Synthetic out-of-range predictions test plumbing/flags only, never forecast accuracy.
Standalone ML installs skip this module because the agent belongs to C.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

rules = pytest.importorskip("src.agent.rules", reason="Run in integration checkout with agent dependencies")
from src.ml import provider

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    features = pd.DataFrame({"wind_speed_ms": [7., 8., 9.], "temperature_c": [10., 10., 10.],
                             "wind_direction_deg": [359., 1., 90.]},
                            index=pd.date_range("2018-01-01", periods=3, freq="h", tz="UTC"))
    features.attrs = {"turbine_id": "Kelmarsh 1", "wind_height_m": 78.5,
                      "source": "test fixture, not real forecast", "kind": "reanalysis"}
    return features


def turbine():
    return SimpleNamespace(rated_power_kw=2050., hub_height_m=78.5, cut_in_ms=3., cut_out_ms=25.,
                           latitude=52.400604, longitude=-.947133, demo_only=False)


def test_raw_provider_values_reach_unchanged_rules(monkeypatch):
    model = provider.load_model(ROOT / "models/power_model.joblib")
    monkeypatch.setattr(model.pipeline, "predict", lambda frame: np.array([-.1, 1.2, np.nan]))
    features = inputs()
    raw_kw = provider.predict(model, features)
    flags, statistics, valid = rules.evaluate(features, raw_kw, turbine())
    assert {flag.code for flag in flags} >= {"negative_power", "above_rated_power", "non_finite_power"}
    np.testing.assert_allclose(raw_kw, [-205., 2460., np.nan], equal_nan=True)
    assert not valid.any()
    assert statistics.min_power_kw == -205.
    assert statistics.max_power_kw == 2460.
    assert statistics.predicted_energy_kwh is None


def test_orchestrator_preserves_raw_with_authoritative_context(monkeypatch):
    orchestrator = pytest.importorskip("src.agent.orchestrator")
    model = provider.load_model(ROOT / "models/power_model.joblib")
    monkeypatch.setattr(model.pipeline, "predict", lambda frame: np.array([-.1, 1.2, np.nan]))
    ml = SimpleNamespace(load_model=lambda path: model, predict=provider.predict)
    data = SimpleNamespace(build_features=lambda weather: weather.copy())
    features = inputs()
    settings = SimpleNamespace(demo=False, model_path=ROOT / "models/power_model.joblib",
                               ml_module="src.ml.provider", openai_model="",
                               turbines=lambda: {"Kelmarsh 1": turbine()})
    monkeypatch.setattr(orchestrator, "load_providers", lambda settings: (data, ml))
    monkeypatch.setattr(orchestrator, "load_weather", lambda *args: features.copy())
    result = orchestrator.run_forecast("Kelmarsh 1", "2018-01-01", 3, settings=settings, with_agent=False)
    assert [point.power_kw for point in result.hourly] == [-205., 2460., None]
    assert {flag.code for flag in result.anomalies} >= {"negative_power", "above_rated_power", "non_finite_power"}
    assert result.statistics.predicted_energy_kwh is None
    # C now supplies authoritative context even when the feature builder omits it.
    features.attrs.pop("turbine_id")
    result = orchestrator.run_forecast("Kelmarsh 1", "2018-01-01", 3, settings=settings, with_agent=False)
    assert [point.power_kw for point in result.hourly] == [-205., 2460., None]
    features.attrs["turbine_id"] = "T1"  # DATA cannot substitute a different turbine.
    result = orchestrator.run_forecast("Kelmarsh 1", "2018-01-01", 3, settings=settings, with_agent=False)
    assert result.hourly[0].power_kw == -205.
