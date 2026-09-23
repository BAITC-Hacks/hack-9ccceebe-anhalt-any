"""Frozen published artifact tests: no training, tuning, or holdout mutation."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
from src.ml import model as core
from src.ml import provider

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model():
    return provider.load_model(ROOT / "models/power_model.joblib")


@pytest.fixture
def features():
    rows = pd.read_csv(ROOT / "data/holdout.csv", nrows=60)
    rows = rows[rows.turbine_id == "Kelmarsh 1"].iloc[[4, 0, 2]]
    frame = rows.set_index(pd.DatetimeIndex(pd.to_datetime(rows.timestamp, utc=True)))[
        ["wind_speed", "temperature", "wind_direction"]].rename(columns={
        "wind_speed": "wind_speed_ms", "temperature": "temperature_c", "wind_direction": "wind_direction_deg"})
    frame.attrs = {"turbine_id": "Kelmarsh 1", "wind_height_m": 78.5}
    return frame


def test_frozen_artifact_shape_order_kw_and_no_input_mutation(model, features):
    before = features.copy(deep=True)
    result = provider.predict(model, features)
    canonical = pd.DataFrame({"timestamp": features.index, "turbine_id": "Kelmarsh 1",
        "wind_speed": features.wind_speed_ms, "temperature": features.temperature_c,
        "wind_direction": features.wind_direction_deg}, index=features.index)
    expected = core.predict(model, canonical).raw_prediction.to_numpy() * 2050
    assert result.shape == (3,) and result.dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(provider.predict(model, features.iloc[::-1]), result[::-1])
    pd.testing.assert_frame_equal(features, before)
    assert features.attrs == before.attrs


def test_raw_negative_above_rated_and_nonfinite_not_clipped(model, features, monkeypatch):
    monkeypatch.setattr(model.pipeline, "predict", lambda frame: np.array([-.1, 1.2, np.nan]))
    result = provider.predict(model, features)
    np.testing.assert_allclose(result, [-205, 2460, np.nan], equal_nan=True)
    assert model.config.clip_normalized is True  # Original DataFrame API behavior unchanged.


def test_timezone_aware_non_utc_same_result(model, features):
    shifted = features.copy()
    shifted.index = shifted.index.tz_convert("Asia/Almaty")
    np.testing.assert_allclose(provider.predict(model, shifted), provider.predict(model, features))


@pytest.mark.parametrize("mutation", [
    lambda f: f.drop(columns="wind_direction_deg"),
    lambda f: f.rename(columns={"wind_speed_ms": "wind_speed"}),
    lambda f: f.assign(power=100),
    lambda f: f.assign(wind_speed_ms="7.0"),
    lambda f: f.assign(temperature_c=np.nan),
    lambda f: f.assign(wind_direction_deg=np.inf),
    lambda f: f.assign(wind_direction_deg=360),
    lambda f: f.assign(wind_speed_ms=-1),
    lambda f: f.assign(wind_speed_ms=True),
    lambda f: f.assign(wind_direction_deg=1 + 2j),
    lambda f: f.iloc[:0],
])
def test_bad_features_rejected(model, features, mutation):
    with pytest.raises(ValueError):
        provider.predict(model, mutation(features))


@pytest.mark.parametrize("turbine", [None, "T1", "T2", "Kelmarsh 99", 1])
def test_no_turbine_guessing_or_goldwind_scaling(model, features, turbine):
    features.attrs["turbine_id"] = turbine
    with pytest.raises(ValueError, match="turbine_id"):
        provider.predict(model, features)


@pytest.mark.parametrize("height", [None, 80, 100, 68.5, np.nan, True, "78.5"])
def test_height_context_required(model, features, height):
    features.attrs["wind_height_m"] = height
    with pytest.raises(ValueError, match="wind_height_m"):
        provider.predict(model, features)


def test_different_confirmed_height(model, features):
    features.attrs.update(turbine_id="Kelmarsh 3", wind_height_m=68.5)
    assert provider.predict(model, features).shape == (3,)


def test_bad_datetime_index(model, features):
    for index in [features.index.tz_localize(None), pd.RangeIndex(3),
                  pd.DatetimeIndex([features.index[0]] * 3),
                  pd.DatetimeIndex([pd.NaT, features.index[1], features.index[2]])]:
        frame = features.copy()
        frame.index = index
        with pytest.raises(ValueError, match="DatetimeIndex"):
            provider.predict(model, frame)


def test_capacity_cannot_be_overridden_to_2500(model, features):
    bad = copy.deepcopy(model)
    bad.metadata["rated_power"]["Kelmarsh 1"] = 2500
    with pytest.raises(ValueError, match="registry|disagree"):
        provider.predict(bad, features)


def test_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="MODEL_PATH"):
        provider.load_model(tmp_path / "missing.joblib")


def test_orchestrator_boolean_selection_keeps_explicit_context(features):
    assert features.loc[np.array([True, False, True])].attrs == features.attrs
