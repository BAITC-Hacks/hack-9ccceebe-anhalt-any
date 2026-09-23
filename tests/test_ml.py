"""Deterministic artificial fixtures ONLY; these are not observed SCADA or accuracy claims."""
import builtins
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from src.ml import MLConfig, train_model, predict, save_model, load_model, aggregate_predictions
from src.ml.data import prepare, chronological_split, choose_target
from src.ml.features import WeatherFeatures
from src.ml.model import _fit
from src.ml.aggregation import circular_mean
from src.ml.evaluation import evaluate_model
from src.ml.adapters import select_forecast_vintage


def fixture_data(n=160, multiple=False):
    t = np.arange(n)
    wind = 2 + (t % 23) * .65
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "wind_speed": wind, "temperature": 10 + np.sin(t / 15), "wind_direction": (t * 17) % 360,
        "turbine_id": "T1", "power": 100 * np.clip((wind - 3) / 9, 0, 1)})
    if multiple:
        second = frame.assign(turbine_id="T2", power=frame.power * 1.7)
        frame = pd.concat([frame, second], ignore_index=True)
    return frame


def config(**kwargs):
    return MLConfig(**({"power_unit": "kW", "target_source": "ARTIFICIAL UNIT TEST FIXTURE ONLY",
                       "backend": "hist", "reports_dir": None} | kwargs))


@pytest.fixture(scope="module")
def trained():
    return train_model(fixture_data(), config())


def test_chronological_multi_turbine():
    data, _ = prepare(fixture_data(multiple=True).sample(frac=1, random_state=42), config(), True)
    a, b, c = chronological_split(data, config())
    assert data.loc[a].timestamp.max() < data.loc[b].timestamp.min()
    assert data.loc[b].timestamp.max() < data.loc[c].timestamp.min()
    assert set(data.loc[a].timestamp).isdisjoint(data.loc[c].timestamp)


def test_no_target_or_future_features():
    data, _ = prepare(fixture_data(), config())
    data["power_lag_1"] = data.power.shift(-1)  # Deliberately unsafe upstream column is ignored.
    transform = WeatherFeatures().fit(data.iloc[:100])
    first = transform.transform(data)
    data["power"] = 1e12
    data["power_lag_1"] = -1e12
    pd.testing.assert_frame_equal(first, transform.transform(data))
    assert not any("power" in c or "timestamp" in c for c in first)


def test_encoder_train_only():
    data, _ = prepare(fixture_data(), config())
    f = WeatherFeatures().fit(data)
    unseen = data.assign(turbine_id="NEW")
    assert f.transform(unseen).turbine_0.eq(0).all()
    assert f.turbines_ == ["T1"]


def test_unverified_capacity_does_not_normalize():
    data, _ = prepare(fixture_data(), config())
    _, kind = choose_target(data, config(rated_power=100))
    assert kind == "absolute_power"


def test_normalized_roundtrip(tmp_path):
    cfg = config(rated_power={"T1": 100}, rated_power_verified=True, rated_power_source="TEST SPEC")
    model = train_model(fixture_data(), cfg)
    output = predict(model, fixture_data().iloc[-4:].drop(columns="power"))
    assert output.predicted_normalized_power.between(0, 1).all()
    np.testing.assert_allclose(output.predicted_power, output.predicted_normalized_power * 100)
    path = tmp_path / "power_model.joblib"
    save_model(model, path)
    pd.testing.assert_frame_equal(output, predict(load_model(path), fixture_data().iloc[-4:]))
    assert path.with_suffix(".metadata.json").exists()


def test_raw_before_clipping(trained):
    from src.ml.model import _postprocess
    raw = np.array([-0.2, 1.2])
    served = _postprocess(raw, config(), "normalized_power")
    np.testing.assert_array_equal(raw, [-0.2, 1.2])
    np.testing.assert_array_equal(served, [0, 1])


def test_direction_mean_and_grouping():
    assert circular_mean([359, 1]) == pytest.approx(0)
    assert np.isnan(circular_mean([0, 180]))
    frame = fixture_data(2).assign(wind_direction=[359, 1], predicted_power=[10, 20])
    for freq in ["daily", "monthly"]:
        result = aggregate_predictions(frame, freq)
        assert result.mean_wind_direction_deg.iloc[0] == pytest.approx(0)
        assert result.predicted_active_power.iloc[0] == 15
        assert result.normalized_active_power.isna().all()
    assert len(aggregate_predictions(frame, "hourly")) == 2


@pytest.mark.parametrize("field,value", [("wind_speed", -1), ("wind_speed", np.nan), ("power", np.nan), ("temperature", np.inf)])
def test_invalid_data_rejected(field, value):
    data = fixture_data()
    data.loc[0, field] = value
    with pytest.raises(ValueError):
        train_model(data, config())


def test_dedup_and_conflicts():
    frame = fixture_data()
    clean, quality = prepare(pd.concat([frame, frame.iloc[:1]]), config(), True)
    assert len(clean) == len(frame)
    assert quality["exact_duplicates_removed"] == 1
    with pytest.raises(ValueError, match="Conflicting"):
        prepare(pd.concat([frame, frame.iloc[:1].assign(power=999)]), config(), True)


def test_timestamp_units_mapping():
    data = pd.DataFrame({"time": ["2024-01-01 05:00"], "ws": [36], "temp": [273.15]})
    cfg = config(columns={"timestamp": "time", "wind_speed": "ws", "temperature": "temp"},
                 timezone="Etc/GMT-5", wind_speed_unit="km/h", temperature_unit="K")
    result, _ = prepare(data, cfg)
    assert result.timestamp.iloc[0] == pd.Timestamp("2024-01-01T00:00Z")
    assert result.wind_speed.iloc[0] == 10
    assert result.temperature.iloc[0] == 0


def test_no_real_target_no_training():
    with pytest.raises(ValueError, match="target_source"):
        train_model(fixture_data().drop(columns="power"), MLConfig(reports_dir=None))


def test_missing_lightgbm_falls_back(monkeypatch):
    original = builtins.__import__
    def import_module(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulate unavailable optional dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", import_module)
    data, _ = prepare(fixture_data(), config())
    model = _fit(data.iloc[:110], data.power.iloc[:110], data.iloc[110:130], data.power.iloc[110:130], config(), "auto")
    assert type(model[-1]).__name__ == "HistGradientBoostingRegressor"
    assert model[-1].early_stopping is False


def test_lightgbm_callbacks():
    pytest.importorskip("lightgbm")
    data, _ = prepare(fixture_data(), config())
    model = _fit(data.iloc[:110], data.power.iloc[:110], data.iloc[110:130], data.power.iloc[110:130], config(), "auto")
    assert type(model[-1]).__name__ == "LGBMRegressor"
    assert model[-1].best_iteration_ > 0


def test_forecast_origin_and_evaluation(trained):
    data = fixture_data().iloc[-10:].copy()
    data["issued_at"] = data.timestamp - np.timedelta64(6, "h")
    result = evaluate_model(trained, data, mode="B", forecast_provenance="TEST FORECAST FIXTURE")
    assert result["mode"] == "B"
    assert result["lead_hours"]["min"] == 6
    data["issued_at"] = data.timestamp + np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="lead time"):
        evaluate_model(trained, data, mode="B", forecast_provenance="TEST")
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_model(trained, fixture_data().iloc[:5])


def test_forecast_vintage():
    frame = fixture_data(3).drop(columns="power")
    frame["issued_at"] = pd.Timestamp("2023-12-31T18:00Z")
    future = frame.assign(issued_at=pd.Timestamp("2024-01-01T01:00Z"), wind_speed=999)
    result = select_forecast_vintage(pd.concat([frame, future]), "2023-12-31T23:00Z")
    assert (result.wind_speed < 999).all()


def test_forecast_origin_must_follow_selection(trained):
    data = fixture_data().iloc[-5:].copy()
    data["issued_at"] = pd.Timestamp("2024-01-01T00:00Z")
    with pytest.raises(ValueError, match="selection cutoff"):
        evaluate_model(trained, data, mode="B", forecast_provenance="TEST")


def test_forecast_training_requires_boundary_gap():
    data = fixture_data()
    data["issued_at"] = data.timestamp - np.timedelta64(6, "h")
    with pytest.raises(ValueError, match="temporal gap"):
        train_model(data, config(mode="B", forecast_provenance="TEST"))


def test_turbine_comparison():
    model = train_model(fixture_data(multiple=True), config())
    comparison = model.metadata["turbine_strategy"]
    assert set(comparison["eligible"]) == {"T1", "T2"}
    assert "per_turbine_validation_RMSE" in comparison
    result = predict(model, fixture_data(2).assign(turbine_id="NEW"))
    assert result.predicted_power.notna().all()


def test_reports(tmp_path):
    model = train_model(fixture_data(), config(reports_dir=str(tmp_path)))
    for name in ["power_curve.png", "actual_vs_predicted.png", "error_vs_wind_speed.png", "time_series_prediction.png"]:
        assert (tmp_path / "figures" / name).stat().st_size > 1000
    assert "MODE A" in (tmp_path / "evaluation.md").read_text()
    assert json.loads((tmp_path / "metrics.json").read_text())["test_used_for_selection"] is False


def test_predictions_preserve_order(trained):
    data = fixture_data().iloc[[159, 150, 151]].drop(columns="power")
    result = predict(trained, data)
    assert list(result.timestamp) == list(data.timestamp)
    assert list(result.index) == list(data.index)


def test_normalized_only():
    frame = fixture_data().rename(columns={"power": "normalized_power"})
    frame.normalized_power /= 100
    model = train_model(frame, config(normalized_target_confirmed=True, power_unit=None))
    output = predict(model, frame.iloc[-3:])
    assert "predicted_normalized_power" in output
    assert "predicted_power" not in output


def test_future_target_cannot_affect_fit_or_selection(trained):
    frame = fixture_data()
    frame.loc[136:, "power"] = 1e6
    other = train_model(frame, config())
    assert other.metadata["selected_candidate"] == trained.metadata["selected_candidate"]
    np.testing.assert_allclose(predict(other, frame).raw_prediction, predict(trained, frame).raw_prediction)
