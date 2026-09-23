"""Artificial software fixtures ONLY; their fit metrics are not station validation."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from src.ml.goldwind_data import GoldwindContract, prepare_hourly, read_organizer_files
from src.ml.goldwind_model import partition, train_model, save_model, evaluate_model
from src.ml.goldwind_provider import load_model, predict


def contract():
    return GoldwindContract(
        input_layout="hourly_aggregates", files={"hourly": "fixture.csv"},
        columns={"timestamp": "datetime", "wind_speed": "ws", "temperature": "temp",
                 "target": "power", "turbine_id": "id", "sample_count": "count"},
        turbine_mapping={"source_a": "T1", "source_b": "T2"},
        dataset_source="ARTIFICIAL UNIT TEST ONLY", metadata_source="TEST ONLY",
        file_identity_source="TEST ONLY", timezone="UTC", timestamp_label="interval_start",
        interval_minutes=60, source_interval_minutes=10, measurement_semantics="interval_mean",
        aggregation_source="TEST equal-duration means", coverage_source="TEST distinct valid 10-minute samples",
        wind_unit="m/s", temperature_unit="C", target_unit="kW",
        wind_height_m={"T1": 80., "T2": 80.}, availability_source="TEST delay",
        release_delay_minutes=10., first_forecast_origin="2025-01-10T00:00Z", forecast_origin_source="TEST C")


def source():
    times = pd.date_range("2025-01-01", periods=120, freq="h").astype(str)
    ws = np.tile(np.arange(12, dtype=float), 10)
    frame = pd.DataFrame({"datetime": times, "ws": ws, "temp": 10., "power": ws * 100., "count": 6})
    return pd.concat([frame.assign(id="source_a"), frame.assign(id="source_b", power=ws * 110.)], ignore_index=True)


def prepared(tmp_path, frame=None, c=None):
    c = c or contract()
    (source() if frame is None else frame).to_csv(tmp_path / "fixture.csv", index=False)
    raw, files = read_organizer_files(c, tmp_path)
    audit, hourly, summary = prepare_hourly(raw, c)
    return audit, hourly, summary, files


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("goldwind_fixture")
    _, hourly, _, files = prepared(directory)
    model = train_model(hourly, contract(), files)
    return model, hourly, files


def features():
    frame = pd.DataFrame({"wind_speed_ms": [9., 1., 5.], "temperature_c": [10., 10., 10.]},
        index=pd.to_datetime(["2025-01-10T03:00Z", "2025-01-10T01:00Z", "2025-01-10T02:00Z"]))
    frame.attrs = {"turbine_id": "T1", "wind_height_m": 80.}
    return frame


@pytest.mark.parametrize('horizon', [24, 48])
def test_real_a_features_b_artifact_c_orchestrator(bundle, tmp_path, monkeypatch, horizon):
    """Cross-module SOFTWARE integration; weather and training are artificial fixtures."""
    from src.data import weather
    from src.forecast.contracts import ForecastProtocol, ForecastSettings, expected_times
    from src.forecast.orchestrator import readiness, run_target_forecast

    model, _, _ = bundle
    artifact = save_model(model, tmp_path / 'model')
    # B's source/calendar is UTC; C's chosen execution calendar is Asia/Almaty.
    protocol = ForecastProtocol(confirmed=True, timezone='Asia/Almaty', daily_origin_hour=23,
                                evidence='ARTIFICIAL SOFTWARE TEST ONLY')
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(protocol.model_dump_json())

    def fixture_weather(lat, lon, origin, hours):
        frame = pd.DataFrame({'wind_speed_ms': np.full(hours, 8.),
                              'temperature_c': np.full(hours, 10.),
                              'weather_issued_at': (origin-pd.Timedelta(6, unit="h")).isoformat(),
                              'weather_available_at': (origin-pd.Timedelta(5, unit="h")).isoformat()},
                             index=expected_times(origin, hours, protocol))
        frame.attrs = {'kind': 'archived_forecast', 'source': 'ARTIFICIAL TEST ONLY',
                       'wind_height_m': 80., 'availability_basis': 'publisher_timestamp',
                       'availability_evidence': 'ARTIFICIAL TEST ONLY'}
        return frame

    monkeypatch.setattr(weather, 'get_forecast_weather', fixture_weather)
    settings = ForecastSettings(model_path=artifact, manifest_path=artifact.parent/'forecast_manifest.json',
                                protocol_path=protocol_path, cache_dir=tmp_path/'cache', results_dir=tmp_path/'results')
    assert readiness(settings)['ready']
    result = run_target_forecast('2026-01-31T23:00:00+05:00', horizon, settings=settings)
    assert result['status'] == 'ok', result.get('errors')
    assert len(result['farm_rows']) == horizon and len(result['rows']) == 2*horizon
    assert result['rows'][0]['valid_time'] == '2026-01-31T19:00:00+00:00'
    for tid in ('T1', 'T2'):
        input_frame = weather.build_features(fixture_weather(0, 0, pd.Timestamp(result['forecast_origin']), horizon))
        input_frame.attrs.update(turbine_id=tid, wind_height_m=80.)
        expected = predict(model, input_frame)
        actual = [row['power_kw'] for row in result['rows'] if row['turbine_id'] == tid]
        np.testing.assert_allclose(actual, expected)


def test_hourly_coverage_and_explicit_identity(tmp_path):
    frame = source()
    frame.loc[0, ["power", "count"]] = [1000., 3]
    frame.loc[1, ["power", "count"]] = [np.nan, 0]
    c = replace(contract(), turbine_mapping={"source_a": "T2", "source_b": "T1"})
    audit, hourly, _, _ = prepared(tmp_path, frame, c)
    row = hourly.loc[(hourly.turbine_id == "T2") & hourly.timestamp.eq(pd.Timestamp("2025-01-01T00:00Z"))].iloc[0]
    assert row.coverage == .5 and not row.complete and row.power == 1000.
    assert row.energy_kwh_observed == 500.
    assert audit.iloc[0].turbine_id == "T2" and audit.iloc[0]["count"] == 3
    assert audit.iloc[0]._source_turbine_id == "source_a"
    missing = hourly.loc[(hourly.turbine_id == "T2") & hourly.timestamp.eq(pd.Timestamp("2025-01-01T01:00Z"))].iloc[0]
    assert pd.isna(missing.power) and pd.isna(missing.energy_kwh_observed)


@pytest.mark.parametrize("count", [-1, 7, 2.5, np.nan])
def test_invalid_aggregate_coverage_rejected(tmp_path, count):
    frame = source()
    frame["count"] = frame["count"].astype(float)
    frame.loc[0, "count"] = count
    with pytest.raises(ValueError, match="sample_count"):
        prepared(tmp_path, frame)


@pytest.mark.parametrize("field", ["timezone", "target_unit", "turbine_mapping", "aggregation_source", "coverage_source", "source_interval_minutes", "first_forecast_origin"])
def test_confirmation_gates_before_fit(bundle, field):
    _, hourly, files = bundle
    with pytest.raises(ValueError):
        train_model(hourly, replace(contract(), **{field: None}), files)


def test_temporal_split_publication_purge_and_february(bundle):
    _, hourly, _ = bundle
    data, split = partition(hourly, contract())
    train, val, test = [data.loc[split[k]] for k in ["training", "validation", "test"]]
    assert train.available_at.max() <= val.timestamp.min()
    assert val.available_at.max() <= test.timestamp.min()
    assert test.available_at.max() <= pd.Timestamp(contract().first_forecast_origin)
    assert not (data.timestamp >= pd.Timestamp("2026-02-01T00:00Z")).any()


def test_model_reload_raw_order_units_and_reproduction(bundle, tmp_path):
    model, hourly, files = bundle
    path = save_model(model, tmp_path / "bundle")
    loaded = load_model(path)
    frame = features()
    raw = predict(loaded, frame)
    assert raw.shape == (3,)
    assert raw.dtype == np.float64
    for turbine in ["T1", "T2"]:
        frame.attrs["turbine_id"] = turbine
        np.testing.assert_allclose(predict(loaded, frame), [predict(loaded, frame.iloc[[i]])[0] for i in range(3)])
    report, _ = evaluate_model(loaded, hourly, contract(), files)
    assert report["metrics"] == model.metadata["metrics"]["test"]
    assert report["mode"] == "A" and report["february_metrics"] is None
    manifest = json.loads((path.parent / "forecast_manifest.json").read_text())
    assert manifest["output_unit"] == "kW" and manifest["rated_power_kw"] == {"T1": 2500., "T2": 2500.}
    assert loaded.pipeline[0].direction_ == False
    assert not any("lag" in f for f in loaded.metadata["feature_names"])
    with pytest.raises(ValueError, match="empty"):
        save_model(loaded, path.parent)
    with pytest.raises(ValueError, match="hashes"):
        evaluate_model(loaded, hourly, contract(), [{"file": "different", "sha256": "a" * 64}])


class RawFixture(RegressorMixin, BaseEstimator):
    def fit(self, X=None, y=None):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.array([-10., 2600., np.nan])[:len(X)]


def test_no_hidden_clipping_or_kw_rescaling(bundle):
    model = deepcopy(bundle[0])
    model.pipeline.steps[-1] = ("regressor", RawFixture().fit())
    np.testing.assert_equal(predict(model, features()), [-10., 2600., np.nan])


@pytest.mark.parametrize("case", ["column", "direction", "unknown", "height", "naive", "duplicate", "half_hour", "nan", "string", "negative"])
def test_provider_rejects_bad_inputs(bundle, case):
    frame = features()
    if case == "column": frame = frame.rename(columns={"wind_speed_ms": "wind_speed"})
    if case == "direction": frame["wind_direction_deg"] = 0.
    if case == "unknown": frame.attrs["turbine_id"] = "Kelmarsh 1"
    if case == "height": frame.attrs["wind_height_m"] = 100.
    if case == "naive": frame.index = frame.index.tz_localize(None)
    if case == "duplicate": frame.index = [frame.index[0]] * 3
    if case == "half_hour": frame.index = frame.index + pd.Timedelta(30, unit="min")
    if case == "nan": frame.iloc[0, 0] = np.nan
    if case == "string": frame["wind_speed_ms"] = "5"
    if case == "negative": frame.iloc[0, 0] = -1.
    with pytest.raises(ValueError):
        predict(bundle[0], frame)


def test_missing_artifact_and_kelmarsh_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Goldwind artifact missing"):
        load_model(tmp_path / "missing.joblib")
    with pytest.raises(ValueError, match="Goldwind"):
        load_model("models/power_model.joblib")


def test_cli_unknown_template_does_not_create_artifact(tmp_path, monkeypatch, capsys):
    from scripts.train_goldwind import main
    monkeypatch.setattr(sys, "argv", ["train", "--contract", "config/goldwind_hourly.template.json", "--output", str(tmp_path / "model")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Confirm organizer metadata" in capsys.readouterr().err
    assert not (tmp_path / "model").exists()


def test_cli_train_evaluate_artificial_fixture_only(tmp_path, monkeypatch, capsys):
    from scripts.train_goldwind import main as train_cli
    from scripts.evaluate_goldwind import main as eval_cli
    source().to_csv(tmp_path / "fixture.csv", index=False)
    cfg = tmp_path / "contract.json"
    cfg.write_text(json.dumps(asdict(contract())))
    out = tmp_path / "test_bundle"
    monkeypatch.setattr(sys, "argv", ["train", "--contract", str(cfg), "--output", str(out)])
    train_cli()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["evaluate", "--contract", str(cfg), "--model", str(out / "power_model.joblib")])
    eval_cli()
    assert json.loads(capsys.readouterr().out)["mode"] == "A"


def test_raw_reaches_existing_deterministic_rules(bundle):
    from types import SimpleNamespace
    rules = pytest.importorskip("src.agent.rules")
    model = deepcopy(bundle[0])
    model.pipeline.steps[-1] = ("regressor", RawFixture().fit())
    frame = features()
    raw = predict(model, frame)
    flags, stats, valid = rules.evaluate(frame, raw, SimpleNamespace(rated_power_kw=2500., cut_in_ms=3., cut_out_ms=25.))
    assert {flag.code for flag in flags} >= {"negative_power", "above_rated_power", "non_finite_power"}
    assert not valid.any() and stats.min_power_kw == -10. and stats.max_power_kw == 2600.


def test_manifest_matches_c_schema(bundle, tmp_path):
    contracts = pytest.importorskip("src.forecast.contracts")
    path = save_model(bundle[0], tmp_path / "bundle")
    manifest = contracts.ModelManifest.model_validate_json((path.parent / "forecast_manifest.json").read_text())
    assert pd.Timestamp(manifest.selection_data_available_until) <= pd.Timestamp(contract().first_forecast_origin)


def test_reload_and_evaluate_never_fit(bundle, tmp_path, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.dummy import DummyRegressor
    from src.ml.features import WeatherFeatures
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference/evaluation must not fit")
    path = save_model(bundle[0], tmp_path / "frozen")
    for estimator in [HistGradientBoostingRegressor, DummyRegressor, WeatherFeatures]:
        monkeypatch.setattr(estimator, "fit", forbidden)
    model = load_model(path)
    predict(model, features())
    evaluate_model(model, bundle[1], contract(), bundle[2])


def test_late_evaluation_cutoff_rejected(bundle):
    model = deepcopy(bundle[0])
    model.metadata["splits"]["test"]["available_until"] = "2026-02-02T00:00:00Z"
    with pytest.raises(ValueError, match="cutoff"):
        predict(model, features())
