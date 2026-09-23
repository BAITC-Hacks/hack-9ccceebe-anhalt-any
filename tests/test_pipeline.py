import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agent.analyzer import analyze_forecast
from src.agent.backtest import load_actuals, run_backtest
from src.agent.orchestrator import run_forecast
from src.agent.rules import evaluate
from src.agent.schemas import Analysis, ForecastError
from src.agent.tools import load_weather, normalize_weather
from src.api.main import app
from src.config import ROOT, Settings, Turbine
from src.data import demo


@pytest.fixture
def settings():
    return Settings(demo=True)


@pytest.fixture
def weather():
    return demo.get_archival_weather(43.645150, 78.535604, date(2026, 2, 10), 24)


def test_demo_e2e_is_offline(monkeypatch, settings):
    monkeypatch.setattr('src.agent.analyzer.OpenAI', Mock(side_effect=AssertionError('network forbidden')))
    result = run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)
    assert result.demo and len(result.hourly) == 24
    assert result.statistics.predicted_energy_kwh == pytest.approx(sum(p.power_kw for p in result.hourly))
    assert result.statistics.valid_hours == 24
    assert result.analysis_source == 'deterministic'
    assert result.statistics.max_power_kw <= 2500
    assert all(point.timestamp.utcoffset().total_seconds() == 0 for point in result.hourly)
    assert 'NaN' not in result.model_dump_json()


@pytest.mark.parametrize('turbine,day,horizon', [('UNKNOWN', '2026-02-10', 24),
    ('T1', 'invalid', 24), ('T1', '2026-02-10', 0), ('T1', '2026-02-10', 73),
    ('T1', '2026-02-10', 1.5), ('T1', '2027-02-10', 24)])
def test_invalid_requests(turbine, day, horizon, settings):
    with pytest.raises(ForecastError):
        run_forecast(turbine, day, horizon, settings=settings, with_agent=False)


def test_t2_and_coordinate_guard(settings):
    result = run_forecast('T2', '2026-02-28', 72, settings=settings, with_agent=False)
    assert len(result.hourly) == 72
    with pytest.raises(ForecastError, match='Coordinates'):
        run_forecast('T1', '2026-02-10', latitude=0, settings=settings, with_agent=False)


def test_production_does_not_fall_back_to_demo():
    with pytest.raises(ForecastError, match='Provider not configured'):
        run_forecast('T1', '2026-02-10', settings=Settings(), with_agent=False)


def test_missing_model(monkeypatch, settings, tmp_path):
    monkeypatch.setattr('src.agent.orchestrator.ROOT', tmp_path)
    with pytest.raises(ForecastError, match='Model file'):
        run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)


def test_rules_and_energy_integrity(weather):
    turbine = Turbine(latitude=0, longitude=0, rated_power_kw=2500)
    power = np.full(24, 1000.0)
    power[:4] = [np.nan, -10, 3000, 0]
    weather.iloc[3, weather.columns.get_loc('wind_speed_ms')] = 15
    weather.iloc[4, weather.columns.get_loc('wind_speed_ms')] = np.nan
    flags, stats, valid = evaluate(weather, power, turbine)
    codes = {f.code for f in flags}
    assert {'non_finite_power', 'negative_power', 'above_rated_power', 'hourly_jump',
            'strong_wind_low_power', 'missing_weather'} <= codes
    assert stats.predicted_energy_kwh is None
    assert valid.sum() == 20
    assert all(len(f.timestamps) <= 3 for f in flags)
    weather.loc[:, 'wind_speed_ms'] = 26
    flags, _, _ = evaluate(weather, np.zeros(24), turbine)
    assert 'strong_wind_low_power' not in {f.code for f in flags}


def test_missing_hour_not_zero_filled(monkeypatch, settings, weather):
    monkeypatch.setattr(demo, 'get_archival_weather', lambda *args: weather.drop(weather.index[5]))
    result = run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)
    assert result.status == 'partial'
    assert result.hourly[5].power_kw is None
    assert result.statistics.predicted_energy_kwh is None
    assert result.statistics.valid_hours == 23


def test_feature_index_cannot_be_reordered(monkeypatch, settings):
    monkeypatch.setattr(demo, 'build_features', lambda df: df.iloc[::-1])
    with pytest.raises(ForecastError, match='preserve'):
        run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)


def test_empty_features(monkeypatch, settings):
    monkeypatch.setattr(demo, 'build_features', lambda df: pd.DataFrame())
    with pytest.raises(ForecastError, match='empty features'):
        run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)


def test_wrong_prediction_shape(monkeypatch, settings):
    monkeypatch.setattr('src.ml.demo.predict', lambda model, features: np.ones((len(features), 1)))
    with pytest.raises(ForecastError, match='1D'):
        run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)


def test_wrong_height(monkeypatch, settings, weather):
    weather.attrs['wind_height_m'] = 10
    monkeypatch.setattr(demo, 'get_archival_weather', lambda *args: weather)
    with pytest.raises(ForecastError, match='hub height'):
        run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)


def test_cache_exact_key_and_provider_outage(tmp_path, weather):
    weather.attrs.update(kind='archived_forecast', issued_at='2026-02-09T12:00:00Z')
    data = SimpleNamespace(get_archival_weather=Mock(return_value=weather))
    settings = Settings(cache_dir=tmp_path, data_module='person_a')
    first = load_weather(settings, data, 43.6, 78.5, date(2026, 2, 10), 24)
    data.get_archival_weather.side_effect = RuntimeError('outage')
    cached = load_weather(settings, data, 43.6, 78.5, date(2026, 2, 10), 24)
    pd.testing.assert_frame_equal(first, cached, check_freq=False)
    assert data.get_archival_weather.call_count == 1
    with pytest.raises(ForecastError, match='unavailable'):
        load_weather(settings, data, 43.7, 78.5, date(2026, 2, 10), 24)


def test_issuance_no_future_leakage(weather):
    weather.attrs.update(kind='archived_forecast', issued_at='2026-02-10T06:00:00Z')
    with pytest.raises(ForecastError, match='later'):
        normalize_weather(weather, date(2026, 2, 10), 24)


def test_duplicate_weather_rejected(weather):
    duplicate = pd.concat([weather, weather.iloc[:1]])
    with pytest.raises(ForecastError, match='duplicate'):
        normalize_weather(duplicate, date(2026, 2, 10), 24)


@pytest.mark.parametrize('mode', ['invalid_json', 'refusal', 'unsupported_code', 'timeout'])
def test_openai_failure_one_call_only(mode, settings):
    result = run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)
    client = Mock()
    if mode in ('invalid_json', 'timeout'):
        client.responses.parse.side_effect = ValueError('do not log this body')
    elif mode == 'refusal':
        client.responses.parse.return_value.output_parsed = None
    else:
        client.responses.parse.return_value.output_parsed = Analysis(summary='test', risk_level='low',
            anomalies=['invented_fault'], recommendation='test', confidence_note='test')
    analysis, source, reason = analyze_forecast({'demo': True, 'weather_kind': 'synthetic'},
        result.anomalies, result.statistics, enabled=True, model='configured-by-test', client=client)
    assert source == 'deterministic' and reason == 'openai_failed_or_invalid'
    assert client.responses.parse.call_count == 1
    assert analysis.risk_level in ('medium', 'high')
    assert len(json.dumps(client.responses.parse.call_args.kwargs['input'])) < 6000


def test_structured_success_preserves_flags_and_risk(settings):
    result = run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)
    client = Mock()
    client.responses.parse.return_value.output_parsed = Analysis(summary='ok', risk_level='low',
        anomalies=[], recommendation='review', confidence_note='overconfident')
    analysis, source, _ = analyze_forecast({'demo': True, 'weather_kind': 'synthetic'},
        result.anomalies, result.statistics, enabled=True, model='configured-by-test', client=client)
    assert source == 'openai'
    assert analysis.anomalies == [flag.code for flag in result.anomalies]
    assert analysis.risk_level in ('medium', 'high')
    assert 'Синтетическое' in analysis.confidence_note


def test_backtest_full_period_no_actuals(settings, tmp_path):
    summary = run_backtest(date(2026, 1, 31), date(2026, 2, 28), 'T1', tmp_path, settings)
    assert summary['days_completed'] == 29
    assert summary['valid_hours'] == 696
    assert summary['mae_kw'] is None
    assert summary['matched_actual_hours'] == 0
    assert len(pd.read_csv(tmp_path / 'results.csv')) == 696


def test_backtest_metrics_actual_alignment(settings, tmp_path):
    result = run_forecast('T1', '2026-02-10', settings=settings, with_agent=False)
    # Test-only artificial values to check joins and arithmetic, not quality evidence.
    actual_path = tmp_path / 'actuals.csv'
    pd.DataFrame([{'timestamp': p.timestamp, 'turbine_id': 'T1', 'actual_power_kw': p.power_kw + 10}
                  for p in reversed(result.hourly)]).to_csv(actual_path, index=False)
    summary = run_backtest(date(2026, 2, 10), date(2026, 2, 10), 'T1', tmp_path / 'out', settings, actual_path)
    assert summary['mae_kw'] == pytest.approx(10)
    assert summary['rmse_kw'] == pytest.approx(10)
    assert summary['bias_kw'] == pytest.approx(-10)
    assert summary['actual_coverage'] == 1


def test_actual_duplicate_rejected(tmp_path):
    path = tmp_path / 'actual.csv'
    path.write_text('timestamp,turbine_id,actual_power_kw\n2026-02-10T00:00:00Z,T1,1\n2026-02-10T00:00:00Z,T1,2\n')
    with pytest.raises(ForecastError, match='duplicate'):
        load_actuals(path)


def test_failed_backtest_reports_missing_days(settings, tmp_path):
    summary = run_backtest(date(2026, 1, 30), date(2026, 1, 31), 'T1', tmp_path, settings)
    assert summary['days_completed'] == 1 and summary['failed_days'] == 1
    assert summary['expected_hours'] == 48 and summary['valid_hours'] == 24


def test_api(monkeypatch):
    monkeypatch.setenv('DEMO_MODE', 'true')
    client = TestClient(app)
    response = client.post('/forecast', json={'turbine_id': 'T1', 'forecast_date': '2026-02-10'})
    assert response.status_code == 200
    assert len(response.json()['hourly']) == 24
    assert client.post('/forecast', json={'turbine_id': 'T1', 'forecast_date': 'bad'}).status_code == 422
    assert client.post('/forecast', json={'turbine_id': 'missing', 'forecast_date': '2026-02-10'}).status_code == 422
