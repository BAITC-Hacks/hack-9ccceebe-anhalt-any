from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.config import ROOT


def test_streamlit_demo():
    app = AppTest.from_file(str(ROOT / 'src/app/streamlit_app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Синтетическое демо').run()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3
    assert app.metric[2].value == '24/24'
    assert any('Синтетическое' in item.value for item in app.info)


def test_streamlit_real_data():
    app = AppTest.from_file(str(ROOT / 'src/app/streamlit_app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Реальные данные Kelmarsh').run()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 6
    assert app.metric[5].value == '144/144'
    assert any('MODE A' in item.value for item in app.info)


def test_target_default_blocks_without_model_and_keeps_other_modes(monkeypatch):
    monkeypatch.setattr('src.forecast.orchestrator.readiness', lambda: {
        'ready': False, 'blockers': ['Целевая модель T1/T2 отсутствует', 'Протокол не подтверждён']})
    run = Mock(side_effect=AssertionError('Blocked mode must not run or fall back'))
    monkeypatch.setattr('src.forecast.orchestrator.run_target_forecast', run)
    app = AppTest.from_file(str(ROOT / 'src/app/streamlit_app.py')).run(timeout=30)
    assert not app.exception
    assert app.sidebar.radio[0].value == 'Прогноз ВЭС T1/T2'
    assert any('Целевая модель' in error.value for error in app.error)
    assert app.button[0].disabled
    assert app.selectbox[0].value == 48
    assert app.text_input[0].value == '2026-01-31T23:00:00+05:00'
    assert set(app.sidebar.radio[0].options) == {
        'Прогноз ВЭС T1/T2', 'Реальные данные Kelmarsh', 'Синтетическое демо'}
    run.assert_not_called()


def target_result(*, missing_turbine=False):
    # UI fixture only: never used as real model or weather evidence.
    times = pd.date_range('2026-02-01', periods=24, freq='h', tz='UTC')
    turbines = ['T1'] if missing_turbine else ['T1', 'T2']
    rows = [{'valid_time': time.isoformat(), 'turbine_id': turbine, 'power_kw': 100.,
             'normalized_power': .04, 'energy_kwh': 100., 'valid': True, 'flags': []}
            for time in times for turbine in turbines]
    farm_rows = [{'valid_time': time.isoformat(), 'power_kw': None if missing_turbine else 200.,
                  'energy_kwh': None if missing_turbine else 200., 'complete': not missing_turbine,
                  'missing_turbines': ['T2'] if missing_turbine else []} for time in times]
    return {'run_id': 'ui-test', 'forecast_origin': '2026-01-31T23:00:00+00:00', 'horizon_h': 24,
            'status': 'partial' if missing_turbine else 'ok', 'rows': rows, 'farm_rows': farm_rows,
            'errors': {'T2': 'Weather unavailable'} if missing_turbine else {},
            'analysis': {'summary': 'UI fixture', 'risk_level': 'high' if missing_turbine else 'low',
                         'anomalies': [], 'recommendation': 'Review', 'confidence_note': 'Fixture only'},
            'analysis_source': 'deterministic', 'cache_hit': False}


@pytest.mark.parametrize('missing_turbine', [False, True])
def test_target_ui_success_and_partial_preserve_farm_coverage(monkeypatch, missing_turbine):
    monkeypatch.setattr('src.forecast.orchestrator.readiness', lambda: {'ready': True, 'blockers': []})
    run = Mock(return_value=target_result(missing_turbine=missing_turbine))
    monkeypatch.setattr('src.forecast.orchestrator.run_target_forecast', run)
    app = AppTest.from_file(str(ROOT / 'src/app/streamlit_app.py')).run(timeout=30)
    app.selectbox[0].set_value(24)
    app.checkbox[0].set_value(True)
    app.checkbox[1].set_value(True)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    run.assert_called_once_with('2026-01-31T23:00:00+05:00', 24, refresh=True, with_agent=True)
    assert len(app.metric) == 3
    assert app.metric[0].value == ('Нет полного расчёта' if missing_turbine else '4,800.0')
    assert app.metric[1].value == ('0/24' if missing_turbine else '24/24')
    assert any('T2: Weather unavailable' in error.value for error in app.error) == missing_turbine


def test_target_api_readiness_and_aware_origin(monkeypatch):
    from src.api import main
    run = Mock(return_value=target_result())
    monkeypatch.setattr(main, 'run_target_forecast', run)
    monkeypatch.setattr(main, 'readiness', lambda: {'ready': False, 'blockers': ['Missing model']})
    client = TestClient(main.app)
    assert client.get('/readiness').json() == {'ready': False, 'blockers': ['Missing model']}
    response = client.post('/target-forecast', json={
        'forecast_origin': '2026-01-31T23:00:00Z', 'horizon_h': 24, 'refresh': True, 'with_agent': True})
    assert response.status_code == 200
    assert response.json()['run_id'] == 'ui-test'
    run.assert_called_once_with('2026-01-31T23:00:00+00:00', 24, refresh=True, with_agent=True)


@pytest.mark.parametrize('origin,horizon', [
    ('2026-01-31T23:00:00', 24), ('not-a-time', 48),
    ('2026-01-31T23:00:00Z', 1), ('2026-01-31T23:00:00Z', 24.0)])
def test_target_api_rejects_invalid_contract_before_inference(monkeypatch, origin, horizon):
    from src.api import main
    run = Mock(side_effect=AssertionError('Invalid contract reached inference'))
    monkeypatch.setattr(main, 'run_target_forecast', run)
    response = TestClient(main.app).post('/target-forecast', json={'forecast_origin': origin, 'horizon_h': horizon})
    assert response.status_code == 422
    run.assert_not_called()
