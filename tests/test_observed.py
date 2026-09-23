from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agent import observed
from src.agent.rules import evaluate
from src.agent.schemas import ForecastError
from src.api.main import app
from src.config import ROOT, Turbine
from src.data.kelmarsh_observed import load_observations


def test_real_native_pipeline_preserves_model_predictions():
    result = observed.run_observed_analysis('Kelmarsh 1', '2017-11-08')
    assert not result.demo and result.mode == 'observed_scada'
    assert len(result.samples) == 144
    assert result.metrics.matched_samples == 144
    assert result.metrics.mae_kw > 0
    assert result.analysis_source == 'deterministic'
    assert 'MODE A' in result.analysis.confidence_note
    frame = load_observations('Kelmarsh 1', date(2017, 11, 8), 24)
    features = frame[list(observed.provider.INPUT_COLUMNS)].copy()
    features.attrs = {'turbine_id': 'Kelmarsh 1', 'wind_height_m': 78.5}
    direct = observed.provider.predict(observed.provider.load_model(ROOT / 'models/power_model.joblib'), features)
    np.testing.assert_allclose([p.power_kw for p in result.samples], direct)
    assert result.metrics.mae_kw == pytest.approx(np.abs(direct-frame.actual_power_kw).mean())


def test_energy_uses_ten_minute_duration():
    frame = pd.DataFrame({'wind_speed_ms': np.full(6, 8.), 'temperature_c': np.full(6, 10.)},
                         index=pd.date_range('2017-11-08', periods=6, freq='10min', tz='UTC'))
    turbine = Turbine(latitude=0, longitude=0, rated_power_kw=2050)
    _, stats, _ = evaluate(frame, np.full(6, 600.), turbine, interval_minutes=10)
    assert stats.predicted_energy_kwh == 600  # Not 3600 kWh.
    assert stats.requested_hours == 1 and stats.valid_hours == 1


def test_missing_native_sample_is_not_filled(monkeypatch):
    original = observed.load_observations
    def missing(*args):
        frame = original(*args)
        frame.iloc[3] = np.nan
        return frame
    monkeypatch.setattr(observed, 'load_observations', missing)
    result = observed.run_observed_analysis('Kelmarsh 1', '2017-11-08')
    assert result.samples[3].power_kw is None
    assert result.statistics.predicted_energy_kwh is None
    assert result.metrics.matched_samples == 143
    assert result.metrics.coverage == pytest.approx(143/144)
    assert result.statistics.valid_hours <= 143/6


def test_actual_power_never_enters_inference(monkeypatch):
    original = observed.provider.predict
    def assert_boundary(model, frame):
        assert set(frame.columns) == set(observed.provider.INPUT_COLUMNS)
        assert frame.attrs['turbine_id'] == 'Kelmarsh 3'
        assert frame.attrs['wind_height_m'] == 68.5
        return original(model, frame)
    monkeypatch.setattr(observed.provider, 'predict', assert_boundary)
    assert len(observed.run_observed_analysis('Kelmarsh 3', '2017-11-08').samples) == 144


def test_goldwind_and_outside_holdout_rejected():
    for turbine, day in [('T1', '2017-11-08'), ('Kelmarsh 1', '2026-02-10'),
                          ('Kelmarsh 1', '2017-09-01'), ('Kelmarsh 1', '2017-11-07')]:
        with pytest.raises(ForecastError):
            observed.run_observed_analysis(turbine, day)


def test_raw_bad_predictions_kept_in_metrics_and_flags(monkeypatch):
    monkeypatch.setattr(observed.provider, 'predict', lambda model, features: np.full(len(features), -205.))
    result = observed.run_observed_analysis('Kelmarsh 1', '2017-11-08')
    assert result.status == 'invalid'
    assert result.metrics.matched_samples == 144
    assert result.metrics.mae_kw is not None
    assert result.statistics.predicted_energy_kwh is None
    assert 'negative_power' in [a.code for a in result.anomalies]
    assert result.samples[0].power_kw == -205


def test_observed_api():
    response = TestClient(app).post('/observed', json={'turbine_id': 'Kelmarsh 1', 'observation_date': '2017-11-08'})
    assert response.status_code == 200
    assert response.json()['sampling_interval_minutes'] == 10
    assert len(response.json()['samples']) == 144


def test_observed_backtest_metrics_match_contiguous_native_run(tmp_path):
    from scripts.run_observed_backtest import run
    two_days = observed.run_observed_analysis('Kelmarsh 1', '2017-11-08', 48)
    summary = run(date(2017, 11, 8), date(2017, 11, 9), 'Kelmarsh 1', tmp_path)
    assert summary['expected_samples'] == 288
    assert summary['raw_rmse_kw'] == pytest.approx(two_days.metrics.rmse_kw)
    assert summary['raw_mae_kw'] == pytest.approx(two_days.metrics.mae_kw)
    assert len(pd.read_csv(tmp_path/'results.csv')) == 288
