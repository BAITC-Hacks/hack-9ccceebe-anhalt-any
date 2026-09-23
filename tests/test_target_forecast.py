"""Synthetic contract fixtures only. They are not evidence of target model accuracy."""
import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.agent.schemas import Analysis, ForecastError
from src.forecast.contracts import ForecastProtocol, ForecastSettings, expected_times
from src.forecast.orchestrator import readiness, run_target_forecast


@pytest.fixture
def setup(tmp_path, monkeypatch):
    policy = ForecastProtocol(confirmed=True, timezone='UTC', daily_origin_hour=23,
                              evidence='Test-only assumed protocol; NOT organizer evidence')
    protocol_path = tmp_path/'protocol.json'
    protocol_path.write_text(policy.model_dump_json())
    artifact = tmp_path/'model.bin'
    artifact.write_bytes(b'test-only weights placeholder')
    registry = {tid:{'latitude':43+i/100, 'longitude':78., 'rated_power_kw':2500.,
                     'hub_height_m':80., 'model_name':'test fixture'} for i,tid in enumerate(['T1','T2'])}
    registry_path = tmp_path/'turbines.json'
    registry_path.write_text(json.dumps(registry))
    manifest = {'model_version':'test-fixture-v1','artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'data_source':'organizer_scada','provenance':'TEST ONLY: no organizer data',
        'turbine_ids':['T1','T2'],'rated_power_kw':{'T1':2500.,'T2':2500.},
        'wind_height_m':{'T1':80.,'T2':80.},'feature_columns':['wind_speed_ms','temperature_c'],
        'output_unit':'kW','normalized_target_definition':'fraction_of_rated_power','timezone':'UTC',
        'interval_minutes':60,'training_data_available_until':'2026-01-30T23:00:00Z',
        'selection_data_available_until':'2026-01-31T22:00:00Z',
        'training_target_interval_end':'2026-01-30T23:00:00Z',
        'selection_target_interval_end':'2026-01-31T21:00:00Z','availability_evidence':'test fixture'}
    manifest_path=tmp_path/'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    data=types.ModuleType('test_target_data')
    data.revision=0
    def get_weather(lat,lon,origin,horizon):
        frame=pd.DataFrame({'wind_speed_ms':np.full(horizon,8.+data.revision),
                            'temperature_c':np.full(horizon,5.),
                            'weather_issued_at':(origin-pd.Timedelta(6, unit="h")).isoformat(),
                            'weather_available_at':(origin-pd.Timedelta(5-data.revision, unit="h")).isoformat()},
                            index=expected_times(origin,horizon,policy))
        frame.attrs={'kind':'archived_forecast','source':'test fixture only','wind_height_m':80.,
                     'availability_basis':'publisher_timestamp','availability_evidence':'test only'}
        return frame
    data.get_forecast_weather=Mock(side_effect=get_weather)
    data.build_features=lambda frame:frame[['wind_speed_ms','temperature_c']].copy()
    ml=types.ModuleType('test_target_ml')
    ml.load_model=Mock(return_value=object())
    def predict(model, frame):
        assert frame.attrs['turbine_id'] in ['T1','T2']
        assert frame.attrs['wind_height_m']==80
        return frame.wind_speed_ms.to_numpy()*100
    ml.predict=Mock(side_effect=predict)
    monkeypatch.setitem(sys.modules,data.__name__,data)
    monkeypatch.setitem(sys.modules,ml.__name__,ml)
    settings=ForecastSettings(data_module=data.__name__,ml_module=ml.__name__,model_path=artifact,
        manifest_path=manifest_path,protocol_path=protocol_path,turbines_file=registry_path,
        cache_dir=tmp_path/'weather-cache',results_dir=tmp_path/'results')
    return settings,data,ml,manifest


def test_24_48_and_farm_alignment(setup):
    settings,data,ml,_=setup
    for horizon in [24,48]:
        result=run_target_forecast('2026-01-31T23:00:00Z',horizon,settings=settings)
        assert result['status']=='ok'
        assert len(result['rows'])==2*horizon and len(result['farm_rows'])==horizon
        assert result['rows'][0]['valid_time']=='2026-02-01T00:00:00+00:00'
        assert result['rows'][0]['lead_hours']==1
        assert result['rows'][-1]['lead_hours']==horizon
        assert all(r['power_kw']==1600 for r in result['farm_rows'])
        assert result['rows'][0]['normalized_power']==pytest.approx(.32)
        assert all(r['energy_kwh']==r['power_kw'] for r in result['farm_rows'])
        assert result['weather_snapshots']['T1']['weather_available_at'] <= result['forecast_origin']


def test_unchanged_refresh_no_new_inference_or_llm(setup,monkeypatch):
    settings,data,ml,_=setup
    analyzer=Mock(return_value=(Analysis(summary='test',risk_level='low',anomalies=[],recommendation='test',confidence_note='test'), 'openai','test'))
    monkeypatch.setattr('src.forecast.orchestrator.analyze_forecast',analyzer)
    first=run_target_forecast('2026-01-31T23:00:00Z',settings=settings,with_agent=True)
    second=run_target_forecast('2026-01-31T23:00:00Z',settings=settings,with_agent=True)
    refreshed=run_target_forecast('2026-01-31T23:00:00Z',settings=settings,refresh=True,with_agent=True)
    assert first['run_id']==second['run_id']==refreshed['run_id']
    assert second['cache_hit'] and refreshed['cache_hit']
    assert ml.predict.call_count==2 and analyzer.call_count==1
    assert data.get_forecast_weather.call_count==4  # Two initial + two explicit refreshes.


def test_updated_inputs_create_new_immutable_version(setup):
    settings,data,ml,_=setup
    first=run_target_forecast('2026-01-31T23:00:00Z',settings=settings)
    first_file=settings.results_dir/first['run_id']/'forecast.json'
    previous=first_file.read_bytes()
    data.revision=1
    updated=run_target_forecast('2026-01-31T23:00:00Z',settings=settings,refresh=True)
    assert updated['run_id']!=first['run_id']
    assert updated['rows'][0]['power_kw']==900
    assert first_file.read_bytes()==previous
    assert ml.predict.call_count==4


def test_future_revision_is_blocked_not_silently_cached(setup):
    settings,data,_,_=setup
    run_target_forecast('2026-01-31T23:00:00Z',settings=settings)
    data.revision=6  # Same old issue, but this correction becomes available AFTER origin.
    result=run_target_forecast('2026-01-31T23:00:00Z',settings=settings,refresh=True)
    assert result['status']=='blocked'
    assert all(r['power_kw'] is None for r in result['rows'])


def test_one_turbine_failure_farm_null(setup):
    settings,data,_,_=setup
    original=data.get_forecast_weather.side_effect
    def fail_t2(lat,*args):
        if lat>43: raise RuntimeError('outage')
        return original(lat,*args)
    data.get_forecast_weather.side_effect=fail_t2
    result=run_target_forecast('2026-01-31T23:00:00Z',settings=settings)
    assert result['status']=='partial'
    assert all(r['power_kw'] is None and r['missing_turbines']==['T2'] for r in result['farm_rows'])
    assert any(r['power_kw']==800 for r in result['rows'])


def test_raw_negative_not_clipped_but_farm_flagged(setup):
    settings,_,ml,_=setup
    ml.predict.side_effect=lambda model,frame:np.full(len(frame),-100.)
    result=run_target_forecast('2026-01-31T23:00:00Z',settings=settings)
    assert result['rows'][0]['power_kw']==-100 and not result['rows'][0]['valid']
    assert result['farm_rows'][0]['raw_power_kw']==-200
    assert result['farm_rows'][0]['power_kw'] is None


@pytest.mark.parametrize('field',['training_data_available_until','selection_data_available_until'])
def test_future_fit_or_selection_data_blocked(setup,field):
    settings,_,ml,manifest=setup
    manifest[field]='2026-02-01T00:00:00Z'
    settings.manifest_path.write_text(json.dumps(manifest))
    result=run_target_forecast('2026-01-31T23:00:00Z',settings=settings)
    assert result['status']=='blocked' and not result['rows']
    ml.load_model.assert_not_called()


def test_hash_mismatch_blocks_unverified_model(setup):
    settings,_,ml,_=setup
    settings.model_path.write_bytes(b'changed weights without manifest update')
    assert not readiness(settings)['ready']
    assert run_target_forecast('2026-01-31T23:00:00Z',settings=settings)['status']=='blocked'
    ml.load_model.assert_not_called()


def test_missing_target_config_never_uses_demo(tmp_path):
    settings=ForecastSettings(protocol_path=tmp_path/'missing',model_path=tmp_path/'missing-model',
                               results_dir=tmp_path/'out')
    result=run_target_forecast('2026-01-31T23:00:00Z',settings=settings)
    assert result['status']=='blocked' and result['rows']==[]
    assert 'substitution is forbidden' in result['note']


def test_invalid_origin_and_horizon(setup):
    settings,*_=setup
    for origin,horizon in [('2026-01-31',24),('2026-01-31T23:15:00Z',24),('2026-01-31T23:00:00Z',12)]:
        with pytest.raises(ForecastError):
            run_target_forecast(origin,horizon,settings=settings)


def test_inference_failure_can_retry_same_weather(setup):
    settings,data,ml,_=setup
    original=ml.predict.side_effect
    ml.predict.side_effect=RuntimeError('temporary failure')
    first=run_target_forecast('2026-01-31T23:00:00Z',settings=settings)
    assert first['status']=='blocked'
    ml.predict.side_effect=original
    recovered=run_target_forecast('2026-01-31T23:00:00Z',settings=settings)
    assert recovered['status']=='ok'


def test_source_target_cannot_be_in_feature_manifest(setup):
    settings,_,_,manifest=setup
    manifest['feature_columns'].append('actual_power_kw')
    settings.manifest_path.write_text(json.dumps(manifest))
    assert not readiness(settings)['ready']


def test_optional_unused_direction_does_not_invalidate_power(setup):
    settings,data,ml,_=setup
    original = data.get_forecast_weather.side_effect
    def weather(*args):
        frame = original(*args)
        frame["wind_direction_deg"] = np.nan
        return frame
    data.get_forecast_weather.side_effect = weather
    result = run_target_forecast("2026-01-31T23:00:00Z", settings=settings)
    assert result["status"] == "ok"
    assert all(row["complete"] for row in result["farm_rows"])


def test_provider_dependency_change_invalidates_numeric_cache(setup, tmp_path):
    settings,data,ml,_=setup
    package = tmp_path / "provider_code"
    package.mkdir()
    adapter = package / "provider.py"
    adapter.write_text("# test adapter\n")
    dependency = package / "features.py"
    dependency.write_text("version = 1\n")
    data.__file__ = str(adapter)
    first = run_target_forecast("2026-01-31T23:00:00Z", settings=settings)
    dependency.write_text("version = 2\n")
    second = run_target_forecast("2026-01-31T23:00:00Z", settings=settings)
    assert first["run_id"] != second["run_id"]
    assert ml.predict.call_count == 4
