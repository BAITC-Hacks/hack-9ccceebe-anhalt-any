"""The team execution calendar covers February without certifying source semantics."""
import json
from datetime import date

import pandas as pd
import pytest

from src.config import ROOT
from src.agent.schemas import ForecastError
from src.forecast.contracts import ForecastProtocol, expected_times
from src.forecast.replay import _origins


@pytest.mark.parametrize('horizon', [24, 48])
def test_team_schedule_covers_every_february_hour(horizon):
    schedule = json.loads((ROOT / 'config/forecast_schedule.json').read_text())
    protocol = ForecastProtocol.model_validate_json((ROOT / 'config/forecast_protocol.json').read_text())
    assert (protocol.timezone, protocol.daily_origin_hour, protocol.first_lead_hour) == (
        schedule['timezone'], schedule['daily_origin_hour'], schedule['first_lead_hour'])
    origins = _origins(date.fromisoformat(schedule['origin_start_date']),
                       date.fromisoformat(schedule['origin_end_date']), protocol)
    assert len(origins) == 29
    assert origins[0] == pd.Timestamp(schedule['first_forecast_origin']) == pd.Timestamp(schedule['first_forecast_origin_utc'])
    start = pd.Timestamp(schedule['evaluation_start_inclusive']).tz_convert('UTC')
    end = pd.Timestamp(schedule['evaluation_end_exclusive']).tz_convert('UTC')
    emitted = pd.DatetimeIndex([t for origin in origins for t in expected_times(origin, horizon, protocol)])
    covered = emitted[(emitted >= start) & (emitted < end)].unique().sort_values()
    assert covered.equals(pd.date_range(start, end, freq='h', inclusive='left'))
    assert len(covered) == 672
    first = expected_times(origins[0], horizon, protocol)
    assert first[0] == start and first[-1] == start + pd.Timedelta(horizon-1, unit='h')
    contract = json.loads((ROOT / 'config/goldwind_dataset.template.json').read_text())
    assert pd.Timestamp(contract['first_forecast_origin']) == origins[0]
    assert contract['forecast_origin_source'] == schedule['forecast_origin_source']


def test_execution_schedule_does_not_certify_raw_source_metadata():
    protocol = ForecastProtocol.model_validate_json((ROOT / 'config/forecast_protocol.json').read_text())
    # Execution time can be fixed while raw time/power semantics are still unknown.
    unconfirmed = protocol.model_copy(update={'confirmed': False})
    with pytest.raises(ForecastError, match='not confirmed'):
        unconfirmed.require_confirmed()
