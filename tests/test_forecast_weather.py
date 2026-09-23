from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agent.schemas import ForecastError
from src.config import Turbine
from src.forecast.contracts import ForecastProtocol, ForecastSettings
from src.forecast.weather import load_weather


ORIGIN = pd.Timestamp("2026-01-31T18:00:00Z")


@pytest.fixture
def context(tmp_path):
    return (ForecastSettings(data_module="example.weather", cache_dir=tmp_path),
            Turbine(latitude=43.645150, longitude=78.535604, rated_power_kw=2500),
            ForecastProtocol(confirmed=True, timezone="Asia/Almaty", daily_origin_hour=23,
                             first_lead_hour=1, evidence="test organizer protocol"))


def weather(issue="2026-01-31T12:00:00Z", available="2026-01-31T14:00:00Z", wind=5.):
    frame = pd.DataFrame({"wind_speed_ms": wind, "temperature_c": 10., "wind_direction_deg": 180.,
                          "weather_issued_at": issue, "weather_available_at": available},
                         index=pd.date_range(ORIGIN + pd.Timedelta(1, unit="h"), periods=24, freq="h"))
    frame.attrs = {"source": "publisher archive", "kind": "archived_forecast", "wind_height_m": 80,
                   "availability_basis": "publisher_timestamp",
                   "availability_evidence": "Publisher revision timestamp field (test fixture)"}
    return frame


def run(context, frame=None, refresh=False, provider=None):
    settings, turbine, protocol = context
    return load_weather(settings, provider or SimpleNamespace(get_forecast_weather=lambda *args: frame),
                        "T1", turbine, ORIGIN, 24, protocol, refresh=refresh)


def test_latest_issue_and_revision_selected_as_a_single_vintage(context):
    old = weather(issue="2026-01-31T06:00:00Z", available="2026-01-31T16:00:00Z", wind=2.)
    initial = weather(wind=5.)
    revision = weather(available="2026-01-31T17:00:00Z", wind=8.)
    future = weather(available="2026-01-31T19:00:00Z", wind=90.)
    raw = pd.concat([future, old, revision, initial])
    frame, meta = run(context, raw)
    assert frame.wind_speed_ms.eq(8.).all()
    assert meta["weather_issued_at"] == "2026-01-31T12:00:00+00:00"
    assert meta["weather_available_at"] == "2026-01-31T17:00:00+00:00"
    assert frame.attrs["turbine_id"] == "T1"
    assert frame.attrs["wind_height_m"] == 80


@pytest.mark.parametrize("issue,available", [("2026-01-31T19:00:00Z", "2026-01-31T20:00:00Z"),
                                           ("2026-01-31T12:00:00Z", "2026-01-31T19:00:00Z")])
def test_future_issue_or_payload_revision_rejected(context, issue, available):
    with pytest.raises(ForecastError, match="No weather vintage"):
        run(context, weather(issue, available))


def test_availability_before_issue_rejected(context):
    with pytest.raises(ForecastError, match="before its issue"):
        run(context, weather(available="2026-01-31T11:00:00Z"))


@pytest.mark.parametrize("column", ["weather_issued_at", "weather_available_at"])
def test_naive_provenance_time_rejected(context, column):
    frame = weather()
    frame[column] = "2026-01-31T12:00:00"
    with pytest.raises(ForecastError, match="timezone-aware"):
        run(context, frame)


def test_naive_index_rejected(context):
    frame = weather()
    frame.index = frame.index.tz_localize(None)
    with pytest.raises(ForecastError, match="timezone-aware"):
        run(context, frame)


@pytest.mark.parametrize("field", ["source", "kind", "availability_basis", "availability_evidence", "wind_height_m"])
def test_missing_provenance_rejected(context, field):
    frame = weather()
    del frame.attrs[field]
    with pytest.raises(ForecastError):
        run(context, frame)


@pytest.mark.parametrize("kind", ["synthetic", "reanalysis", "observed_scada"])
def test_non_archived_weather_rejected(context, kind):
    frame = weather()
    frame.attrs["kind"] = kind
    with pytest.raises(ForecastError, match="archived_forecast"):
        run(context, frame)


def test_height_mismatch_rejected(context):
    frame = weather()
    frame.attrs["wind_height_m"] = 100
    with pytest.raises(ForecastError, match="hub height"):
        run(context, frame)


def test_duplicate_hours_within_vintage_rejected(context):
    frame = weather()
    with pytest.raises(ForecastError, match="duplicate valid_time"):
        run(context, pd.concat([frame, frame.iloc[:1]]))


def test_off_hour_valid_time_rejected(context):
    frame = weather()
    frame.index = frame.index + pd.Timedelta(30, unit="min")
    with pytest.raises(ForecastError, match="hourly boundaries"):
        run(context, frame)


def test_missing_hours_are_not_filled_from_older_vintage(context):
    old = weather(issue="2026-01-31T06:00:00Z", wind=2.)
    latest = weather(wind=9.).iloc[1:]
    frame, _ = run(context, pd.concat([old, latest]))
    assert len(frame) == 24
    assert frame.iloc[0].isna().all()
    assert frame.wind_speed_ms.iloc[1:].eq(9.).all()


def test_invalid_numeric_weather_stays_missing(context):
    frame = weather()
    frame.loc[frame.index[0], "wind_speed_ms"] = np.inf
    normalized, _ = run(context, frame)
    assert np.isnan(normalized.wind_speed_ms.iloc[0])


def test_empty_weather_rejected(context):
    with pytest.raises(ForecastError, match="no archived weather"):
        run(context, weather().iloc[:0])


def test_same_values_row_order_and_refresh_have_same_immutable_version(context):
    raw = weather()
    original = raw.copy(deep=True)
    first, meta1 = run(context, raw)
    path = Path(meta1["snapshot_path"])
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    second, meta2 = run(context, raw.iloc[::-1], refresh=True)
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(first, second)
    assert meta1["weather_version"] == meta2["weather_version"]
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime


def test_cache_first_does_not_call_provider(context):
    _, saved = run(context, weather())
    def fail(*args):
        raise AssertionError("provider should not be called")
    cached, meta = run(context, provider=SimpleNamespace(get_forecast_weather=fail))
    assert meta["cache_hit"]
    assert saved["weather_version"] == meta["weather_version"]
    assert cached.wind_speed_ms.eq(5.).all()


def test_refresh_new_eligible_revision_preserves_previous_snapshot(context):
    _, old = run(context, weather())
    frame, new = run(context, weather(available="2026-01-31T17:00:00Z", wind=7.), refresh=True)
    assert frame.wind_speed_ms.eq(7.).all()
    assert old["weather_version"] != new["weather_version"]
    assert Path(old["snapshot_path"]).is_file()
    _, cached = run(context)
    assert cached["weather_version"] == new["weather_version"]


def test_refresh_failure_does_not_fall_back_or_expose_credentials(context):
    _, saved = run(context, weather())
    def fail(*args):
        raise RuntimeError("secret token")
    with pytest.raises(ForecastError, match="no stale-cache fallback") as error:
        run(context, refresh=True, provider=SimpleNamespace(get_forecast_weather=fail))
    assert "secret token" not in str(error.value)
    _, cached = run(context)
    assert saved["weather_version"] == cached["weather_version"]


def test_corrupted_snapshot_rejected(context):
    _, meta = run(context, weather())
    path = Path(meta["snapshot_path"])
    path.write_text(path.read_text().replace("5.0", "6.0"))
    with pytest.raises(ForecastError, match="cache is invalid"):
        run(context, weather())


def test_request_identity_includes_protocol(context):
    _, first = run(context, weather())
    settings, turbine, protocol = context
    changed = protocol.model_copy(update={"evidence": "reconfirmed protocol"})
    _, second = run((settings, turbine, changed), weather())
    assert first["weather_version"] == second["weather_version"]
    assert first["snapshot_path"] != second["snapshot_path"]


@pytest.mark.parametrize("column", ["wind_speed_ms", "temperature_c", "weather_issued_at", "weather_available_at"])
def test_missing_required_column_rejected(context, column):
    with pytest.raises(ForecastError, match="requires wind_speed_ms"):
        run(context, weather().drop(columns=column))


def test_exact_origin_availability_is_allowed(context):
    _, meta = run(context, weather(available=ORIGIN.isoformat()))
    assert meta["weather_available_at"] == ORIGIN.isoformat()


def test_48_hour_horizon_reindexes_and_preserves_missing(context):
    settings, turbine, protocol = context
    raw = weather()
    frame, _ = load_weather(settings, SimpleNamespace(get_forecast_weather=lambda *args: raw),
                            "T1", turbine, ORIGIN, 48, protocol)
    assert len(frame) == 48
    assert frame.index[0] == ORIGIN + pd.Timedelta(1, unit="h")
    assert frame.index[-1] == ORIGIN + pd.Timedelta(48, unit="h")
    assert frame.iloc[24:].isna().all().all()


def test_provider_requires_new_forecast_interface(context):
    with pytest.raises(ForecastError, match="get_forecast_weather"):
        run(context, provider=SimpleNamespace(get_archival_weather=lambda *args: weather()))


def test_refresh_cannot_backdate_changes_to_known_payload_revision(context):
    _, original = run(context, weather())
    with pytest.raises(ForecastError, match="without a new payload availability"):
        run(context, weather(wind=8.), refresh=True)
    _, cached = run(context)
    assert cached["weather_version"] == original["weather_version"]


def test_refresh_cannot_regress_to_older_vintage(context):
    run(context, weather(available="2026-01-31T17:00:00Z", wind=8.))
    with pytest.raises(ForecastError, match="regressed"):
        run(context, weather(), refresh=True)


def test_refresh_cannot_silently_change_source(context):
    run(context, weather())
    new = weather(available="2026-01-31T17:00:00Z")
    new.attrs["source"] = "different publisher"
    with pytest.raises(ForecastError, match="source changed"):
        run(context, new, refresh=True)


def test_uncontracted_provider_features_and_metadata_are_not_cached(context):
    raw = weather()
    raw["future_actual_power"] = 2500.
    raw.attrs["unverified_detail"] = "not part of normalized weather contract"
    frame, meta = run(context, raw)
    assert list(frame.columns) == ["wind_speed_ms", "temperature_c", "wind_direction_deg"]
    assert "unverified_detail" not in frame.attrs
    assert "unverified_detail" not in Path(meta["snapshot_path"]).read_text()
