import json
from datetime import date, timedelta

import pandas as pd
import pytest

from src.agent.schemas import ForecastError
from src.forecast.contracts import ForecastProtocol, ForecastSettings, expected_times
from src.forecast import replay


@pytest.fixture
def setup(tmp_path):
    policy = ForecastProtocol(confirmed=True, timezone="Asia/Almaty", daily_origin_hour=0,
                              first_lead_hour=1, evidence="Test fixture only")
    path = tmp_path / "protocol.json"
    path.write_text(policy.model_dump_json())
    settings = ForecastSettings(protocol_path=path, model_path=tmp_path / "missing.joblib",
        manifest_path=tmp_path / "missing.json", results_dir=tmp_path / "runner", cache_dir=tmp_path / "cache")
    return settings, policy, tmp_path / "replay"


def result_for(origin, horizon, policy, *, negative=False):
    rows, farms = [], []
    run_id = "fixture-" + origin.isoformat()
    for stamp in expected_times(origin, horizon, policy):
        lead = int((stamp - origin).total_seconds() / 3600)
        for tid in ("T1", "T2"):
            power = -10.0 if negative and tid == "T1" else 100.0
            rows.append(dict(run_id=run_id, forecast_origin=origin.isoformat(),
                weather_issued_at=(origin.to_pydatetime()-timedelta(hours=6)).isoformat(), weather_available_at=(origin.to_pydatetime()-timedelta(hours=3)).isoformat(),
                valid_time=stamp.isoformat(), lead_hours=lead, turbine_id=tid, normalized_power=power/2500,
                power_kw=power, energy_kwh=power, model_version="fixture-model", weather_version="fixture-weather",
                flags=["negative"] if power < 0 else [], valid=power >= 0))
        farms.append(dict(run_id=run_id, forecast_origin=origin.isoformat(), valid_time=stamp.isoformat(), lead_hours=lead,
            raw_power_kw=90.0 if negative else 200.0, power_kw=None if negative else 200.0,
            energy_kwh=None if negative else 200.0, complete=not negative, missing_turbines=[],
            incomplete_turbines=["T1"] if negative else [], flags=["negative"] if negative else [],
            model_version="fixture-model", weather_versions={"T1": "fixture-weather", "T2": "fixture-weather"}))
    return {"run_id": run_id, "status": "partial" if negative else "ok", "rows": rows, "farm_rows": farms, "errors": {}}


def actuals_file(path, rows):
    pd.DataFrame(rows, columns=["timestamp", "turbine_id", "actual_power_kw"]).to_csv(path, index=False)
    return path


def test_full_replay_preserves_origins_overlap_and_local_february(setup, monkeypatch):
    settings, policy, output = setup
    calls = []
    def runner(origin, horizon, **kwargs):
        calls.append((origin, kwargs))
        return result_for(origin, horizon, policy)
    monkeypatch.setattr(replay, "run_target_forecast", runner)
    summary = replay.run_february_replay(output, settings=settings, horizon_h=48)
    assert summary["status"] == "ok"
    assert len(calls) == 29
    assert calls[0][0] == pd.Timestamp("2026-01-30T19:00:00Z")
    assert calls[-1][0] == pd.Timestamp("2026-02-27T19:00:00Z")
    assert summary["coverage"]["all_emitted_forecast_rows"] == 29 * 48 * 2
    assert summary["coverage"]["expected_forecast_rows"] == 2686
    assert summary["coverage"]["expected_unique_hours"] == 672
    assert summary["coverage"]["missing_forecast_rows"] == 0
    assert summary["metrics"] is None
    assert summary["configuration"]["evaluation_window"]["start_inclusive"] == "2026-01-31T19:00:00+00:00"
    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions.duplicated(["turbine_id", "valid_time"]).any()
    assert predictions.horizon_h.eq(48).all()
    assert not predictions.duplicated(["forecast_origin", "turbine_id", "valid_time"]).any()
    assert set(summary["files"]) == {"predictions.csv", "farm.csv"}
    assert all(len(info["sha256"]) == 64 for info in summary["files"].values())


def test_negative_raw_predictions_and_actuals_are_scored_including_farm(setup, monkeypatch, tmp_path):
    settings, policy, output = setup
    monkeypatch.setattr(replay, "run_target_forecast", lambda origin, horizon, **kwargs: result_for(origin, horizon, policy, negative=True))
    actuals = actuals_file(tmp_path / "actual.csv", [("2026-02-01T01:00:00+05:00", "T1", -5), ("2026-02-01T01:00:00+05:00", "T2", 90)])
    summary = replay.run_february_replay(output, settings=settings, horizon_h=24, actuals_path=actuals,
        start=date(2026, 2, 1), end=date(2026, 2, 1))
    assert summary["status"] == "partial"
    assert summary["coverage"]["finite_raw_forecast_rows"] == 48
    assert summary["coverage"]["physically_valid_forecast_rows"] == 24
    assert summary["coverage"]["finite_raw_farm_forecast_rows"] == 24
    assert summary["coverage"]["physically_complete_farm_rows"] == 0
    assert summary["metrics"]["per_turbine"]["T1"]["mae_kw"] == 5
    assert summary["metrics"]["overall"]["mae_kw"] == 7.5
    assert summary["metrics"]["farm"]["mae_kw"] == 5
    assert summary["metrics"]["farm"]["matched_forecast_rows"] == 1


def test_actuals_match_many_vintages_without_selecting_best(setup, monkeypatch, tmp_path):
    settings, policy, output = setup
    def runner(origin, horizon, **kwargs):
        result = result_for(origin, horizon, policy)
        if origin == pd.Timestamp("2026-01-31T19:00:00Z"):
            for row in result["rows"]:
                row["power_kw"] = 120
            for row in result["farm_rows"]:
                row["raw_power_kw"] = row["power_kw"] = 240
        return result
    monkeypatch.setattr(replay, "run_target_forecast", runner)
    actuals = actuals_file(tmp_path / "actual.csv", [("2026-02-01T02:00:00+05:00", tid, 90) for tid in ("T1", "T2")])
    summary = replay.run_february_replay(output, settings=settings, actuals_path=actuals,
        start=date(2026, 1, 31), end=date(2026, 2, 1))
    assert summary["metrics"]["overall"]["matched_forecast_rows"] == 4
    assert summary["metrics"]["overall"]["mae_kw"] == 20
    assert summary["metrics"]["unique_turbine_hours_matched"] == 2
    assert summary["metrics"]["per_horizon_bucket"]["first_24h"]["mae_kw"] == 30
    assert summary["metrics"]["per_horizon_bucket"]["second_24h"]["mae_kw"] == 10


@pytest.mark.parametrize("rows,match", [
    ([("2026-02-01 01:00:00", "T1", 1)], "timezone-aware"),
    ([("2026-02-01T01:30:00Z", "T1", 1)], "hourly boundaries"),
    ([("2026-02-01T01:00:00Z", "T3", 1)], "T1 or T2"),
    ([("2026-02-01T01:00:00Z", "T1", float("inf"))], "finite"),
    ([("2026-02-01T01:00:00Z", "T1", 1), ("2026-02-01T06:00:00+05:00", "T1", 1)], "Duplicate"),
])
def test_bad_actuals_rejected(tmp_path, rows, match):
    path = actuals_file(tmp_path / "actual.csv", rows)
    with pytest.raises(ForecastError, match=match):
        replay.load_actuals(path)


def test_missing_actual_turbine_never_becomes_zero_farm_actual(setup, monkeypatch, tmp_path):
    settings, policy, output = setup
    monkeypatch.setattr(replay, "run_target_forecast", lambda origin, horizon, **kwargs: result_for(origin, horizon, policy))
    actuals = actuals_file(tmp_path / "actual.csv", [("2026-02-01T01:00:00+05:00", "T1", 90)])
    summary = replay.run_february_replay(output, settings=settings, actuals_path=actuals,
        start=date(2026, 2, 1), end=date(2026, 2, 1))
    assert summary["metrics"]["farm"]["matched_forecast_rows"] == 0
    assert summary["metrics"]["farm"]["mae_kw"] is None


def test_blocked_protocol_emits_no_fake_forecasts(setup, monkeypatch):
    settings, policy, output = setup
    policy.confirmed = False
    settings.protocol_path.write_text(policy.model_dump_json())
    monkeypatch.setattr(replay, "run_target_forecast", lambda *args, **kwargs: pytest.fail("Runner must not run without confirmed policy"))
    summary = replay.run_february_replay(output, settings=settings)
    assert summary["status"] == "blocked"
    assert summary["runs"] == []
    assert pd.read_csv(output / "predictions.csv").empty
    assert pd.read_csv(output / "farm.csv").empty
    assert json.loads((output / "manifest.json").read_text())["metrics"] is None


def test_runner_failure_does_not_stop_next_origin(setup, monkeypatch):
    settings, policy, output = setup
    calls = []
    def runner(origin, horizon, **kwargs):
        calls.append(origin)
        if len(calls) == 1:
            raise ForecastError("Archive temporarily unavailable")
        return result_for(origin, horizon, policy)
    monkeypatch.setattr(replay, "run_target_forecast", runner)
    summary = replay.run_february_replay(output, settings=settings,
        start=date(2026, 2, 1), end=date(2026, 2, 2))
    assert len(calls) == 2
    assert summary["status"] == "partial"
    assert len(summary["failures"]) == 1
    assert summary["coverage"]["missing_forecast_rows"] == 96


def test_duplicate_predictions_are_contract_failure_and_not_scored(setup, monkeypatch, tmp_path):
    settings, policy, output = setup
    def runner(origin, horizon, **kwargs):
        result = result_for(origin, horizon, policy)
        result["rows"].append(result["rows"][0].copy())
        return result
    monkeypatch.setattr(replay, "run_target_forecast", runner)
    actuals = actuals_file(tmp_path / "actual.csv", [("2026-02-01T01:00:00+05:00", "T1", 90)])
    summary = replay.run_february_replay(output, settings=settings, actuals_path=actuals,
        start=date(2026, 2, 1), end=date(2026, 2, 1))
    assert summary["status"] == "blocked"
    assert summary["metrics"]["overall"]["matched_forecast_rows"] == 0
    assert pd.read_csv(output / "predictions.csv").empty
    assert len(summary["failures"]) == 1


@pytest.mark.parametrize("field,value", [
    ("valid_time", "2026-03-01T00:00:00Z"),
    ("valid_time", "2026-02-01 01:00:00"),
    ("lead_hours", 999),
    ("turbine_id", "Kelmarsh-1"),
    ("forecast_origin", "2026-02-02T00:00:00Z"),
])
def test_malformed_forecast_rows_are_never_written_or_scored(setup, monkeypatch, field, value):
    settings, policy, output = setup
    def runner(origin, horizon, **kwargs):
        result = result_for(origin, horizon, policy)
        result["rows"][0][field] = value
        return result
    monkeypatch.setattr(replay, "run_target_forecast", runner)
    summary = replay.run_february_replay(output, settings=settings,
        start=date(2026, 2, 1), end=date(2026, 2, 1))
    assert summary["status"] == "blocked"
    assert pd.read_csv(output / "predictions.csv").empty
    assert len(summary["failures"]) == 1


def test_unexpected_provider_exception_does_not_leak_credentials(setup, monkeypatch):
    settings, _, output = setup
    def runner(*args, **kwargs):
        raise RuntimeError("https://example.invalid/?api_key=private-test-key")
    monkeypatch.setattr(replay, "run_target_forecast", runner)
    summary = replay.run_february_replay(output, settings=settings,
        start=date(2026, 2, 1), end=date(2026, 2, 1))
    assert summary["status"] == "blocked"
    assert "private-test-key" not in (output / "manifest.json").read_text()


def test_cli_blocked_exit_code(setup, monkeypatch, capsys):
    from scripts import replay_february
    settings, policy, output = setup
    policy.confirmed = False
    settings.protocol_path.write_text(policy.model_dump_json())
    monkeypatch.setattr(replay_february.ForecastSettings, "from_env", lambda: settings)
    monkeypatch.setattr("sys.argv", ["replay_february.py", "--output-dir", str(output)])
    assert replay_february.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_origins_follow_local_calendar_across_dst():
    protocol = ForecastProtocol(confirmed=True, timezone="Europe/Berlin", daily_origin_hour=0,
                                evidence="Fixture only")
    origins = replay._origins(date(2026, 3, 28), date(2026, 3, 30), protocol)
    assert (origins[1] - origins[0]).total_seconds() == 24 * 3600
    assert (origins[2] - origins[1]).total_seconds() == 23 * 3600
    protocol.daily_origin_hour = 2
    with pytest.raises(ForecastError, match="nonexistent"):
        replay._origins(date(2026, 3, 29), date(2026, 3, 29), protocol)


def test_zero_lead_protocol_bucket_is_first_24_samples(setup, monkeypatch, tmp_path):
    settings, policy, output = setup
    policy.first_lead_hour = 0
    settings.protocol_path.write_text(policy.model_dump_json())
    monkeypatch.setattr(replay, "run_target_forecast", lambda origin, horizon, **kwargs: result_for(origin, horizon, policy))
    actuals = actuals_file(tmp_path / "actual.csv", [("2026-02-02T00:00:00+05:00", "T1", 90)])
    summary = replay.run_february_replay(output, settings=settings, actuals_path=actuals,
        start=date(2026, 2, 1), end=date(2026, 2, 1))
    assert summary["metrics"]["per_horizon_bucket"]["first_24h"]["matched_forecast_rows"] == 0
    assert summary["metrics"]["per_horizon_bucket"]["second_24h"]["matched_forecast_rows"] == 1
