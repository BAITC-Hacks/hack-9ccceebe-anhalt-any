"""Deterministic parser fixtures only. No organizer target, model fitting, or accuracy claim."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from src.ml.goldwind_data import GoldwindContract, prepare_hourly, read_organizer_files, training_rows


def contract(**changes):
    return GoldwindContract(**({
        "files": {"T1": "one.csv", "T2": "two.csv"},
        "columns": {"timestamp": "time", "wind_speed": "ws", "temperature": "temp", "target": "value"},
        "dataset_source": "ARTIFICIAL PARSER TEST", "metadata_source": "TEST ONLY",
        "file_identity_source": "TEST ONLY", "timezone": "UTC", "timestamp_label": "interval_start",
        "interval_minutes": 30, "measurement_semantics": "interval_mean", "wind_unit": "m/s",
        "temperature_unit": "C", "target_unit": "kW", "wind_height_m": {"T1": 80., "T2": 80.},
        "availability_source": "TEST release rule", "release_delay_minutes": 10.} | changes))


def raw(times=None, values=None):
    times = times or ["2025-01-01 00:00", "2025-01-01 00:30"]
    frame = pd.DataFrame({"time": times, "ws": [6., 8.], "temp": [10., 12.],
                          "value": values or [1000., 2000.]})
    return pd.concat([frame.assign(_organizer_turbine="T1"), frame.assign(_organizer_turbine="T2")], ignore_index=True)


def first_hour(hourly):
    return hourly.loc[(hourly.turbine_id == "T1") & (hourly.timestamp == pd.Timestamp("2025-01-01T00:00Z"))].iloc[0]


def test_hourly_mean_energy_and_full_coverage():
    original = raw()
    audit, hourly, summary = prepare_hourly(original, contract())
    row = first_hour(hourly)
    assert row.power == 1500 and row.energy_kwh_observed == 1500
    assert row.wind_speed == 7 and row.temperature == 11 and row.coverage == 1
    assert row.available_at == pd.Timestamp("2025-01-01T01:10Z")
    assert summary["complete_hours"] == 2
    pd.testing.assert_frame_equal(original, raw())
    assert audit.value.tolist() == original.value.tolist()


def test_reordered_source_index_preserves_turbine_identity():
    source = raw()
    source.loc[source._organizer_turbine.eq("T2"), "value"] = [300., 400.]
    # A caller may inspect/sort rows without resetting their original labels.
    reordered = source.iloc[[2, 3, 0, 1]]
    original = reordered.copy(deep=True)
    audit, hourly, _ = prepare_hourly(reordered, contract())
    assert audit.turbine_id.tolist() == ["T2", "T2", "T1", "T1"]
    assert audit.power_kw.tolist() == [300., 400., 1000., 2000.]
    complete = hourly.loc[hourly.complete].set_index("turbine_id")
    assert complete.loc["T1", "power"] == 1500.
    assert complete.loc["T2", "power"] == 350.
    pd.testing.assert_frame_equal(reordered, original)


def test_partial_hour_no_zero_fill_or_energy_extrapolation():
    data = raw()
    data.loc[1, "value"] = np.nan
    _, hourly, _ = prepare_hourly(data, contract())
    row = first_hour(hourly)
    assert row.power == 1000 and row.energy_kwh_observed == 500
    assert row.coverage == .5 and not row.complete
    missing = hourly.loc[hourly.source_rows.eq(0)]
    assert missing.power.isna().all() and missing.energy_kwh_observed.isna().all()


@pytest.mark.parametrize("unit,expected", [("fraction", 1250.), ("percent", 12.5), ("kW", .5), ("W", .0005), ("MW", 500.)])
def test_target_scale_explicit_never_inferred(unit, expected):
    c = contract(target_unit=unit, normalized_reference_kw={"T1": 2500., "T2": 2500.})
    _, hourly, _ = prepare_hourly(raw(values=[.5, .5]), c)
    assert first_hour(hourly).power == pytest.approx(expected)


def test_unknown_normalization_rejected_despite_known_nameplate():
    with pytest.raises(ValueError, match="reference"):
        contract(target_unit="fraction").validate()


def test_end_timestamp_and_declared_unit_conversion():
    data = raw(times=["2025-01-01 00:30", "2025-01-01 01:00"])
    data["ws"], data["temp"] = 36., 273.15
    _, hourly, _ = prepare_hourly(data, contract(timestamp_label="interval_end", wind_unit="km/h", temperature_unit="K"))
    row = first_hour(hourly)
    assert row.wind_speed == 10 and row.temperature == 0 and row.complete


def test_explicit_non_utc_timezone():
    data = raw(times=["2025-01-01 05:00", "2025-01-01 05:30"])
    _, hourly, _ = prepare_hourly(data, contract(timezone="Etc/GMT-5"))
    assert first_hour(hourly).complete


def test_ambiguous_date_order_never_guessed():
    data = raw(times=["01/01/2025 00:00", "01/01/2025 00:30"])
    with pytest.raises(ValueError, match="Non-ISO"):
        prepare_hourly(data, contract())
    _, hourly, _ = prepare_hourly(data, contract(timestamp_format="%d/%m/%Y %H:%M"))
    assert first_hour(hourly).complete


def test_duplicates_are_audited_and_conflicts_fail():
    data = raw()
    audit, hourly, summary = prepare_hourly(pd.concat([data, data.iloc[:1]], ignore_index=True), contract())
    assert len(audit) == 5 and summary["exact_duplicates_excluded"] == 1
    assert first_hour(hourly).power == 1500
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_hourly(pd.concat([data, data.iloc[:1].assign(value=999)], ignore_index=True), contract())


def test_negative_above_nominal_and_zero_preserved_not_shutdown_inferred():
    audit, _, report = prepare_hourly(raw(values=[-10., 3000.]), contract())
    assert audit.power_kw.tolist() == [-10., 3000., -10., 3000.]
    assert report["flags"]["negative_power"] == 2 and report["flags"]["above_rated_power"] == 2
    assert "does not establish" in report["shutdown_status"]


def test_forecast_origin_and_release_availability_gate():
    c = contract(first_forecast_origin="2025-01-01T01:05:00Z", forecast_origin_source="TEST C agreement")
    _, hourly, _ = prepare_hourly(raw(), c)
    with pytest.raises(ValueError, match="available"):
        training_rows(hourly, c)
    selected = training_rows(hourly, replace(c, first_forecast_origin="2025-01-01T01:10:00Z"))
    assert len(selected) == 2 and selected.available_at.max() <= pd.Timestamp(c.first_forecast_origin) + pd.Timedelta(5, unit="min")


def test_origin_absence_blocks_training_but_not_audit():
    _, hourly, _ = prepare_hourly(raw(), contract())
    with pytest.raises(ValueError, match="C must confirm"):
        training_rows(hourly, contract())


def test_february_excluded_even_if_files_include_it():
    data = pd.concat([raw(), raw(times=["2026-02-01 00:00", "2026-02-01 00:30"])], ignore_index=True)
    audit, hourly, report = prepare_hourly(data, contract())
    assert report["rows_outside_training_period"] == 4 and len(audit) == 8
    assert hourly.timestamp.max() < pd.Timestamp("2026-02-01T00:00Z")


@pytest.mark.parametrize("changes", [{"timezone": None}, {"target_unit": None}, {"timestamp_label": None},
    {"interval_minutes": 7}, {"measurement_semantics": "instantaneous"}, {"release_delay_minutes": None},
    {"minimum_hour_coverage": .9}, {"wind_height_m": None}])
def test_unknown_or_unsupported_metadata_fails(changes):
    with pytest.raises(ValueError):
        contract(**changes).validate()


def test_available_at_column_must_not_precede_interval_end():
    c = contract(columns=contract().columns | {"available_at": "release"}, release_delay_minutes=None)
    data = raw().assign(release="2024-01-01T00:00Z")
    with pytest.raises(ValueError, match="available before"):
        prepare_hourly(data, c)


def test_file_identity_validation(tmp_path):
    c = contract(columns=contract().columns | {"turbine_id": "id"})
    for file in c.files.values():
        raw().iloc[:2].assign(id="T2").to_csv(tmp_path / file, index=False)
    with pytest.raises(ValueError, match="identity"):
        read_organizer_files(c, tmp_path)


def test_cli_preserves_raw_and_blocks_unknown_template(tmp_path, monkeypatch, capsys):
    from scripts.prepare_goldwind import main
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prepare_goldwind", "--contract", str(cfg), "--output", str(tmp_path / "audit")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and "Confirm organizer metadata" in capsys.readouterr().err
    assert not (tmp_path / "audit").exists()


def test_cli_complete_preparation_no_training(tmp_path, monkeypatch):
    from scripts.prepare_goldwind import main
    c = contract()
    for file in c.files.values():
        raw().iloc[:2].drop(columns="_organizer_turbine").to_csv(tmp_path / file, index=False)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(asdict(c)), encoding="utf-8")
    out = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["prepare_goldwind", "--contract", str(cfg), "--output", str(out)])
    main()
    report = json.loads((out / "preparation_report.json").read_text())
    assert report["model_trained"] is False and "training_blocked" in report
    assert (out / "source_rows.csv").exists() and not (out / "eligible_hourly.csv").exists()
    with pytest.raises(SystemExit):
        main()  # No audit overwrite.
