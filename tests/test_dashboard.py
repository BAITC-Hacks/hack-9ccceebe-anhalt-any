"""Dashboard API checks; synthetic results never certify target forecast readiness."""
import json
import socket
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.config import Settings
from src.forecast.contracts import ForecastSettings


def test_dashboard_redirect_and_assets():
    client = TestClient(main.app)
    redirect = client.get("/", follow_redirects=False)
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "/ui/"
    for path, media_type in (("/ui/", "text/html"), ("/ui/ui.css", "text/css"),
                             ("/ui/app.js", "javascript")):
        response = client.get(path)
        assert response.status_code == 200
        assert media_type in response.headers["content-type"]
        assert response.content
    assert client.get("/ui/%2e%2e/%2e%2e/config/turbines.json").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}


def test_dashboard_config_returns_only_public_registry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-must-not-be-published")
    response = TestClient(main.app).get("/dashboard/config")
    assert response.status_code == 200
    config = response.json()
    assert set(config) == {"turbines", "observed_turbines", "first_forecast_origin"}
    assert config["first_forecast_origin"] == "2026-01-31T23:00:00+05:00"
    assert set(config["turbines"]) == {"T1", "T2"}
    assert config["turbines"]["T1"]["rated_power_kw"] == 2500
    assert config["observed_turbines"] == [f"Kelmarsh {i}" for i in range(1, 7)]
    assert "test-secret" not in json.dumps(config)


@pytest.mark.parametrize("horizon", [24, 48])
def test_explicit_demo_is_local_and_does_not_enable_target(monkeypatch, tmp_path, horizon):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("DATA_MODULE", "external_data_must_not_load")
    monkeypatch.setenv("ML_MODULE", "external_ml_must_not_load")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-never-used")
    monkeypatch.setenv("OPENAI_MODEL", "configured-but-never-used")
    network = Mock(side_effect=AssertionError("Demo must stay local"))
    openai = Mock(side_effect=AssertionError("Demo must not call an agent"))
    monkeypatch.setattr(socket.socket, "connect", network)
    monkeypatch.setattr("src.agent.analyzer.OpenAI", openai)
    target_settings = ForecastSettings(model_path=tmp_path / "missing-model",
                                       protocol_path=tmp_path / "missing-protocol",
                                       results_dir=tmp_path / "results")
    monkeypatch.setattr(ForecastSettings, "from_env", classmethod(lambda cls: target_settings))
    client = TestClient(main.app)
    before = client.get("/readiness").json()
    assert before["ready"] is False

    response = client.post("/demo-forecast", json={"forecast_date": "2026-02-01", "horizon_h": horizon})
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "demo" and payload["horizon_h"] == horizon
    assert set(payload["results"]) == {"T1", "T2"}
    for turbine, result in payload["results"].items():
        assert result["request"]["turbine_id"] == turbine
        assert result["demo"] is True
        assert result["weather_kind"] == "synthetic"
        assert result["analysis_source"] == "deterministic"
        assert result["analysis_reason"] == "agent_disabled"
        assert len(result["hourly"]) == horizon

    assert Settings.from_env().demo is False
    assert client.get("/readiness").json() == before
    target = client.post("/target-forecast", json={
        "forecast_origin": "2026-01-31T23:00:00Z", "horizon_h": horizon}).json()
    assert target["status"] == "blocked"
    assert target["rows"] == [] and target["farm_rows"] == []
    assert "substitution is forbidden" in target["note"]
    network.assert_not_called()
    openai.assert_not_called()


@pytest.mark.parametrize("payload", [
    {"forecast_date": "2026-02-01", "horizon_h": value} for value in [1, 72, 24.0, "24", True]
] + [
    {"forecast_date": "bad-date", "horizon_h": 24},
    {"forecast_date": "2026-02-01", "horizon_h": 24, "with_agent": True},
])
def test_demo_rejects_invalid_contract_before_inference(monkeypatch, payload):
    run = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(main, "run_forecast", run)
    assert TestClient(main.app).post("/demo-forecast", json=payload).status_code == 422
    run.assert_not_called()


def test_demo_unsupported_date_returns_error_without_fallback(monkeypatch):
    target = Mock(side_effect=AssertionError("Demo must not substitute target data"))
    observed = Mock(side_effect=AssertionError("Demo must not substitute observed data"))
    monkeypatch.setattr(main, "run_target_forecast", target)
    monkeypatch.setattr(main, "run_observed_analysis", observed)
    response = TestClient(main.app).post("/demo-forecast", json={
        "forecast_date": "2030-02-01", "horizon_h": 24})
    assert response.status_code == 422
    assert "Demo dates supported" in response.json()["detail"]
    target.assert_not_called()
    observed.assert_not_called()
