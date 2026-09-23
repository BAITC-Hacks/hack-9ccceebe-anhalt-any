from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from src.agent.observed import kelmarsh_turbines, run_observed_analysis
from src.agent.orchestrator import run_forecast
from src.agent.schemas import (ForecastError, ForecastRequest, ForecastResult,
                               ObservedRequest, ObservedResult)
from src.config import ROOT, Settings, Turbine, team_forecast_origin
from src.forecast.orchestrator import readiness, run_target_forecast

app = FastAPI(title="AI Energy Agent", version="0.1.0")


@app.get("/", include_in_schema=False)
def dashboard_redirect():
    return RedirectResponse(url="/ui/")


# Resolve assets from the package, so launching uvicorn from another directory works.
app.mount("/ui", StaticFiles(directory=ROOT / "src/app/web", html=True), name="ui")


@app.get("/health")
def health():
    return {"status": "ok"}


class DashboardConfig(BaseModel):
    turbines: dict[str, Turbine]
    observed_turbines: list[str]
    first_forecast_origin: AwareDatetime


@app.get("/dashboard/config", response_model=DashboardConfig)
def dashboard_config():
    try:
        registry = Settings.from_env().turbines()
        return {"turbines": {name: registry[name] for name in ("T1", "T2")},
                "observed_turbines": list(kelmarsh_turbines()),
                "first_forecast_origin": team_forecast_origin()}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail="Cannot read valid dashboard turbine configuration") from exc


class DemoForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_date: date
    horizon_h: int = Field(default=24, strict=True)

    @field_validator("horizon_h")
    @classmethod
    def demo_horizon(cls, value):
        if value not in (24, 48):
            raise ValueError("Dashboard demo horizon must be exactly 24 or 48 hours")
        return value


class DemoForecastResult(BaseModel):
    mode: Literal["demo"] = "demo"
    horizon_h: int
    results: dict[str, ForecastResult]


@app.post("/demo-forecast", response_model=DemoForecastResult)
def demo_forecast(request: DemoForecastRequest):
    # Demo is selected per request; it never changes production/target environment.
    settings = Settings.from_env(demo=True)
    try:
        results = {
            turbine_id: run_forecast(turbine_id=turbine_id,
                                     forecast_date=request.forecast_date,
                                     horizon_h=request.horizon_h,
                                     settings=settings, with_agent=False)
            for turbine_id in ("T1", "T2")
        }
        return {"mode": "demo", "horizon_h": request.horizon_h, "results": results}
    except ForecastError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class TargetForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_origin: AwareDatetime
    horizon_h: int = Field(default=48, strict=True)
    refresh: bool = False
    with_agent: bool = False

    @field_validator("horizon_h")
    @classmethod
    def target_horizon(cls, value):
        if value not in (24, 48):
            raise ValueError("Target forecast horizon must be exactly 24 or 48 hours")
        return value


@app.get("/readiness")
def target_readiness():
    return readiness()


@app.post("/target-forecast")
def target_forecast(request: TargetForecastRequest):
    try:
        return run_target_forecast(request.forecast_origin.isoformat(), request.horizon_h,
                                   refresh=request.refresh, with_agent=request.with_agent)
    except ForecastError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/forecast", response_model=ForecastResult)
def forecast(request: ForecastRequest, with_agent: bool = False):
    try:
        return run_forecast(**request.model_dump(), settings=Settings.from_env(), with_agent=with_agent)
    except ForecastError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/observed", response_model=ObservedResult)
def observed_analysis(request: ObservedRequest, with_agent: bool = False):
    try:
        return run_observed_analysis(**request.model_dump(), with_agent=with_agent)
    except ForecastError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
