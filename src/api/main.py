from fastapi import FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from src.agent.orchestrator import run_forecast
from src.agent.schemas import ForecastError, ForecastRequest, ForecastResult
from src.config import Settings
from src.forecast.orchestrator import readiness, run_target_forecast

app = FastAPI(title="AI Energy Agent", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


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


from src.agent.observed import run_observed_analysis
from src.agent.schemas import ObservedRequest, ObservedResult


@app.post("/observed", response_model=ObservedResult)
def observed_analysis(request: ObservedRequest, with_agent: bool = False):
    try:
        return run_observed_analysis(**request.model_dump(), with_agent=with_agent)
    except ForecastError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
