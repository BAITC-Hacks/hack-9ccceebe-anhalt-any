from fastapi import FastAPI, HTTPException

from src.agent.orchestrator import run_forecast
from src.agent.schemas import ForecastError, ForecastRequest, ForecastResult
from src.config import Settings

app = FastAPI(title="AI Energy Agent", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResult)
def forecast(request: ForecastRequest, with_agent: bool = False):
    try:
        return run_forecast(**request.model_dump(), settings=Settings.from_env(), with_agent=with_agent)
    except ForecastError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
