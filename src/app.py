"""HTTP API + HTML dispatcher console over the shared ForecastAgent.

GET  /               -> src/web/index.html
GET  /api/status     -> model / LLM / weather source status
POST /api/forecast   -> run the full forecast cycle (no LLM)
POST /api/update     -> re-check archived weather, recompute if it changed (no LLM)
POST /api/chat       -> LLM orchestrator; returns the forecast too if the chat built one
"""

from __future__ import annotations

import logging
import threading
from datetime import date
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from src.agents.orchestrator import ForecastOrchestrator
    from src.agents.pipeline import TURBINE_IDS, ForecastResult
    from src.tools.inference_tool import ModelArtifactError
    from src.tools.weather_tool import MODEL_NAME
except ModuleNotFoundError:
    from agents.orchestrator import ForecastOrchestrator
    from agents.pipeline import TURBINE_IDS, ForecastResult
    from tools.inference_tool import ModelArtifactError
    from tools.weather_tool import MODEL_NAME

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INDEX_HTML = Path(__file__).with_name("web") / "index.html"
POINT_COLUMNS = ["target_time", "turbine_id", "lead_hours", "predicted_power", "wind_speed_100m", "temperature_2m"]


class ForecastRequest(BaseModel):
    issue_date: date
    horizon_hours: Literal[24, 48] = 48
    turbine_ids: list[Literal[1, 2]] = Field(default_factory=lambda: list(TURBINE_IDS), min_length=1)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    forecast_context: dict[str, Any] | None = None


def forecast_payload(result: ForecastResult) -> dict[str, Any]:
    """Summary (the same one the LLM sees) + full steps + hourly points for the chart."""
    points = []
    if result.forecast is not None:
        frame = result.forecast.loc[:, POINT_COLUMNS].copy()
        frame["target_time"] = frame["target_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        points = frame.to_dict(orient="records")
    return {"summary": result.summary(), "steps": result.steps, "points": points}


def create_app(orchestrator: ForecastOrchestrator) -> FastAPI:
    agent = orchestrator.agent  # one ForecastAgent for the buttons and the chat tools
    lock = threading.Lock()  # the agent keeps state; serialize runs
    api = FastAPI(title="NEXCEL Wind AI")

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_HTML)

    @api.get("/api/status")
    def status() -> dict[str, Any]:
        try:
            model: dict[str, Any] = {"available": True, **agent.predictor.metadata()}
            model["model_path"] = Path(model["model_path"]).name
        except ModelArtifactError as error:
            model = {"available": False, "error": str(error)}
        return {
            "model": model,
            "weather_source": f"{MODEL_NAME} · UTC",
            "llm_available": not orchestrator.mock_mode and orchestrator.client is not None,
        }

    @api.post("/api/forecast")
    def forecast(request: ForecastRequest) -> dict[str, Any]:
        with lock:
            result = agent.run(request.issue_date, request.horizon_hours, tuple(request.turbine_ids))
        return forecast_payload(result)

    @api.post("/api/update")
    def update(request: ForecastRequest) -> dict[str, Any]:
        with lock:
            result = agent.check_update(request.issue_date, request.horizon_hours, tuple(request.turbine_ids))
        return forecast_payload(result)

    @api.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        history = [message.model_dump() for message in request.history]
        with lock:
            before = agent.last_result
            try:
                answer = orchestrator.chat(request.message, history, request.forecast_context)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                LOGGER.exception("Unable to process chat message")
                raise HTTPException(status_code=500, detail=str(error)) from error
            result = agent.last_result
        built = result is not None and result is not before
        return {"answer": answer, "forecast": forecast_payload(result) if built else None}

    return api


app = create_app(ForecastOrchestrator())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
