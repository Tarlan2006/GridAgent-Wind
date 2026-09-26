"""HTTP API поверх общего ForecastAgent: фейковая модель и погода, без сети и LLM."""

from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient
from test_agent_cycle import (
    FakeClient,
    FakeWeatherService,
    WindScaledModel,
    _message,
    _tool_call,
)

from src.agents.orchestrator import DEMO_RESPONSE, ForecastOrchestrator
from src.agents.pipeline import ForecastAgent
from src.app import create_app
from src.tools.feature_tool import MODEL_FEATURE_COLUMNS
from src.tools.inference_tool import WindPowerPredictor


def make_client(tmp_path, *, model=True, llm_replies=None):
    model_path = tmp_path / "wind_power_lgb.pkl"
    if model:
        joblib.dump({"model": WindScaledModel(), "feature_names": MODEL_FEATURE_COLUMNS.copy(),
                     "objective": "huber", "validation_metrics": {"mae": 0.25, "rmse": 0.3}}, model_path)
    agent = ForecastAgent(FakeWeatherService(), WindPowerPredictor(model_path), output_dir=tmp_path / "out")
    orchestrator = ForecastOrchestrator(agent=agent)
    orchestrator.mock_mode = llm_replies is None
    orchestrator.api_key = "test"
    orchestrator.client = FakeClient(llm_replies) if llm_replies else None
    return TestClient(create_app(orchestrator)), agent


def test_index_serves_dispatcher_console(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "NEXCEL Wind AI" in response.text and "/api/forecast" in response.text


@pytest.mark.parametrize("turbine_ids", [[1], [2], [1, 2]])
@pytest.mark.parametrize("horizon", [24, 48])
def test_forecast_endpoint_returns_summary_steps_and_points(tmp_path, turbine_ids, horizon):
    client, _ = make_client(tmp_path)
    body = {"issue_date": "2026-01-31", "horizon_hours": horizon, "turbine_ids": turbine_ids}
    payload = client.post("/api/forecast", json=body).json()

    assert payload["summary"]["status"] in {"ok", "warn"}
    assert [s["stage"] for s in payload["steps"]] == ["context", "weather", "features", "model", "validation"]
    assert all("ms" in s for s in payload["steps"])
    assert len(payload["points"]) == horizon * len(turbine_ids)
    assert payload["points"][0]["target_time"] == "2026-02-01T00:00:00Z"
    assert set(payload["points"][0]) == {"target_time", "turbine_id", "lead_hours", "predicted_power",
                                         "wind_speed_100m", "temperature_2m"}


def test_update_endpoint_reports_unchanged_weather(tmp_path):
    client, _ = make_client(tmp_path)
    body = {"issue_date": "2026-01-31", "horizon_hours": 24, "turbine_ids": [1]}
    client.post("/api/forecast", json=body)
    steps = client.post("/api/update", json=body).json()["steps"]
    assert steps[-1]["stage"] == "weather_update" and "не изменился" in steps[-1]["detail"]


def test_missing_model_is_visible_in_payload_and_status(tmp_path):
    client, _ = make_client(tmp_path, model=False)
    payload = client.post("/api/forecast", json={"issue_date": "2026-01-31"}).json()
    assert payload["summary"]["status"] == "error"
    assert payload["steps"][-1]["stage"] == "context" and payload["points"] == []

    status = client.get("/api/status").json()
    assert status["model"]["available"] is False and "not available" in status["model"]["error"]


def test_invalid_request_is_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/api/forecast", json={"issue_date": "31.01.2026"}).status_code == 422
    assert client.post("/api/forecast", json={"issue_date": "2026-01-31", "horizon_hours": 12}).status_code == 422
    assert client.post("/api/forecast", json={"issue_date": "2026-01-31", "turbine_ids": [3]}).status_code == 422


def test_status_reports_model_and_llm(tmp_path):
    client, _ = make_client(tmp_path)
    status = client.get("/api/status").json()
    assert status["model"]["available"] is True
    assert status["model"]["validation_metrics"]["mae"] == 0.25
    assert status["llm_available"] is False


def test_chat_in_demo_mode_returns_demo_response(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/chat", json={"message": "привет"}).json()
    assert response == {"answer": DEMO_RESPONSE, "forecast": None}


def test_chat_forecast_is_returned_for_the_chart(tmp_path):
    call = _tool_call("run_forecast_cycle", {"issue_date": "2026-02-03", "horizon_hours": 24, "turbine_ids": [2]})
    client, _agent = make_client(
        tmp_path,
        llm_replies=[_message(tool_calls=[call]), _message(content="Готово")],
    )

    response = client.post("/api/chat", json={
        "message": "Прогноз турбины 2 на сутки от 3 февраля",
        "history": [{"role": "assistant", "content": "Здравствуйте"}],
        "forecast_context": None,
    }).json()

    assert response["answer"] == "Готово"
    assert response["forecast"]["summary"]["issue_date"] == "2026-02-03"
    assert {p["turbine_id"] for p in response["forecast"]["points"]} == {2}
    assert len(response["forecast"]["points"]) == 24
