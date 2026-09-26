"""End-to-end ForecastAgent и LLM-цикла на фейковой модели и погоде (без сети и без LLM)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.agents.orchestrator import MAX_TOOL_ROUNDS, ForecastOrchestrator
from src.agents.pipeline import ForecastAgent
from src.tools.feature_tool import MODEL_FEATURE_COLUMNS
from src.tools.inference_tool import WindPowerPredictor
from src.tools.weather_tool import AS_OF_BASE_VARIABLES, MAX_LEAD_DAYS

STAGES = ["context", "weather", "features", "model", "validation"]


class WindScaledModel:
    """Модель-заглушка по контракту артефакта: мощность растёт с ветром на 100 м."""

    def predict(self, features):
        return np.asarray(features["wind_speed_100m"], dtype=float) / 15.0


class FakeWeatherService:
    def __init__(self, base_wind=8.0, drop_hours=0):
        self.base_wind = base_wind
        self.drop_hours = drop_hours
        self.calls = 0

    def get_weather(self, turbine_id, start_date, end_date, *, force_refresh=False):
        self.calls += 1
        timestamps = pd.date_range(
            pd.Timestamp(start_date, tz="UTC"),
            pd.Timestamp(end_date + timedelta(days=1), tz="UTC"),
            freq="1h",
            inclusive="left",
        )
        wind = self.base_wind + 3 * np.sin(np.arange(len(timestamps)) / 5) + turbine_id
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "wind_speed_10m": wind * 0.7,
            "wind_speed_80m": wind * 0.95,
            "wind_speed_100m": wind,
            "wind_direction_10m": 200.0,
            "wind_direction_100m": 210.0,
            "temperature_2m": -5.0,
            "turbine_id": turbine_id,
        })
        for lead_day in range(MAX_LEAD_DAYS + 1):
            suffix = "" if lead_day == 0 else f"_previous_day{lead_day}"
            for variable in AS_OF_BASE_VARIABLES:
                frame[f"{variable}{suffix}"] = frame[variable]
        target_start = pd.Timestamp(start_date, tz="UTC") + timedelta(days=1)
        missing = (frame["timestamp"] >= target_start) & (
            frame["timestamp"] < target_start + timedelta(hours=self.drop_hours)
        )
        return frame.loc[~missing].reset_index(drop=True)


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "wind_power_lgb.pkl"
    joblib.dump({
        "model": WindScaledModel(),
        "feature_names": MODEL_FEATURE_COLUMNS.copy(),
        "objective": "huber",
        "validation_metrics": {"mae": 0.25, "rmse": 0.3},
    }, path)
    return path


def make_agent(model_path, tmp_path, **weather_kwargs):
    return ForecastAgent(
        FakeWeatherService(**weather_kwargs),
        WindPowerPredictor(model_path),
        output_dir=tmp_path / "forecasts",
    )


@pytest.fixture
def agent(model_path, tmp_path):
    return make_agent(model_path, tmp_path)


@pytest.mark.parametrize("turbine_ids", [(1,), (2,), (1, 2)])
@pytest.mark.parametrize("horizon", [24, 48])
def test_cycle_for_each_turbine_and_horizon(agent, tmp_path, turbine_ids, horizon):
    result = agent.run(date(2026, 1, 31), horizon, turbine_ids)

    assert result.status in {"ok", "warn"}
    assert [s["stage"] for s in result.steps] == STAGES
    assert len(result.forecast) == horizon * len(turbine_ids)
    assert set(result.forecast["turbine_id"]) == set(turbine_ids)
    assert result.forecast["target_time"].min() == pd.Timestamp("2026-02-01", tz="UTC")
    assert result.forecast["lead_hours"].between(1, horizon).all()
    assert result.forecast["predicted_power"].between(0, 1).all()
    turbines = "".join(str(t) for t in turbine_ids)
    assert (tmp_path / "forecasts" / f"forecast_2026-01-31_t{turbines}_{horizon}h.csv").is_file()

    summary = result.summary()
    assert set(summary["turbines"]) == {f"turbine_{t}" for t in turbine_ids}
    assert set(summary["turbines"][f"turbine_{turbine_ids[0]}"]) >= {
        "mean_power", "peak_power", "peak_time_utc", "mean_wind_100m_ms",
    }
    serialized = json.dumps(summary, ensure_ascii=False, default=str)
    assert "weather_data" not in serialized and len(serialized) < 4000  # компактно для LLM


def test_shutdown_wind_zeroes_power_and_is_flagged(model_path, tmp_path):
    agent = make_agent(model_path, tmp_path, base_wind=27.0)
    result = agent.run(date(2026, 2, 5), 24, (1,))

    checks = {c["name"]: c["status"] for c in result.checks}
    assert result.status == "warn"
    assert checks["shutdown_wind"] == "warn"
    assert (result.forecast.loc[result.forecast["wind_speed_100m"] > 25, "predicted_power"] == 0).all()


def test_next_issue_is_compared_with_previous_issue(agent):
    agent.run(date(2026, 1, 31), 48)
    result = agent.run(date(2026, 2, 1), 48)

    assert result.changes["hours_compared"] == 48  # 2 февраля × 2 турбины
    assert result.changes["against"] == "прогноз от 2026-01-31"
    assert any(c["name"] == "divergence_from_previous_issue" for c in result.checks)


def test_previous_issue_combines_all_runs_of_that_day(agent, model_path, tmp_path):
    agent.run(date(2026, 1, 31), 24, (1,))  # покрывает только 1 февраля — общих часов нет
    agent.run(date(2026, 1, 31), 48, (2,))  # покрывает 2 февраля для турбины 2
    result = agent.run(date(2026, 2, 1), 48)
    assert result.changes["hours_compared"] == 24

    restarted = make_agent(model_path, tmp_path)  # после перезапуска — из CSV
    assert restarted.run(date(2026, 2, 1), 48).changes["hours_compared"] == 24


def test_check_update_recomputes_only_when_weather_changes(agent):
    first = agent.run(date(2026, 1, 31), 48)

    same = agent.check_update(date(2026, 1, 31), 48)
    assert same is first
    assert "не изменился" in same.steps[-1]["detail"]

    agent.weather_service.base_wind = 12.0
    updated = agent.check_update(date(2026, 1, 31), 48)
    assert updated is not first
    assert [s["stage"] for s in updated.steps] == ["weather_update", "context", "features", "model", "validation"]
    assert updated.changes["vs_previous_run"]["mean_abs_diff"] > 0


def test_missing_model_fails_before_weather(tmp_path):
    agent = ForecastAgent(FakeWeatherService(), WindPowerPredictor(tmp_path / "missing.pkl"), tmp_path)
    result = agent.run(date(2026, 1, 31), 48)

    assert result.status == "error"
    assert result.steps[-1]["stage"] == "context"
    assert "Model artifact is not available" in result.steps[-1]["detail"]
    assert agent.weather_service.calls == 0
    assert result.forecast is None


def test_incomplete_weather_fails_loud(model_path, tmp_path):
    agent = make_agent(model_path, tmp_path, drop_hours=3)
    result = agent.run(date(2026, 1, 31), 24, (2,))

    assert result.status == "error"
    assert result.steps[-1]["stage"] == "weather"
    assert "Incomplete as-of forecast" in result.steps[-1]["detail"]


def _message(content=None, tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": content, "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in tool_calls or []
        ]},
    )


def _tool_call(name, arguments):
    return SimpleNamespace(id=f"call_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])


def _orchestrator(agent, replies):
    orchestrator = ForecastOrchestrator(agent=agent)
    orchestrator.api_key = "test"
    orchestrator.mock_mode = False
    orchestrator.client = FakeClient(replies)
    return orchestrator


@pytest.mark.parametrize("turbine_ids", [[1], [2], [1, 2]])
def test_chat_forecast_goes_through_shared_agent(agent, turbine_ids):
    call = _tool_call("run_forecast_cycle", {"issue_date": "2026-01-31", "horizon_hours": 24,
                                             "turbine_ids": turbine_ids})
    orchestrator = _orchestrator(agent, [_message(tool_calls=[call]), _message(content="Готово")])

    assert orchestrator.chat("Прогноз от 31 января на сутки", []) == "Готово"
    tool_message = orchestrator.client.requests[1]["messages"][-1]
    payload = json.loads(tool_message["content"])
    assert tool_message["role"] == "tool"
    assert set(payload["turbines"]) == {f"turbine_{t}" for t in turbine_ids}
    assert "weather_data" not in tool_message["content"]
    assert agent.last_result.turbine_ids == tuple(turbine_ids)  # UI видит тот же результат


def test_chat_passes_forecast_context_without_recompute(agent):
    summary = agent.run(date(2026, 1, 31), 24).summary()
    calls_before = agent.weather_service.calls
    orchestrator = _orchestrator(agent, [_message(content="Пик у турбины 1 ...")])

    orchestrator.chat("Когда пик?", [], forecast_context=summary)
    system_messages = [m["content"] for m in orchestrator.client.requests[0]["messages"] if m["role"] == "system"]
    assert any("Последний построенный прогноз" in m and "turbine_1" in m for m in system_messages)
    assert agent.weather_service.calls == calls_before


def test_chat_reports_bad_arguments_to_llm(agent):
    call = _tool_call("run_forecast_cycle", {"issue_date": "31.01.2026", "horizon_hours": 24})
    orchestrator = _orchestrator(agent, [_message(tool_calls=[call]), _message(content="ok")])

    orchestrator.chat("прогноз", [])
    assert "error" in json.loads(orchestrator.client.requests[1]["messages"][-1]["content"])


def test_chat_stops_after_max_tool_rounds(agent):
    call = _tool_call("check_weather_update", {"issue_date": "2026-01-31", "horizon_hours": 48})
    orchestrator = _orchestrator(agent, [_message(tool_calls=[call])])

    answer = orchestrator.chat("зацикливайся", [])
    assert "Превышено" in answer
    assert len(orchestrator.client.requests) == MAX_TOOL_ROUNDS
