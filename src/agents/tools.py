"""Tool adapters between the agent orchestrator and deterministic services."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

try:
    from src.agents.pipeline import ALLOWED_HORIZONS, TURBINE_IDS, ForecastAgent
    from src.tools.weather_tool import HistoricalWeatherService
except ModuleNotFoundError:
    from agents.pipeline import ALLOWED_HORIZONS, TURBINE_IDS, ForecastAgent
    from tools.weather_tool import HistoricalWeatherService

MAX_FORECAST_HORIZON_HOURS = 48

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_tool",
            "description": (
                "Только для вопросов про погоду: почасовой архивный прогноз погоды "
                "по координатам одной турбины. Для прогноза мощности не используй."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turbine_id": {
                        "type": "integer",
                        "enum": [1, 2],
                        "description": "Идентификатор турбины.",
                    },
                    "target_date": {
                        "type": "string",
                        "description": (
                            "Время начала в ISO 8601. Если передана только дата, "
                            "используется 00:00 UTC."
                        ),
                    },
                    "horizon_hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 48,
                        "default": 48,
                    },
                },
                "required": ["turbine_id", "target_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_forecast_cycle",
            "description": (
                "Главный инструмент прогноза. Сам получает архивный прогноз погоды, "
                "готовит признаки, запускает модель, проверяет результат и сравнивает "
                "с прогнозом от предыдущей даты. Возвращает компактную сводку по каждой "
                "турбине. Прогноз от даты D покрывает часы с D+1 00:00 UTC."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_date": {
                        "type": "string",
                        "description": "Дата выпуска прогноза, YYYY-MM-DD.",
                    },
                    "horizon_hours": {"type": "integer", "enum": list(ALLOWED_HORIZONS)},
                    "turbine_ids": {
                        "type": "array",
                        "items": {"type": "integer", "enum": list(TURBINE_IDS)},
                        "description": "Турбины; по умолчанию обе.",
                    },
                },
                "required": ["issue_date", "horizon_hours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_weather_update",
            "description": (
                "Заново скачать погоду для уже построенного прогноза. Если входные "
                "данные изменились — пересчитать и вернуть, что изменилось."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "horizon_hours": {"type": "integer", "enum": list(ALLOWED_HORIZONS)},
                    "turbine_ids": {
                        "type": "array",
                        "items": {"type": "integer", "enum": list(TURBINE_IDS)},
                    },
                },
                "required": ["issue_date", "horizon_hours"],
            },
        },
    },
]


def _parse_turbine_id(turbine_id: int | str) -> int:
    """Validate a turbine ID without coercing arbitrary user input."""
    if isinstance(turbine_id, bool):
        raise TypeError("turbine_id must be an integer")
    try:
        parsed = int(turbine_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("turbine_id must be 1 or 2") from exc
    if parsed not in (1, 2):
        raise ValueError("turbine_id must be 1 or 2")
    return parsed


def _parse_target_timestamp(target_date: str) -> pd.Timestamp:
    """Parse a requested UTC forecast start timestamp."""
    timestamp = pd.to_datetime(target_date, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise ValueError("target_date must be an ISO 8601 date or timestamp")
    return pd.Timestamp(timestamp)


def get_weather_tool(
    turbine_id: int | str,
    target_date: str,
    horizon_hours: int = MAX_FORECAST_HORIZON_HOURS,
) -> dict[str, Any]:
    """Get hourly weather records for one historical forecast horizon."""
    turbine = _parse_turbine_id(turbine_id)
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int):
        raise TypeError("horizon_hours must be an integer from 1 to 48")
    if not 1 <= horizon_hours <= MAX_FORECAST_HORIZON_HOURS:
        raise ValueError("horizon_hours must be an integer from 1 to 48")

    start = _parse_target_timestamp(target_date)
    end = start + timedelta(hours=horizon_hours - 1)
    weather_service = HistoricalWeatherService()
    weather = weather_service.get_weather(turbine, start.date(), end.date())
    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="raise",
        utc=True,
    )
    selected = weather.loc[
        weather["timestamp"].between(start, end, inclusive="both")
    ].copy()
    if len(selected) != horizon_hours:
        raise RuntimeError(
            "Weather service did not return complete hourly coverage for "
            f"{start.isoformat()} through {end.isoformat()}"
        )

    selected["timestamp"] = selected["timestamp"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "turbine_id": turbine,
        "start_timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": horizon_hours,
        "weather_data": selected.to_dict(orient="records"),
    }


def _parse_turbine_ids(turbine_ids: list[int | str] | None) -> tuple[int, ...]:
    if turbine_ids is None:
        return TURBINE_IDS
    if not isinstance(turbine_ids, list) or not turbine_ids:
        raise TypeError("turbine_ids must be a non-empty list of 1 and/or 2")
    return tuple(sorted({_parse_turbine_id(turbine_id) for turbine_id in turbine_ids}))


def run_forecast_cycle(
    agent: ForecastAgent,
    issue_date: str,
    horizon_hours: int,
    turbine_ids: list[int | str] | None = None,
) -> dict[str, Any]:
    """Run the whole pipeline in Python and return only its compact summary."""
    result = agent.run(date.fromisoformat(issue_date), int(horizon_hours), _parse_turbine_ids(turbine_ids))
    return result.summary()


def check_weather_update(
    agent: ForecastAgent,
    issue_date: str,
    horizon_hours: int,
    turbine_ids: list[int | str] | None = None,
) -> dict[str, Any]:
    """Recompute only if the archived weather changed since the last run."""
    result = agent.check_update(date.fromisoformat(issue_date), int(horizon_hours), _parse_turbine_ids(turbine_ids))
    return result.summary()
