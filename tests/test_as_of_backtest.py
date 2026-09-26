"""Tests for leakage-safe weather selection and walk-forward backtesting."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from backtest_runner import run_backtest
from src.agents.pipeline import ForecastAgent, ForecastResult
from src.tools.feature_tool import RAW_WEATHER_COLUMNS
from src.tools.weather_tool import AS_OF_BASE_VARIABLES, get_forecast_as_of


class PreviousRunsWeather:
    """In-memory previous-runs cache with distinguishable lead-day values."""

    def get_weather(self, turbine_id, start_date, end_date, *, force_refresh=False):
        timestamps = pd.date_range(
            pd.Timestamp(start_date, tz="UTC"),
            pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1),
            freq="1h",
            inclusive="left",
        )
        data: dict[str, object] = {"timestamp": timestamps, "turbine_id": turbine_id}
        for lead_day in range(4):
            suffix = "" if lead_day == 0 else f"_previous_day{lead_day}"
            for index, variable in enumerate(AS_OF_BASE_VARIABLES):
                data[f"{variable}{suffix}"] = np.full(len(timestamps), lead_day * 100 + index)
        return pd.DataFrame(data)


def _as_of_weather(issue_time, horizon_hours):
    timestamps = pd.date_range(
        pd.Timestamp(issue_time) + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="1h",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "turbine_id": 1,
            "wind_speed_10m": 5.0,
            "wind_speed_80m": 8.0,
            "wind_speed_100m": 9.0,
            "wind_direction_10m": 180.0,
            "wind_direction_100m": 190.0,
            "temperature_2m": 1.0,
        }
    )


def test_as_of_forecast_uses_previous_day_column() -> None:
    forecast = get_forecast_as_of(
        1,
        "2026-01-31T12:00:00Z",
        60,
        service=PreviousRunsWeather(),
    )

    february_first = forecast.loc[forecast["timestamp"].dt.date == date(2026, 2, 1)]
    assert (february_first["wind_speed_10m"] == 100).all()
    assert (february_first["lead_days"] == 1).all()


def test_weather_network_failure_retries_offline_cache() -> None:
    calls: list[bool] = []

    def fetch(turbine_id, issue_time, horizon_hours, **kwargs):
        calls.append(kwargs["offline"])
        if not kwargs["offline"]:
            raise requests.ConnectionError("offline test")
        return _as_of_weather(issue_time, horizon_hours)

    agent = ForecastAgent()
    result = ForecastResult(date(2026, 1, 31), 24, (1,))
    with patch("src.agents.pipeline.get_forecast_as_of", side_effect=fetch):
        weather, _ = agent._fetch_weather(result)

    assert calls == [False, True]
    assert len(weather) == 24
    assert list(weather.columns) == ["timestamp", "turbine_id", *RAW_WEATHER_COLUMNS]
    assert result.steps[-1]["stage"] == "weather_fallback_cache"


class FakeBacktestAgent:
    def __init__(self) -> None:
        self.issue_dates: list[date] = []

    def run(self, issue_date, horizon_hours):
        self.issue_dates.append(issue_date)
        start = pd.Timestamp(issue_date + pd.Timedelta(days=1), tz="UTC")
        timestamps = pd.date_range(start, periods=horizon_hours, freq="1h")
        forecast = pd.DataFrame(
            {
                "target_time": np.repeat(timestamps, 2),
                "turbine_id": [1, 2] * horizon_hours,
                "predicted_power": [0.4, 0.6] * horizon_hours,
            }
        )
        return SimpleNamespace(
            status="ok",
            forecast=forecast,
            turbine_ids=(1, 2),
            issue_date=issue_date,
            steps=[{"stage": "weather", "status": "ok"}],
        )


def test_backtest_keeps_only_each_target_day() -> None:
    agent = FakeBacktestAgent()
    submission, runs = run_backtest("2026-02-01", "2026-02-02", agent=agent)

    assert agent.issue_dates == [date(2026, 1, 31), date(2026, 2, 1)]
    assert len(submission) == 96
    assert submission.columns.tolist() == [
        "timestamp",
        "turbine_id",
        "predicted_normalized_power",
    ]
    assert len(runs) == 2
