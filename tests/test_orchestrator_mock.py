"""Sanity-тест: ForecastOrchestrator.chat() в MOCK_MODE не падает и не бьёт по OpenAI API."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["MOCK_MODE"] = "true"

from agents.orchestrator import DEMO_RESPONSE, ForecastOrchestrator


def test_chat_returns_demo_response_in_mock_mode():
    orchestrator = ForecastOrchestrator()
    result = orchestrator.chat("Построй прогноз на 24 часа", history=[])
    assert result == DEMO_RESPONSE


def test_chat_handles_empty_history():
    orchestrator = ForecastOrchestrator()
    result = orchestrator.chat("привет", history=None)
    assert isinstance(result, str) and result
