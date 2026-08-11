"""
Tests for the Technical Analysis Agent (argus/agents/technical.py).
"""

import pytest
from argus.agents.technical import TechnicalStatisticalAgent
from argus.schemas.signals import Signal


@pytest.fixture
def agent() -> TechnicalStatisticalAgent:
    """Returns:
        A fresh TechnicalStatisticalAgent instance.
    """
    return TechnicalStatisticalAgent()


def test_bullish_signal(agent: TechnicalStatisticalAgent) -> None:
    """Oversold RSI, positive MACD, and strong momentum together yield a BULLISH signal."""
    session_state = {
        "rsi_14": 28.0,
        "macd_histogram": 0.15,
        "bb_percent_b": -0.2,
        "adx_14": 35.0,
        "vwap_distance": 0.008,
        "volume_ratio": 1.8,
        "momentum_30m": 0.012,
        "momentum_1d": 0.025,
        "close": 100.0,
    }
    result = agent.analyze("TEST", session_state)
    assert result.signal == Signal.BULLISH
    assert result.net_score > 0
    assert result.conviction > 0.5


def test_bearish_signal(agent: TechnicalStatisticalAgent) -> None:
    """Inverting every bullish indicator flips the signal to BEARISH."""
    session_state = {
        "rsi_14": 72.0,  # Opposite of 28
        "macd_histogram": -0.15,  # Opposite of 0.15
        "bb_percent_b": 1.2,  # Opposite of -0.2 (midline is 0.5, so 0.5 + 0.7 = 1.2)
        "adx_14": 35.0,  # Trend strength
        "vwap_distance": -0.008,  # Opposite
        "volume_ratio": 1.8,  # High volume confirming the move
        "momentum_30m": -0.012,  # Opposite
        "momentum_1d": -0.025,  # Opposite
        "close": 100.0,
    }
    result = agent.analyze("TEST", session_state)
    assert result.signal == Signal.BEARISH
    assert result.net_score < 0
    assert result.conviction > 0.5


def test_neutral_signal(agent: TechnicalStatisticalAgent) -> None:
    """All-flat indicators with a weak trend keep the net score within the neutral band."""
    session_state = {
        "rsi_14": 50.0,
        "macd_histogram": 0.0,
        "bb_percent_b": 0.5,
        "adx_14": 15.0,  # Weak trend
        "vwap_distance": 0.0,
        "volume_ratio": 1.0,
        "momentum_30m": 0.0,
        "momentum_1d": 0.0,
        "close": 100.0,
    }
    result = agent.analyze("TEST", session_state)
    assert result.signal == Signal.NEUTRAL
    assert abs(result.net_score) < 0.12


def test_zero_api_calls(agent: TechnicalStatisticalAgent) -> None:
    """The technical agent evaluates purely statistically, never calling an LLM."""
    session_state = {
        "rsi_14": 50.0,
        "macd_histogram": 0.0,
        "bb_percent_b": 0.5,
        "adx_14": 20.0,
        "vwap_distance": 0.0,
        "volume_ratio": 1.0,
        "momentum_30m": 0.0,
        "momentum_1d": 0.0,
        "close": 100.0,
    }
    result = agent.analyze("TEST", session_state)
    assert result.api_calls_used == 0
