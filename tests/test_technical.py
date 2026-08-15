"""
Tests for the Technical Analysis Agent (argus/agents/technical.py).
"""

import math

import pytest
from pydantic import ValidationError

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
        "atr_pct": 0.02,
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
        "atr_pct": 0.02,
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
        "atr_pct": 0.01,
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
        "atr_pct": 0.01,
        "momentum_30m": 0.0,
        "momentum_1d": 0.0,
        "close": 100.0,
    }
    result = agent.analyze("TEST", session_state)
    assert result.api_calls_used == 0


def test_batch_analyze_reports_dropped_tickers_in_errors(agent: TechnicalStatisticalAgent) -> None:
    """A ticker missing a required indicator is omitted from signals and named in errors."""
    complete_state = {
        "rsi_14": 50.0,
        "macd_histogram": 0.0,
        "bb_percent_b": 0.5,
        "adx_14": 20.0,
        "vwap_distance": 0.0,
        "volume_ratio": 1.0,
        "atr_pct": 0.01,
        "momentum_30m": 0.0,
        "momentum_1d": 0.0,
        "close": 100.0,
    }
    incomplete_state = {k: v for k, v in complete_state.items() if k != "adx_14"}

    signals, errors = agent.batch_analyze({"GOOD": complete_state, "BAD": incomplete_state})

    assert "GOOD" in signals
    assert "BAD" not in signals
    assert any("BAD" in e for e in errors)


def _complete_state(**overrides: float) -> dict:
    """Returns:
        A fully-populated session state dict, overridden per-field by kwargs.
    """
    base = {
        "rsi_14": 50.0,
        "macd_histogram": 0.0,
        "bb_percent_b": 0.5,
        "adx_14": 20.0,
        "vwap_distance": 0.0,
        "volume_ratio": 1.0,
        "atr_pct": 0.01,
        "momentum_30m": 0.0,
        "momentum_1d": 0.0,
        "close": 100.0,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("bad_rsi", [-50.0, 150.0, 1e9])
def test_out_of_range_rsi_raises_instead_of_producing_a_signal(
    agent: TechnicalStatisticalAgent, bad_rsi: float
) -> None:
    """An RSI outside [0, 100] fails TechnicalSignal validation rather than scoring."""
    with pytest.raises(ValidationError):
        agent.analyze("TEST", _complete_state(rsi_14=bad_rsi))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_macd_histogram_is_rejected_at_the_missing_key_gate(
    agent: TechnicalStatisticalAgent, bad_value: float
) -> None:
    """A NaN/inf MACD histogram is skipped, the same as a missing required key."""
    result = agent.analyze("TEST", _complete_state(macd_histogram=bad_value))
    assert result is None


def test_absent_volume_ratio_skips_rather_than_applying_the_floor(
    agent: TechnicalStatisticalAgent,
) -> None:
    """A missing volume_ratio drops the ticker instead of silently applying the 0.7 floor."""
    state = _complete_state()
    del state["volume_ratio"]
    result = agent.analyze("TEST", state)
    assert result is None
