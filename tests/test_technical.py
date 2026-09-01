import math

import pytest
from pydantic import ValidationError

import argus.agents.technical as technical_module
from argus.agents.technical import TechnicalStatisticalAgent
from argus.schemas import signals as signals_module
from argus.schemas.signals import Signal


@pytest.fixture
def agent() -> TechnicalStatisticalAgent:
    """Returns a fresh TechnicalStatisticalAgent instance."""
    return TechnicalStatisticalAgent()


def _complete_state(**overrides: float) -> dict:
    """Returns a fully-populated session state dict, overridden per-field by kwargs."""
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
        "timestamp": "2024-01-02T10:00:00-05:00",
    }
    base.update(overrides)
    return base


def test_required_indicator_keys_is_the_shared_schema_contract():
    """technical.py's frozenset is the same object as the schema's, not a copy that can drift."""
    assert technical_module._REQUIRED_INDICATOR_KEYS is signals_module.SESSION_STATE_REQUIRED_KEYS


def test_bullish_signal(agent: TechnicalStatisticalAgent) -> None:
    """Oversold RSI, positive MACD, and strong momentum together yield a BULLISH signal."""
    session_state = _complete_state(
        rsi_14=28.0,
        macd_histogram=0.15,
        bb_percent_b=-0.2,
        adx_14=35.0,
        vwap_distance=0.008,
        volume_ratio=1.8,
        atr_pct=0.02,
        momentum_30m=0.012,
        momentum_1d=0.025,
    )
    result = agent.analyze("TEST", session_state)
    assert result.signal == Signal.BULLISH
    assert result.net_score > 0
    assert result.conviction > 0.5


def test_bearish_signal(agent: TechnicalStatisticalAgent) -> None:
    """Inverting every bullish indicator flips the signal to BEARISH."""
    session_state = _complete_state(
        rsi_14=72.0,
        macd_histogram=-0.15,
        # Mirrored about the 0.5 midline rather than negated: 0.5 + 0.7 = 1.2
        bb_percent_b=1.2,
        adx_14=35.0,
        vwap_distance=-0.008,
        volume_ratio=1.8,
        atr_pct=0.02,
        momentum_30m=-0.012,
        momentum_1d=-0.025,
    )
    result = agent.analyze("TEST", session_state)
    assert result.signal == Signal.BEARISH
    assert result.net_score < 0
    assert result.conviction > 0.5


def test_neutral_signal(agent: TechnicalStatisticalAgent) -> None:
    """All-flat indicators with a weak trend keep the net score within the neutral band."""
    session_state = _complete_state(adx_14=15.0)  # weak trend
    result = agent.analyze("TEST", session_state)
    assert result.signal == Signal.NEUTRAL
    assert abs(result.net_score) < 0.12


def test_zero_api_calls(agent: TechnicalStatisticalAgent) -> None:
    """The technical agent evaluates purely statistically, never calling an LLM."""
    result = agent.analyze("TEST", _complete_state())
    assert result.api_calls_used == 0


def test_batch_analyze_reports_dropped_tickers_in_errors(agent: TechnicalStatisticalAgent) -> None:
    """A ticker missing a required indicator is omitted from signals and named in errors."""
    complete_state = _complete_state()
    incomplete_state = {k: v for k, v in complete_state.items() if k != "adx_14"}

    signals, errors = agent.batch_analyze({"GOOD": complete_state, "BAD": incomplete_state})

    assert "GOOD" in signals
    assert "BAD" not in signals
    assert any("BAD" in e for e in errors)


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
