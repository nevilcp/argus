"""
Tests for the Portfolio Manager Agent and its pure-Python helpers.
"""

from datetime import datetime

import pytest

from argus.agents.portfolio import half_kelly_weight, build_signal_table
from argus.orchestration.state import TickerSnapshot
from argus.schemas.signals import RiskAssessment, RiskVerdict, Signal, MacroContext, Regime

def test_half_kelly_weight_bullish():
    """A half-Kelly weight above the position cap clips to that cap."""
    # b=2, q=0.4: full kelly 0.4, half kelly 0.2, clipped to the 0.15 cap
    res = half_kelly_weight(0.6, 0.08, 0.04, 0.15)
    assert res == 0.15

def test_half_kelly_weight_bearish():
    """A negative half-Kelly weight floors at 0.0 rather than going short."""
    # b=2, q=0.7: full kelly is -0.05, so half kelly is negative
    res = half_kelly_weight(0.3, 0.08, 0.04, 0.15)
    assert res == 0.0

def test_half_kelly_weight_moderate():
    """A half-Kelly weight within the cap passes through unclipped."""
    # b=2, q=0.5: full kelly 0.25 halves to 0.125, under the 0.15 cap
    res = half_kelly_weight(0.5, 0.08, 0.04, 0.15)
    assert pytest.approx(res) == 0.125

class MockSignal:
    """Minimal stand-in for a per-agent Signal result.

    Args:
        signal: Signal direction.
        conviction: Confidence score for the signal.
    """

    def __init__(self, signal, conviction, agents_present=None):
        self.signal = signal
        self.conviction = conviction
        self.agents_present = agents_present if agents_present is not None else []

def test_build_signal_table():
    """Vetoed positions are excluded from the table; others render with their approved weight."""
    macro = MacroContext(
        fed_funds=5.25,
        cpi_yoy=3.2,
        unemployment=3.8,
        t10y2y=-0.5,
        consumer_sentiment=70.0,
        vix_level=12.5,
        macro_regime=Regime.EXPANSION,
        regime_confidence=0.8,
        model_healthy=True,
        interest_rate_trend="STABLE",
        yield_curve_shape="INVERTED",
        vix_regime="LOW",
        vix_percentile=15.0,
        inflation_trajectory="FALLING",
        sector_rotation_signal="GROWTH_FAVORED",
        agent_multipliers={"fundamental": 1.0, "technical": 1.0, "sentiment": 1.0},
        timestamp=datetime.now(),
    )

    snapshots = {
        "AAPL": TickerSnapshot(
            fundamental=MockSignal(Signal.BULLISH, 0.8),
            technical=MockSignal(Signal.NEUTRAL, 0.5),
            sentiment=MockSignal(Signal.BULLISH, 0.7),
            aggregated=MockSignal(Signal.BULLISH, 0.85, agents_present=["fundamental", "technical", "sentiment"]),
            risk=RiskAssessment(verdict=RiskVerdict.APPROVE, proposed_weight=0.15, approved_weight=0.15, var_99=0.02, stop_loss=150.0, portfolio_beta=1.1, api_calls_used=0, timestamp=datetime.now()),
        ),
        "TSLA": TickerSnapshot(
            risk=RiskAssessment(verdict=RiskVerdict.VETO, proposed_weight=0.15, approved_weight=0.0, var_99=0.08, stop_loss=None, portfolio_beta=2.0, veto_reasons=["Too risky"], api_calls_used=0, timestamp=datetime.now()),
        ),
        "MSFT": TickerSnapshot(
            fundamental=MockSignal(Signal.BEARISH, 0.9),
            risk=RiskAssessment(verdict=RiskVerdict.REDUCE, proposed_weight=0.15, approved_weight=0.05, var_99=0.05, stop_loss=280.0, portfolio_beta=0.9, api_calls_used=0, timestamp=datetime.now()),
        ),
    }

    table = build_signal_table(snapshots, macro)

    assert "AAPL:" in table
    assert "MSFT:" in table
    assert "TSLA:" not in table
    # .value, not str(Enum) — py>=3.11 would otherwise leak the class name into the prompt
    assert "MACRO: EXPANSION | VIX 12.50 (LOW) | GROWTH_FAVORED" in table
    assert "FUND=BULLISH(0.80)" in table
    assert "TECH=NEUTRAL(0.50)" in table
    assert "FUND=BEARISH(0.90)" in table
    assert "TECH=N/A" in table # MSFT missing tech signal
    assert "Cap=5.0%" in table # MSFT reduced weight
    assert "Evidence=3/3" in table # AAPL has all three specialists present
    assert "Evidence=0/3" in table # MSFT has no aggregated signal
