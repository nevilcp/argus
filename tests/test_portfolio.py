from datetime import datetime

import pytest

from argus.agents.portfolio import (
    SYSTEM_PROMPT,
    PortfolioManagerAgent,
    build_signal_table,
    half_kelly_weight,
)
from argus.config import settings
from argus.governor import BOOTSTRAP_LIMITS, estimate_tokens
from argus.orchestration.state import TickerSnapshot
from argus.params import PORTFOLIO
from argus.schemas.signals import (
    MacroContext,
    Regime,
    RiskVerdict,
    Signal,
)
from argus.seams import FixtureLLMClient
from tests.helpers.builders import risk_assessment


def test_half_kelly_weight_bullish():
    res = half_kelly_weight(0.6, 0.08, 0.04, 0.15)
    assert res == 0.15


def test_half_kelly_weight_bearish():
    res = half_kelly_weight(0.3, 0.08, 0.04, 0.15)
    assert res == 0.0


def test_half_kelly_weight_moderate():
    res = half_kelly_weight(0.5, 0.08, 0.04, 0.15)
    assert pytest.approx(res) == 0.125


class MockSignal:
    """Minimal stand-in for a per-agent Signal result."""

    def __init__(self, signal, conviction, agents_present=None):
        self.signal = signal
        self.conviction = conviction
        self.agents_present = agents_present if agents_present is not None else []


def test_build_signal_table():
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
            risk=risk_assessment(
                RiskVerdict.APPROVE,
                0.15,
                proposed_weight=0.15,
                api_calls_used=0,
                var_99=0.02,
                stop_loss=150.0,
                portfolio_beta=1.1,
            ),
        ),
        "TSLA": TickerSnapshot(
            risk=risk_assessment(
                RiskVerdict.VETO,
                0.0,
                proposed_weight=0.15,
                api_calls_used=0,
                var_99=0.08,
                stop_loss=None,
                portfolio_beta=2.0,
                veto_reasons=["Too risky"],
            ),
        ),
        "MSFT": TickerSnapshot(
            fundamental=MockSignal(Signal.BEARISH, 0.9),
            risk=risk_assessment(
                RiskVerdict.REDUCE,
                0.05,
                proposed_weight=0.15,
                api_calls_used=0,
                var_99=0.05,
                stop_loss=280.0,
                portfolio_beta=0.9,
            ),
        ),
    }

    table = build_signal_table(snapshots, macro)

    assert "AAPL:" in table
    assert "MSFT:" in table
    assert "TSLA:" not in table
    assert "MACRO: EXPANSION | VIX 12.50 (LOW) | GROWTH_FAVORED" in table
    assert "FUND=BULLISH(0.80)" in table
    assert "TECH=NEUTRAL(0.50)" in table
    assert "FUND=BEARISH(0.90)" in table
    assert "TECH=N/A" in table
    assert "Cap=5.0%" in table
    assert "Evidence=3/3" in table
    assert "Evidence=0/3" in table


def test_full_universe_prompt_reservation_fits_the_tpm_floor():
    snapshots = {
        ticker: TickerSnapshot(
            fundamental=MockSignal(Signal.BULLISH, 0.8),
            technical=MockSignal(Signal.NEUTRAL, 0.5),
            sentiment=MockSignal(Signal.BULLISH, 0.7),
            aggregated=MockSignal(Signal.BULLISH, 0.85, agents_present=["fundamental", "technical", "sentiment"]),
            risk=risk_assessment(
                RiskVerdict.APPROVE,
                0.05,
                proposed_weight=0.15,
                api_calls_used=0,
                var_99=0.02,
                stop_loss=150.0,
                portfolio_beta=1.1,
            ),
        )
        for ticker in settings.ARGUS_UNIVERSE
    }
    agent = PortfolioManagerAgent(llm_client=FixtureLLMClient({}))
    prompt = agent._build_allocation_prompt(
        100_000,
        0.9,
        0.9,
        {},
        build_signal_table(snapshots, None),
        dict.fromkeys(snapshots, 0.05),
        ["x" * 200] * 3,
        ["x" * 200] * 3,
    )

    reserved = estimate_tokens(SYSTEM_PROMPT, prompt, PORTFOLIO.llm_max_tokens)

    floor = BOOTSTRAP_LIMITS[settings.ARGUS_PORTFOLIO_MODEL]["tokens_per_minute"]
    assert reserved <= floor


def test_the_prompt_quotes_the_same_advisor_note_limit_the_schema_enforces():
    assert f"Maximum {PORTFOLIO.advisor_note_char_limit} characters" in SYSTEM_PROMPT
