"""
tests/test_position_sizing.py

Regression tests for PA-2, PA-3, PA-4: the Half-Kelly anchor must come from a
measured win rate (not direction-blind conviction) and must be omitted when no
such measurement exists, and the capital base for allocation_pct/allocation_usd
must be total wealth rather than wealth pre-scaled by invest_pct.
"""

import json
from datetime import datetime

from argus.agents.portfolio import PortfolioManagerAgent
from argus.orchestration.state import TickerSnapshot
from argus.schemas.signals import AggregatedSignal, RiskAssessment, RiskVerdict, Signal
from argus.seams import FixtureLLMClient

TIMESTAMP = datetime(2026, 1, 1)


def _risk(approved_weight: float = 0.15) -> RiskAssessment:
    return RiskAssessment(
        verdict=RiskVerdict.APPROVE,
        approved_weight=approved_weight,
        proposed_weight=approved_weight,
        var_99=0.05,
        portfolio_beta=1.0,
        timestamp=TIMESTAMP,
    )


def _agg(
    ticker: str,
    signal: Signal,
    weighted_votes: dict[str, float],
    reliability: dict[str, float] | None = None,
    reliability_n: dict[str, int] | None = None,
) -> AggregatedSignal:
    return AggregatedSignal(
        ticker=ticker,
        signal=signal,
        conviction=0.6,
        weighted_votes=weighted_votes,
        agents_present=list(weighted_votes),
        reliability=reliability or {},
        reliability_n=reliability_n or {},
    )


def _agent_with_capture(body: dict) -> tuple[PortfolioManagerAgent, list[str]]:
    """Builds a PortfolioManagerAgent whose FixtureLLMClient records the user prompt."""
    prompts: list[str] = []

    def key_fn(user_prompt: str) -> str:
        prompts.append(user_prompt)
        return "only"

    llm_client = FixtureLLMClient({"only": json.dumps(body)}, key_fn=key_fn)
    return PortfolioManagerAgent(llm_client=llm_client), prompts


def test_bearish_ticker_receives_no_kelly_anchor():
    """A BEARISH ticker is never listed under HALF-KELLY ANCHORS, even with outcome data."""
    snapshots = {
        "BULL": TickerSnapshot(
            risk=_risk(),
            aggregated=_agg(
                "BULL",
                Signal.BULLISH,
                {"technical": 0.5},
                reliability={"technical": 0.65},
                reliability_n={"technical": 20},
            ),
        ),
        "BEAR": TickerSnapshot(
            risk=_risk(),
            aggregated=_agg(
                "BEAR",
                Signal.BEARISH,
                {"technical": 0.5},
                reliability={"technical": 0.65},
                reliability_n={"technical": 20},
            ),
        ),
    }
    body = {
        "portfolio": [],
        "cash_reserve_pct": 1.0,
        "expected_sharpe": None,
        "rebalance_trigger": "MONTHLY",
    }
    agent, prompts = _agent_with_capture(body)
    agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.5, "risk_tolerance": "MODERATE"},
        snapshots=snapshots,
        macro=None,
    )

    assert len(prompts) == 1
    assert "BULL: half-Kelly suggests" in prompts[0]
    assert "BEAR: half-Kelly suggests" not in prompts[0]


def test_kelly_anchor_block_omitted_when_no_agent_has_data():
    """The whole HALF-KELLY ANCHORS block is omitted, with an explanation, when n == 0 for every agent."""
    snapshots = {
        "AAPL": TickerSnapshot(
            risk=_risk(),
            aggregated=_agg(
                "AAPL",
                Signal.BULLISH,
                {"technical": 0.5},
                reliability={"technical": 0.5, "fundamental": 0.5, "sentiment": 0.5},
                reliability_n={"technical": 0, "fundamental": 0, "sentiment": 0},
            ),
        ),
    }
    body = {
        "portfolio": [],
        "cash_reserve_pct": 1.0,
        "expected_sharpe": None,
        "rebalance_trigger": "MONTHLY",
    }
    agent, prompts = _agent_with_capture(body)
    agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.5, "risk_tolerance": "MODERATE"},
        snapshots=snapshots,
        macro=None,
    )

    assert len(prompts) == 1
    assert "HALF-KELLY ANCHORS: omitted" in prompts[0]
    assert "half-Kelly suggests" not in prompts[0]


def test_kelly_anchor_uses_primary_drivers_measured_win_rate_not_conviction():
    """The anchor is derived from the argmax(weighted_votes) agent's reliability rate."""
    snapshots = {
        "AAPL": TickerSnapshot(
            risk=_risk(),
            aggregated=_agg(
                "AAPL",
                Signal.BULLISH,
                {"technical": 0.9, "fundamental": 0.1},
                reliability={"technical": 0.7, "fundamental": 0.5},
                reliability_n={"technical": 30, "fundamental": 0},
            ),
        ),
    }
    body = {
        "portfolio": [],
        "cash_reserve_pct": 1.0,
        "expected_sharpe": None,
        "rebalance_trigger": "MONTHLY",
    }
    agent, prompts = _agent_with_capture(body)
    agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.5, "risk_tolerance": "MODERATE"},
        snapshots=snapshots,
        macro=None,
    )

    # b=2, p=0.7, q=0.3: full Kelly (2*0.7-0.3)/2=0.55, half=0.275, clipped to the 0.15 cap
    assert "AAPL: half-Kelly suggests 15.0%" in prompts[0]


def test_capital_base_is_total_wealth_not_wealth_times_invest_pct():
    """user_investable_capital and allocation_usd are based on total_wealth, not total_wealth * invest_pct.

    Reproduces PA-2/PA-3: previously invest_pct was applied once to shrink the
    capital base and again by the LLM's deployment target, so requesting 80%
    deployment produced roughly invest_pct^2 (64%) of true total wealth.
    """
    snapshots = {
        "AAPL": TickerSnapshot(risk=_risk()),
    }
    body = {
        "portfolio": [
            {
                "ticker": "AAPL",
                "allocation_pct": 0.10,
                "stop_loss": 100.0,
                "thesis": "Bullish",
                "composite_conviction": 0.7,
                "time_horizon": "3-6 months",
            }
        ],
        "cash_reserve_pct": 0.9,
        "expected_sharpe": None,
        "rebalance_trigger": "MONTHLY",
    }
    agent, prompts = _agent_with_capture(body)
    alloc = agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.8, "risk_tolerance": "MODERATE"},
        snapshots=snapshots,
        macro=None,
    )

    assert alloc is not None
    assert alloc.user_investable_capital == 100_000.0
    assert alloc.portfolio[0].allocation_usd == 10_000.0
    assert "capital_usd: $100,000" in prompts[0]
