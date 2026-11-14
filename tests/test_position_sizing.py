"""Regression tests for the Half-Kelly anchor and the allocation capital base."""

import json

from argus.agents.portfolio import PortfolioManagerAgent
from argus.schemas.signals import AggregatedSignal, Signal, TickerSnapshot
from argus.seams import FixtureLLMClient
from tests.helpers.builders import risk_assessment, user_profile

_ALL_CASH_BODY = {
    "portfolio": [],
    "cash_reserve_pct": 1.0,
    "expected_sharpe": None,
    "rebalance_trigger": "MONTHLY",
}


def _snapshot(
    ticker: str,
    signal: Signal,
    weighted_votes: dict[str, float],
    reliability: dict[str, float] | None = None,
    reliability_n: dict[str, int] | None = None,
) -> TickerSnapshot:
    """Build a risk-approved snapshot with the aggregate used by Kelly sizing."""
    return TickerSnapshot(
        risk=risk_assessment(),
        aggregated=AggregatedSignal(
            ticker=ticker,
            signal=signal,
            conviction=0.6,
            weighted_votes=weighted_votes,
            agents_present=list(weighted_votes),
            reliability=reliability or {},
            reliability_n=reliability_n or {},
        ),
    )


def _agent_with_capture(body: dict) -> tuple[PortfolioManagerAgent, list[str]]:
    """Build a PortfolioManagerAgent whose FixtureLLMClient records the user prompt."""
    prompts: list[str] = []

    def key_fn(user_prompt: str) -> str:
        prompts.append(user_prompt)
        return "only"

    llm_client = FixtureLLMClient({"only": json.dumps(body)}, key_fn=key_fn)
    return PortfolioManagerAgent(llm_client=llm_client), prompts


def test_bearish_ticker_receives_no_kelly_anchor():
    snapshots = {
        "BULL": _snapshot(
            "BULL",
            Signal.BULLISH,
            {"technical": 0.5},
            reliability={"technical": 0.65},
            reliability_n={"technical": 20},
        ),
        "BEAR": _snapshot(
            "BEAR",
            Signal.BEARISH,
            {"technical": 0.5},
            reliability={"technical": 0.65},
            reliability_n={"technical": 20},
        ),
    }
    agent, prompts = _agent_with_capture(_ALL_CASH_BODY)
    agent.allocate(user_profile=user_profile(), snapshots=snapshots, macro=None)

    assert len(prompts) == 1
    assert "BULL: half-Kelly suggests" in prompts[0]
    assert "BEAR: half-Kelly suggests" not in prompts[0]


def test_kelly_anchor_block_omitted_when_no_agent_has_data():
    snapshots = {
        "AAPL": _snapshot(
            "AAPL",
            Signal.BULLISH,
            {"technical": 0.5},
            reliability={"technical": 0.5, "fundamental": 0.5, "sentiment": 0.5},
            reliability_n={"technical": 0, "fundamental": 0, "sentiment": 0},
        ),
    }
    agent, prompts = _agent_with_capture(_ALL_CASH_BODY)
    agent.allocate(user_profile=user_profile(), snapshots=snapshots, macro=None)

    assert len(prompts) == 1
    assert "HALF-KELLY ANCHORS: omitted" in prompts[0]
    assert "half-Kelly suggests" not in prompts[0]


def test_kelly_anchor_uses_primary_drivers_measured_win_rate_not_conviction():
    snapshots = {
        "AAPL": _snapshot(
            "AAPL",
            Signal.BULLISH,
            {"technical": 0.9, "fundamental": 0.1},
            reliability={"technical": 0.7, "fundamental": 0.5},
            reliability_n={"technical": 30, "fundamental": 0},
        ),
    }
    agent, prompts = _agent_with_capture(_ALL_CASH_BODY)
    agent.allocate(user_profile=user_profile(), snapshots=snapshots, macro=None)

    assert "AAPL: half-Kelly suggests 15.0%" in prompts[0]


def test_capital_base_is_total_wealth_not_wealth_times_invest_pct():
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
        user_profile=user_profile(invest_pct=0.8),
        snapshots={"AAPL": TickerSnapshot(risk=risk_assessment())},
        macro=None,
    )

    assert alloc is not None
    assert alloc.user_investable_capital == 100_000.0
    assert alloc.portfolio[0].allocation_usd == 10_000.0
    assert "capital_usd: $100,000" in prompts[0]
