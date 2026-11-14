"""Regression tests for PortfolioManagerAgent.allocate()'s in-code enforcement."""

import json

import pytest
from pydantic import ValidationError

from argus.agents.portfolio import PortfolioManagerAgent
from argus.governor import RateLimitExceeded
from argus.infra.diagnostics import Diagnostics
from argus.params import PORTFOLIO, SYSTEM
from argus.schemas.signals import (
    PortfolioAllocation,
    PositionAllocation,
    ProposedPosition,
    RiskVerdict,
    TickerSnapshot,
)
from argus.seams import FixtureLLMClient
from tests.helpers.builders import risk_assessment, user_profile


class _CapturingLLMClient:
    """A stub LLMClient that records the prompt it was called with."""

    def __init__(self, response_json: dict) -> None:
        self._response_json = response_json
        self.last_prompt: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_prompt = user_prompt
        return json.dumps(self._response_json)


def _approved(*tickers: str, weight: float = 0.10) -> dict[str, TickerSnapshot]:
    """Build risk-approved snapshots sharing one per-position cap."""
    return {
        ticker: TickerSnapshot(risk=risk_assessment(RiskVerdict.APPROVE, weight, proposed_weight=weight))
        for ticker in tickers
    }


def _position(ticker: str, allocation_pct: float, stop_loss: float = 10.0, thesis: str = "Bullish") -> dict:
    """Build one advisory position in the model response."""
    return {
        "ticker": ticker,
        "allocation_pct": allocation_pct,
        "stop_loss": stop_loss,
        "thesis": thesis,
        "composite_conviction": 0.7,
        "time_horizon": "3-6 months",
    }


def _llm_response(portfolio: list[dict]) -> FixtureLLMClient:
    """Wrap `portfolio` positions in a full allocator response body, keyed "only"."""
    body = {
        "portfolio": portfolio,
        "cash_reserve_pct": 0.5,
        "expected_sharpe": None,
        "rebalance_trigger": "MONTHLY",
    }
    return FixtureLLMClient({"only": json.dumps(body)}, key_fn=lambda _prompt: "only")


def test_allocate_reraises_on_governor_exhaustion_without_retrying():

    class _RateLimitedLLMClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            raise RateLimitExceeded("gpt-oss-120b quota exhausted")

    llm = _RateLimitedLLMClient()
    agent = PortfolioManagerAgent(llm_client=llm)

    with pytest.raises(RateLimitExceeded):
        agent.allocate(user_profile=user_profile(), snapshots=_approved("AAPL"), macro=None)

    assert llm.calls == 1


def test_allocate_does_not_repair_a_bad_response(monkeypatch):
    monkeypatch.setattr("argus.structured_output.time.sleep", lambda *_: None)
    llm_client = _CapturingLLMClient({"cash_reserve_pct": 0.5})

    agent = PortfolioManagerAgent(llm_client=llm_client)
    alloc = agent.allocate(user_profile=user_profile(), snapshots=_approved("AAPL"), macro=None)

    assert alloc is None
    assert "Your previous response was invalid" not in llm_client.last_prompt


def test_allocate_enforces_caps_vetoes_and_fabrications_in_code():
    snapshots = _approved("OVER_CAP")
    snapshots["VETOED"] = TickerSnapshot(risk=risk_assessment(RiskVerdict.VETO, 0.0, proposed_weight=0.10))

    llm_client = _llm_response(
        [
            _position("OVER_CAP", 0.15, stop_loss=100.0),
            _position("VETOED", 0.08, stop_loss=50.0, thesis="Should have been skipped"),
            _position("FABRICATED", 0.05, stop_loss=20.0, thesis="Never analyzed"),
        ]
    )

    agent = PortfolioManagerAgent(llm_client=llm_client)
    diagnostics = Diagnostics()
    alloc = agent.allocate(
        user_profile=user_profile(),
        snapshots=snapshots,
        macro=None,
        diagnostics=diagnostics,
    )

    assert alloc is not None
    positions = {pos.ticker: pos for pos in alloc.portfolio}

    assert positions["OVER_CAP"].allocation_pct == 0.10
    assert positions["VETOED"].allocation_pct == 0.0
    assert "FABRICATED" not in positions

    assert any("clamped OVER_CAP allocation from 0.1500 to risk cap 0.1000" in a for a in diagnostics.messages)
    assert any("zeroed VETOED allocation (risk verdict VETO)" in a for a in diagnostics.messages)
    assert any("dropped FABRICATED" in a for a in diagnostics.messages)
    assert any("cash_reserve_pct forced to residual 0.9000 (LLM proposed 0.5000)" in a for a in diagnostics.messages)
    assert len(diagnostics.messages) == 4


def test_allocate_names_the_json_parse_stage_on_exhausted_retries(monkeypatch):
    monkeypatch.setattr("argus.structured_output.time.sleep", lambda *_: None)
    llm_client = FixtureLLMClient({"only": "not valid json"}, key_fn=lambda _prompt: "only")

    agent = PortfolioManagerAgent(llm_client=llm_client)
    diagnostics = Diagnostics()
    alloc = agent.allocate(
        user_profile=user_profile(),
        snapshots=_approved("OVER_CAP"),
        macro=None,
        diagnostics=diagnostics,
    )

    assert alloc is None
    assert any("json_parse" in a for a in diagnostics.messages)


def test_allocate_names_the_schema_validation_stage_on_exhausted_retries(monkeypatch):
    monkeypatch.setattr("argus.structured_output.time.sleep", lambda *_: None)
    llm_client = FixtureLLMClient(
        {"only": json.dumps({"cash_reserve_pct": "not_a_number"})}, key_fn=lambda _prompt: "only"
    )

    agent = PortfolioManagerAgent(llm_client=llm_client)
    diagnostics = Diagnostics()
    alloc = agent.allocate(
        user_profile=user_profile(),
        snapshots=_approved("OVER_CAP"),
        macro=None,
        diagnostics=diagnostics,
    )

    assert alloc is None
    assert any("schema_validation" in a for a in diagnostics.messages)


def test_allocation_pct_cap_matches_system_max_single_position_pct():
    le_constraint = next(m for m in PositionAllocation.model_fields["allocation_pct"].metadata if hasattr(m, "le"))
    assert le_constraint.le == SYSTEM.max_single_position_pct


def test_proposed_position_allocation_pct_bounded_at_proposal():
    base = {
        "ticker": "AAPL",
        "thesis": "Bullish",
        "composite_conviction": 0.5,
        "time_horizon": "3-6 months",
    }

    with pytest.raises(ValidationError, match=r".*"):
        ProposedPosition(**base, allocation_pct=-0.01)

    with pytest.raises(ValidationError, match=r".*"):
        ProposedPosition(**base, allocation_pct=SYSTEM.max_single_position_pct + 0.01)

    ProposedPosition(**base, allocation_pct=SYSTEM.max_single_position_pct)


def test_portfolio_allocation_has_no_expected_sharpe_field():
    assert "expected_sharpe" not in PortfolioAllocation.model_fields


def test_allocate_clamps_cash_reserve_to_schema_floor_at_high_deployment():
    caps = {f"T{i}": 0.15 for i in range(6)}
    caps["T6"] = 0.051

    snapshots = {
        ticker: TickerSnapshot(risk=risk_assessment(RiskVerdict.APPROVE, cap, proposed_weight=cap))
        for ticker, cap in caps.items()
    }
    llm_client = _llm_response([_position(ticker, cap) for ticker, cap in caps.items()])

    agent = PortfolioManagerAgent(llm_client=llm_client)
    alloc = agent.allocate(
        user_profile=user_profile(invest_pct=0.95),
        snapshots=snapshots,
        macro=None,
    )

    assert alloc is not None
    assert alloc.cash_reserve_pct == PORTFOLIO.cash_reserve_floor_pct


def test_allocate_states_achievable_deployment_ceiling_not_raw_invest_pct():
    llm_client = _CapturingLLMClient({"portfolio": [], "cash_reserve_pct": 1.0, "rebalance_trigger": "MONTHLY"})

    agent = PortfolioManagerAgent(llm_client=llm_client)
    agent.allocate(
        user_profile=user_profile(invest_pct=0.95),
        snapshots=_approved("A", "B", "C", weight=0.15),
        macro=None,
    )

    assert "deployment_ceiling: 45%" in llm_client.last_prompt
    assert "invest_pct: 95%" in llm_client.last_prompt


@pytest.mark.parametrize("total_wealth", [None, 0.0, -5.0])
def test_allocate_without_positive_total_wealth_returns_none_and_records_it(total_wealth):
    llm = _CapturingLLMClient({})
    agent = PortfolioManagerAgent(llm_client=llm)
    diagnostics = Diagnostics()
    profile = {**user_profile(), "total_wealth": total_wealth}

    for snapshots in (_approved("AAPL"), {}):
        diagnostics.messages.clear()
        alloc = agent.allocate(user_profile=profile, snapshots=snapshots, macro=None, diagnostics=diagnostics)

        assert alloc is None
        assert len(diagnostics.messages) == 1
        assert diagnostics.messages[0].startswith("portfolio_allocation:")
        assert "total_wealth" in diagnostics.messages[0]
    assert llm.last_prompt is None
