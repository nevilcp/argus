"""
tests/test_portfolio_enforcement.py

Regression tests for PA-1: PortfolioManagerAgent.allocate() must enforce risk
verdicts and caps in code, not merely display them in the prompt. Reproduces
the audit's finding that an over-cap allocation, a VETO'd ticker, and a
fabricated ticker all validated cleanly.

Also covers PA-6: when all 3 retry attempts fail, allocate() must say which
of its three stages (LLM call, JSON parse, schema validation) was failing,
not a single generic message regardless of cause.
"""

import json
from datetime import datetime

from argus.agents.portfolio import PortfolioManagerAgent
from argus.orchestration.state import TickerSnapshot
from argus.params import PORTFOLIO, SYSTEM
from argus.schemas.signals import PortfolioAllocation, PositionAllocation, RiskAssessment, RiskVerdict
from argus.seams import FixtureLLMClient

TIMESTAMP = datetime(2026, 1, 1)


class _CapturingLLMClient:
    """A stub LLMClient that records the prompt it was called with.

    Unlike FixtureLLMClient (keyed lookup by design), tests that need to
    inspect the exact prompt text sent to the model use this instead.
    """

    def __init__(self, response_json: dict) -> None:
        self._response_json = response_json
        self.last_prompt: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_prompt = user_prompt
        return json.dumps(self._response_json)


def _risk(verdict: RiskVerdict, approved_weight: float, proposed_weight: float) -> RiskAssessment:
    return RiskAssessment(
        verdict=verdict,
        approved_weight=approved_weight,
        proposed_weight=proposed_weight,
        var_99=0.05,
        portfolio_beta=1.0,
        timestamp=TIMESTAMP,
    )


def _llm_response(portfolio: list[dict]) -> FixtureLLMClient:
    body = {
        "portfolio": portfolio,
        "cash_reserve_pct": 0.5,
        "expected_sharpe": None,
        "rebalance_trigger": "MONTHLY",
    }
    return FixtureLLMClient({"only": json.dumps(body)}, key_fn=lambda _prompt: "only")


def test_allocate_enforces_caps_vetoes_and_fabrications_in_code():
    """An over-cap allocation is clamped, a VETO'd ticker is zeroed-not-dropped,

    and a fabricated ticker absent from snapshots is dropped — each producing
    an adjustments entry rather than validating silently.
    """
    snapshots = {
        "OVER_CAP": TickerSnapshot(risk=_risk(RiskVerdict.APPROVE, approved_weight=0.10, proposed_weight=0.10)),
        "VETOED": TickerSnapshot(risk=_risk(RiskVerdict.VETO, approved_weight=0.0, proposed_weight=0.10)),
    }

    llm_client = _llm_response(
        [
            {
                "ticker": "OVER_CAP",
                "allocation_pct": 0.15,
                "stop_loss": 100.0,
                "thesis": "Bullish",
                "composite_conviction": 0.7,
                "time_horizon": "3-6 months",
            },
            {
                "ticker": "VETOED",
                "allocation_pct": 0.08,
                "stop_loss": 50.0,
                "thesis": "Should have been skipped",
                "composite_conviction": 0.6,
                "time_horizon": "3-6 months",
            },
            {
                "ticker": "FABRICATED",
                "allocation_pct": 0.05,
                "stop_loss": 20.0,
                "thesis": "Never analyzed",
                "composite_conviction": 0.5,
                "time_horizon": "3-6 months",
            },
        ]
    )

    agent = PortfolioManagerAgent(llm_client=llm_client)
    adjustments: list[str] = []
    alloc = agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.5, "risk_tolerance": "MODERATE"},
        snapshots=snapshots,
        macro=None,
        adjustments=adjustments,
    )

    assert alloc is not None
    positions = {pos.ticker: pos for pos in alloc.portfolio}

    assert positions["OVER_CAP"].allocation_pct == 0.10
    assert positions["VETOED"].allocation_pct == 0.0
    assert "FABRICATED" not in positions

    assert any("clamped OVER_CAP allocation from 0.1500 to risk cap 0.1000" in a for a in adjustments)
    assert any("zeroed VETOED allocation (risk verdict VETO)" in a for a in adjustments)
    assert any("dropped FABRICATED" in a for a in adjustments)
    assert len(adjustments) == 3


def test_allocate_names_the_json_parse_stage_on_exhausted_retries(monkeypatch):
    """A response that never parses as JSON is reported as a json_parse failure, not a generic one."""
    monkeypatch.setattr("argus.agents.portfolio.time.sleep", lambda *_: None)
    llm_client = FixtureLLMClient({"only": "not valid json"}, key_fn=lambda _prompt: "only")

    agent = PortfolioManagerAgent(llm_client=llm_client)
    adjustments: list[str] = []
    alloc = agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.5, "risk_tolerance": "MODERATE"},
        snapshots={"OVER_CAP": TickerSnapshot(risk=_risk(RiskVerdict.APPROVE, approved_weight=0.10, proposed_weight=0.10))},
        macro=None,
        adjustments=adjustments,
    )

    assert alloc is None
    assert any("json_parse" in a for a in adjustments)


def test_allocate_names_the_schema_validation_stage_on_exhausted_retries(monkeypatch):
    """Valid JSON that fails PortfolioAllocation's schema is reported as schema_validation, not json_parse."""
    monkeypatch.setattr("argus.agents.portfolio.time.sleep", lambda *_: None)
    # Missing the required "portfolio" list entirely -> model_validate raises, json.loads does not
    llm_client = FixtureLLMClient(
        {"only": json.dumps({"cash_reserve_pct": "not_a_number"})}, key_fn=lambda _prompt: "only"
    )

    agent = PortfolioManagerAgent(llm_client=llm_client)
    adjustments: list[str] = []
    alloc = agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.5, "risk_tolerance": "MODERATE"},
        snapshots={"OVER_CAP": TickerSnapshot(risk=_risk(RiskVerdict.APPROVE, approved_weight=0.10, proposed_weight=0.10))},
        macro=None,
        adjustments=adjustments,
    )

    assert alloc is None
    assert any("schema_validation" in a for a in adjustments)


def test_allocation_pct_cap_matches_system_max_single_position_pct():
    """PA-6 regression: PositionAllocation.allocation_pct's upper bound must track
    SYSTEM.max_single_position_pct rather than duplicate it as a bare literal that a
    param change would silently stop enforcing.
    """
    le_constraint = next(
        m for m in PositionAllocation.model_fields["allocation_pct"].metadata if hasattr(m, "le")
    )
    assert le_constraint.le == SYSTEM.max_single_position_pct


def test_portfolio_allocation_has_no_expected_sharpe_field():
    """PA-5 regression: expected_sharpe used to pass through from the LLM's raw JSON with no
    enforcement, unlike every other numeric field on this schema. Dropped rather than left
    unenforced — no agent in the system produces a genuine expected-return estimate to
    compute a real Sharpe figure from.
    """
    assert "expected_sharpe" not in PortfolioAllocation.model_fields


def test_allocate_clamps_cash_reserve_to_schema_floor_at_high_deployment():
    """PA-4 regression: cash_reserve_pct must clamp to PORTFOLIO.cash_reserve_floor_pct rather
    than leave a residual just under the schema's floor, which used to fail model_validate
    (and surface as a 500) on a near-fully-deployed session.
    """
    tickers = [f"T{i}" for i in range(6)]
    caps = {t: 0.15 for t in tickers}
    caps["T6"] = 0.051
    tickers.append("T6")

    snapshots = {
        t: TickerSnapshot(risk=_risk(RiskVerdict.APPROVE, approved_weight=caps[t], proposed_weight=caps[t]))
        for t in tickers
    }
    portfolio = [
        {
            "ticker": t,
            "allocation_pct": caps[t],
            "stop_loss": 10.0,
            "thesis": "Bullish",
            "composite_conviction": 0.7,
            "time_horizon": "3-6 months",
        }
        for t in tickers
    ]
    llm_client = _llm_response(portfolio)

    agent = PortfolioManagerAgent(llm_client=llm_client)
    alloc = agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.95, "risk_tolerance": "MODERATE"},
        snapshots=snapshots,
        macro=None,
    )

    assert alloc is not None
    assert alloc.cash_reserve_pct == PORTFOLIO.cash_reserve_floor_pct


def test_allocate_states_achievable_deployment_ceiling_not_raw_invest_pct():
    """PA-4 regression: the prompt must state the achievable deployment ceiling
    (min(invest_pct, sum of approved Caps)), not raw invest_pct, so ALLOCATION RULE 3's
    target is reachable for a small approved universe under per-position caps.
    """
    snapshots = {
        t: TickerSnapshot(risk=_risk(RiskVerdict.APPROVE, approved_weight=0.15, proposed_weight=0.15))
        for t in ("A", "B", "C")
    }
    llm_client = _CapturingLLMClient(
        {"portfolio": [], "cash_reserve_pct": 1.0, "rebalance_trigger": "MONTHLY"}
    )

    agent = PortfolioManagerAgent(llm_client=llm_client)
    agent.allocate(
        user_profile={"total_wealth": 100_000.0, "invest_pct": 0.95, "risk_tolerance": "MODERATE"},
        snapshots=snapshots,
        macro=None,
    )

    # 3 tickers capped at 0.15 each => achievable ceiling is 0.45, not the requested 0.95
    assert "deployment_ceiling: 45%" in llm_client.last_prompt
    assert "invest_pct: 95%" in llm_client.last_prompt
