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
from argus.schemas.signals import RiskAssessment, RiskVerdict
from argus.seams import FixtureLLMClient

TIMESTAMP = datetime(2026, 1, 1)


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

    and a fabricated ticker absent from all_signals is dropped — each producing
    an adjustments entry rather than validating silently.
    """
    all_signals = {
        "OVER_CAP": {"risk": _risk(RiskVerdict.APPROVE, approved_weight=0.10, proposed_weight=0.10)},
        "VETOED": {"risk": _risk(RiskVerdict.VETO, approved_weight=0.0, proposed_weight=0.10)},
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
        all_signals=all_signals,
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
        all_signals={"OVER_CAP": {"risk": _risk(RiskVerdict.APPROVE, approved_weight=0.10, proposed_weight=0.10)}},
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
        all_signals={"OVER_CAP": {"risk": _risk(RiskVerdict.APPROVE, approved_weight=0.10, proposed_weight=0.10)}},
        macro=None,
        adjustments=adjustments,
    )

    assert alloc is None
    assert any("schema_validation" in a for a in adjustments)
