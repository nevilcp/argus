"""
tests/test_state.py

Tests for TickerSnapshot and build_ticker_snapshots.
"""

from datetime import datetime

from argus.orchestration.state import TickerSnapshot, build_ticker_snapshots
from argus.schemas.signals import RiskAssessment, RiskVerdict

TIMESTAMP = datetime(2026, 1, 1)


def _risk(verdict: RiskVerdict) -> RiskAssessment:
    return RiskAssessment(
        verdict=verdict,
        approved_weight=0.0 if verdict == RiskVerdict.VETO else 0.10,
        proposed_weight=0.10,
        timestamp=TIMESTAMP,
    )


def test_risk_approved_true_for_approve_and_reduce():
    """APPROVE and REDUCE verdicts permit an equity allocation."""
    assert TickerSnapshot(risk=_risk(RiskVerdict.APPROVE)).risk_approved
    assert TickerSnapshot(risk=_risk(RiskVerdict.REDUCE)).risk_approved


def test_risk_approved_false_for_veto_or_missing():
    """A VETO verdict, or no risk assessment at all, blocks an equity allocation."""
    assert not TickerSnapshot(risk=_risk(RiskVerdict.VETO)).risk_approved
    assert not TickerSnapshot().risk_approved


def test_build_ticker_snapshots_covers_whole_universe_with_absent_specialists_empty():
    """Every universe ticker gets a snapshot, even one no specialist produced output for."""
    state = {
        "universe": ["AAPL", "UNCOVERED"],
        "risk_assessments": {"AAPL": _risk(RiskVerdict.APPROVE)},
    }

    snapshots = build_ticker_snapshots(state)

    assert set(snapshots) == {"AAPL", "UNCOVERED"}
    assert snapshots["AAPL"].risk_approved
    assert snapshots["UNCOVERED"] == TickerSnapshot()
