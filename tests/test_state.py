from argus.orchestration.state import build_ticker_snapshots
from argus.schemas.signals import RiskVerdict, TickerSnapshot
from tests.helpers.builders import risk_assessment


def test_risk_approved_true_for_approve_and_reduce():
    """APPROVE and REDUCE verdicts permit an equity allocation."""
    assert TickerSnapshot(
        risk=risk_assessment(RiskVerdict.APPROVE, 0.10, proposed_weight=0.10)
    ).risk_approved
    assert TickerSnapshot(
        risk=risk_assessment(RiskVerdict.REDUCE, 0.10, proposed_weight=0.10)
    ).risk_approved


def test_risk_approved_false_for_veto_or_missing():
    """A VETO verdict, or no risk assessment at all, blocks an equity allocation."""
    assert not TickerSnapshot(
        risk=risk_assessment(RiskVerdict.VETO, 0.0, proposed_weight=0.10)
    ).risk_approved
    assert not TickerSnapshot().risk_approved


def test_build_ticker_snapshots_covers_whole_universe_with_absent_specialists_empty():
    """Every universe ticker gets a snapshot, even one no specialist produced output for."""
    state = {
        "universe": ["AAPL", "UNCOVERED"],
        "risk_assessments": {
            "AAPL": risk_assessment(RiskVerdict.APPROVE, 0.10, proposed_weight=0.10)
        },
    }

    snapshots = build_ticker_snapshots(state)

    assert set(snapshots) == {"AAPL", "UNCOVERED"}
    assert snapshots["AAPL"].risk_approved
    assert snapshots["UNCOVERED"] == TickerSnapshot()
