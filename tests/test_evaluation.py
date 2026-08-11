"""
tests/test_evaluation.py

Tests for argus/backtesting/evaluation.py — see
docs/adr/0012-pre-registered-evaluation.md for what these metrics mean and
why. Metric functions are tested against synthetic, deterministic data (no
network); collect_paired_outcomes/evaluate_decisions and
system_behavior_report are tested end-to-end against the one real fixture
session with a fake MarketDataProvider standing in for
LiveMarketDataProvider (see tests/test_reconciliation.py's _FakeMarketData,
reused here).
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from argus.backtesting.evaluation import (
    PairedOutcome,
    bootstrap_ci,
    check_replay_determinism,
    collect_paired_outcomes,
    evaluate_decisions,
    hit_rate_with_deadband,
    rank_information_coefficient,
    signed_conviction,
    system_behavior_report,
)
from argus.backtesting.replay import SessionResult, replay_session
from argus.schemas.signals import ARGUSDecision, Signal
import pytest

from tests.test_reconciliation import (
    _allocation,
    _FakeMarketData,
    _macro,
    _technical,
    _ten_day_series,
)
from argus.orchestration.aggregator import HybridSignalAggregator

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _decision_with_signal(
    signal: Signal, conviction: float, ticker: str = "TEST", price: float = 100.0
) -> ARGUSDecision:
    technical = _technical(signal, conviction, ticker=ticker, price=price)
    macro = _macro()
    aggregated = HybridSignalAggregator().aggregate(technical, macro, None, None)
    return ARGUSDecision(
        ticker=ticker,
        session_timestamp=datetime(2026, 1, 1),
        technical=technical,
        macro=macro,
        aggregated=aggregated,
        allocation=_allocation(ticker=ticker),
    )


# ---------------------------------------------------------------------------
# signed_conviction
# ---------------------------------------------------------------------------


def test_signed_conviction_signs_by_direction():
    bullish = _decision_with_signal(Signal.BULLISH, 0.8)
    bearish = _decision_with_signal(Signal.BEARISH, 0.6)
    neutral = _decision_with_signal(Signal.NEUTRAL, 0.3)

    assert signed_conviction(bullish) == bullish.aggregated.conviction
    assert signed_conviction(bearish) == -bearish.aggregated.conviction
    assert signed_conviction(neutral) == 0.0


def test_signed_conviction_none_without_aggregated_signal():
    decision = ARGUSDecision(ticker="TEST", session_timestamp=datetime.now())
    assert signed_conviction(decision) is None


# ---------------------------------------------------------------------------
# rank_information_coefficient
# ---------------------------------------------------------------------------


def _pairs(convictions: list[float], returns: list[float]) -> list[PairedOutcome]:
    return [
        PairedOutcome(ticker=f"T{i}", signed_conviction=c, forward_return=r, holding_days=5)
        for i, (c, r) in enumerate(zip(convictions, returns))
    ]


def test_rank_ic_perfect_positive_correlation():
    pairs = _pairs([0.1, 0.3, 0.5, 0.7, 0.9], [-0.02, -0.01, 0.0, 0.01, 0.02])
    rho, p_value = rank_information_coefficient(pairs)
    assert rho == pytest.approx(1.0)
    assert p_value is not None


def test_rank_ic_perfect_negative_correlation():
    pairs = _pairs([0.1, 0.3, 0.5, 0.7, 0.9], [0.02, 0.01, 0.0, -0.01, -0.02])
    rho, _ = rank_information_coefficient(pairs)
    assert rho == pytest.approx(-1.0)


def test_rank_ic_undefined_below_two_pairs():
    assert rank_information_coefficient(_pairs([0.5], [0.01])) == (None, None)
    assert rank_information_coefficient([]) == (None, None)


def test_rank_ic_undefined_for_constant_series():
    pairs = _pairs([0.5, 0.5, 0.5], [0.01, -0.02, 0.03])
    assert rank_information_coefficient(pairs) == (None, None)


# ---------------------------------------------------------------------------
# hit_rate_with_deadband
# ---------------------------------------------------------------------------


def test_hit_rate_counts_matching_signs_outside_deadband():
    pairs = _pairs(
        convictions=[0.5, -0.5, 0.5, -0.5],
        returns=[0.05, -0.05, -0.05, 0.05],  # first two "hit", last two "miss"
    )
    hit_rate, n_scored = hit_rate_with_deadband(pairs, deadband=0.01)
    assert n_scored == 4
    assert hit_rate == 0.5


def test_hit_rate_excludes_returns_inside_deadband():
    pairs = _pairs([0.5, -0.5], [0.005, -0.005])
    hit_rate, n_scored = hit_rate_with_deadband(pairs, deadband=0.01)
    assert n_scored == 0
    assert hit_rate is None


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_deterministic_and_brackets_zero_pairs():
    assert bootstrap_ci(_pairs([0.5], [0.01]), lambda p: rank_information_coefficient(p)[0]) == (None, None)


def test_bootstrap_ci_reproducible_with_fixed_seed():
    pairs = _pairs([0.1, 0.3, 0.5, 0.7, 0.9, -0.2], [-0.03, -0.01, 0.0, 0.02, 0.04, 0.01])
    stat_fn = lambda p: rank_information_coefficient(p)[0]  # noqa: E731

    first = bootstrap_ci(pairs, stat_fn, n_resamples=500, seed=42)
    second = bootstrap_ci(pairs, stat_fn, n_resamples=500, seed=42)

    assert first == second
    lower, upper = first
    assert lower is not None and upper is not None
    assert lower <= upper


# ---------------------------------------------------------------------------
# collect_paired_outcomes / evaluate_decisions
# ---------------------------------------------------------------------------


def test_collect_paired_outcomes_pairs_conviction_with_realized_return():
    decision = _decision_with_signal(Signal.BULLISH, 0.8, ticker="TEST", price=100.0)
    market_data = _ten_day_series(datetime(2026, 1, 1), start_price=100.0, ticker="TEST")

    pairs = collect_paired_outcomes([decision], market_data, horizon_days=5)

    assert len(pairs) == 1
    assert pairs[0].signed_conviction == decision.aggregated.conviction
    assert pairs[0].forward_return == (105.0 - 100.0) / 100.0


def test_collect_paired_outcomes_skips_decisions_with_no_aggregated_signal():
    decision = ARGUSDecision(ticker="TEST", session_timestamp=datetime(2026, 1, 1))
    market_data = _ten_day_series(datetime(2026, 1, 1), ticker="TEST")

    assert collect_paired_outcomes([decision], market_data, horizon_days=5) == []


def test_evaluate_decisions_end_to_end_on_multiple_tickers():
    decisions = [
        _decision_with_signal(Signal.BULLISH, 0.8, ticker="A", price=100.0),
        _decision_with_signal(Signal.BEARISH, 0.6, ticker="B", price=100.0),
    ]
    closes_a = pd.Series(
        [100.0 + i for i in range(10)], index=pd.date_range(datetime(2026, 1, 1), periods=10, freq="D")
    )
    closes_b = pd.Series(
        [100.0 - i for i in range(10)], index=pd.date_range(datetime(2026, 1, 1), periods=10, freq="D")
    )
    market_data = _FakeMarketData({"A": closes_a, "B": closes_b})

    result = evaluate_decisions(decisions, market_data, horizon_days=5, deadband=0.01)

    assert result.n == 2
    # A: bullish and price rose; B: bearish and price fell -> both "hit"
    assert result.hit_rate == 1.0


# ---------------------------------------------------------------------------
# system_behavior_report / check_replay_determinism (real fixture session)
# ---------------------------------------------------------------------------


def test_system_behavior_report_on_replayed_fixture_session():
    result = replay_session(FIXTURES_DIR, closed_loop=False)

    report = system_behavior_report(result)

    assert report.tickers_total == len(result.universe)
    assert report.decisions_built == report.tickers_total
    assert report.schema_validity == 1.0
    assert report.constraint_violations == 0
    assert report.total_api_calls > 0
    assert report.api_calls_per_decision is not None
    assert report.retries_instrumented is False
    assert report.tokens_instrumented is False


def test_system_behavior_report_counts_reduce_verdicts():
    decisions = [_decision_with_signal(Signal.BULLISH, 0.8, ticker="TEST")]
    session = SessionResult(
        session_dir=FIXTURES_DIR,
        universe=["TEST"],
        final_state={"decisions": decisions, "errors": []},
    )
    report = system_behavior_report(session)
    assert report.tickers_total == 1
    assert report.decisions_built == 1
    assert report.reduce_verdicts == 0  # no risk assessment attached in this synthetic decision


def test_replay_determinism_holds_for_the_real_fixture_session():
    assert check_replay_determinism(FIXTURES_DIR) is True
