"""Tests for argus/backtesting/evaluation.py.

Metric functions are tested against synthetic, deterministic data (no
network); collect_paired_outcomes, evaluate_decisions, and
system_behavior_report are tested end-to-end against the one real fixture
session, with a fake MarketDataProvider standing in for
LiveMarketDataProvider (see tests/test_reconciliation.py's _FakeMarketData,
reused here).
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

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
    trade_level_win_loss_stats,
)
from argus.backtesting.replay import SessionResult, replay_session
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.schemas.signals import ARGUSDecision, Signal
from tests.test_reconciliation import (
    _allocation,
    _FakeMarketData,
    _macro,
    _technical,
    _ten_day_series,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _decision_with_signal(
    signal: Signal, conviction: float, ticker: str = "TEST", price: float = 100.0
) -> ARGUSDecision:
    """Builds a decision whose aggregated signal comes from the real aggregator."""
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
    """Signed conviction is positive for BULLISH, negative for BEARISH, zero for NEUTRAL."""
    bullish = _decision_with_signal(Signal.BULLISH, 0.8)
    bearish = _decision_with_signal(Signal.BEARISH, 0.6)
    neutral = _decision_with_signal(Signal.NEUTRAL, 0.3)

    assert signed_conviction(bullish) == bullish.aggregated.conviction
    assert signed_conviction(bearish) == -bearish.aggregated.conviction
    assert signed_conviction(neutral) == 0.0


def test_signed_conviction_none_without_aggregated_signal():
    """A decision with no aggregated signal has no signed conviction."""
    decision = ARGUSDecision(ticker="TEST", session_timestamp=datetime.now())
    assert signed_conviction(decision) is None


# ---------------------------------------------------------------------------
# rank_information_coefficient
# ---------------------------------------------------------------------------


def _pairs(convictions: list[float], returns: list[float]) -> list[PairedOutcome]:
    """Zips convictions with forward returns into PairedOutcomes on synthetic tickers."""
    return [
        PairedOutcome(ticker=f"T{i}", signed_conviction=c, forward_return=r, holding_days=5)
        for i, (c, r) in enumerate(zip(convictions, returns))
    ]


def test_rank_ic_perfect_positive_correlation():
    """Rank IC is 1.0 when conviction and forward return are perfectly rank-aligned."""
    pairs = _pairs([0.1, 0.3, 0.5, 0.7, 0.9], [-0.02, -0.01, 0.0, 0.01, 0.02])
    rho, p_value = rank_information_coefficient(pairs)
    assert rho == pytest.approx(1.0)
    assert p_value is not None


def test_rank_ic_perfect_negative_correlation():
    """Rank IC is -1.0 when conviction and forward return are perfectly rank-inverted."""
    pairs = _pairs([0.1, 0.3, 0.5, 0.7, 0.9], [0.02, 0.01, 0.0, -0.01, -0.02])
    rho, _ = rank_information_coefficient(pairs)
    assert rho == pytest.approx(-1.0)


def test_rank_ic_undefined_below_two_pairs():
    """Rank IC is undefined with fewer than two pairs."""
    assert rank_information_coefficient(_pairs([0.5], [0.01])) == (None, None)
    assert rank_information_coefficient([]) == (None, None)


def test_rank_ic_undefined_for_constant_series():
    """Rank IC is undefined when one of the series has zero variance."""
    pairs = _pairs([0.5, 0.5, 0.5], [0.01, -0.02, 0.03])
    assert rank_information_coefficient(pairs) == (None, None)


# ---------------------------------------------------------------------------
# hit_rate_with_deadband
# ---------------------------------------------------------------------------


def test_hit_rate_counts_matching_signs_outside_deadband():
    """Hit rate counts only pairs where conviction sign matches return sign."""
    pairs = _pairs(
        convictions=[0.5, -0.5, 0.5, -0.5],
        returns=[0.05, -0.05, -0.05, 0.05],  # First two "hit", last two "miss"
    )
    hit_rate, n_scored = hit_rate_with_deadband(pairs, deadband=0.01)
    assert n_scored == 4
    assert hit_rate == 0.5


def test_hit_rate_excludes_returns_inside_deadband():
    """Pairs whose return falls inside the deadband are excluded from scoring."""
    pairs = _pairs([0.5, -0.5], [0.005, -0.005])
    hit_rate, n_scored = hit_rate_with_deadband(pairs, deadband=0.01)
    assert n_scored == 0
    assert hit_rate is None


# ---------------------------------------------------------------------------
# trade_level_win_loss_stats
# ---------------------------------------------------------------------------


def test_trade_level_win_loss_stats_empty_for_no_pairs():
    """Trade-level stats are an empty dict when there are no paired outcomes."""
    assert trade_level_win_loss_stats([]) == {}


def test_trade_level_win_loss_stats_computes_win_rate_and_profit_factor():
    """Trade-level stats key on forward_return sign, independent of conviction."""
    pairs = _pairs(convictions=[0.5, -0.5, 0.5], returns=[0.05, -0.02, -0.01])
    stats = trade_level_win_loss_stats(pairs)
    assert stats["total_trades"] == 3
    assert stats["win_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["avg_win_pct"] == pytest.approx(0.05)
    assert stats["avg_loss_pct"] == pytest.approx(-0.015)
    assert stats["avg_holding_days"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def _rank_ic(pairs: list[PairedOutcome]) -> float | None:
    """The rank IC alone, dropping the p-value, as bootstrap_ci's statistic argument."""
    return rank_information_coefficient(pairs)[0]


def test_bootstrap_ci_is_deterministic_and_brackets_zero_pairs():
    """A bootstrap CI over too few pairs to compute the statistic is (None, None)."""
    assert bootstrap_ci(_pairs([0.5], [0.01]), _rank_ic) == (None, None)


def test_bootstrap_ci_reproducible_with_fixed_seed():
    """The same seed produces identical bootstrap CIs across runs."""
    pairs = _pairs([0.1, 0.3, 0.5, 0.7, 0.9, -0.2], [-0.03, -0.01, 0.0, 0.02, 0.04, 0.01])

    first = bootstrap_ci(pairs, _rank_ic, n_resamples=500, seed=42)
    second = bootstrap_ci(pairs, _rank_ic, n_resamples=500, seed=42)

    assert first == second
    lower, upper = first
    assert lower is not None and upper is not None
    assert lower <= upper


# ---------------------------------------------------------------------------
# collect_paired_outcomes / evaluate_decisions
# ---------------------------------------------------------------------------


def test_collect_paired_outcomes_pairs_conviction_with_realized_return():
    """Each decision is paired with its forward return over the given horizon."""
    decision = _decision_with_signal(Signal.BULLISH, 0.8, ticker="TEST", price=100.0)
    market_data = _ten_day_series(datetime(2026, 1, 1), start_price=100.0, ticker="TEST")

    pairs = collect_paired_outcomes([decision], market_data, horizon_days=5)

    assert len(pairs) == 1
    assert pairs[0].signed_conviction == decision.aggregated.conviction
    assert pairs[0].forward_return == (105.0 - 100.0) / 100.0


def test_collect_paired_outcomes_skips_decisions_with_no_aggregated_signal():
    """Decisions without an aggregated signal produce no paired outcome."""
    decision = ARGUSDecision(ticker="TEST", session_timestamp=datetime(2026, 1, 1))
    market_data = _ten_day_series(datetime(2026, 1, 1), ticker="TEST")

    assert collect_paired_outcomes([decision], market_data, horizon_days=5) == []


def test_evaluate_decisions_end_to_end_on_multiple_tickers():
    """Evaluation over multiple tickers scores each decision against its own price series."""
    decisions = [
        _decision_with_signal(Signal.BULLISH, 0.8, ticker="A", price=100.0),
        _decision_with_signal(Signal.BEARISH, 0.6, ticker="B", price=100.0),
    ]
    dates = pd.date_range(datetime(2026, 1, 1), periods=10, freq="D")
    rising = pd.Series([100.0 + i for i in range(10)], index=dates)
    falling = pd.Series([100.0 - i for i in range(10)], index=dates)
    market_data = _FakeMarketData({"A": rising, "B": falling})

    result = evaluate_decisions(decisions, market_data, horizon_days=5, deadband=0.01)

    assert result.n == 2
    # A: bullish and price rose, B: bearish and price fell, so both "hit"
    assert result.hit_rate == 1.0


# ---------------------------------------------------------------------------
# system_behavior_report / check_replay_determinism (real fixture session)
# ---------------------------------------------------------------------------


def test_system_behavior_report_on_replayed_fixture_session():
    """The behavior report accounts for every ticker in a replayed fixture session."""
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
    """The behavior report counts zero reduce verdicts when no risk assessment is attached."""
    decisions = [_decision_with_signal(Signal.BULLISH, 0.8, ticker="TEST")]
    session = SessionResult(
        session_dir=FIXTURES_DIR,
        universe=["TEST"],
        final_state={"decisions": decisions, "errors": []},
    )
    report = system_behavior_report(session)
    assert report.tickers_total == 1
    assert report.decisions_built == 1
    assert report.reduce_verdicts == 0  # No risk assessment attached in this synthetic decision


def test_replay_determinism_holds_for_the_real_fixture_session():
    """Replaying the same fixture session twice yields identical decisions."""
    assert check_replay_determinism(FIXTURES_DIR) is True
