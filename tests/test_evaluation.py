"""Tests for argus/backtesting/evaluation.py."""

from datetime import datetime

import pandas as pd
import pytest

from argus.backtesting.evaluation import (
    EvaluationResult,
    PairedOutcome,
    bootstrap_ci,
    check_replay_determinism,
    collect_paired_outcomes,
    evaluate_decisions,
    hit_rate_with_deadband,
    pre_registered_verdict,
    rank_information_coefficient,
    signed_conviction,
    system_behavior_report,
    trade_level_win_loss_stats,
)
from argus.backtesting.replay import SessionResult, replay_session
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.schemas.signals import ARGUSDecision, Signal
from tests.helpers.builders import macro_context, position_allocation, technical_signal
from tests.helpers.market_data import FakeMarketData, ten_day_series
from tests.helpers.paths import FIXTURES_DIR


def _decision_with_signal(
    signal: Signal, conviction: float, ticker: str = "TEST", price: float = 100.0
) -> ARGUSDecision:
    """Build a decision whose aggregated signal comes from the real aggregator."""
    technical = technical_signal(signal, conviction, ticker=ticker, current_price=price)
    macro = macro_context()
    aggregated = HybridSignalAggregator().aggregate(technical, macro, None, None)
    return ARGUSDecision(
        ticker=ticker,
        session_timestamp=datetime(2026, 1, 1),
        technical=technical,
        macro=macro,
        aggregated=aggregated,
        allocation=position_allocation(ticker),
    )


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


def _pairs(convictions: list[float], returns: list[float]) -> list[PairedOutcome]:
    """Zip convictions with forward returns into PairedOutcomes on synthetic tickers."""
    return [
        PairedOutcome(ticker=f"T{i}", signed_conviction=c, forward_return=r, holding_days=5)
        for i, (c, r) in enumerate(zip(convictions, returns, strict=True))
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


def test_hit_rate_counts_matching_signs_outside_deadband():
    pairs = _pairs(
        convictions=[0.5, -0.5, 0.5, -0.5],
        returns=[0.05, -0.05, -0.05, 0.05],
    )
    hit_rate, n_scored = hit_rate_with_deadband(pairs, deadband=0.01)
    assert n_scored == 4
    assert hit_rate == 0.5


def test_hit_rate_excludes_returns_inside_deadband():
    pairs = _pairs([0.5, -0.5], [0.005, -0.005])
    hit_rate, n_scored = hit_rate_with_deadband(pairs, deadband=0.01)
    assert n_scored == 0
    assert hit_rate is None


def test_hit_rate_excludes_neutral_pairs():
    pairs = _pairs([0.0], [-0.05])
    hit_rate, n_scored = hit_rate_with_deadband(pairs, deadband=0.01)
    assert n_scored == 0
    assert hit_rate is None


def test_hit_rate_scores_only_the_directional_pairs_in_a_mixed_set():
    pairs = _pairs(
        convictions=[0.5, -0.5, 0.0, 0.0],
        returns=[0.05, 0.05, -0.05, 0.05],  # long hits, short misses, neutrals ignored
    )
    hit_rate, n_scored = hit_rate_with_deadband(pairs, deadband=0.01)
    assert n_scored == 2
    assert hit_rate == 0.5


def test_trade_level_win_loss_stats_empty_for_no_pairs():
    assert trade_level_win_loss_stats([]) == {}


def test_trade_level_win_loss_stats_computes_win_rate_and_profit_factor():
    pairs = _pairs(convictions=[0.5, -0.5, 0.5], returns=[0.05, -0.02, -0.01])
    stats = trade_level_win_loss_stats(pairs)
    assert stats["total_trades"] == 3
    assert stats["win_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["avg_win_pct"] == pytest.approx(0.05)
    assert stats["avg_loss_pct"] == pytest.approx(-0.015)
    assert stats["avg_holding_days"] == pytest.approx(5.0)


def _rank_ic(pairs: list[PairedOutcome]) -> float | None:
    """Return rank correlation alone for bootstrap resampling."""
    return rank_information_coefficient(pairs)[0]


def test_bootstrap_ci_is_deterministic_and_brackets_zero_pairs():
    assert bootstrap_ci(_pairs([0.5], [0.01]), _rank_ic) == (None, None)


def test_bootstrap_ci_reproducible_with_fixed_seed():
    pairs = _pairs([0.1, 0.3, 0.5, 0.7, 0.9, -0.2], [-0.03, -0.01, 0.0, 0.02, 0.04, 0.01])

    first = bootstrap_ci(pairs, _rank_ic, n_resamples=500, seed=42)
    second = bootstrap_ci(pairs, _rank_ic, n_resamples=500, seed=42)

    assert first == second
    lower, upper = first
    assert lower is not None
    assert upper is not None
    assert lower <= upper


def test_collect_paired_outcomes_pairs_conviction_with_realized_return():
    decision = _decision_with_signal(Signal.BULLISH, 0.8, ticker="TEST", price=100.0)
    market_data = ten_day_series(datetime(2026, 1, 1), start_price=100.0, ticker="TEST")

    pairs = collect_paired_outcomes([decision], market_data, horizon_days=5)

    assert len(pairs) == 1
    assert pairs[0].signed_conviction == decision.aggregated.conviction
    assert pairs[0].forward_return == (105.0 - 100.0) / 100.0


def test_collect_paired_outcomes_skips_decisions_with_no_aggregated_signal():
    decision = ARGUSDecision(ticker="TEST", session_timestamp=datetime(2026, 1, 1))
    market_data = ten_day_series(datetime(2026, 1, 1), ticker="TEST")

    assert collect_paired_outcomes([decision], market_data, horizon_days=5) == []


def test_collect_paired_outcomes_fetches_each_ticker_once():
    decisions = [
        _decision_with_signal(Signal.BULLISH, 0.8, ticker="A", price=100.0),
        _decision_with_signal(Signal.BULLISH, 0.7, ticker="A", price=100.0),
        _decision_with_signal(Signal.BEARISH, 0.6, ticker="B", price=100.0),
    ]
    decisions[2] = decisions[2].model_copy(update={"allocation": None})
    dates = pd.date_range(datetime(2026, 1, 1), periods=10, freq="D")
    market_data = FakeMarketData.from_closes({"A": pd.Series([100.0 + i for i in range(10)], index=dates)})

    pairs = collect_paired_outcomes(decisions, market_data, horizon_days=5)

    assert len(pairs) == 2
    assert market_data.calls["ohlcv_daily"] == 1


def test_evaluate_decisions_end_to_end_on_multiple_tickers():
    decisions = [
        _decision_with_signal(Signal.BULLISH, 0.8, ticker="A", price=100.0),
        _decision_with_signal(Signal.BEARISH, 0.6, ticker="B", price=100.0),
    ]
    dates = pd.date_range(datetime(2026, 1, 1), periods=10, freq="D")
    rising = pd.Series([100.0 + i for i in range(10)], index=dates)
    falling = pd.Series([100.0 - i for i in range(10)], index=dates)
    market_data = FakeMarketData.from_closes({"A": rising, "B": falling})

    result = evaluate_decisions(decisions, market_data, horizon_days=5, deadband=0.01)

    assert result.n == 2
    assert result.hit_rate == 1.0


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
    assert report.reduce_verdicts == 0


def test_replay_determinism_holds_for_the_real_fixture_session():
    assert check_replay_determinism(FIXTURES_DIR) is True


def _result(rank_ic: float | None, ci_lower: float | None) -> EvaluationResult:
    return EvaluationResult(
        n=10,
        rank_ic=rank_ic,
        rank_ic_p_value=None,
        rank_ic_ci=(ci_lower, None),
        hit_rate=None,
        hit_rate_n=0,
        hit_rate_ci=(None, None),
    )


@pytest.mark.parametrize(
    ("open_ic", "closed_ic", "closed_ci_lower", "cleared"),
    [
        (0.1, 0.3, 0.05, True),
        (0.1, 0.3, -0.05, False),  # better, but the CI includes zero
        (0.1, 0.3, 0.0, False),  # the CI must exclude zero, not just touch it
        (0.3, 0.1, 0.05, False),  # CI excludes zero but closed-loop did not beat open-loop
        (0.1, None, None, False),  # undefined closed-loop IC
        (None, 0.3, 0.05, False),  # undefined open-loop IC
        (0.1, 0.3, None, False),  # undefined CI
    ],
)
def test_pre_registered_verdict(open_ic, closed_ic, closed_ci_lower, cleared):
    verdict = pre_registered_verdict(_result(open_ic, None), _result(closed_ic, closed_ci_lower))

    assert verdict is cleared
