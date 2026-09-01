"""Tests for orchestration/reconciliation.py.

Covers leave-one-out credit assignment, entry/exit price outcome
computation, and reading decisions back out of the LangGraph checkpoint.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite import SqliteSaver

from argus.memory.cultural import CulturalMemoryManager
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.orchestration.graph import build_checkpoint_serde, build_graph
from argus.orchestration.reconciliation import (
    ReconciliationReport,
    compact_decisions_jsonl,
    compute_realized_return,
    credit_primary_driver,
    load_decisions_from_checkpoints,
    load_decisions_from_jsonl,
    prune_checkpoints,
    reconcile_decision,
    reconcile_decisions,
    run_reconciliation_pass,
)
from argus.orchestration.state import ARGUSState
from argus.risk import paper_book
from argus.risk.paper_book import PaperBook
from argus.schemas.signals import (
    ARGUSDecision,
    FundamentalSignal,
    MacroContext,
    PositionAllocation,
    Regime,
    SectorSignal,
    SentimentSignal,
    Signal,
    TechnicalSignal,
    VixRegime,
    YieldCurve,
)
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _technical(signal: Signal, conviction: float, ticker: str = "TEST", price: float = 100.0) -> TechnicalSignal:
    """Builds a TechnicalSignal with fixed indicators, varying only the given fields."""
    return TechnicalSignal(
        ticker=ticker,
        current_price=price,
        rsi_14=50.0,
        macd_histogram=0.0,
        bb_percent_b=0.5,
        atr_pct=0.02,
        adx_14=20.0,
        vwap_distance=0.0,
        volume_ratio=1.0,
        momentum_30m=0.0,
        momentum_1d=0.0,
        signal=signal,
        conviction=conviction,
        net_score=0.0,
        timestamp=datetime.now(),
    )


def _fundamental(signal: Signal, conviction: float, ticker: str = "TEST") -> FundamentalSignal:
    """Builds a FundamentalSignal with fixed fields, varying only the given ones."""
    return FundamentalSignal(
        ticker=ticker,
        sector="Technology",
        industry="Software",
        signal=signal,
        conviction=conviction,
        moat_score=5.0,
        reasoning="test",
        data_as_of_date=datetime.now().date(),
        timestamp=datetime.now(),
    )


def _sentiment(signal: Signal, conviction: float, ticker: str = "TEST") -> SentimentSignal:
    """Builds a SentimentSignal with fixed field values, varying only the given fields."""
    return SentimentSignal(
        ticker=ticker,
        finbert_net_score=0.0,
        pct_positive=0.3,
        pct_negative=0.3,
        news_volume_7d=5,
        upcoming_catalyst=False,
        signal=signal,
        conviction=conviction,
        sentiment_decay_risk="LOW",
        reasoning="test",
        timestamp=datetime.now(),
    )


def _macro(regime: Regime = Regime.EXPANSION) -> MacroContext:
    """Builds a MacroContext with fixed field values, varying only the regime."""
    return MacroContext(
        fed_funds=4.0,
        cpi_yoy=3.0,
        unemployment=4.0,
        t10y2y=0.1,
        consumer_sentiment=60.0,
        vix_level=18.0,
        macro_regime=regime,
        regime_confidence=0.7,
        model_healthy=True,
        interest_rate_trend="STABLE",
        yield_curve_shape=YieldCurve.NORMAL,
        vix_regime=VixRegime.MEDIUM,
        vix_percentile=40.0,
        inflation_trajectory="STABLE",
        sector_rotation_signal=SectorSignal.GROWTH_FAVORED,
        agent_multipliers={"fundamental": 1.0, "technical": 1.0, "sentiment": 1.0},
        timestamp=datetime.now(),
    )


def _allocation(ticker: str = "TEST") -> PositionAllocation:
    """Builds a PositionAllocation with fixed field values, varying only the ticker."""
    return PositionAllocation(
        ticker=ticker,
        allocation_pct=0.10,
        allocation_usd=10_000.0,
        stop_loss=90.0,
        thesis="test thesis",
        composite_conviction=0.7,
        time_horizon="30 days",
    )


class _FakeMarketData:
    """Minimal MarketDataProvider double: only ohlcv_daily is used by reconciliation."""

    def __init__(self, closes: dict[str, pd.Series]) -> None:
        """Stores the fixed per-ticker close series to serve from ohlcv_daily."""
        self._closes = closes

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Returns the fixed close series for the given ticker, ignoring period."""
        return pd.DataFrame({"close": self._closes[ticker]})


def _ten_day_series(start: datetime, start_price: float = 100.0, ticker: str = "TEST") -> _FakeMarketData:
    """Builds a 10-day daily close series rising by 1.0 per day from start_price."""
    dates = pd.date_range(start=start, periods=10, freq="D")
    closes = pd.Series([start_price + i for i in range(10)], index=dates)
    return _FakeMarketData({ticker: closes})


def _aggregated_decision(
    technical: TechnicalSignal,
    macro: MacroContext,
    fundamental: FundamentalSignal | None = None,
    sentiment: SentimentSignal | None = None,
    *,
    reliability: dict[str, float] | None = None,
    session_timestamp: datetime | None = None,
    allocation: PositionAllocation | None = None,
) -> ARGUSDecision:
    """Builds a TEST decision whose aggregated signal is the real aggregator's output."""
    aggregated = HybridSignalAggregator().aggregate(
        technical, macro, fundamental, sentiment, reliability=reliability
    )
    return ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime.now() if session_timestamp is None else session_timestamp,
        technical=technical,
        fundamental=fundamental,
        sentiment=sentiment,
        macro=macro,
        aggregated=aggregated,
        allocation=allocation,
    )


def _write_decisions_log(path: Path, *decisions: ARGUSDecision) -> Path:
    """Writes `decisions` as one JSON object per line at `path` and returns that path."""
    with open(path, "w", encoding="utf-8") as f:
        for decision in decisions:
            f.write(decision.model_dump_json() + "\n")
    return path


# ---------------------------------------------------------------------------
# credit_primary_driver
# ---------------------------------------------------------------------------


def test_credit_primary_driver_returns_unknown_with_no_signals():
    """A decision with no agent signals credits no driver."""
    decision = ARGUSDecision(ticker="TEST", session_timestamp=datetime.now())
    assert credit_primary_driver(decision) == "unknown"


def test_credit_primary_driver_returns_the_sole_signal_when_only_one_present():
    """With only one agent signal present, that agent is credited by default."""
    decision = _aggregated_decision(_technical(Signal.BULLISH, 0.8), _macro())
    assert credit_primary_driver(decision) == "technical"


def test_credit_primary_driver_credits_the_dominant_agent():
    """Leave-one-out ablation credits whichever agent's removal swings conviction most."""
    decision = _aggregated_decision(
        _technical(Signal.BULLISH, 0.9), _macro(), _fundamental(Signal.BULLISH, 0.1)
    )
    assert credit_primary_driver(decision) == "technical"


def test_credit_primary_driver_is_symmetric_under_relabeling():
    """Swapping which agent is dominant should swap the credited driver."""
    decision = _aggregated_decision(
        _technical(Signal.BULLISH, 0.1), _macro(), _fundamental(Signal.BULLISH, 0.9)
    )
    assert credit_primary_driver(decision) == "fundamental"


def test_credit_primary_driver_replays_the_baseline_reliability_in_each_ablation():
    """Ablation must reuse the exact reliability dict the baseline aggregation used.

    Regression test for X3: without this, every ablated rerun silently falls
    back to aggregate()'s unweighted reliability_mult=1.0 default, so the
    baseline (computed WITH reliability) and each ablated rerun (computed
    WITHOUT it) can disagree on consensus direction before any agent is even
    removed — flipping credit to argmax(baseline_votes) regardless of which
    agent was actually ablated.
    """
    reliability = {"technical": 0.95, "fundamental": 0.20, "sentiment": 0.50}
    decision = _aggregated_decision(
        _technical(Signal.BULLISH, 0.9),
        _macro(),
        _fundamental(Signal.BEARISH, 0.85),
        _sentiment(Signal.NEUTRAL, 0.3),
        reliability=reliability,
    )

    seen_reliability: list[object] = []

    class _SpyAggregator(HybridSignalAggregator):
        def aggregate(self, *args, **kwargs):
            seen_reliability.append(kwargs.get("reliability"))
            return super().aggregate(*args, **kwargs)

    credit_primary_driver(decision, aggregator=_SpyAggregator())

    assert seen_reliability
    assert all(r == reliability for r in seen_reliability)


# ---------------------------------------------------------------------------
# compute_realized_return
# ---------------------------------------------------------------------------


def test_compute_realized_return_pairs_entry_and_horizon_close():
    """Realized return pairs the entry-day close with the close N horizon days later."""
    start = datetime(2026, 1, 1)
    technical = _technical(Signal.BULLISH, 0.8, price=100.0)
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=technical,
        allocation=_allocation(),
    )
    market_data = _ten_day_series(start, start_price=100.0)

    outcome = compute_realized_return(decision, market_data, horizon_days=5)

    assert outcome is not None
    actual_return_pct, holding_days, exit_reason = outcome
    assert actual_return_pct == pytest.approx((105.0 - 100.0) / 100.0)
    assert holding_days == 5
    assert "5d" in exit_reason


def test_compute_realized_return_none_without_allocation():
    """A decision with no position allocation has no realized return to compute."""
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime(2026, 1, 1),
        technical=_technical(Signal.BULLISH, 0.8),
    )
    market_data = _ten_day_series(datetime(2026, 1, 1))
    assert compute_realized_return(decision, market_data, horizon_days=5) is None


def test_compute_realized_return_none_without_technical_signal():
    """A decision with no technical signal has no entry price to compute a return from."""
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime(2026, 1, 1),
        allocation=_allocation(),
    )
    market_data = _ten_day_series(datetime(2026, 1, 1))
    assert compute_realized_return(decision, market_data, horizon_days=5) is None


def test_compute_realized_return_none_with_zero_weight_allocation():
    """A decision allocated 0% has no position to reconcile.

    Same as a decision with no allocation at all. Regression test: filtering
    0%-weight positions belongs at the reconciliation boundary,
    not at graph.py's decision-building step (which would break the audit
    trail of "we evaluated this ticker and chose not to hold").
    """
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime(2026, 1, 1),
        technical=_technical(Signal.BULLISH, 0.8),
        allocation=PositionAllocation(
            ticker="TEST",
            allocation_pct=0.0,
            allocation_usd=0.0,
            stop_loss=90.0,
            thesis="no position taken",
            composite_conviction=0.0,
            time_horizon="30 days",
        ),
    )
    market_data = _ten_day_series(datetime(2026, 1, 1))
    assert compute_realized_return(decision, market_data, horizon_days=5) is None


def test_compute_realized_return_none_when_horizon_not_yet_reached():
    """No return is computed until price data reaches the requested horizon."""
    start = datetime(2026, 1, 1)
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=_technical(Signal.BULLISH, 0.8),
        allocation=_allocation(),
    )
    market_data = _ten_day_series(start)  # only 10 days, short of the 30-day horizon requested below
    assert compute_realized_return(decision, market_data, horizon_days=30) is None


# ---------------------------------------------------------------------------
# reconcile_decision / reconcile_decisions
# ---------------------------------------------------------------------------


def test_reconcile_decision_stores_outcome_with_ablated_primary_driver():
    """Reconciling a decision stores its outcome tagged with the credited driver."""
    start = datetime(2026, 1, 1)
    decision = _aggregated_decision(
        _technical(Signal.BULLISH, 0.9, price=100.0),
        _macro(),
        _fundamental(Signal.BULLISH, 0.1),
        session_timestamp=start,
        allocation=_allocation(),
    )
    market_data = _ten_day_series(start, start_price=100.0)
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)

    stored = reconcile_decision(decision, market_data, cultural, horizon_days=5)

    assert stored is True
    cultural.store_trade_outcome.assert_called_once()
    kwargs = cultural.store_trade_outcome.call_args.kwargs
    assert kwargs["decision"] is decision
    assert kwargs["actual_return_pct"] == pytest.approx(0.05)
    assert kwargs["holding_days"] == 5
    assert kwargs["primary_driver"] == "technical"


def test_reconcile_decision_returns_false_and_stores_nothing_with_no_position():
    """A decision with no position produces no stored outcome."""
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime(2026, 1, 1),
        technical=_technical(Signal.BULLISH, 0.8),
    )
    market_data = _ten_day_series(datetime(2026, 1, 1))
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)

    assert reconcile_decision(decision, market_data, cultural, horizon_days=5) is False
    cultural.store_trade_outcome.assert_not_called()


def test_reconcile_decisions_counts_only_the_ones_actually_reconciled():
    """The batch count reflects only decisions that actually had an outcome to store."""
    start = datetime(2026, 1, 1)
    reconcilable = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=_technical(Signal.BULLISH, 0.8, price=100.0),
        allocation=_allocation(),
    )
    not_reconcilable = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=_technical(Signal.BULLISH, 0.8),
        # no allocation -> nothing to reconcile
    )
    market_data = _ten_day_series(start, start_price=100.0)
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)
    cultural.already_reconciled.return_value = set()

    count = reconcile_decisions(
        [reconcilable, not_reconcilable], market_data, cultural, horizon_days=5
    )

    assert count == 1
    assert cultural.store_trade_outcome.call_count == 1


def test_reconcile_decisions_skips_decisions_already_reconciled():
    """A decision cultural already has a stored outcome for is not touched again."""
    start = datetime(2026, 1, 1)
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=_technical(Signal.BULLISH, 0.8, price=100.0),
        allocation=_allocation(),
    )
    market_data = _ten_day_series(start, start_price=100.0)
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)
    cultural.already_reconciled.return_value = {decision.decision_id}

    count = reconcile_decisions([decision], market_data, cultural, horizon_days=5)

    assert count == 0
    cultural.store_trade_outcome.assert_not_called()


def test_reconcile_decisions_fetches_each_ticker_price_history_once():
    """Two decisions on the same ticker share a single ohlcv_daily fetch."""
    start = datetime(2026, 1, 1)
    first = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=_technical(Signal.BULLISH, 0.8, price=100.0),
        allocation=_allocation(),
    )
    second = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=_technical(Signal.BULLISH, 0.8, price=101.0),
        allocation=_allocation(),
    )
    market_data = _ten_day_series(start, start_price=100.0)
    market_data.ohlcv_daily = mock.Mock(wraps=market_data.ohlcv_daily)
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)
    cultural.already_reconciled.return_value = set()

    count = reconcile_decisions([first, second], market_data, cultural, horizon_days=5)

    assert count == 2
    market_data.ohlcv_daily.assert_called_once_with("TEST")


# ---------------------------------------------------------------------------
# load_decisions_from_checkpoints
# ---------------------------------------------------------------------------


def test_load_decisions_from_checkpoints_round_trips_a_real_graph_run(tmp_path):
    """Checkpoints round-trip real ARGUSDecision objects, not degraded dicts."""
    universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "XOM"]

    def per_ticker_llm(fixture_file: str) -> FixtureLLMClient:
        with open(FIXTURES_DIR / "llm_responses" / fixture_file) as f:
            responses = json.load(f)

        def key_fn(user_prompt: str) -> str:
            for ticker in universe:
                if ticker in user_prompt:
                    return ticker
            raise KeyError(user_prompt[:200])

        return FixtureLLMClient(responses, key_fn=key_fn)

    def portfolio_llm() -> FixtureLLMClient:
        with open(FIXTURES_DIR / "llm_responses" / "portfolio.json") as f:
            response = json.load(f)
        return FixtureLLMClient({"only": json.dumps(response)}, key_fn=lambda _p: "only")

    with open(FIXTURES_DIR / "market_data" / "session_states.json") as f:
        session_states = json.load(f)

    db_path = str(tmp_path / "argus_graph_test.db")
    graph = build_graph(
        market_data=FixtureMarketDataProvider(fixtures_dir=FIXTURES_DIR / "market_data"),
        fundamental_llm=per_ticker_llm("fundamental.json"),
        sentiment_llm=per_ticker_llm("sentiment.json"),
        portfolio_llm=portfolio_llm(),
        checkpoint_db_path=db_path,
    )

    state = ARGUSState(
        ticker=universe[0],
        universe=universe,
        total_wealth=100_000.0,
        invest_pct=0.8,
        risk_tolerance="MODERATE",
        backtest_mode=False,
        session_seed=None,
        as_of=None,
        price_history={},
        session_states=session_states,
        macro_context=None,
        technical_signals={},
        fundamental_signals={},
        sentiment_signals={},
        cultural_memory={"wisdom": [], "warnings": []},
        risk_assessments={},
        aggregated_signals={},
        portfolio_allocation=None,
        decisions=[],
        errors=[],
    )

    with mock.patch("argus.orchestration.graph.get_cultural_memory") as mock_get_cultural_memory:
        mock_get_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
            store_decision_snapshot=mock.Mock(),
        )
        graph.invoke({**state}, {"configurable": {"thread_id": "test-thread"}})

    decisions = load_decisions_from_checkpoints(db_path)

    assert decisions, "expected at least one decision to round-trip"
    assert all(isinstance(d, ARGUSDecision) for d in decisions)
    assert {d.ticker for d in decisions} <= set(universe)


# ---------------------------------------------------------------------------
# compact_decisions_jsonl
# ---------------------------------------------------------------------------


def test_compact_decisions_jsonl_drops_sessions_before_cutoff(tmp_path):
    """Decisions older than cutoff are dropped; everything at or after it is kept."""
    path = _write_decisions_log(
        tmp_path / "decisions.jsonl",
        ARGUSDecision(ticker="OLD", session_timestamp=datetime(2026, 1, 1)),
        ARGUSDecision(ticker="EXACT", session_timestamp=datetime(2026, 1, 10)),
        ARGUSDecision(ticker="NEW", session_timestamp=datetime(2026, 1, 20)),
    )

    retained = compact_decisions_jsonl(str(path), cutoff=datetime(2026, 1, 10))

    assert retained == 2
    tickers = {d.ticker for d in load_decisions_from_jsonl(str(path))}
    assert tickers == {"EXACT", "NEW"}


def test_compact_decisions_jsonl_missing_file_is_a_noop():
    """A path that doesn't exist yet returns 0 and writes nothing."""
    assert compact_decisions_jsonl("/nonexistent/decisions.jsonl", cutoff=datetime(2026, 1, 1)) == 0


def test_compact_decisions_jsonl_nothing_to_drop_leaves_the_file_untouched(tmp_path):
    """When every decision is at or after cutoff, the file is not rewritten."""
    path = _write_decisions_log(
        tmp_path / "decisions.jsonl",
        ARGUSDecision(ticker="NEW", session_timestamp=datetime(2026, 1, 20)),
    )
    original_mtime = path.stat().st_mtime_ns

    retained = compact_decisions_jsonl(str(path), cutoff=datetime(2026, 1, 1))

    assert retained == 1
    assert path.stat().st_mtime_ns == original_mtime


# ---------------------------------------------------------------------------
# prune_checkpoints
# ---------------------------------------------------------------------------


def _put_checkpoint(
    db_path: str, thread_id: str, ts: datetime, decisions: list[ARGUSDecision] | None = None
) -> None:
    """Writes one checkpoint for `thread_id` stamped with `ts`.

    Carries `decisions` if given.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint: Checkpoint = {
        "v": 1,
        "ts": ts.isoformat(),
        "id": thread_id,
        "channel_values": {} if decisions is None else {"decisions": decisions},
        "channel_versions": {} if decisions is None else {"decisions": "1"},
        "versions_seen": {},
        "updated_channels": None,
    }
    metadata: CheckpointMetadata = {"source": "input", "step": 1}

    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        SqliteSaver(conn, serde=build_checkpoint_serde()).put(config, checkpoint, metadata, {})
    finally:
        conn.close()


def test_prune_checkpoints_deletes_only_threads_older_than_cutoff(tmp_path):
    """A thread whose newest checkpoint predates cutoff is deleted.

    A newer thread survives.
    """
    db_path = str(tmp_path / "argus_graph.db")
    _put_checkpoint(db_path, "stale-thread", datetime(2026, 1, 1))
    _put_checkpoint(db_path, "fresh-thread", datetime(2026, 1, 20))

    deleted = prune_checkpoints(db_path, cutoff=datetime(2026, 1, 10))

    assert deleted == 1
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpoint_threads = {
        row[0] for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    }
    conn.close()
    assert checkpoint_threads == {"fresh-thread"}


def test_prune_checkpoints_missing_db_is_a_noop():
    """A checkpoint database that doesn't exist yet returns 0 rather than raising."""
    assert prune_checkpoints("/nonexistent/argus_graph.db", cutoff=datetime(2026, 1, 1)) == 0


def test_prune_checkpoints_nothing_stale_deletes_nothing(tmp_path):
    """Every thread newer than cutoff survives untouched."""
    db_path = str(tmp_path / "argus_graph.db")
    _put_checkpoint(db_path, "fresh-thread", datetime.now())

    deleted = prune_checkpoints(db_path, cutoff=datetime.now() - timedelta(days=30))

    assert deleted == 0


# ---------------------------------------------------------------------------
# run_reconciliation_pass: the end-to-end pass that reads decisions, reconciles
# them, and bounds every store it touches.
# ---------------------------------------------------------------------------


def _matured_decision(ticker: str, start: datetime, price: float = 100.0) -> ARGUSDecision:
    """A decision with a position that has cleared a 5-day horizon by `start` + 5 days."""
    return ARGUSDecision(
        ticker=ticker,
        session_timestamp=start,
        technical=_technical(Signal.BULLISH, 0.8, ticker=ticker, price=price),
        allocation=_allocation(ticker=ticker),
    )


def _autospec_cultural(expired: int = 0) -> mock.MagicMock:
    """An autospec CulturalMemoryManager stubbed for reconcile_decisions and expiry."""
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)
    cultural.already_reconciled.return_value = set()
    cultural.expire_pending_snapshots.return_value = expired
    return cultural


def test_run_reconciliation_pass_over_matured_decisions_produces_outcome_and_equity(tmp_path):
    """A full pass over matured decisions stores the outcome and updates paper equity.

    The decision's realized return is compounded onto paper equity.
    """
    start = datetime(2026, 1, 1)
    decisions_log = _write_decisions_log(
        tmp_path / "decisions.jsonl", _matured_decision("TEST", start)
    )

    book_path = tmp_path / "paper_equity.json"
    paper_book.save(PaperBook(equity=100_000.0, high_water_mark=100_000.0), str(book_path))

    market_data = _ten_day_series(start, start_price=100.0)
    cultural = _autospec_cultural()

    report = run_reconciliation_pass(
        market_data,
        cultural,
        str(book_path),
        decisions_log_path=str(decisions_log),
        horizon_days=5,
    )

    assert isinstance(report, ReconciliationReport)
    assert report.decisions_loaded == 1
    assert report.outcomes_stored == 1
    assert report.paper_book_updated is True
    assert report.equity == pytest.approx(105_000.0)
    assert report.drawdown == pytest.approx(0.0)
    assert report.decisions_compacted is not None
    assert report.checkpoints_pruned is None
    assert report.pending_snapshots_expired == 0
    assert report.errors == []
    cultural.store_trade_outcome.assert_called_once()

    reloaded = paper_book.load(str(book_path))
    assert reloaded.equity == pytest.approx(105_000.0)


def test_run_reconciliation_pass_reading_from_checkpoints_bounds_checkpoints(tmp_path):
    """With only a checkpoint path, decisions are read from it and it alone is bounded."""
    start = datetime(2026, 1, 1)
    db_path = str(tmp_path / "argus_graph.db")
    _put_checkpoint(db_path, "session-thread", start, [_matured_decision("TEST", start)])

    market_data = _ten_day_series(start, start_price=100.0)
    cultural = _autospec_cultural()

    report = run_reconciliation_pass(
        market_data,
        cultural,
        str(tmp_path / "paper_equity.json"),
        checkpoint_db_path=db_path,
        horizon_days=5,
    )

    assert report.decisions_loaded == 1
    assert report.outcomes_stored == 1
    assert report.decisions_compacted is None
    assert report.checkpoints_pruned is not None
    assert report.errors == []


def test_run_reconciliation_pass_given_both_paths_bounds_both(tmp_path):
    """With both paths given, the decisions log wins as the read source.

    Both stores — the decisions log and the checkpoint database — are bounded.
    """
    start = datetime(2026, 1, 1)
    decisions_log = _write_decisions_log(
        tmp_path / "decisions.jsonl", _matured_decision("TEST", start)
    )

    db_path = str(tmp_path / "argus_graph.db")
    _put_checkpoint(db_path, "stale-thread", datetime(2020, 1, 1))

    market_data = _ten_day_series(start, start_price=100.0)
    cultural = _autospec_cultural()

    report = run_reconciliation_pass(
        market_data,
        cultural,
        str(tmp_path / "paper_equity.json"),
        decisions_log_path=str(decisions_log),
        checkpoint_db_path=db_path,
        horizon_days=5,
    )

    # The checkpoint's only thread carries no decisions -> the log (with one) is the read source
    assert report.decisions_loaded == 1
    assert report.decisions_compacted is not None
    assert report.checkpoints_pruned is not None
    assert report.checkpoints_pruned >= 1
    assert report.errors == []


def test_run_reconciliation_pass_one_store_failing_leaves_the_others_bounded_and_names_it(tmp_path):
    """A corrupt checkpoint database fails only that store.

    decisions.jsonl compaction still runs.
    """
    start = datetime(2026, 1, 1)
    decisions_log = _write_decisions_log(
        tmp_path / "decisions.jsonl", _matured_decision("TEST", start)
    )

    db_path = tmp_path / "argus_graph.db"
    db_path.write_bytes(b"not a sqlite database")

    market_data = _ten_day_series(start, start_price=100.0)
    cultural = _autospec_cultural(expired=3)

    report = run_reconciliation_pass(
        market_data,
        cultural,
        str(tmp_path / "paper_equity.json"),
        decisions_log_path=str(decisions_log),
        checkpoint_db_path=str(db_path),
        horizon_days=5,
    )

    assert report.decisions_compacted is not None
    assert report.checkpoints_pruned is None
    assert report.pending_snapshots_expired == 3
    assert report.outcomes_stored == 1
    assert len(report.errors) == 1
    assert "checkpoint" in report.errors[0]


def test_run_reconciliation_pass_paper_book_not_partially_applied_when_its_step_fails(
    tmp_path, monkeypatch
):
    """A paper-book save failure leaves the persisted book exactly as it was.

    The failure is named in the report's errors.
    """
    start = datetime(2026, 1, 1)
    decisions_log = _write_decisions_log(
        tmp_path / "decisions.jsonl", _matured_decision("TEST", start)
    )

    book_path = tmp_path / "paper_equity.json"
    paper_book.save(PaperBook(equity=12_345.0, high_water_mark=12_345.0), str(book_path))
    original_bytes = book_path.read_bytes()

    def _boom(_book, _path) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(paper_book, "save", _boom)

    market_data = _ten_day_series(start, start_price=100.0)
    cultural = _autospec_cultural()

    report = run_reconciliation_pass(
        market_data,
        cultural,
        str(book_path),
        decisions_log_path=str(decisions_log),
        horizon_days=5,
    )

    assert report.paper_book_updated is False
    assert report.equity == 0.0
    assert len(report.errors) == 1
    assert "paper-book update failed" in report.errors[0]
    assert book_path.read_bytes() == original_bytes
    # Independent of the failed paper-book step, the other stores still ran
    assert report.decisions_compacted is not None
    assert report.pending_snapshots_expired == 0
