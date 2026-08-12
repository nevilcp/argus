"""
tests/test_reconciliation.py

Tests for argus/orchestration/reconciliation.py: leave-one-out credit
assignment, entry/exit price outcome computation, and reading decisions back
out of the LangGraph checkpoint.
"""

from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from argus.memory.cultural import CulturalMemoryManager
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.orchestration.graph import build_graph
from argus.orchestration.reconciliation import (
    compute_realized_return,
    credit_primary_driver,
    load_decisions_from_checkpoints,
    reconcile_decision,
    reconcile_decisions,
)
from argus.orchestration.state import ARGUSState
from argus.schemas.signals import (
    ARGUSDecision,
    FundamentalSignal,
    MacroContext,
    PositionAllocation,
    Regime,
    SectorSignal,
    Signal,
    TechnicalSignal,
    VixRegime,
    YieldCurve,
)
from argus.seams import FixtureLLMClient, FixtureMarketDataProvider

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _technical(signal: Signal, conviction: float, ticker: str = "TEST", price: float = 100.0) -> TechnicalSignal:
    """Builds a TechnicalSignal with fixed indicator values, varying only the given fields.

    Args:
        signal: Signal direction to assign.
        conviction: Confidence score to assign.
        ticker: Ticker symbol.
        price: Current price.

    Returns:
        A TechnicalSignal fixture.
    """
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
    """Builds a FundamentalSignal with fixed field values, varying only the given fields.

    Args:
        signal: Signal direction to assign.
        conviction: Confidence score to assign.
        ticker: Ticker symbol.

    Returns:
        A FundamentalSignal fixture.
    """
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


def _macro(regime: Regime = Regime.EXPANSION) -> MacroContext:
    """Builds a MacroContext with fixed field values, varying only the regime.

    Args:
        regime: Macro regime to assign.

    Returns:
        A MacroContext fixture.
    """
    return MacroContext(
        fed_funds=4.0,
        cpi_yoy=3.0,
        unemployment=4.0,
        t10y2y=0.1,
        consumer_sentiment=60.0,
        vix_level=18.0,
        macro_regime=regime,
        regime_confidence=0.7,
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
    """Builds a PositionAllocation with fixed field values, varying only the ticker.

    Args:
        ticker: Ticker symbol.

    Returns:
        A PositionAllocation fixture.
    """
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
        """Stores the fixed per-ticker close series to serve from ohlcv_daily.

        Args:
            closes: Ticker to close-price series.
        """
        self._closes = closes

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Returns the fixed close series for the given ticker, ignoring period.

        Args:
            ticker: Ticker to look up.
            period: Lookback period (ignored).

        Returns:
            A DataFrame with the fixed close series.
        """
        return pd.DataFrame({"close": self._closes[ticker]})


def _ten_day_series(start: datetime, start_price: float = 100.0, ticker: str = "TEST") -> _FakeMarketData:
    """Builds a 10-day daily close series rising by 1.0 per day from start_price.

    Args:
        start: First date in the series.
        start_price: Close price on the first day.
        ticker: Ticker symbol the series is keyed under.

    Returns:
        A _FakeMarketData backed by the generated series.
    """
    dates = pd.date_range(start=start, periods=10, freq="D")
    closes = pd.Series([start_price + i for i in range(10)], index=dates)
    return _FakeMarketData({ticker: closes})


# ---------------------------------------------------------------------------
# credit_primary_driver
# ---------------------------------------------------------------------------


def test_credit_primary_driver_returns_unknown_with_no_signals():
    """A decision with no agent signals credits no driver."""
    decision = ARGUSDecision(ticker="TEST", session_timestamp=datetime.now())
    assert credit_primary_driver(decision) == "unknown"


def test_credit_primary_driver_returns_the_sole_signal_when_only_one_present():
    """With only one agent signal present, that agent is credited by default."""
    technical = _technical(Signal.BULLISH, 0.8)
    macro = _macro()
    aggregator = HybridSignalAggregator()
    aggregated = aggregator.aggregate(technical, macro, None, None)

    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime.now(),
        technical=technical,
        macro=macro,
        aggregated=aggregated,
    )
    assert credit_primary_driver(decision) == "technical"


def test_credit_primary_driver_credits_the_dominant_agent():
    """Leave-one-out ablation credits whichever agent's removal swings conviction most."""
    technical = _technical(Signal.BULLISH, 0.9)
    fundamental = _fundamental(Signal.BULLISH, 0.1)
    macro = _macro()
    aggregator = HybridSignalAggregator()
    aggregated = aggregator.aggregate(technical, macro, fundamental, None)

    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime.now(),
        technical=technical,
        fundamental=fundamental,
        macro=macro,
        aggregated=aggregated,
    )
    assert credit_primary_driver(decision) == "technical"


def test_credit_primary_driver_is_symmetric_under_relabeling():
    """Swapping which agent is dominant should swap the credited driver."""
    technical = _technical(Signal.BULLISH, 0.1)
    fundamental = _fundamental(Signal.BULLISH, 0.9)
    macro = _macro()
    aggregator = HybridSignalAggregator()
    aggregated = aggregator.aggregate(technical, macro, fundamental, None)

    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime.now(),
        technical=technical,
        fundamental=fundamental,
        macro=macro,
        aggregated=aggregated,
    )
    assert credit_primary_driver(decision) == "fundamental"


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
    """Reconciling a decision stores the realized outcome tagged with its credited driver."""
    start = datetime(2026, 1, 1)
    technical = _technical(Signal.BULLISH, 0.9, price=100.0)
    fundamental = _fundamental(Signal.BULLISH, 0.1)
    macro = _macro()
    aggregated = HybridSignalAggregator().aggregate(technical, macro, fundamental, None)

    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=technical,
        fundamental=fundamental,
        macro=macro,
        aggregated=aggregated,
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

    count = reconcile_decisions(
        [reconcilable, not_reconcilable], market_data, cultural, horizon_days=5
    )

    assert count == 1
    assert cultural.store_trade_outcome.call_count == 1


# ---------------------------------------------------------------------------
# load_decisions_from_checkpoints
# ---------------------------------------------------------------------------


def test_load_decisions_from_checkpoints_round_trips_a_real_graph_run(tmp_path):
    """Checkpoints round-trip real ARGUSDecision objects, not degraded dicts, per ticker."""
    import json

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
            get_agent_accuracy=mock.Mock(return_value=0.5),
            store_decision_snapshot=mock.Mock(),
        )
        graph.invoke({**state}, {"configurable": {"thread_id": "test-thread"}})

    decisions = load_decisions_from_checkpoints(db_path)

    assert decisions, "expected at least one decision to round-trip"
    assert all(isinstance(d, ARGUSDecision) for d in decisions)
    assert {d.ticker for d in decisions} <= set(universe)
