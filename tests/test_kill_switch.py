"""
tests/test_kill_switch.py

Tests for argus/risk/kill_switch.py (drawdown/VIX circuit breaker) and
argus/risk/paper_book.py (the synthetic equity curve that feeds it).

fetch_vix is mocked throughout; no test starts the daemon thread or sleeps —
_check() is driven directly, following the pattern used to reproduce these
findings originally.
"""

from __future__ import annotations

from datetime import datetime
from unittest import mock

import pandas as pd
import pytest

import argus.risk.kill_switch as kill_switch_module
from argus.config import settings
from argus.orchestration.collector import run_collection_cycle
from argus.risk import paper_book
from argus.risk.kill_switch import KillSwitch, initialize_kill_switch
from argus.risk.paper_book import PaperBook, compute_run_returns
from argus.schemas.signals import ARGUSDecision, PositionAllocation, Signal, TechnicalSignal


@pytest.fixture(autouse=True)
def _reset_kill_switch_singleton() -> None:
    """Clears the module-level KillSwitch singleton before and after each test."""
    kill_switch_module._kill_switch = None
    yield
    kill_switch_module._kill_switch = None


def _make_ks(risk_tolerance: str = "MODERATE", inception: float = 100_000.0) -> KillSwitch:
    """Builds a KillSwitch with inception state set directly, bypassing start()'s thread.

    Args:
        risk_tolerance: Risk tier string.
        inception: Seed value for inception/current/high-water-mark.

    Returns:
        A KillSwitch ready for _check() to be driven directly.
    """
    ks = KillSwitch(risk_tolerance)
    ks._portfolio_inception_value = inception
    ks._current_portfolio_value = inception
    ks._high_water_mark = inception
    return ks


# ---------------------------------------------------------------------------
# Drawdown thresholds (KS-4, KS-5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tolerance,threshold",
    [("CONSERVATIVE", 0.08), ("MODERATE", 0.12), ("AGGRESSIVE", 0.18)],
)
@mock.patch("argus.data.fetchers.fetch_vix", return_value=15.0)
def test_drawdown_halts_at_tier_threshold(mock_vix, tolerance, threshold):
    """The drawdown halt fires once realized drawdown reaches each tier's own threshold."""
    ks = _make_ks(tolerance, inception=100_000.0)
    ks.update_portfolio_value(100_000.0 * (1 - threshold))
    ks._check()
    assert ks.is_halted


@mock.patch("argus.data.fetchers.fetch_vix", return_value=15.0)
def test_drawdown_does_not_halt_below_threshold(mock_vix):
    """A drawdown just short of the tier threshold does not halt."""
    ks = _make_ks("MODERATE", inception=100_000.0)
    ks.update_portfolio_value(100_000.0 * (1 - 0.12) + 1.0)
    ks._check()
    assert not ks.is_halted


@mock.patch("argus.data.fetchers.fetch_vix", return_value=15.0)
def test_zero_portfolio_value_registers_as_real_drawdown(mock_vix):
    """KS-4: a literal 0.0 portfolio value must be treated as a real value, not 'unset'."""
    ks = _make_ks("CONSERVATIVE", inception=100_000.0)
    ks.update_portfolio_value(0.0)
    ks._check()
    assert ks.is_halted
    assert ks.status.realized_drawdown == pytest.approx(1.0)


@mock.patch("argus.data.fetchers.fetch_vix", return_value=15.0)
def test_peak_to_trough_drawdown_not_inception_relative(mock_vix):
    """KS-5: drawdown is measured from the running peak, not the stale inception value.

    100k -> 150k (new peak, no drawdown) -> 105k: -5% vs inception (a gain,
    wouldn't halt under the old logic) but -30% vs the $150k peak, well past
    even AGGRESSIVE's 18% threshold.
    """
    ks = _make_ks("AGGRESSIVE", inception=100_000.0)

    ks.update_portfolio_value(150_000.0)
    ks._check()
    assert not ks.is_halted

    ks.update_portfolio_value(105_000.0)
    ks._check()
    assert ks.is_halted
    assert ks.status.realized_drawdown == pytest.approx((150_000.0 - 105_000.0) / 150_000.0)


def test_reset_clears_halt_and_reseeds_high_water_mark():
    """reset() clears the halt/blackout and re-bases both inception and the peak."""
    ks = _make_ks("MODERATE", inception=100_000.0)
    with mock.patch("argus.data.fetchers.fetch_vix", return_value=15.0):
        ks.update_portfolio_value(80_000.0)
        ks._check()
    assert ks.is_halted

    ks.reset(120_000.0)

    assert not ks.is_halted
    assert ks.new_positions_allowed
    assert ks.status.realized_drawdown == 0.0


# ---------------------------------------------------------------------------
# VIX blackout gate (KS-3)
# ---------------------------------------------------------------------------


def test_vix_blackout_sets_and_clears_on_genuine_readings():
    """The blackout sets on a high reading and clears on a later genuine low reading."""
    ks = _make_ks("MODERATE")
    with mock.patch("argus.data.fetchers.fetch_vix", return_value=40.0):
        ks._check()
    assert not ks.new_positions_allowed

    with mock.patch("argus.data.fetchers.fetch_vix", return_value=15.0):
        ks._check()
    assert ks.new_positions_allowed


def test_vix_fetch_failure_does_not_clear_an_active_blackout():
    """KS-3: a fetch failure must never clear a blackout that's already set."""
    ks = _make_ks("MODERATE")
    with mock.patch("argus.data.fetchers.fetch_vix", return_value=40.0):
        ks._check()
    assert not ks.new_positions_allowed

    with mock.patch("argus.data.fetchers.fetch_vix", side_effect=RuntimeError("network down")):
        ks._check()
    assert not ks.new_positions_allowed


def test_vix_fetch_failure_sets_blackout_after_n_consecutive_failures():
    """KS-3: repeated fetch failures fail closed instead of silently trusting a stale reading."""
    ks = _make_ks("MODERATE")
    assert ks.new_positions_allowed

    with mock.patch("argus.data.fetchers.fetch_vix", side_effect=RuntimeError("down")):
        for _ in range(KillSwitch._VIX_FAILURE_HALT_THRESHOLD - 1):
            ks._check()
            assert ks.new_positions_allowed
        ks._check()

    assert not ks.new_positions_allowed


def test_vix_fetch_success_resets_failure_counter():
    """A single genuine reading between failures resets the consecutive-failure count."""
    ks = _make_ks("MODERATE")
    with mock.patch("argus.data.fetchers.fetch_vix", side_effect=RuntimeError("down")):
        for _ in range(KillSwitch._VIX_FAILURE_HALT_THRESHOLD - 1):
            ks._check()

    with mock.patch("argus.data.fetchers.fetch_vix", return_value=15.0):
        ks._check()
    assert ks._consecutive_vix_failures == 0

    with mock.patch("argus.data.fetchers.fetch_vix", side_effect=RuntimeError("down")):
        for _ in range(KillSwitch._VIX_FAILURE_HALT_THRESHOLD - 1):
            ks._check()
    assert ks.new_positions_allowed  # only 4 consecutive failures since the reset, not 5


# ---------------------------------------------------------------------------
# Halt persistence and restore-on-init
# ---------------------------------------------------------------------------


def test_halt_restores_from_a_persisted_dump(tmp_path, monkeypatch):
    """A halt dump written by one KillSwitch instance restores into a fresh one."""
    monkeypatch.chdir(tmp_path)
    seed = _make_ks("MODERATE", inception=100_000.0)
    seed._current_portfolio_value = 80_000.0
    seed._halt_reason = "Drawdown 20.0% >= 12% limit (MODERATE)"
    seed._halt_time = datetime.now()
    seed._persist_halt_event(0.20, 15.0)

    halt_file = kill_switch_module._find_latest_halt_file()
    assert halt_file is not None

    restored = KillSwitch("MODERATE")
    restored._restore_halt_from_file(halt_file)

    assert restored.is_halted
    assert restored._halt_reason == seed._halt_reason


def test_initialize_kill_switch_restores_halt_on_reinit(tmp_path, monkeypatch):
    """initialize_kill_switch() re-applies a leftover halt file rather than starting clean."""
    monkeypatch.chdir(tmp_path)
    seed = _make_ks("MODERATE", inception=100_000.0)
    seed._current_portfolio_value = 80_000.0
    seed._halt_reason = "Drawdown 20.0% >= 12% limit (MODERATE)"
    seed._halt_time = datetime.now()
    seed._persist_halt_event(0.20, 15.0)

    with mock.patch.object(KillSwitch, "start"):
        ks = initialize_kill_switch("MODERATE", portfolio_value=100_000.0)

    assert ks.is_halted
    assert ks._halt_reason == seed._halt_reason


# ---------------------------------------------------------------------------
# Collector gate (KS-2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collector_skips_when_kill_switch_halted():
    """KS-2: run_collection_cycle refuses to run while this process's kill switch is halted."""
    ks = _make_ks("MODERATE")
    ks._halted.set()
    ks._halt_reason = "test halt"
    kill_switch_module._kill_switch = ks

    pipeline = mock.Mock()
    pipeline.run_once = mock.AsyncMock(side_effect=AssertionError("must not run while halted"))

    result = await run_collection_cycle(
        universe=["AAPL"],
        total_wealth=100_000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
        pipeline=pipeline,
        compiled_graph=mock.Mock(),
    )

    assert result.ran is False
    assert "halted" in result.reason
    pipeline.run_once.assert_not_called()


@pytest.mark.asyncio
async def test_collector_runs_when_no_kill_switch_initialized():
    """A process with no kill switch initialized (get_kill_switch() is None) is ungated."""
    assert kill_switch_module.get_kill_switch() is None

    pipeline = mock.Mock()
    pipeline.run_once = mock.AsyncMock(return_value={})

    result = await run_collection_cycle(
        universe=["AAPL"],
        total_wealth=100_000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
        pipeline=pipeline,
        compiled_graph=mock.Mock(),
    )

    pipeline.run_once.assert_awaited_once()
    assert result.ran is False  # no session data for AAPL in the empty {} session_states
    assert "halted" not in result.reason


# ---------------------------------------------------------------------------
# paper_book: compute_run_returns
# ---------------------------------------------------------------------------


def _technical(ticker: str, price: float) -> TechnicalSignal:
    """Builds a minimal TechnicalSignal, varying only ticker and current_price."""
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
        signal=Signal.BULLISH,
        conviction=0.7,
        net_score=0.0,
        timestamp=datetime.now(),
    )


def _allocation(ticker: str, pct: float) -> PositionAllocation:
    """Builds a minimal PositionAllocation, varying only ticker and allocation_pct."""
    return PositionAllocation(
        ticker=ticker,
        allocation_pct=pct,
        allocation_usd=pct * 100_000.0,
        stop_loss=90.0,
        thesis="test thesis",
        composite_conviction=0.7,
        time_horizon="30 days",
    )


def _decision(ticker: str, session_timestamp: datetime, price: float, pct: float) -> ARGUSDecision:
    """Builds a minimal reconciliation-eligible ARGUSDecision."""
    return ARGUSDecision(
        ticker=ticker,
        session_timestamp=session_timestamp,
        technical=_technical(ticker, price),
        allocation=_allocation(ticker, pct),
    )


class _FakePrices:
    """Minimal MarketDataProvider double: only ohlcv_daily is used by paper_book."""

    def __init__(self, closes: dict[str, pd.Series]) -> None:
        self._closes = closes

    def ohlcv_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        return pd.DataFrame({"close": self._closes[ticker]})


def _series(start: datetime, start_price: float, days: int = 10, drift: float = 1.0) -> pd.Series:
    """Builds a daily close series rising by `drift` per day from start_price."""
    dates = pd.date_range(start=start, periods=days, freq="D")
    return pd.Series([start_price + drift * i for i in range(days)], index=dates)


def test_compute_run_returns_weights_by_allocation_pct():
    """A run's return is the allocation-weighted average of its matured decisions."""
    start = datetime(2026, 1, 1)
    aaa = _decision("AAA", start, price=100.0, pct=0.1)  # +5% over 5 days
    bbb = _decision("BBB", start, price=200.0, pct=0.15)  # +10% over 5 days
    prices = _FakePrices(
        {
            "AAA": _series(start, 100.0, drift=1.0),
            "BBB": _series(start, 200.0, drift=4.0),
        }
    )

    results = compute_run_returns([aaa, bbb], prices, horizon_days=5)

    assert len(results) == 1
    run_timestamp, run_return = results[0]
    assert run_timestamp == start
    expected = (0.05 * 0.1 + 0.10 * 0.15) / (0.1 + 0.15)
    assert run_return == pytest.approx(expected)


def test_compute_run_returns_partial_maturity_uses_only_matured_subset():
    """A run with one matured and one unmatured decision isn't diluted by the unmatured one."""
    start = datetime(2026, 1, 1)
    matured = _decision("AAA", start, price=100.0, pct=0.1)  # +5% over 5 days
    unmatured = _decision("BBB", start, price=100.0, pct=0.15)
    prices = _FakePrices(
        {
            "AAA": _series(start, 100.0, days=10, drift=1.0),
            "BBB": _series(start, 100.0, days=2, drift=1.0),  # short of the 5-day horizon
        }
    )

    results = compute_run_returns([matured, unmatured], prices, horizon_days=5)

    assert len(results) == 1
    _, run_return = results[0]
    assert run_return == pytest.approx(0.05)


def test_compute_run_returns_omits_runs_with_no_matured_decisions():
    """A run where nothing has matured yet is deferred, not reported as a 0.0 return."""
    start = datetime(2026, 1, 1)
    decision = _decision("AAA", start, price=100.0, pct=0.1)
    prices = _FakePrices({"AAA": _series(start, 100.0, days=2, drift=1.0)})

    assert compute_run_returns([decision], prices, horizon_days=30) == []


def test_compute_run_returns_groups_by_session_timestamp():
    """Decisions from two different runs produce two separate results, in order."""
    run1 = datetime(2026, 1, 1)
    run2 = datetime(2026, 1, 8)
    d1 = _decision("AAA", run1, price=100.0, pct=0.15)
    d2 = _decision("AAA", run2, price=100.0, pct=0.15)
    prices = _FakePrices({"AAA": _series(run1, 100.0, days=20, drift=1.0)})

    results = compute_run_returns([d1, d2], prices, horizon_days=5)

    assert [r[0] for r in results] == [run1, run2]


def test_compute_run_returns_skips_decisions_that_took_no_position():
    """A decision with a zero-weight allocation contributes nothing to its run."""
    start = datetime(2026, 1, 1)
    zero_weight = _decision("AAA", start, price=100.0, pct=0.0)
    prices = _FakePrices({"AAA": _series(start, 100.0, drift=1.0)})

    assert compute_run_returns([zero_weight], prices, horizon_days=5) == []


# ---------------------------------------------------------------------------
# paper_book: PaperBook
# ---------------------------------------------------------------------------


def test_paperbook_apply_run_idempotent():
    """Re-applying the same run_timestamp is a no-op the second time."""
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    ts = datetime(2026, 1, 1)

    assert book.apply_run(ts, 0.05) is True
    assert book.equity == pytest.approx(105_000.0)

    assert book.apply_run(ts, 0.05) is False
    assert book.equity == pytest.approx(105_000.0)


def test_paperbook_drawdown_from_peak():
    """drawdown_from_peak reflects the highest equity reached, not the latest run alone."""
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    book.apply_run(datetime(2026, 1, 1), 0.20)
    book.apply_run(datetime(2026, 1, 8), -0.10)

    assert book.high_water_mark == pytest.approx(120_000.0)
    assert book.equity == pytest.approx(108_000.0)
    assert book.drawdown_from_peak() == pytest.approx((120_000.0 - 108_000.0) / 120_000.0)


def test_paperbook_save_load_round_trip(tmp_path):
    """A saved PaperBook loads back with identical field values."""
    path = str(tmp_path / "paper_equity.json")
    book = PaperBook(
        equity=95_000.0,
        high_water_mark=110_000.0,
        last_run_timestamp=datetime(2026, 1, 1),
        runs_applied={"2026-01-01T00:00:00"},
    )

    paper_book.save(book, path)
    loaded = paper_book.load(path)

    assert loaded.equity == pytest.approx(95_000.0)
    assert loaded.high_water_mark == pytest.approx(110_000.0)
    assert loaded.last_run_timestamp == datetime(2026, 1, 1)
    assert loaded.runs_applied == {"2026-01-01T00:00:00"}


def test_paperbook_load_missing_file_starts_fresh_at_total_wealth(tmp_path):
    """A missing paper_equity.json is a fresh deployment, not an error."""
    path = str(tmp_path / "does_not_exist.json")

    book = paper_book.load(path)

    assert book.equity == pytest.approx(settings.ARGUS_TOTAL_WEALTH)
    assert book.high_water_mark == pytest.approx(settings.ARGUS_TOTAL_WEALTH)
    assert book.runs_applied == set()
