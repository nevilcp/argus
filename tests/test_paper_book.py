"""
tests/test_paper_book.py

Tests for argus/risk/paper_book.py: compute_run_returns and the PaperBook
equity curve that feeds the kill switch.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import argus.risk.kill_switch as kill_switch_module
from argus.config import settings
from argus.risk import paper_book
from argus.risk.paper_book import PaperBook, compute_run_returns
from argus.schemas.signals import ARGUSDecision, PositionAllocation, Signal, TechnicalSignal


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


def test_compute_run_returns_skips_runs_inside_the_previous_kept_runs_horizon():
    """A run timestamped inside the previous kept run's horizon is omitted, not compounded."""
    run1 = datetime(2026, 1, 1)
    run2 = run1 + timedelta(days=2)  # inside run1's 5-day horizon
    d1 = _decision("AAA", run1, price=100.0, pct=0.1)
    d2 = _decision("AAA", run2, price=100.0, pct=0.1)
    prices = _FakePrices({"AAA": _series(run1, 100.0, days=20, drift=1.0)})

    results = compute_run_returns([d1, d2], prices, horizon_days=5)

    assert [r[0] for r in results] == [run1]


def test_compute_run_returns_keeps_runs_exactly_horizon_days_apart():
    """A run landing exactly on the previous kept run's horizon boundary is kept, not skipped."""
    run1 = datetime(2026, 1, 1)
    run2 = run1 + timedelta(days=5)
    d1 = _decision("AAA", run1, price=100.0, pct=0.1)
    d2 = _decision("AAA", run2, price=100.0, pct=0.1)
    prices = _FakePrices({"AAA": _series(run1, 100.0, days=20, drift=1.0)})

    results = compute_run_returns([d1, d2], prices, horizon_days=5)

    assert [r[0] for r in results] == [run1, run2]


def test_compute_run_returns_and_paperbook_survive_the_audit_simulation():
    """120 daily runs at -0.1%/day must compound to their true cumulative decline,
    not a MODERATE-halt-tripping blowup from compounding ~24x overlapping 5-day windows."""
    start = datetime(2026, 1, 1)
    decay = 0.999
    days = 130
    closes = pd.Series(
        [100.0 * (decay**i) for i in range(days)],
        index=pd.date_range(start=start, periods=days, freq="D"),
    )
    prices = _FakePrices({"AAA": closes})
    decisions = [
        _decision("AAA", start + timedelta(days=i), price=100.0 * (decay**i), pct=0.15)
        for i in range(120)
    ]

    results = compute_run_returns(decisions, prices, horizon_days=5)

    # Non-overlapping: one kept run every 5 days, not one per day
    assert len(results) == 24

    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    for run_timestamp, run_return in results:
        book.apply_run(run_timestamp, run_return)

    assert book.drawdown_from_peak() == pytest.approx(1 - decay**120, abs=1e-4)
    assert book.drawdown_from_peak() < kill_switch_module.KILL_SWITCH.moderate_drawdown_halt


# ---------------------------------------------------------------------------
# paper_book: PaperBook
# ---------------------------------------------------------------------------


def test_paperbook_apply_run_floors_equity_at_zero():
    """A run_return that would drive equity negative floors it at 0 instead."""
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)

    book.apply_run(datetime(2026, 1, 1), -1.5)

    assert book.equity == 0.0


def test_paperbook_rebase_floors_equity_and_hwm_at_zero():
    """rebase() to a negative inception value floors equity/high_water_mark at 0."""
    book = PaperBook(equity=50_000.0, high_water_mark=50_000.0)

    book.rebase(-1.0)

    assert book.equity == 0.0
    assert book.high_water_mark == 0.0


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


def test_paperbook_rebase_sets_equity_and_hwm_and_leaves_runs_applied():
    """rebase() replaces equity/high_water_mark but leaves runs_applied untouched,
    so a run already compounded into the old curve is never re-applied against the new base."""
    book = PaperBook(
        equity=88_000.0, high_water_mark=100_000.0, runs_applied={"2026-01-01T00:00:00"}
    )

    book.rebase(120_000.0, rebased_at=datetime(2026, 2, 1))

    assert book.equity == pytest.approx(120_000.0)
    assert book.high_water_mark == pytest.approx(120_000.0)
    assert book.rebased_at == datetime(2026, 2, 1)
    assert book.runs_applied == {"2026-01-01T00:00:00"}


def test_paperbook_rebase_then_reapplying_an_already_applied_run_is_a_noop():
    """After a rebase, re-running compute_run_returns over old decisions can't drag the
    rebased equity back down, since the run is already in runs_applied."""
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    ts = datetime(2026, 1, 1)
    book.apply_run(ts, -0.12)

    book.rebase(150_000.0)

    assert book.apply_run(ts, -0.12) is False
    assert book.equity == pytest.approx(150_000.0)


def test_paperbook_save_load_round_trip_includes_rebased_at(tmp_path):
    """rebased_at survives a save/load round trip."""
    path = str(tmp_path / "paper_equity.json")
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    book.rebase(150_000.0, rebased_at=datetime(2026, 2, 1))

    paper_book.save(book, path)
    loaded = paper_book.load(path)

    assert loaded.rebased_at == datetime(2026, 2, 1)


def test_paperbook_load_missing_file_starts_fresh_at_total_wealth(tmp_path):
    """A missing paper_equity.json is a fresh deployment, not an error."""
    path = str(tmp_path / "does_not_exist.json")

    book = paper_book.load(path)

    assert book.equity == pytest.approx(settings.ARGUS_TOTAL_WEALTH)
    assert book.high_water_mark == pytest.approx(settings.ARGUS_TOTAL_WEALTH)
    assert book.runs_applied == set()


def test_paperbook_load_truncated_existing_file_raises(tmp_path):
    """An existing-but-unparseable file is a hard failure, not a silent fresh start.

    Silently starting fresh here would reset equity (erasing the drawdown
    history the kill switch's drawdown gate reads) and empty runs_applied
    (so an already-applied run in decisions.jsonl gets compounded again).
    """
    path = tmp_path / "paper_equity.json"
    good_book = PaperBook(equity=95_000.0, high_water_mark=110_000.0)
    good_book.apply_run(datetime(2026, 1, 1), -0.05)
    paper_book.save(good_book, str(path))
    original = path.read_bytes()
    path.write_bytes(original[: len(original) // 2])  # simulate a crash mid-write

    with pytest.raises(ValueError, match=r"line \d+ column \d+"):
        paper_book.load(str(path))


def test_paperbook_save_is_crash_safe_no_leftover_tmp_file(tmp_path):
    """save() leaves no .tmp artifact behind — readers only ever see the final file or nothing."""
    path = tmp_path / "paper_equity.json"

    paper_book.save(PaperBook(equity=100_000.0, high_water_mark=100_000.0), str(path))

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_paperbook_prune_runs_applied_drops_only_entries_older_than_cutoff():
    """Entries before cutoff are dropped; entries at or after it survive."""
    book = PaperBook(
        equity=100_000.0,
        high_water_mark=100_000.0,
        runs_applied={
            "2026-01-01T00:00:00",
            "2026-01-10T00:00:00",
            "2026-01-20T00:00:00",
        },
    )

    dropped = book.prune_runs_applied(cutoff=datetime(2026, 1, 10))

    assert dropped == 1
    assert book.runs_applied == {"2026-01-10T00:00:00", "2026-01-20T00:00:00"}


def test_paperbook_prune_runs_applied_leaves_equity_and_hwm_untouched():
    """Pruning runs_applied is bookkeeping only — it must not touch the equity curve itself."""
    book = PaperBook(equity=88_000.0, high_water_mark=100_000.0)
    book.apply_run(datetime(2026, 1, 1), -0.12)

    book.prune_runs_applied(cutoff=datetime(2026, 2, 1))

    assert book.equity == pytest.approx(88_000.0 * 0.88)
    assert book.runs_applied == set()
