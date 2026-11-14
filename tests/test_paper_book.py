"""Matured run returns and synthetic paper equity regressions."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import argus.risk.kill_switch as kill_switch_module
from argus.config import settings
from argus.risk import paper_book
from argus.risk.paper_book import PaperBook, compute_run_returns
from argus.schemas.signals import ARGUSDecision
from tests.helpers.builders import argus_decision, position_allocation, technical_signal
from tests.helpers.market_data import FakeMarketData


def _decision(ticker: str, session_timestamp: datetime, price: float, pct: float) -> ARGUSDecision:
    """Build a minimal reconciliation-eligible ARGUSDecision."""
    return argus_decision(
        ticker,
        session_timestamp,
        technical=technical_signal(ticker=ticker, current_price=price),
        allocation=position_allocation(ticker, pct),
    )


def _series(start: datetime, start_price: float, days: int = 10, drift: float = 1.0) -> pd.Series:
    """Build a daily close series rising by `drift` per day from start_price."""
    dates = pd.date_range(start=start, periods=days, freq="D")
    return pd.Series([start_price + drift * i for i in range(days)], index=dates)


def test_compute_run_returns_weights_by_allocation_pct():
    start = datetime(2026, 1, 1)
    aaa = _decision("AAA", start, price=100.0, pct=0.1)
    bbb = _decision("BBB", start, price=200.0, pct=0.15)
    prices = FakeMarketData.from_closes(
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
    start = datetime(2026, 1, 1)
    matured = _decision("AAA", start, price=100.0, pct=0.1)
    unmatured = _decision("BBB", start, price=100.0, pct=0.15)
    prices = FakeMarketData.from_closes(
        {
            "AAA": _series(start, 100.0, days=10, drift=1.0),
            "BBB": _series(start, 100.0, days=2, drift=1.0),
        }
    )

    results = compute_run_returns([matured, unmatured], prices, horizon_days=5)

    assert len(results) == 1
    _, run_return = results[0]
    assert run_return == pytest.approx(0.05)


def test_compute_run_returns_omits_runs_with_no_matured_decisions():
    start = datetime(2026, 1, 1)
    decision = _decision("AAA", start, price=100.0, pct=0.1)
    prices = FakeMarketData.from_closes({"AAA": _series(start, 100.0, days=2, drift=1.0)})

    assert compute_run_returns([decision], prices, horizon_days=30) == []


def test_compute_run_returns_groups_by_session_timestamp():
    run1 = datetime(2026, 1, 1)
    run2 = datetime(2026, 1, 8)
    d1 = _decision("AAA", run1, price=100.0, pct=0.15)
    d2 = _decision("AAA", run2, price=100.0, pct=0.15)
    prices = FakeMarketData.from_closes({"AAA": _series(run1, 100.0, days=20, drift=1.0)})

    results = compute_run_returns([d1, d2], prices, horizon_days=5)

    assert [r[0] for r in results] == [run1, run2]


def test_compute_run_returns_skips_decisions_that_took_no_position():
    start = datetime(2026, 1, 1)
    zero_weight = _decision("AAA", start, price=100.0, pct=0.0)
    prices = FakeMarketData.from_closes({"AAA": _series(start, 100.0, drift=1.0)})

    assert compute_run_returns([zero_weight], prices, horizon_days=5) == []


def test_compute_run_returns_skips_runs_inside_the_previous_kept_runs_horizon():
    run1 = datetime(2026, 1, 1)
    run2 = run1 + timedelta(days=2)
    d1 = _decision("AAA", run1, price=100.0, pct=0.1)
    d2 = _decision("AAA", run2, price=100.0, pct=0.1)
    prices = FakeMarketData.from_closes({"AAA": _series(run1, 100.0, days=20, drift=1.0)})

    results = compute_run_returns([d1, d2], prices, horizon_days=5)

    assert [r[0] for r in results] == [run1]


def test_compute_run_returns_keeps_runs_exactly_horizon_days_apart():
    run1 = datetime(2026, 1, 1)
    run2 = run1 + timedelta(days=5)
    d1 = _decision("AAA", run1, price=100.0, pct=0.1)
    d2 = _decision("AAA", run2, price=100.0, pct=0.1)
    prices = FakeMarketData.from_closes({"AAA": _series(run1, 100.0, days=20, drift=1.0)})

    results = compute_run_returns([d1, d2], prices, horizon_days=5)

    assert [r[0] for r in results] == [run1, run2]


def test_compute_run_returns_and_paperbook_survive_the_audit_simulation():
    start = datetime(2026, 1, 1)
    decay = 0.999
    days = 130
    closes = pd.Series(
        [100.0 * (decay**i) for i in range(days)],
        index=pd.date_range(start=start, periods=days, freq="D"),
    )
    prices = FakeMarketData.from_closes({"AAA": closes})
    decisions = [_decision("AAA", start + timedelta(days=i), price=100.0 * (decay**i), pct=0.15) for i in range(120)]

    results = compute_run_returns(decisions, prices, horizon_days=5)

    assert len(results) == 24

    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    for run_timestamp, run_return in results:
        book.apply_run(run_timestamp, run_return)

    assert book.drawdown_from_peak() == pytest.approx(1 - decay**120, abs=1e-4)
    assert book.drawdown_from_peak() < kill_switch_module.KILL_SWITCH.moderate_drawdown_halt


def test_paperbook_apply_run_floors_equity_at_zero():
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)

    book.apply_run(datetime(2026, 1, 1), -1.5)

    assert book.equity == 0.0


def test_paperbook_rebase_floors_equity_and_hwm_at_zero():
    book = PaperBook(equity=50_000.0, high_water_mark=50_000.0)

    book.rebase(-1.0)

    assert book.equity == 0.0
    assert book.high_water_mark == 0.0


def test_paperbook_apply_run_idempotent():
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    ts = datetime(2026, 1, 1)

    assert book.apply_run(ts, 0.05) is True
    assert book.equity == pytest.approx(105_000.0)

    assert book.apply_run(ts, 0.05) is False
    assert book.equity == pytest.approx(105_000.0)


def test_paperbook_drawdown_from_peak():
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    book.apply_run(datetime(2026, 1, 1), 0.20)
    book.apply_run(datetime(2026, 1, 8), -0.10)

    assert book.high_water_mark == pytest.approx(120_000.0)
    assert book.equity == pytest.approx(108_000.0)
    assert book.drawdown_from_peak() == pytest.approx((120_000.0 - 108_000.0) / 120_000.0)


def test_paperbook_save_load_round_trip(tmp_path):
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
    book = PaperBook(equity=88_000.0, high_water_mark=100_000.0, runs_applied={"2026-01-01T00:00:00"})

    book.rebase(120_000.0, rebased_at=datetime(2026, 2, 1))

    assert book.equity == pytest.approx(120_000.0)
    assert book.high_water_mark == pytest.approx(120_000.0)
    assert book.rebased_at == datetime(2026, 2, 1)
    assert book.runs_applied == {"2026-01-01T00:00:00"}


def test_paperbook_rebase_then_reapplying_an_already_applied_run_is_a_noop():
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    ts = datetime(2026, 1, 1)
    book.apply_run(ts, -0.12)

    book.rebase(150_000.0)

    assert book.apply_run(ts, -0.12) is False
    assert book.equity == pytest.approx(150_000.0)


def test_paperbook_save_load_round_trip_includes_rebased_at(tmp_path):
    path = str(tmp_path / "paper_equity.json")
    book = PaperBook(equity=100_000.0, high_water_mark=100_000.0)
    book.rebase(150_000.0, rebased_at=datetime(2026, 2, 1))

    paper_book.save(book, path)
    loaded = paper_book.load(path)

    assert loaded.rebased_at == datetime(2026, 2, 1)


def test_paperbook_load_missing_file_starts_fresh_at_total_wealth(tmp_path):
    path = str(tmp_path / "does_not_exist.json")

    book = paper_book.load(path)

    assert book.equity == pytest.approx(settings.ARGUS_TOTAL_WEALTH)
    assert book.high_water_mark == pytest.approx(settings.ARGUS_TOTAL_WEALTH)
    assert book.runs_applied == set()


def test_paperbook_load_truncated_existing_file_raises(tmp_path):
    path = tmp_path / "paper_equity.json"
    good_book = PaperBook(equity=95_000.0, high_water_mark=110_000.0)
    good_book.apply_run(datetime(2026, 1, 1), -0.05)
    paper_book.save(good_book, str(path))
    original = path.read_bytes()
    path.write_bytes(original[: len(original) // 2])

    with pytest.raises(ValueError, match=r"line \d+ column \d+"):
        paper_book.load(str(path))


def test_paperbook_save_is_crash_safe_no_leftover_tmp_file(tmp_path):
    path = tmp_path / "paper_equity.json"

    paper_book.save(PaperBook(equity=100_000.0, high_water_mark=100_000.0), str(path))

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_paperbook_prune_runs_applied_drops_only_entries_older_than_cutoff():
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
    book = PaperBook(equity=88_000.0, high_water_mark=100_000.0)
    book.apply_run(datetime(2026, 1, 1), -0.12)

    book.prune_runs_applied(cutoff=datetime(2026, 2, 1))

    assert book.equity == pytest.approx(88_000.0 * 0.88)
    assert book.runs_applied == set()
