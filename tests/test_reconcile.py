"""Tests for orchestration/reconcile.py.

Covers realized-return computation, reconciling decisions into stored
outcomes, and the end-to-end reconciliation pass.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest import mock

import pandas as pd
import pytest

from argus.memory.cultural import CulturalMemoryManager
from argus.orchestration.decision_log import load_decisions_from_jsonl
from argus.orchestration.reconcile import (
    ReconciliationReport,
    reconcile_decision,
    reconcile_decisions,
    run_reconciliation_pass,
)
from argus.outcomes import compute_realized_return
from argus.params import RECONCILIATION
from argus.risk import paper_book
from argus.risk.paper_book import PaperBook
from argus.schemas.signals import (
    ARGUSDecision,
    PositionAllocation,
    Signal,
)
from tests.helpers.builders import (
    aggregated_decision,
    fundamental_signal,
    macro_context,
    matured_decision,
    position_allocation,
    technical_signal,
)
from tests.helpers.decision_store import put_checkpoint, write_decisions_log
from tests.helpers.market_data import FakeMarketData, ten_day_series

# ---------------------------------------------------------------------------
# compute_realized_return
# ---------------------------------------------------------------------------


def test_compute_realized_return_pairs_entry_and_horizon_close():
    """Realized return pairs the entry-day close with the close N horizon days later."""
    start = datetime(2026, 1, 1)
    technical = technical_signal(Signal.BULLISH, 0.8, current_price=100.0)
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=technical,
        allocation=position_allocation(),
    )
    market_data = ten_day_series(start, start_price=100.0)

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
        technical=technical_signal(Signal.BULLISH, 0.8),
    )
    market_data = ten_day_series(datetime(2026, 1, 1))
    assert compute_realized_return(decision, market_data, horizon_days=5) is None


def test_compute_realized_return_none_without_technical_signal():
    """A decision with no technical signal has no entry price to compute a return from."""
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime(2026, 1, 1),
        allocation=position_allocation(),
    )
    market_data = ten_day_series(datetime(2026, 1, 1))
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
        technical=technical_signal(Signal.BULLISH, 0.8),
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
    market_data = ten_day_series(datetime(2026, 1, 1))
    assert compute_realized_return(decision, market_data, horizon_days=5) is None


def test_compute_realized_return_none_when_horizon_not_yet_reached():
    """No return is computed until price data reaches the requested horizon."""
    start = datetime(2026, 1, 1)
    decision = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=technical_signal(Signal.BULLISH, 0.8),
        allocation=position_allocation(),
    )
    market_data = ten_day_series(start)  # only 10 days, short of the 30-day horizon requested below
    assert compute_realized_return(decision, market_data, horizon_days=30) is None


# ---------------------------------------------------------------------------
# reconcile_decision / reconcile_decisions
# ---------------------------------------------------------------------------


def test_reconcile_decision_stores_outcome_with_ablated_primary_driver():
    """Reconciling a decision stores its outcome tagged with the credited driver."""
    start = datetime(2026, 1, 1)
    decision = aggregated_decision(
        technical_signal(Signal.BULLISH, 0.9, current_price=100.0),
        macro_context(),
        fundamental_signal(Signal.BULLISH, 0.1),
        session_timestamp=start,
        allocation=position_allocation(),
    )
    market_data = ten_day_series(start, start_price=100.0)
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
        technical=technical_signal(Signal.BULLISH, 0.8),
    )
    market_data = ten_day_series(datetime(2026, 1, 1))
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)

    assert reconcile_decision(decision, market_data, cultural, horizon_days=5) is False
    cultural.store_trade_outcome.assert_not_called()


def test_reconcile_decisions_counts_only_the_ones_actually_reconciled():
    """The batch count reflects only decisions that actually had an outcome to store."""
    start = datetime(2026, 1, 1)
    reconcilable = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=technical_signal(Signal.BULLISH, 0.8, current_price=100.0),
        allocation=position_allocation(),
    )
    not_reconcilable = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=technical_signal(Signal.BULLISH, 0.8),
        # no allocation -> nothing to reconcile
    )
    market_data = ten_day_series(start, start_price=100.0)
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
        technical=technical_signal(Signal.BULLISH, 0.8, current_price=100.0),
        allocation=position_allocation(),
    )
    market_data = ten_day_series(start, start_price=100.0)
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
        technical=technical_signal(Signal.BULLISH, 0.8, current_price=100.0),
        allocation=position_allocation(),
    )
    second = ARGUSDecision(
        ticker="TEST",
        session_timestamp=start,
        technical=technical_signal(Signal.BULLISH, 0.8, current_price=101.0),
        allocation=position_allocation(),
    )
    market_data = ten_day_series(start, start_price=100.0)
    market_data.ohlcv_daily = mock.Mock(wraps=market_data.ohlcv_daily)
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)
    cultural.already_reconciled.return_value = set()

    count = reconcile_decisions([first, second], market_data, cultural, horizon_days=5)

    assert count == 2
    market_data.ohlcv_daily.assert_called_once_with("TEST")


# ---------------------------------------------------------------------------
# run_reconciliation_pass: the end-to-end pass that reads decisions, reconciles
# them, and bounds every store it touches.
# ---------------------------------------------------------------------------


def _autospec_cultural(expired: int = 0) -> mock.MagicMock:
    """An autospec CulturalMemoryManager stubbed for reconcile_decisions and expiry."""
    cultural = mock.create_autospec(CulturalMemoryManager, instance=True)
    cultural.already_reconciled.return_value = set()
    cultural.expire_pending_snapshots.return_value = expired
    return cultural


class _FakeCultural:
    """A stateful CulturalMemoryManager double that actually tracks stored outcomes.

    Unlike _autospec_cultural's fixed return_value, already_reconciled here
    reflects real state across calls within one test — needed to exercise
    resolved_ids across two run_reconciliation_pass calls.
    """

    def __init__(self) -> None:
        """Starts with no stored outcomes."""
        self._stored_ids: set[str] = set()

    def already_reconciled(self, decision_ids: list[str]) -> set[str]:
        """Returns the subset of decision_ids this fake has a stored outcome for."""
        return {d for d in decision_ids if d in self._stored_ids}

    def store_trade_outcome(self, decision: ARGUSDecision, **_kwargs: object) -> None:
        """Records decision.decision_id as resolved."""
        self._stored_ids.add(decision.decision_id)

    def expire_pending_snapshots(self, _cutoff: datetime) -> int:
        """No PENDING snapshots to expire in this fake."""
        return 0


def test_run_reconciliation_pass_over_matured_decisions_produces_outcome_and_equity(tmp_path):
    """A full pass over matured decisions stores the outcome and updates paper equity.

    The decision's realized return is compounded onto paper equity.
    """
    start = datetime(2026, 1, 1)
    decisions_log = write_decisions_log(
        tmp_path / "decisions.jsonl", matured_decision("TEST", start)
    )

    book_path = tmp_path / "paper_equity.json"
    paper_book.save(PaperBook(equity=100_000.0, high_water_mark=100_000.0), str(book_path))

    market_data = ten_day_series(start, start_price=100.0)
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


def test_run_reconciliation_pass_survives_across_runs_when_resolution_is_unavailable(tmp_path):
    """A decision whose horizon has passed but whose outcome couldn't be computed is not pruned.

    Reproduces the observed "Reconciled 0/N" failure mode: the decision's
    session_timestamp is already well past the ordinary retention cutoff, but
    its price series never reaches the target exit date (as if the reconcile
    pass keeps finding nothing to work with, run after run). It must survive
    compaction rather than being destroyed before it ever gets a chance to
    resolve.
    """
    horizon_days = RECONCILIATION.horizon_days
    old_start = datetime.now() - timedelta(
        days=horizon_days + RECONCILIATION.retention_margin_days + 5
    )
    decision = ARGUSDecision(
        ticker="UNRESOLVED",
        session_timestamp=old_start,
        technical=technical_signal(Signal.BULLISH, 0.8, ticker="UNRESOLVED"),
        allocation=position_allocation(ticker="UNRESOLVED"),
    )
    decisions_log = write_decisions_log(tmp_path / "decisions.jsonl", decision)

    book_path = tmp_path / "paper_equity.json"
    paper_book.save(PaperBook(equity=100_000.0, high_water_mark=100_000.0), str(book_path))

    # Only one day of price history: the horizon's target exit date is never reached.
    market_data = FakeMarketData.from_closes({"UNRESOLVED": pd.Series([100.0], index=[old_start])})
    cultural = _autospec_cultural()

    report = run_reconciliation_pass(
        market_data,
        cultural,
        str(book_path),
        decisions_log_path=str(decisions_log),
        horizon_days=horizon_days,
    )

    assert report.outcomes_stored == 0
    cultural.store_trade_outcome.assert_not_called()
    assert report.decisions_compacted == 1
    assert report.decisions_retired_unresolved == 0
    survivors = {d.ticker for d in load_decisions_from_jsonl(str(decisions_log))}
    assert survivors == {"UNRESOLVED"}


def test_run_reconciliation_pass_never_double_applies_a_run_whose_other_ticker_resolved_first(
    tmp_path,
):
    """Protecting an unresolved decision past cutoff must not let its run be applied twice.

    One run has two decisions: FAST matures and gets reconciled and compacted
    out of decisions.jsonl on the first pass, while SLOW stays unresolved and
    is protected past the same cutoff. The run is partially applied (FAST
    only) on the first pass. When SLOW finally matures on a second pass, its
    now-solo recomputation of the same run_timestamp must not compound onto
    equity again — runs_applied must still remember this run, even though its
    session_timestamp is already older than the routine retention cutoff.
    """
    horizon_days = RECONCILIATION.horizon_days
    old_start = datetime.now() - timedelta(
        days=horizon_days + RECONCILIATION.retention_margin_days + 1
    )
    fast = ARGUSDecision(
        ticker="FAST",
        session_timestamp=old_start,
        technical=technical_signal(Signal.BULLISH, 0.8, ticker="FAST", current_price=100.0),
        allocation=position_allocation(ticker="FAST"),
    )
    slow = ARGUSDecision(
        ticker="SLOW",
        session_timestamp=old_start,
        technical=technical_signal(Signal.BULLISH, 0.8, ticker="SLOW", current_price=100.0),
        allocation=position_allocation(ticker="SLOW"),
    )
    decisions_log = write_decisions_log(tmp_path / "decisions.jsonl", fast, slow)

    book_path = tmp_path / "paper_equity.json"
    paper_book.save(PaperBook(equity=100_000.0, high_water_mark=100_000.0), str(book_path))

    fast_prices = pd.Series(
        [100.0 + i for i in range(20)],
        index=pd.date_range(start=old_start, periods=20, freq="D"),
    )
    closes = {"FAST": fast_prices, "SLOW": pd.Series([100.0], index=[old_start])}
    market_data = FakeMarketData.from_closes(closes)
    cultural = _FakeCultural()

    first = run_reconciliation_pass(
        market_data,
        cultural,
        str(book_path),
        decisions_log_path=str(decisions_log),
        horizon_days=horizon_days,
    )

    assert first.outcomes_stored == 1
    equity_after_first_pass = first.equity
    assert equity_after_first_pass != pytest.approx(100_000.0)
    assert {d.ticker for d in load_decisions_from_jsonl(str(decisions_log))} == {"SLOW"}

    # SLOW's price history finally extends far enough to mature.
    closes["SLOW"] = pd.Series(
        [100.0 + i for i in range(20)],
        index=pd.date_range(start=old_start, periods=20, freq="D"),
    )

    second = run_reconciliation_pass(
        market_data,
        cultural,
        str(book_path),
        decisions_log_path=str(decisions_log),
        horizon_days=horizon_days,
    )

    assert second.outcomes_stored == 1
    assert second.equity == pytest.approx(equity_after_first_pass), (
        "SLOW maturing alone must not re-compound a run already applied via FAST"
    )


def test_run_reconciliation_pass_reading_from_checkpoints_bounds_checkpoints(tmp_path):
    """With only a checkpoint path, decisions are read from it and it alone is bounded."""
    start = datetime(2026, 1, 1)
    db_path = str(tmp_path / "argus_graph.db")
    put_checkpoint(db_path, "session-thread", start, [matured_decision("TEST", start)])

    market_data = ten_day_series(start, start_price=100.0)
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
    decisions_log = write_decisions_log(
        tmp_path / "decisions.jsonl", matured_decision("TEST", start)
    )

    db_path = str(tmp_path / "argus_graph.db")
    put_checkpoint(db_path, "stale-thread", datetime(2020, 1, 1))

    market_data = ten_day_series(start, start_price=100.0)
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
    decisions_log = write_decisions_log(
        tmp_path / "decisions.jsonl", matured_decision("TEST", start)
    )

    db_path = tmp_path / "argus_graph.db"
    db_path.write_bytes(b"not a sqlite database")

    market_data = ten_day_series(start, start_price=100.0)
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


def test_run_reconciliation_pass_truncated_paper_equity_file_neither_resets_nor_double_applies(
    tmp_path,
):
    """A truncated paper_equity.json fails the paper-book step instead of silently starting
    fresh — which would both reset equity and re-apply a run already reflected in it."""
    start = datetime(2026, 1, 1)
    decisions_log = write_decisions_log(
        tmp_path / "decisions.jsonl", matured_decision("TEST", start)
    )

    book_path = tmp_path / "paper_equity.json"
    good_book = PaperBook(equity=12_345.0, high_water_mark=15_000.0)
    good_book.apply_run(start, -0.05)  # this run is already reflected in equity and runs_applied
    paper_book.save(good_book, str(book_path))
    full = book_path.read_bytes()
    truncated = full[: len(full) // 2]
    book_path.write_bytes(truncated)

    market_data = ten_day_series(start, start_price=100.0)
    cultural = _autospec_cultural()

    report = run_reconciliation_pass(
        market_data,
        cultural,
        str(book_path),
        decisions_log_path=str(decisions_log),
        horizon_days=5,
    )

    assert report.paper_book_updated is False
    assert report.equity == 0.0  # unset default — neither reset to fresh, nor double-applied
    assert len(report.errors) == 1
    assert "paper-book update failed" in report.errors[0]
    # The persisted file is untouched: still truncated, not silently overwritten by a fresh book.
    assert book_path.read_bytes() == truncated


def test_run_reconciliation_pass_paper_book_not_partially_applied_when_its_step_fails(
    tmp_path, monkeypatch
):
    """A paper-book save failure leaves the persisted book exactly as it was.

    The failure is named in the report's errors.
    """
    start = datetime(2026, 1, 1)
    decisions_log = write_decisions_log(
        tmp_path / "decisions.jsonl", matured_decision("TEST", start)
    )

    book_path = tmp_path / "paper_equity.json"
    paper_book.save(PaperBook(equity=12_345.0, high_water_mark=12_345.0), str(book_path))
    original_bytes = book_path.read_bytes()

    def _boom(_book, _path) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(paper_book, "save", _boom)

    market_data = ten_day_series(start, start_price=100.0)
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


# ---------------------------------------------------------------------------
# ReconciliationReport.summary_lines
# ---------------------------------------------------------------------------


def test_summary_lines_for_a_pass_that_only_reconciled_and_expired():
    """Steps that did not run, or failed, contribute no line; the two always-present ones do."""
    report = ReconciliationReport(
        decisions_loaded=4, outcomes_stored=3, pending_snapshots_expired=2
    )

    assert report.summary_lines() == [
        "Reconciled 3/4 decision(s)",
        "Expired 2 stale PENDING snapshot(s)",
    ]


def test_summary_lines_reports_every_step_that_ran():
    """Each optional step adds its own line, in pass order."""
    report = ReconciliationReport(
        decisions_loaded=4,
        outcomes_stored=3,
        paper_book_updated=True,
        equity=101_234.5,
        drawdown=0.0312,
        decisions_compacted=7,
        decisions_retired_unresolved=1,
        checkpoints_pruned=2,
        pending_snapshots_expired=0,
    )

    assert report.summary_lines() == [
        "Reconciled 3/4 decision(s)",
        "Paper equity: $101,234.50 (drawdown=3.1% from peak)",
        "Compacted decisions.jsonl: 7 decision(s) retained",
        "Retired 1 decision(s) as permanently unresolved",
        "Pruned 2 stale checkpoint thread(s)",
        "Expired 0 stale PENDING snapshot(s)",
    ]
