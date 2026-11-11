"""
argus/orchestration/reconcile.py

Closes the decision -> outcome loop: realized-return computation against a
MarketDataProvider, credit assignment, and the one reconciliation sequence
that the API's daily loop (orchestration/scheduler.py) and
scripts/reconcile_outcomes.py both run.

Decisions are read back from the decision log (or the LangGraph checkpoint),
credit assignment uses leave-one-out ablation rather than exact Shapley, and
horizon_days is provisional pending a pre-registered evaluation.

Responsibilities:
  - reconcile_decision / reconcile_decisions: compute each matured decision's
    outcome and hand it to cultural.store_trade_outcome
  - run_reconciliation_pass: compose the above plus the paper-book update and
    every store's bounding step into one pass
  - ReconciliationReport: what that pass did, and how to summarise it

Not responsible for:
  - Reading or compacting the decision log (see orchestration/decision_log.py)
  - Crediting an agent (see orchestration/attribution.py)
  - Deciding what a "good" outcome is, or evaluation metrics (see
    backtesting/evaluation.py and backtesting/metrics.py)
  - Scheduling reconciliation runs (see scripts/reconcile_outcomes.py)

Dependencies:
  - argus.orchestration.attribution, decision_log and checkpoint
  - argus.outcomes (compute_realized_return, needs_reconciliation, close_prices)
  - argus.memory.cultural (CulturalMemoryManager)
  - argus.ports (MarketDataProvider)
  - argus.risk.paper_book (the paper-book update in run_reconciliation_pass)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from argus.infra.clock import now_local
from argus.memory.cultural import CulturalMemoryManager
from argus.orchestration.attribution import credit_primary_driver
from argus.orchestration.checkpoint import prune_checkpoints, retention_cutoff
from argus.orchestration.decision_log import (
    compact_decisions_jsonl,
    load_decisions_from_checkpoints,
    load_decisions_from_jsonl,
)
from argus.outcomes import close_prices, compute_realized_return, needs_reconciliation
from argus.params import RECONCILIATION
from argus.ports import MarketDataProvider
from argus.risk import paper_book
from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger(__name__)


def reconcile_decision(
    decision: ARGUSDecision,
    market_data: MarketDataProvider,
    cultural: CulturalMemoryManager,
    horizon_days: int = RECONCILIATION.horizon_days,
    *,
    prices: pd.Series | None = None,
) -> bool:
    """Reconciles a single decision: computes its outcome and stores it if the horizon has passed.

    Args:
        decision: Completed ARGUSDecision to reconcile.
        market_data: Source of the ticker's daily close price series.
        cultural: CulturalMemoryManager to persist the outcome to.
        horizon_days: Calendar days after session_timestamp defining the
            target exit date.
        prices: Pre-fetched close-price Series for decision.ticker. Passed by
            reconcile_decisions() so decisions sharing a ticker issue one
            market_data fetch between them instead of one each; fetched here
            when omitted.

    Returns:
        True if an outcome was computed and handed to
        cultural.store_trade_outcome (which itself still drops |return| <=
        RECONCILIATION.min_abs_return_for_storage trades — see its
        docstring). False if the decision had nothing to reconcile or its
        horizon hasn't passed yet.
    """
    outcome = compute_realized_return(decision, market_data, horizon_days, prices=prices)
    if outcome is None:
        return False

    actual_return_pct, holding_days, exit_reason = outcome
    primary_driver = credit_primary_driver(decision)

    cultural.store_trade_outcome(
        decision=decision,
        actual_return_pct=actual_return_pct,
        holding_days=holding_days,
        exit_reason=exit_reason,
        primary_driver=primary_driver,
    )
    logger.info(
        "%s (%s): %+.2f%% over %d days, primary_driver=%s",
        decision.ticker,
        decision.decision_id,
        actual_return_pct * 100,
        holding_days,
        primary_driver,
    )
    return True


def reconcile_decisions(
    decisions: list[ARGUSDecision],
    market_data: MarketDataProvider,
    cultural: CulturalMemoryManager,
    horizon_days: int = RECONCILIATION.horizon_days,
) -> int:
    """Reconciles every not-yet-reconciled decision whose horizon has passed.

    Skips decisions cultural already has a stored outcome for (see
    CulturalMemoryManager.already_reconciled) and fetches each ticker's price
    history at most once regardless of how many decisions share it. This runs
    on a daily schedule against a decision history that only grows — without
    both, every past decision gets re-fetched and re-processed on every run.

    Args:
        decisions: Decisions to attempt reconciliation for, e.g. from
            load_decisions_from_checkpoints().
        market_data: Source of each ticker's daily close price series.
        cultural: CulturalMemoryManager to persist outcomes to.
        horizon_days: Calendar days after session_timestamp defining each
            decision's target exit date.

    Returns:
        Count of decisions actually stored via cultural.store_trade_outcome.
    """
    already_done = cultural.already_reconciled([d.decision_id for d in decisions])
    candidates = [
        d for d in decisions if d.decision_id not in already_done and needs_reconciliation(d)
    ]

    prices_by_ticker: dict[str, pd.Series | None] = {}
    stored = 0
    for decision in candidates:
        prices = close_prices(decision.ticker, market_data, prices_by_ticker)
        if prices is None:
            continue
        if reconcile_decision(decision, market_data, cultural, horizon_days, prices=prices):
            stored += 1

    return stored


@dataclass
class ReconciliationReport:
    """Outcome of one run_reconciliation_pass() call.

    Attributes:
        decisions_loaded: Count of decisions read from the configured
            decision source (decisions.jsonl or the checkpoint database).
        outcomes_stored: Count of decisions actually reconciled and stored
            via cultural.store_trade_outcome.
        paper_book_updated: Whether the paper-book update step completed.
            False if it raised — equity/drawdown are then still their unset
            defaults, not a real (zeroed-out) portfolio value, so a caller
            syncing a kill switch off `equity` must gate on this first.
        equity: Paper-book equity after this pass, or 0.0 if
            paper_book_updated is False.
        drawdown: Paper-book drawdown from peak after this pass, or 0.0 if
            paper_book_updated is False.
        runs_applied_pruned: Count of applied-run entries pruned from the
            paper book.
        pending_snapshots_expired: Count of PENDING cultural-memory
            snapshots expired by this pass.
        decisions_compacted: Retained decisions.jsonl count, or None if
            decisions_log_path wasn't given.
        decisions_retired_unresolved: Count of decisions.jsonl decisions
            dropped for aging past RECONCILIATION.unresolved_retirement_days
            without ever receiving an outcome, or None if decisions_log_path
            wasn't given. Distinct from decisions_compacted, which counts what
            survived — this counts a deliberate, unresolved retirement rather
            than a routine, resolved prune.
        checkpoints_pruned: Deleted checkpoint-thread count, or None if
            checkpoint_db_path wasn't given.
        errors: One entry per independent step that failed; every other
            step still ran.
    """

    decisions_loaded: int = 0
    outcomes_stored: int = 0
    paper_book_updated: bool = False
    equity: float = 0.0
    drawdown: float = 0.0
    runs_applied_pruned: int = 0
    pending_snapshots_expired: int = 0
    decisions_compacted: int | None = None
    decisions_retired_unresolved: int | None = None
    checkpoints_pruned: int | None = None
    errors: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        """Describes what the pass did, one line per step that ran.

        The paper-book line is omitted when that step failed, because equity and
        drawdown are then unset defaults, not a measured portfolio. ``errors`` is
        left out: each caller reports those at its own severity.

        Returns:
            Human-readable lines for the API log and the CLI to print as they are.
        """
        lines = [f"Reconciled {self.outcomes_stored}/{self.decisions_loaded} decision(s)"]
        if self.paper_book_updated:
            lines.append(
                f"Paper equity: ${self.equity:,.2f} (drawdown={self.drawdown:.1%} from peak)"
            )
        if self.decisions_compacted is not None:
            lines.append(
                f"Compacted decisions.jsonl: {self.decisions_compacted} decision(s) retained"
            )
        if self.decisions_retired_unresolved:
            lines.append(
                f"Retired {self.decisions_retired_unresolved} decision(s) as permanently unresolved"
            )
        if self.checkpoints_pruned is not None:
            lines.append(f"Pruned {self.checkpoints_pruned} stale checkpoint thread(s)")
        lines.append(f"Expired {self.pending_snapshots_expired} stale PENDING snapshot(s)")
        return lines


@contextmanager
def _pass_step(report: ReconciliationReport, label: str) -> Iterator[None]:
    """Runs one independently-failable step of a reconciliation pass.

    A raise is logged and recorded on the report instead of propagating, so one
    store failing to bound itself never skips the next one.

    Args:
        report: Report the failure is recorded on.
        label: Step name, used verbatim in both the log line and the recorded error.
    """
    try:
        yield
    except Exception as exc:
        logger.exception("%s failed", label)
        report.errors.append(f"{label} failed: {exc}")


def run_reconciliation_pass(
    market_data: MarketDataProvider,
    cultural: CulturalMemoryManager,
    paper_book_path: str,
    *,
    decisions_log_path: str | None = None,
    checkpoint_db_path: str | None = None,
    horizon_days: int = RECONCILIATION.horizon_days,
) -> ReconciliationReport:
    """Runs the one reconciliation sequence shared by the API loop and the reconcile script.

    Loads decisions, reconciles the matured ones against market_data, updates
    the paper book, and bounds every growing store whose path was supplied.
    Kill-switch sync is deliberately not part of this: it stays at the API's
    call site, fed by the returned report's equity, so a scheduled runner
    with no daemon to sync simply doesn't call it — that difference is one
    line at the call site rather than an argument silently omitted here.

    The decisions log wins as the read source when decisions_log_path is
    given; the checkpoint database otherwise. Every store whose path is
    known is bounded regardless of which one was the read source, so a
    caller supplying both (the API loop) gets both bounded and a caller
    supplying only one (a collector-only deployment) gets exactly that one.

    Args:
        market_data: Source of each ticker's daily close price series.
        cultural: CulturalMemoryManager to persist outcomes to and expire
            PENDING snapshots from.
        paper_book_path: Path the paper equity curve is loaded from and
            saved to.
        decisions_log_path: Path to a decisions.jsonl log. Read source when
            given, and compacted at the end of the pass. Omit for a
            deployment with no jsonl log.
        checkpoint_db_path: Path to the LangGraph checkpoint database. Read
            source when decisions_log_path is omitted; pruned at the end of
            the pass whenever given. Omit for a deployment with no
            checkpoint database.
        horizon_days: Calendar days after session_timestamp defining each
            decision's target exit date.

    Returns:
        A ReconciliationReport describing what happened. Failures in the
        paper-book update and in each independently-bounded store
        (decisions.jsonl compaction, checkpoint pruning, PENDING snapshot
        expiry) are caught, logged, and recorded in `errors` — one store
        failing never skips another. The paper-book update is one atomic
        load/compute/apply/prune/save step: on failure nothing from it is
        saved, so the persisted book is never left partially applied.

    Raises:
        ValueError: Neither decisions_log_path nor checkpoint_db_path was
            given, so there is no decision source to read.
        Exception: Whatever loading decisions or reconcile_decisions raises —
            unlike the bounded stores below, these aren't independent of the
            rest of the pass, so a failure here is the caller's to handle.
    """
    if decisions_log_path:
        decisions = load_decisions_from_jsonl(decisions_log_path)
    elif checkpoint_db_path:
        decisions = load_decisions_from_checkpoints(checkpoint_db_path)
    else:
        raise ValueError(
            "run_reconciliation_pass requires decisions_log_path or checkpoint_db_path"
        )

    report = ReconciliationReport(decisions_loaded=len(decisions))
    report.outcomes_stored = reconcile_decisions(
        decisions, market_data=market_data, cultural=cultural, horizon_days=horizon_days
    )

    cutoff = retention_cutoff(horizon_days)
    unresolved_cutoff = now_local() - timedelta(
        days=horizon_days + RECONCILIATION.unresolved_retirement_days
    )

    with _pass_step(report, "paper-book update"):
        book = paper_book.load(paper_book_path)
        for run_timestamp, run_return in paper_book.compute_run_returns(
            decisions, market_data, horizon_days
        ):
            book.apply_run(run_timestamp, run_return)
        # unresolved_cutoff, not cutoff: an unresolved decision can now outlive
        # cutoff in decisions.jsonl (see compact_decisions_jsonl), so a run
        # sharing its session_timestamp must stay in runs_applied at least as
        # long — otherwise a later pass that recomputes the run from the
        # decisions still on disk would compound it onto equity a second time.
        report.runs_applied_pruned = book.prune_runs_applied(unresolved_cutoff)
        paper_book.save(book, paper_book_path)
        report.equity = book.equity
        report.drawdown = book.drawdown_from_peak()
        report.paper_book_updated = True

    if decisions_log_path:
        with _pass_step(report, "decisions.jsonl compaction"):
            resolved_ids = cultural.already_reconciled([d.decision_id for d in decisions])
            compaction = compact_decisions_jsonl(
                decisions_log_path, cutoff, unresolved_cutoff, resolved_ids
            )
            report.decisions_compacted = compaction.retained
            report.decisions_retired_unresolved = compaction.retired_unresolved

    if checkpoint_db_path:
        with _pass_step(report, "checkpoint pruning"):
            report.checkpoints_pruned = prune_checkpoints(checkpoint_db_path, cutoff)

    with _pass_step(report, "PENDING snapshot expiry"):
        report.pending_snapshots_expired = cultural.expire_pending_snapshots(cutoff)

    return report
