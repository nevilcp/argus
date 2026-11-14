"""Decision settlement and coordinated retention across persistent stores."""

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
    """Reconcile a single decision: computes its outcome and store it if the horizon has passed."""
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
    """Reconcile every not-yet-reconciled decision whose horizon has passed."""
    already_done = cultural.already_reconciled([d.decision_id for d in decisions])
    candidates = [d for d in decisions if d.decision_id not in already_done and needs_reconciliation(d)]

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
    """Outcome of one run_reconciliation_pass() call."""

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
        """Report completed pass steps, omitting paper equity when its update failed."""
        lines = [f"Reconciled {self.outcomes_stored}/{self.decisions_loaded} decision(s)"]
        if self.paper_book_updated:
            lines.append(f"Paper equity: ${self.equity:,.2f} (drawdown={self.drawdown:.1%} from peak)")
        if self.decisions_compacted is not None:
            lines.append(f"Compacted decisions.jsonl: {self.decisions_compacted} decision(s) retained")
        if self.decisions_retired_unresolved:
            lines.append(f"Retired {self.decisions_retired_unresolved} decision(s) as permanently unresolved")
        if self.checkpoints_pruned is not None:
            lines.append(f"Pruned {self.checkpoints_pruned} stale checkpoint thread(s)")
        lines.append(f"Expired {self.pending_snapshots_expired} stale PENDING snapshot(s)")
        return lines


@contextmanager
def _pass_step(report: ReconciliationReport, label: str) -> Iterator[None]:
    """Run one independently-failable step of a reconciliation pass."""
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
    """Settle matured decisions and bound supplied stores, leaving daemon sync to the API caller."""
    if decisions_log_path:
        decisions = load_decisions_from_jsonl(decisions_log_path)
    elif checkpoint_db_path:
        decisions = load_decisions_from_checkpoints(checkpoint_db_path)
    else:
        raise ValueError("run_reconciliation_pass requires decisions_log_path or checkpoint_db_path")

    report = ReconciliationReport(decisions_loaded=len(decisions))
    report.outcomes_stored = reconcile_decisions(
        decisions, market_data=market_data, cultural=cultural, horizon_days=horizon_days
    )

    cutoff = retention_cutoff(horizon_days)
    unresolved_cutoff = now_local() - timedelta(days=horizon_days + RECONCILIATION.unresolved_retirement_days)

    with _pass_step(report, "paper-book update"):
        book = paper_book.load(paper_book_path)
        for run_timestamp, run_return in paper_book.compute_run_returns(decisions, market_data, horizon_days):
            book.apply_run(run_timestamp, run_return)
        # Retain applied-run identities as long as unresolved decisions to prevent later double compounding.
        report.runs_applied_pruned = book.prune_runs_applied(unresolved_cutoff)
        paper_book.save(book, paper_book_path)
        report.equity = book.equity
        report.drawdown = book.drawdown_from_peak()
        report.paper_book_updated = True

    if decisions_log_path:
        with _pass_step(report, "decisions.jsonl compaction"):
            resolved_ids = cultural.already_reconciled([d.decision_id for d in decisions])
            compaction = compact_decisions_jsonl(decisions_log_path, cutoff, unresolved_cutoff, resolved_ids)
            report.decisions_compacted = compaction.retained
            report.decisions_retired_unresolved = compaction.retired_unresolved

    if checkpoint_db_path:
        with _pass_step(report, "checkpoint pruning"):
            report.checkpoints_pruned = prune_checkpoints(checkpoint_db_path, cutoff)

    with _pass_step(report, "PENDING snapshot expiry"):
        report.pending_snapshots_expired = cultural.expire_pending_snapshots(cutoff)

    return report
