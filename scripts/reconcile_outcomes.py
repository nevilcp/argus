"""
scripts/reconcile_outcomes.py

CLI entry point for run_reconciliation_pass
(argus/orchestration/reconciliation.py) — the scheduled GitHub Actions
runner's reconciliation pass for deployments with no long-lived API process
to run api/main.py's own daily loop.

    .venv/bin/python scripts/reconcile_outcomes.py [--db PATH] [--horizon-days N]
    .venv/bin/python scripts/reconcile_outcomes.py --decisions-log PATH

Meant to run periodically (e.g. a daily cron) against the live checkpoint
database, or — for the unattended collector, which doesn't carry the full
checkpoint DB around — against its decisions.jsonl log.

Responsibilities:
  - Parse CLI args into run_reconciliation_pass's store paths and print its
    returned report

Not responsible for:
  - The reconciliation sequence itself, or which stores get bounded (see
    argus/orchestration/reconciliation.py, which api/main.py's own
    reconciliation loop also calls)
"""

from __future__ import annotations

import argparse
import logging

from argus.config import settings
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.reconciliation import run_reconciliation_pass
from argus.params import RECONCILIATION
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger("argus.reconcile_outcomes")


def main() -> None:
    """Parses CLI args, runs one reconciliation pass, and prints its report."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=f"{settings.ARGUS_DATA_DIR}/argus_graph.db",
        help="Path to the LangGraph checkpoint database",
    )
    parser.add_argument(
        "--decisions-log",
        default=None,
        help="Path to a decisions.jsonl log; if given, read from this instead of --db",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=RECONCILIATION.horizon_days,
        help="Calendar days after a decision's session_timestamp to look up its exit price",
    )
    args = parser.parse_args()

    report = run_reconciliation_pass(
        LiveMarketDataProvider(),
        get_cultural_memory(),
        f"{settings.ARGUS_DATA_DIR}/paper_equity.json",
        decisions_log_path=args.decisions_log,
        checkpoint_db_path=args.db,
        horizon_days=args.horizon_days,
    )

    source = args.decisions_log or args.db
    print(f"Loaded {report.decisions_loaded} decision(s) from {source}")
    print(
        f"Reconciled {report.outcomes_stored}/{report.decisions_loaded} decision(s) "
        f"(horizon={args.horizon_days}d)"
    )
    if report.paper_book_updated:
        print(f"Paper equity: ${report.equity:,.2f} (drawdown={report.drawdown:.1%} from peak)")
    if report.decisions_compacted is not None:
        print(f"Compacted {args.decisions_log}: {report.decisions_compacted} decision(s) retained")
    if report.checkpoints_pruned is not None:
        print(f"Pruned {report.checkpoints_pruned} stale checkpoint thread(s) from {args.db}")
    print(f"Expired {report.pending_snapshots_expired} stale PENDING snapshot(s)")
    for error in report.errors:
        print(f"WARNING: {error}")


if __name__ == "__main__":
    main()
