"""
scripts/reconcile_outcomes.py

CLI entry point for argus/orchestration/reconciliation.py — reads every
decision back out of the LangGraph checkpoint (argus_graph.db) or a
decisions.jsonl log, reconciles whichever ones have cleared
RECONCILIATION.horizon_days against live market data, and reports how many
outcomes were stored to cultural memory. Also applies matured runs onto the
persisted paper equity curve (argus/risk/paper_book.py) so the scheduled
GitHub Actions runner — which has no long-lived KillSwitch daemon to feed —
still keeps that curve current for whichever process reads it next. Finally
prunes decisions.jsonl, the checkpoint DB, PENDING snapshots, and
runs_applied back to a bounded window (PR 6), the same cleanup api/main.py's
_reconcile_once performs, since this script is the only reconcile pass a
collector-only deployment (no long-lived API process) ever gets.

    .venv/bin/python scripts/reconcile_outcomes.py [--db PATH] [--horizon-days N]
    .venv/bin/python scripts/reconcile_outcomes.py --decisions-log PATH

Meant to run periodically (e.g. a daily cron) against the live checkpoint
database, or — for the unattended collector, which doesn't carry the full
checkpoint DB around — against its decisions.jsonl log.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

from argus.config import settings
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.reconciliation import (
    compact_decisions_jsonl,
    load_decisions_from_checkpoints,
    load_decisions_from_jsonl,
    prune_checkpoints,
    reconcile_decisions,
)
from argus.params import RECONCILIATION
from argus.risk import paper_book
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger("argus.reconcile_outcomes")


def main() -> None:
    """Parses CLI args, reconciles cleared decisions, and prints the outcome count."""
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

    if args.decisions_log:
        decisions = load_decisions_from_jsonl(args.decisions_log)
        source = args.decisions_log
    else:
        decisions = load_decisions_from_checkpoints(args.db)
        source = args.db
    print(f"Loaded {len(decisions)} decision(s) from {source}")

    market_data = LiveMarketDataProvider()
    stored = reconcile_decisions(
        decisions,
        market_data=market_data,
        cultural=get_cultural_memory(),
        horizon_days=args.horizon_days,
    )
    print(f"Reconciled {stored}/{len(decisions)} decision(s) (horizon={args.horizon_days}d)")

    book_path = f"{settings.ARGUS_DATA_DIR}/paper_equity.json"
    book = paper_book.load(book_path)
    for run_timestamp, run_return in paper_book.compute_run_returns(
        decisions, market_data, args.horizon_days
    ):
        book.apply_run(run_timestamp, run_return)

    cutoff = datetime.now() - timedelta(
        days=args.horizon_days + RECONCILIATION.retention_margin_days
    )
    book.prune_runs_applied(cutoff)
    paper_book.save(book, book_path)
    print(
        f"Paper equity: ${book.equity:,.2f} "
        f"(drawdown={book.drawdown_from_peak():.1%} from peak ${book.high_water_mark:,.2f})"
    )

    if args.decisions_log:
        kept = compact_decisions_jsonl(args.decisions_log, cutoff)
        print(f"Compacted {args.decisions_log}: {kept} decision(s) retained")

    pruned = prune_checkpoints(args.db, cutoff)
    print(f"Pruned {pruned} stale checkpoint thread(s) from {args.db}")

    expired = get_cultural_memory().expire_pending_snapshots(cutoff)
    print(f"Expired {expired} stale PENDING snapshot(s)")


if __name__ == "__main__":
    main()
