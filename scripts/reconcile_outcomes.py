"""
scripts/reconcile_outcomes.py

CLI entry point for argus/orchestration/reconciliation.py — reads every
decision back out of the LangGraph checkpoint (argus_graph.db) or a
decisions.jsonl log, reconciles whichever ones have cleared
RECONCILIATION.horizon_days against live market data, and reports how many
outcomes were stored to cultural memory.

    .venv/bin/python scripts/reconcile_outcomes.py [--db PATH] [--horizon-days N]
    .venv/bin/python scripts/reconcile_outcomes.py --decisions-log PATH

Meant to run periodically (e.g. a daily cron) against the live checkpoint
database, or — for the unattended collector, which doesn't carry the full
checkpoint DB around — against its decisions.jsonl log.
"""

from __future__ import annotations

import argparse
import logging

from argus.memory.cultural import get_cultural_memory
from argus.orchestration.reconciliation import (
    load_decisions_from_checkpoints,
    load_decisions_from_jsonl,
    reconcile_decisions,
)
from argus.params import RECONCILIATION
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger("argus.reconcile_outcomes")


def main() -> None:
    """Parses CLI args, reconciles cleared decisions, and prints the outcome count."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="argus_graph.db", help="Path to the LangGraph checkpoint database")
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

    stored = reconcile_decisions(
        decisions,
        market_data=LiveMarketDataProvider(),
        cultural=get_cultural_memory(),
        horizon_days=args.horizon_days,
    )
    print(f"Reconciled {stored}/{len(decisions)} decision(s) (horizon={args.horizon_days}d)")


if __name__ == "__main__":
    main()
