"""
scripts/reconcile_outcomes.py

CLI entry point for run_reconciliation_pass
(argus/orchestration/reconcile.py) — the scheduled GitHub Actions
runner's reconciliation pass for deployments with no long-lived API process
to run the API's own daily loop (argus/orchestration/scheduler.py).

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
    argus/orchestration/reconcile.py, which the API's own
    reconciliation loop also calls)
"""

from __future__ import annotations

import argparse

from argus.config import settings
from argus.infra.log import configure_logging
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.reconcile import run_reconciliation_pass
from argus.params import RECONCILIATION
from argus.ports import MarketDataProvider
from argus.seams import LiveMarketDataProvider


def main(*, market_data: MarketDataProvider | None = None) -> None:
    """Parses CLI args, runs one reconciliation pass, and prints its report.

    Args:
        market_data: Price source; defaults to live fetches. Tests inject a fixture-backed one.
    """
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=str(settings.data_path("argus_graph.db")),
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
        market_data or LiveMarketDataProvider(),
        get_cultural_memory(),
        str(settings.data_path("paper_equity.json")),
        decisions_log_path=args.decisions_log,
        checkpoint_db_path=args.db,
        horizon_days=args.horizon_days,
    )

    source = args.decisions_log or args.db
    print(
        f"Loaded {report.decisions_loaded} decision(s) from {source} (horizon={args.horizon_days}d)"
    )
    for line in report.summary_lines():
        print(line)
    for error in report.errors:
        print(f"WARNING: {error}")


if __name__ == "__main__":
    main()
