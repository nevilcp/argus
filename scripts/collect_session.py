"""
scripts/collect_session.py

CLI entry point for argus/orchestration/collector.py — runs one unattended
fetch-and-analyze cycle: sweep the MFT pipeline, invoke the graph over
whichever tickers actually got session data, and append the resulting
decisions to a JSONL log.

    .venv/bin/python -m scripts.collect_session [--universe AAPL,MSFT,...]

This is the same cycle api/main.py's background collector loop runs on a
timer; this script is what the scheduled GitHub Actions workflow invokes
once per cron tick. Exits 0 for a no-op (market holiday, off-hours tick) or
a successful cycle — neither is a failure. Exits 1 for a DEGRADED cycle: the
graph ran but too few decisions came back with a real allocation to be worth
reconciling (see argus.orchestration.collector.CycleOutcome and
params.COLLECTOR), the same "ran but produced nothing" failure the secrets
preflight in collector.yml catches upfront, caught here after the fact.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from pathlib import Path

from argus.config import settings
from argus.data.pipeline import MFTDataPipeline
from argus.orchestration.collector import CycleOutcome, run_collection_cycle
from argus.orchestration.graph import build_graph

logger = logging.getLogger("argus.collect_session")


def _data_path(filename: str) -> str:
    """Resolves a filename under the currently configured ARGUS_DATA_DIR."""
    return f"{settings.ARGUS_DATA_DIR}/{filename}"


def main() -> None:
    """Parses CLI args, runs one collection cycle, and prints the outcome."""
    logging.basicConfig(level=settings.ARGUS_LOG_LEVEL)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        default=None,
        help="Comma-separated tickers (default: ARGUS_UNIVERSE / Settings default)",
    )
    parser.add_argument("--total-wealth", type=float, default=settings.ARGUS_TOTAL_WEALTH)
    parser.add_argument("--invest-pct", type=float, default=settings.ARGUS_INVEST_PCT)
    parser.add_argument("--risk-tolerance", default=settings.ARGUS_RISK_TOLERANCE)
    parser.add_argument(
        "--buffer-db",
        default=_data_path("ohlcv_buffer.db"),
        help="Persistent intraday candle buffer path",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=_data_path("argus_graph.db"),
        help="LangGraph checkpoint database path",
    )
    parser.add_argument(
        "--decisions-log",
        default=_data_path("decisions.jsonl"),
        help="JSONL file each cycle's decisions are appended to",
    )
    parser.add_argument(
        "--result-out",
        default=_data_path("collector_result.json"),
        help="Where to write this cycle's CollectionResult as JSON, for the "
        "scheduled workflow to fold into status.json",
    )
    args = parser.parse_args()

    if args.universe:
        universe = [t.strip() for t in args.universe.split(",") if t.strip()]
    else:
        universe = settings.ARGUS_UNIVERSE

    pipeline = MFTDataPipeline(tickers=list(universe), db_path=args.buffer_db)
    compiled_graph = build_graph(checkpoint_db_path=args.checkpoint_db)
    result = asyncio.run(
        run_collection_cycle(
            universe=universe,
            total_wealth=args.total_wealth,
            invest_pct=args.invest_pct,
            risk_tolerance=args.risk_tolerance,
            pipeline=pipeline,
            compiled_graph=compiled_graph,
            decisions_log_path=args.decisions_log,
            checkpoint_db_path=args.checkpoint_db,
        )
    )
    pipeline.buffer.close()

    result_path = Path(args.result_out)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(dataclasses.asdict(result)))

    print(f"ran={result.ran} reason={result.reason!r} outcome={result.outcome.value}")
    if result.ran:
        print(
            f"tickers_with_session_data={result.tickers_with_session_data} "
            f"decisions_logged={result.decisions_logged} "
            f"decisions_with_allocation={result.decisions_with_allocation} "
            f"macro_regime={result.macro_regime}"
        )
        if result.degraded_inputs:
            print(f"degraded_inputs={result.degraded_inputs}")

    if result.outcome is CycleOutcome.DEGRADED:
        print(
            "::error::collection cycle DEGRADED — "
            f"only {result.decisions_with_allocation}/{result.decisions_logged} "
            "decision(s) carried a real allocation; nothing worth reconciling this cycle"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
