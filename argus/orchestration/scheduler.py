"""
argus/orchestration/scheduler.py

The two long-lived loops an always-on API process runs in the background: the
unattended collection cycle on a fixed interval, and the daily reconciliation
pass at a fixed hour.

Responsibilities:
  - collector_loop: run run_collection_cycle_from_settings every
    ARGUS_COLLECTOR_INTERVAL_SECONDS, reporting each result to the caller
  - reconcile_loop / reconcile_once: run one reconciliation pass a day at
    ARGUS_RECONCILE_HOUR_ET off the event loop, then sync the kill switch
  - seconds_until_next_reconcile: the DST-safe wait until that hour

Not responsible for:
  - Starting, tracking or cancelling the tasks that run these loops (see
    api/lifespan.py)
  - What one collection cycle or one reconciliation pass does (see
    orchestration/collector.py and orchestration/reconcile.py)

Dependencies:
  - argus.orchestration.collector, argus.orchestration.reconcile
  - argus.risk.kill_switch (get_kill_switch)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from argus.config import settings
from argus.data.pipeline import MFTDataPipeline
from argus.infra.clock import now_market
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.collector import CollectionResult, run_collection_cycle_from_settings
from argus.orchestration.reconcile import run_reconciliation_pass
from argus.risk.kill_switch import get_kill_switch
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger(__name__)


async def collector_loop(
    pipeline: MFTDataPipeline,
    compiled_graph,
    analyze_lock: asyncio.Semaphore,
    on_result: Callable[[CollectionResult], None],
) -> None:
    """Runs one collection cycle on a fixed interval for as long as the app is alive.

    Args:
        pipeline: The shared MFT pipeline instance, so this loop reuses the
            same warm buffer the live /analyze cache draws from.
        compiled_graph: The shared, already-built graph — the same instance
            /analyze invokes, so this loop reopens no extra SqliteSaver
            connection on every cycle.
        analyze_lock: The semaphore /analyze also holds around its graph run,
            so a cycle skips rather than runs concurrently with a request.
        on_result: Called with each completed cycle's result, so the caller
            can surface the latest one (e.g. on /pipeline/status).
    """
    while True:
        try:
            result = await run_collection_cycle_from_settings(
                pipeline=pipeline, compiled_graph=compiled_graph, analyze_lock=analyze_lock
            )
            on_result(result)
            logger.info("cycle result: %s", result)
        except Exception:
            logger.exception("cycle failed")
        await asyncio.sleep(settings.ARGUS_COLLECTOR_INTERVAL_SECONDS)


def reconcile_once() -> None:
    """Runs one reconciliation pass, then syncs the kill switch off its resulting equity.

    Synchronous by design — `reconcile_loop` runs this via `asyncio.to_thread`
    since it performs per-ticker yfinance fetches directly, which would
    otherwise block the event loop for the whole app for the duration of a run.
    Kill-switch sync stays here rather than in run_reconciliation_pass: this is
    the caller with a daemon to sync to, and it's skipped when the paper-book
    step itself failed — syncing a fresh report's zeroed-out default equity
    would read as a total portfolio loss.
    """
    report = run_reconciliation_pass(
        LiveMarketDataProvider(),
        get_cultural_memory(),
        str(settings.data_path("paper_equity.json")),
        decisions_log_path=str(settings.data_path("decisions.jsonl")),
        checkpoint_db_path=str(settings.data_path("argus_graph.db")),
    )
    for line in report.summary_lines():
        logger.info("%s", line)
    for error in report.errors:
        logger.error("%s", error)

    if report.paper_book_updated:
        ks = get_kill_switch()
        if ks is not None:
            ks.update_portfolio_value(report.equity)


def seconds_until_next_reconcile(now_et: datetime, hour: int) -> float:
    """Computes seconds until the next `hour:00` ET, safe across DST transitions.

    `target - now_et` alone is unsafe here: both are built from the same
    ZoneInfo instance, and aware-datetime subtraction with a shared tzinfo
    object compares wall-clock fields directly rather than each side's own
    UTC offset, so the result silently drops or adds an hour across a DST
    boundary. Converting both to UTC first sidesteps that.

    Args:
        now_et: Current time in America/New_York.
        hour: Target hour (0-23) in America/New_York.

    Returns:
        Seconds until the next occurrence of that hour, today or tomorrow.
    """
    target = now_et.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now_et:
        target += timedelta(days=1)
    return (target.astimezone(UTC) - now_et.astimezone(UTC)).total_seconds()


async def reconcile_loop() -> None:
    """Runs the reconciliation pass once a day at settings.ARGUS_RECONCILE_HOUR_ET."""
    while True:
        now_et = now_market()
        await asyncio.sleep(seconds_until_next_reconcile(now_et, settings.ARGUS_RECONCILE_HOUR_ET))

        try:
            await asyncio.to_thread(reconcile_once)
        except Exception:
            logger.exception("cycle failed")
