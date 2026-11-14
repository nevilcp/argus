"""Background collection and daily reconciliation scheduling."""

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
    """Run one collection cycle on a fixed interval for as long as the app is alive."""
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
    """Run one reconciliation pass, then syncs the kill switch off its resulting equity."""
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
    """Compute seconds until the next Eastern reconciliation hour using UTC elapsed time across DST."""
    target = now_et.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now_et:
        target += timedelta(days=1)
    return (target.astimezone(UTC) - now_et.astimezone(UTC)).total_seconds()


async def reconcile_loop() -> None:
    """Run the reconciliation pass once a day at settings.ARGUS_RECONCILE_HOUR_ET."""
    while True:
        now_et = now_market()
        await asyncio.sleep(seconds_until_next_reconcile(now_et, settings.ARGUS_RECONCILE_HOUR_ET))

        try:
            await asyncio.to_thread(reconcile_once)
        except Exception:
            logger.exception("cycle failed")
