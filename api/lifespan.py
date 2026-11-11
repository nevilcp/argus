"""
api/lifespan.py

Process startup and shutdown for the API, and the runtime state they build.

Responsibilities:
  - RuntimeState / `runtime`: the process-wide singletons the routes read (live
    session cache, MFT pipeline, background tasks, shared graph, analyze lock)
  - lifespan: run the startup guards, build the graph, start the background loops,
    seed the kill switch, and tear it all down in order on shutdown
  - Keep the MFT pipeline, collector and reconcile loops alive and loud when they crash

Not responsible for:
  - The startup checks themselves (see api/guards.py)
  - The loops' bodies (see argus/orchestration/scheduler.py)
  - Routes (see api/main.py)

Dependencies:
  - argus.orchestration.scheduler, argus.data.pipeline, argus.risk.kill_switch
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI

from api import guards
from argus.agents.macro_hmm import RegimeClassifier
from argus.config import settings
from argus.data.live_session_cache import LiveSessionCache
from argus.data.pipeline import MFTDataPipeline
from argus.governor import governor
from argus.infra.log import configure_logging
from argus.orchestration.collector import CollectionResult
from argus.orchestration.graph import build_graph
from argus.orchestration.scheduler import collector_loop, reconcile_loop
from argus.risk import paper_book
from argus.risk.kill_switch import get_kill_switch, initialize_kill_switch
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    """The API process's mutable singletons, filled in by `lifespan` and read by the routes.

    Every field is None (or False) until startup builds it, which is what lets a
    route answer 503 "not yet initialized" instead of failing on a missing attribute.

    Attributes:
        live_cache: Latest intraday feature dict per ticker, published by the pipeline.
        pipeline: The MFT pipeline; tickers are registered dynamically per /analyze
            request in addition to the ARGUS_UNIVERSE seed.
        pipeline_task: The task running the pipeline.
        collector_task: The unattended collector loop, if enabled.
        reconcile_task: The daily reconciliation loop, if enabled.
        graph: Built once at startup against the same persistent checkpoint path the
            collector uses, so /analyze and the collector share one decision history.
        macro_classifier_fitted: Whether the committed HMM artifact loaded.
        last_collection_result: The collector loop's most recent cycle.
        analyze_lock: Serializes graph invocations across /analyze and the collector
            loop — both call the same governor-gated graph, and a second concurrent
            run only pushes both toward RateLimitExceeded rather than adding real
            throughput. Held only around the graph invoke itself.
    """

    live_cache: LiveSessionCache | None = None
    pipeline: MFTDataPipeline | None = None
    pipeline_task: asyncio.Task | None = None
    collector_task: asyncio.Task | None = None
    reconcile_task: asyncio.Task | None = None
    graph: Any = None
    macro_classifier_fitted: bool = False
    last_collection_result: CollectionResult | None = None
    analyze_lock: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))

    def record_collection(self, result: CollectionResult) -> None:
        """Remembers the collector loop's latest result for /pipeline/status."""
        self.last_collection_result = result


runtime = RuntimeState()


async def _mft_session_callback(session_states: dict) -> None:
    """Receives compressed technical feature dicts from the MFT pipeline on every sweep.

    Publishes them into the live cache so the next /analyze call picks up fresh
    intraday indicators without re-fetching historical data. Eviction of
    tickers no longer tracked happens inside LiveSessionCache.publish, keyed
    off the pipeline's current tracked universe.

    Args:
        session_states: Mapping of ticker → technical feature dict from MFTDataPipeline.
    """
    if runtime.live_cache is not None and runtime.pipeline is not None:
        runtime.live_cache.publish(session_states, runtime.pipeline.tickers)

    logger.info(
        "Live session cache updated: %d ticker(s)",
        len(runtime.live_cache) if runtime.live_cache is not None else 0,
    )


def _log_task_exception(task: asyncio.Task, name: str) -> None:
    """Logs a background task's exception instead of letting it vanish silently.

    An `asyncio.create_task()` result with no held reference can be garbage
    collected mid-flight, and any exception it raises is otherwise only
    reported (noisily, to stderr) once the loop shuts down. Every background
    task here is submitted with this as a done-callback.

    Args:
        task: The finished task.
        name: Label used in the log line, e.g. "MFTDataPipeline".
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("[%s] background task crashed: %s", name, exc, exc_info=exc)


def _start_background_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task:
    """Schedules a coroutine as a background task that logs its own crash.

    Args:
        coro: The coroutine to run for the lifetime of the app.
        name: Label used in the crash log line, e.g. "MFTDataPipeline".

    Returns:
        The created task, which the caller must keep a reference to.
    """
    task = asyncio.create_task(coro)
    task.add_done_callback(lambda t: _log_task_exception(t, name))
    return task


def _start_background_loops() -> None:
    """Creates the live cache and MFT pipeline, then launches the enabled loops."""
    # Seeded from ARGUS_UNIVERSE so the pipeline starts collecting immediately
    # rather than waiting for a first /analyze call to register any tickers
    pipeline = MFTDataPipeline(
        tickers=list(settings.ARGUS_UNIVERSE), market_data=LiveMarketDataProvider()
    )
    runtime.pipeline = pipeline
    runtime.live_cache = LiveSessionCache(interval_minutes=pipeline.interval_minutes)
    runtime.pipeline_task = _start_background_task(
        pipeline.start(on_session_ready=_mft_session_callback), "MFTDataPipeline"
    )
    logger.info(
        "MFT pipeline background task launched: %d ticker(s).",
        len(settings.ARGUS_UNIVERSE),
    )

    if settings.ARGUS_COLLECTOR_ENABLED:
        runtime.collector_task = _start_background_task(
            collector_loop(
                pipeline, runtime.graph, runtime.analyze_lock, runtime.record_collection
            ),
            "CollectorLoop",
        )
        logger.info(
            "Unattended collector loop launched (interval=%ds).",
            settings.ARGUS_COLLECTOR_INTERVAL_SECONDS,
        )

    if settings.ARGUS_RECONCILE_ENABLED:
        runtime.reconcile_task = _start_background_task(reconcile_loop(), "ReconcileLoop")
        logger.info(
            "Daily reconciliation loop launched (hour=%d ET).",
            settings.ARGUS_RECONCILE_HOUR_ET,
        )


def _seed_kill_switch() -> None:
    """Starts the kill switch from the persisted paper equity curve.

    Falls back to ARGUS_TOTAL_WEALTH if no curve exists yet, so a process
    restart doesn't silently forget an already-realized drawdown;
    /kill-switch/reset re-bases if a real session base is known instead.
    """
    book = paper_book.load(str(settings.data_path("paper_equity.json")))
    initialize_kill_switch(
        risk_tolerance=settings.ARGUS_RISK_TOLERANCE, portfolio_value=book.high_water_mark
    )
    ks = get_kill_switch()
    if ks is not None:
        ks.update_portfolio_value(book.equity)
    logger.info(
        "Kill switch initialized (paper equity=$%.2f, peak=$%.2f).",
        book.equity,
        book.high_water_mark,
    )


async def _stop_background_loops() -> None:
    """Stops the pipeline, loops, kill-switch daemon and candle buffer, in that order."""
    logger.info("Stopping background tasks...")
    if runtime.pipeline is not None:
        await runtime.pipeline.stop()

    for task in (runtime.pipeline_task, runtime.collector_task, runtime.reconcile_task):
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    ks = get_kill_switch()
    if ks is not None:
        await asyncio.to_thread(ks.stop, 5.0)

    if runtime.pipeline is not None:
        # Pipeline-owned so it can wait out an orphaned buffer worker rather than
        # racing it — cancelling the loops above can't stop one already mid-`to_thread`
        await runtime.pipeline.close_buffer()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Starts background tasks on startup and stops them cleanly on shutdown."""
    configure_logging()
    guards.warn_on_permissive_security_defaults()
    guards.assert_single_worker()
    guards.assert_registered_models()
    guards.acquire_process_lock()

    runtime.graph = build_graph(checkpoint_db_path=str(settings.data_path("argus_graph.db")))

    runtime.macro_classifier_fitted = RegimeClassifier.load(settings.ARGUS_HMM_MODEL_PATH).is_fitted
    if runtime.macro_classifier_fitted:
        logger.info("Macro HMM artifact loaded: %s", settings.ARGUS_HMM_MODEL_PATH)
    else:
        logger.warning(
            "Macro HMM artifact not loaded; classifier will use the rule-based fallback."
        )

    _start_background_loops()
    _seed_kill_switch()
    logger.info("Governor initialized: %s", governor.get_usage_report())

    yield

    await _stop_background_loops()
    guards.release_process_lock()
    logger.info("Complete.")
