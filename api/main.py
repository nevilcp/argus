"""
api/main.py

FastAPI gateway service for the ARGUS multi-agent decision system.

Exposes REST API endpoints to orchestrate execution graphs and audit system
safety/governor statistics. There is no live /backtest endpoint; a
multi-year walk-forward backtest isn't offered, so scripts/replay_backtest.py
replays recorded fixtures through this same graph over the ~60-day window
that is genuinely available.

Responsibilities:
  - Route analysis requests to the LangGraph execution pipeline
  - Append every /analyze decision to the same decisions.jsonl log the
    unattended collector writes to (see argus/orchestration/collector.py),
    so reconciliation sees both entry points into the graph
  - Enforce kill-switch and VIX blackout checks before every allocation request
  - Expose health, memory, governor, and kill-switch management endpoints
  - Host the MFT pipeline as a background asyncio task and maintain a live
    session state cache consumed by /analyze
  - Optionally run the unattended collector and daily reconciliation loops
    (ARGUS_COLLECTOR_ENABLED / ARGUS_RECONCILE_ENABLED) so the system keeps
    accumulating decisions and outcomes without a human calling /analyze,
    pruning every store the reconcile loop touches back to a bounded window
    (PR 6) as part of the same daily pass
  - Serialize graph invocations across /analyze and the collector loop, and
    guard against a second process sharing this one's data directory

Not responsible for:
  - Agent logic (see argus/agents/)
  - Signal computation (see argus/orchestration/)
  - Execution order routing
"""

import asyncio
import fcntl
import logging
import os
import sys
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import argus
from argus.agents.macro import MacroStatisticalAgent
from argus.config import settings
from argus.data.live_session_cache import LiveSessionCache
from argus.data.pipeline import MFTDataPipeline
from argus.data.tickers import TICKER_PATTERN
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.collector import (
    CollectionResult,
    append_decisions_jsonl,
    run_collection_cycle,
)
from argus.orchestration.governor import REGISTERED_MODELS, RateLimitExceeded, UnregisteredModel, governor
from argus.orchestration.graph import build_graph
from argus.orchestration.reconciliation import run_reconciliation_pass
from argus.orchestration.state import ARGUSState
from argus.params import SYSTEM
from argus.risk import paper_book
from argus.risk.kill_switch import get_kill_switch, initialize_kill_switch
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger("argus.api")

_ET = ZoneInfo("America/New_York")

_live_cache: LiveSessionCache | None = None

# Initialized during lifespan startup; tickers are registered dynamically per
# /analyze request in addition to the ARGUS_UNIVERSE seed
_mft_pipeline: MFTDataPipeline | None = None
_pipeline_task: asyncio.Task | None = None
_collector_task: asyncio.Task | None = None
_reconcile_task: asyncio.Task | None = None
_last_collection_result: CollectionResult | None = None

# Serializes graph invocations across /analyze and the unattended collector
# loop (API-3) — both call the same governor-gated graph, and a second
# concurrent run only pushes both toward RateLimitExceeded rather than adding
# real throughput. Held only around the graph invoke itself.
_analyze_semaphore = asyncio.Semaphore(1)

# Holds the open file object backing the process-exclusivity flock (API-2) for
# as long as this process runs; garbage-collecting it would release the lock
_process_lock_file = None

# Built once in lifespan startup, pointed at the same persistent checkpoint
# path the collector loop uses, so /analyze and the unattended collector
# share one decision history instead of two disjoint SQLite files
_graph = None

# A throwaway MacroStatisticalAgent constructed solely to surface whether the
# committed HMM artifact loaded, for /pipeline/status. Cheap: the constructor
# only loads a joblib file, unlike the old fit_on_history() network fetch this
# replaced (see the deleted startup block below).
_macro_status_agent: MacroStatisticalAgent | None = None


async def _mft_session_callback(session_states: dict) -> None:
    """Receives compressed technical feature dicts from the MFT pipeline on every sweep.

    Publishes them into the module-level live cache so the next /analyze call
    picks up fresh intraday indicators without re-fetching historical data.
    Eviction of tickers no longer tracked (API-9) happens inside
    LiveSessionCache.publish, keyed off the pipeline's current tracked
    universe.

    Args:
        session_states: Mapping of ticker → technical feature dict from MFTDataPipeline.
    """
    if _live_cache is not None and _mft_pipeline is not None:
        _live_cache.publish(session_states, _mft_pipeline.tickers)

    logger.info(
        "[MFT] Live session cache updated: %d ticker(s)",
        len(_live_cache) if _live_cache is not None else 0,
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


async def _collector_loop(pipeline: MFTDataPipeline, compiled_graph) -> None:
    """Runs run_collection_cycle on a fixed interval for as long as the app is alive.

    Args:
        pipeline: The shared MFT pipeline instance, so this loop reuses the
            same warm buffer the live /analyze cache draws from.
        compiled_graph: The shared, already-built graph (module-level
            ``_graph``) — the same instance /analyze invokes, so this loop
            reopens no extra SqliteSaver connection on every cycle.
    """
    global _last_collection_result
    while True:
        try:
            _last_collection_result = await run_collection_cycle(
                universe=list(settings.ARGUS_UNIVERSE),
                total_wealth=settings.ARGUS_TOTAL_WEALTH,
                invest_pct=settings.ARGUS_INVEST_PCT,
                risk_tolerance=settings.ARGUS_RISK_TOLERANCE,
                pipeline=pipeline,
                compiled_graph=compiled_graph,
                decisions_log_path=f"{settings.ARGUS_DATA_DIR}/decisions.jsonl",
                analyze_lock=_analyze_semaphore,
            )
            logger.info("[Collector] cycle result: %s", _last_collection_result)
        except Exception:
            logger.exception("[Collector] cycle failed")
        await asyncio.sleep(settings.ARGUS_COLLECTOR_INTERVAL_SECONDS)


def _reconcile_once() -> None:
    """Runs one reconciliation pass, then syncs the kill switch off its resulting equity.

    Synchronous by design — `_reconcile_loop` runs this via `asyncio.to_thread`
    since it performs per-ticker yfinance fetches directly, which would
    otherwise block the event loop for the whole app for the duration of a run.
    Kill-switch sync stays here rather than in run_reconciliation_pass: this is
    the caller with a daemon to sync to, and it's skipped when the paper-book
    step itself failed — syncing a fresh report's zeroed-out default equity
    would read as a total portfolio loss (issue #77).
    """
    report = run_reconciliation_pass(
        LiveMarketDataProvider(),
        get_cultural_memory(),
        f"{settings.ARGUS_DATA_DIR}/paper_equity.json",
        decisions_log_path=f"{settings.ARGUS_DATA_DIR}/decisions.jsonl",
        checkpoint_db_path=f"{settings.ARGUS_DATA_DIR}/argus_graph.db",
    )
    logger.info(
        "[Reconcile] stored %d/%d outcome(s)", report.outcomes_stored, report.decisions_loaded
    )
    # Only meaningful once the paper-book step actually ran (see docstring) —
    # otherwise these are still their zeroed-out defaults, not a real value
    if report.paper_book_updated:
        logger.info(
            "[Reconcile] equity=$%.2f (drawdown=%.1f%%)", report.equity, report.drawdown * 100
        )
    if report.decisions_compacted is not None:
        logger.info("[Reconcile] decisions.jsonl compacted: %d retained", report.decisions_compacted)
    if report.checkpoints_pruned is not None:
        logger.info("[Reconcile] checkpoint threads pruned: %d", report.checkpoints_pruned)
    logger.info("[Reconcile] PENDING snapshots expired: %d", report.pending_snapshots_expired)
    for error in report.errors:
        logger.error("[Reconcile] %s", error)

    if report.paper_book_updated:
        ks = get_kill_switch()
        if ks is not None:
            ks.update_portfolio_value(report.equity)


def _seconds_until_next_reconcile(now_et: datetime, hour: int) -> float:
    """Seconds to sleep until the next `hour:00` ET, correct across DST transitions (API-13).

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
    return (target.astimezone(timezone.utc) - now_et.astimezone(timezone.utc)).total_seconds()


async def _reconcile_loop() -> None:
    """Runs the reconciliation pass once a day at settings.ARGUS_RECONCILE_HOUR_ET."""
    while True:
        now_et = datetime.now(_ET)
        await asyncio.sleep(_seconds_until_next_reconcile(now_et, settings.ARGUS_RECONCILE_HOUR_ET))

        try:
            await asyncio.to_thread(_reconcile_once)
        except Exception:
            logger.exception("[Reconcile] cycle failed")


def _configure_logging() -> None:
    """Attaches a stream handler to the ``argus`` logger tree so ARGUS_LOG_LEVEL takes effect (OBS-1).

    Under uvicorn the root logger has no handlers at level WARNING, so every
    argus.* `logger.info`/`debug` call is silently dropped and
    ARGUS_LOG_LEVEL is inert. `propagate = False` stops uvicorn's own
    logging config from resetting this. Idempotent, so re-running lifespan
    (e.g. across tests) doesn't stack duplicate handlers.
    """
    argus_logger = logging.getLogger("argus")
    argus_logger.setLevel(settings.ARGUS_LOG_LEVEL)
    argus_logger.propagate = False
    if not argus_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        argus_logger.addHandler(handler)


def _assert_single_worker() -> None:
    """Fails fast on the two argv/env spellings of "more than one worker" this can detect.

    The rate governor and kill switch are in-process singletons (see
    argus/orchestration/governor.py). A second worker process would run its
    own ignorant copy of both, silently doubling real Groq usage against the
    same account and gating risk independently rather than jointly — the
    Actions collector (scripts/reconcile_outcomes.py) is already a second,
    independent governor against that same key, and this assertion exists so
    a deployment doesn't add a third by accident. See Dockerfile.api's CMD.

    This is a best-effort pre-flight only, not the real guard: it can't see
    gunicorn's `-w`, a gunicorn config file, or a programmatic
    `uvicorn.run(workers=N)`. `_acquire_process_lock` (API-2) is what actually
    catches those, and the case that costs real money — two containers on one
    data volume — which no argv/env spelling of "workers" describes at all.

    Raises:
        RuntimeError: If ``--workers``/``--workers=N`` in argv, or
            ``WEB_CONCURRENCY`` in the environment, requests a value other
            than "1".
    """
    argv = sys.argv
    if "--workers" in argv:
        idx = argv.index("--workers")
        if idx + 1 < len(argv) and argv[idx + 1] != "1":
            raise RuntimeError(
                f"ARGUS must run with --workers 1 (in-process governor/kill-switch "
                f"singletons); got --workers {argv[idx + 1]}. This is only a pre-flight "
                f"check of argv/env — it can't see a gunicorn config file or "
                f"uvicorn.run(workers=N), and _acquire_process_lock's flock is the real guard."
            )
    for arg in argv:
        if arg.startswith("--workers=") and arg.removeprefix("--workers=") != "1":
            raise RuntimeError(
                f"ARGUS must run with --workers 1 (in-process governor/kill-switch "
                f"singletons); got {arg}. This is only a pre-flight check of argv/env — it "
                f"can't see a gunicorn config file or uvicorn.run(workers=N), and "
                f"_acquire_process_lock's flock is the real guard."
            )

    web_concurrency = os.environ.get("WEB_CONCURRENCY")
    if web_concurrency is not None and web_concurrency != "1":
        raise RuntimeError(
            f"ARGUS must run with WEB_CONCURRENCY=1 (in-process governor/kill-switch "
            f"singletons); got WEB_CONCURRENCY={web_concurrency}. This is only a pre-flight "
            f"check of argv/env — it can't see a gunicorn config file or "
            f"uvicorn.run(workers=N), and _acquire_process_lock's flock is the real guard."
        )


def _acquire_process_lock() -> None:
    """Takes an exclusive, non-blocking flock on ``${ARGUS_DATA_DIR}/argus.lock`` (API-2).

    This is the real single-process guard, unlike `_assert_single_worker`'s
    argv/env pre-flight: it catches every way two ARGUS processes could end up
    sharing one data directory — including two whole containers — not just
    the worker-count spellings that pre-flight can parse. Note this only runs
    during lifespan startup, so ``uvicorn --reload``'s parent process (which
    doesn't run lifespan) is not itself protected; the reloaded child is.
    Scoping the lock file to ``ARGUS_DATA_DIR`` also correctly leaves
    ``scripts/collect_session.py``'s GitHub Actions runner unblocked — it
    writes to its own runner-local directory, not this deployment's, and is
    already an accepted second governor/kill-switch process (see
    ``argus/orchestration/collector.py``'s module docstring).

    Raises:
        RuntimeError: If another process already holds the lock.
    """
    global _process_lock_file
    lock_path = Path(settings.ARGUS_DATA_DIR) / "argus.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_file.close()
        raise RuntimeError(
            f"Another ARGUS process already holds {lock_path} — refusing to start a "
            f"second process against the same data directory."
        ) from exc
    _process_lock_file = lock_file


def _warn_on_permissive_security_defaults() -> None:
    """Logs a WARNING for each security-relevant setting still at its permissive default (DEP-8).

    ARGUS_CORS_ORIGINS defaulting to ``["*"]`` and a blank ARGUS_API_KEY are
    both fine for local development behind a trusted network boundary, but
    silent in a deployment actually exposed to the internet. Neither default
    changes here — only whether the deployment says so out loud.
    """
    if settings.ARGUS_CORS_ORIGINS == ["*"]:
        logger.warning(
            "[Startup] ARGUS_CORS_ORIGINS is at its default (\"*\"): every origin is allowed. "
            "Set it explicitly for a deployment reachable outside a trusted network."
        )
    if not settings.ARGUS_API_KEY:
        logger.warning(
            "[Startup] ARGUS_API_KEY is unset: /analyze and /kill-switch/reset accept requests "
            "with no X-API-Key check. Set it for a deployment reachable outside a trusted network."
        )


def _assert_registered_models() -> None:
    """Fails fast if a configured agent model has no rate-limit profile (GOV-11).

    Converts what would otherwise be a first-call UnregisteredModel deep inside
    a request into a boot-time failure, so a typo'd or unsupported
    ARGUS_*_MODEL setting is caught before the process ever serves traffic.

    Raises:
        UnregisteredModel: If any of ARGUS_FUNDAMENTAL_MODEL, ARGUS_SENTIMENT_MODEL,
            or ARGUS_PORTFOLIO_MODEL is absent from REGISTERED_MODELS.
    """
    for model in (
        settings.ARGUS_FUNDAMENTAL_MODEL,
        settings.ARGUS_SENTIMENT_MODEL,
        settings.ARGUS_PORTFOLIO_MODEL,
    ):
        if model not in REGISTERED_MODELS:
            raise UnregisteredModel(f"{model!r} is configured but has no registered rate-limit profile")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Starts background tasks on startup and stops them cleanly on shutdown."""
    global _mft_pipeline, _pipeline_task, _collector_task, _reconcile_task, _graph, _macro_status_agent, _live_cache

    _configure_logging()
    _warn_on_permissive_security_defaults()
    _assert_single_worker()
    _assert_registered_models()
    _acquire_process_lock()

    _graph = build_graph(checkpoint_db_path=f"{settings.ARGUS_DATA_DIR}/argus_graph.db")

    _macro_status_agent = MacroStatisticalAgent()
    if _macro_status_agent.classifier.is_fitted:
        logger.info("[Startup] Macro HMM artifact loaded: %s", settings.ARGUS_HMM_MODEL_PATH)
    else:
        logger.warning(
            "[Startup] Macro HMM artifact not loaded; classifier will use the rule-based fallback."
        )

    # Seeded from ARGUS_UNIVERSE so the pipeline starts collecting immediately
    # rather than waiting for a first /analyze call to register any tickers
    _mft_pipeline = MFTDataPipeline(tickers=list(settings.ARGUS_UNIVERSE))
    _live_cache = LiveSessionCache(interval_minutes=_mft_pipeline.interval_minutes)
    _pipeline_task = asyncio.create_task(_mft_pipeline.start(on_session_ready=_mft_session_callback))
    _pipeline_task.add_done_callback(lambda t: _log_task_exception(t, "MFTDataPipeline"))
    logger.info(
        "[Startup] MFT pipeline background task launched: %d ticker(s).",
        len(settings.ARGUS_UNIVERSE),
    )

    if settings.ARGUS_COLLECTOR_ENABLED:
        _collector_task = asyncio.create_task(_collector_loop(_mft_pipeline, _graph))
        _collector_task.add_done_callback(lambda t: _log_task_exception(t, "CollectorLoop"))
        logger.info(
            "[Startup] Unattended collector loop launched (interval=%ds).",
            settings.ARGUS_COLLECTOR_INTERVAL_SECONDS,
        )

    if settings.ARGUS_RECONCILE_ENABLED:
        _reconcile_task = asyncio.create_task(_reconcile_loop())
        _reconcile_task.add_done_callback(lambda t: _log_task_exception(t, "ReconcileLoop"))
        logger.info(
            "[Startup] Daily reconciliation loop launched (hour=%d ET).",
            settings.ARGUS_RECONCILE_HOUR_ET,
        )

    # Seeded from the persisted paper equity curve (falls back to
    # ARGUS_TOTAL_WEALTH if none exists yet) so a process restart doesn't
    # silently forget an already-realized drawdown; /kill-switch/reset
    # re-bases if a real session base is known instead
    _book = paper_book.load(f"{settings.ARGUS_DATA_DIR}/paper_equity.json")
    initialize_kill_switch(
        risk_tolerance=settings.ARGUS_RISK_TOLERANCE, portfolio_value=_book.high_water_mark
    )
    _ks = get_kill_switch()
    if _ks is not None:
        _ks.update_portfolio_value(_book.equity)
    logger.info(
        "[Startup] Kill switch initialized (paper equity=$%.2f, peak=$%.2f).",
        _book.equity,
        _book.high_water_mark,
    )

    logger.info("[Startup] Governor initialized: %s", governor.get_usage_report())

    yield

    logger.info("[Shutdown] Stopping background tasks...")
    if _mft_pipeline is not None:
        await _mft_pipeline.stop()

    for task in (_pipeline_task, _collector_task, _reconcile_task):
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    _ks = get_kill_switch()
    if _ks is not None:
        await asyncio.to_thread(_ks.stop, 5.0)

    if _mft_pipeline is not None:
        # Pipeline-owned so it can wait out an orphaned buffer worker rather than
        # racing it — cancelling the loops above can't stop one already mid-`to_thread` (MFT-16)
        await _mft_pipeline.close_buffer()

    if _process_lock_file is not None:
        fcntl.flock(_process_lock_file, fcntl.LOCK_UN)
        _process_lock_file.close()

    logger.info("[Shutdown] Complete.")


app = FastAPI(title="ARGUS API", version=argus.__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.ARGUS_CORS_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Gates mutating endpoints (/analyze, /kill-switch/reset) behind a shared secret.

    A blank ARGUS_API_KEY (the default) disables the check entirely, so local
    development and any deployment that hasn't set one keep working unchanged
    — this is a minimum bar for a publicly reachable deployment, not a
    substitute for running behind a real auth layer.

    Args:
        x_api_key: Value of the X-API-Key request header, if present.

    Raises:
        HTTPException 401: If ARGUS_API_KEY is set and the header doesn't match.
    """
    if not settings.ARGUS_API_KEY:
        return
    if x_api_key != settings.ARGUS_API_KEY:
        raise HTTPException(401, "Missing or invalid X-API-Key header.")


class AnalysisRequest(BaseModel):
    """Request payload defining parameters for stock universe valuation and rebalancing."""

    tickers: list[str] = Field(min_length=1, max_length=20)
    total_wealth: float = Field(gt=1000)
    invest_pct: float = Field(gt=0.05, le=0.95)
    risk_tolerance: Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"] = "MODERATE"

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, tickers: list[str]) -> list[str]:
        """Strips whitespace, upper-cases, validates ticker shape, then dedupes (API-1, API-6).

        Runs in that order deliberately: pydantic's StringConstraints checks
        `pattern` against the raw, pre-transform string even when combined
        with `to_upper`/`strip_whitespace`, so a case/whitespace-only field
        constraint can't do this — the transform has to happen before the
        pattern check runs.

        Raises:
            ValueError: If any entry doesn't match TICKER_PATTERN after normalization.
        """
        normalized = [t.strip().upper() for t in tickers]
        invalid = [t for t in normalized if not TICKER_PATTERN.match(t)]
        if invalid:
            raise ValueError(f"Invalid ticker symbol(s): {invalid}")
        seen: set[str] = set()
        deduped = []
        for t in normalized:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped


class AnalysisResponse(BaseModel):
    """Synthesized portfolio allocation recommendation and system diagnostics."""

    session_id: str
    portfolio: list[dict]
    cash_reserve_pct: float
    macro_regime: str
    vix_level: float
    governor_report: dict
    timestamp: str
    errors: list[str] = []


def _background_task_status(task: asyncio.Task | None) -> dict:
    """Reports a background task's liveness, distinguishing "never enabled" from "died" (OBS-2).

    Args:
        task: The task to inspect, or None if the loop it would run was never
            started (e.g. ARGUS_COLLECTOR_ENABLED is False).

    Returns:
        Dict with ``enabled``, ``alive``, and — only when dead — ``error``.
    """
    if task is None:
        return {"enabled": False, "alive": True}
    if not task.done():
        return {"enabled": True, "alive": True}
    if task.cancelled():
        error = "task was cancelled"
    else:
        exc = task.exception()
        error = str(exc) if exc else "task exited unexpectedly"
    return {"enabled": True, "alive": False, "error": error}


@app.get("/health")
async def health():
    """Validates connectivity to model backends, API quotas, and system diagnostics.

    Unlike the P0 gaps it replaces, this reads background-task liveness, the
    kill switch's halt state, and cache freshness — not just in-memory
    governor counters — so a dead pipeline/collector/reconcile loop shows up
    here instead of staying invisible behind a green /health (OBS-2).

    Raises:
        HTTPException 503: If any background task (MFT pipeline, collector,
            or reconciliation loop) has stopped running.
    """
    try:
        llama_cap = governor.get_remaining_capacity(settings.ARGUS_PORTFOLIO_MODEL)
        can_make_calls = llama_cap > 0
    except Exception:
        can_make_calls = False

    background_tasks = {
        "mft_pipeline": _background_task_status(_pipeline_task),
        "collector": _background_task_status(_collector_task),
        "reconcile": _background_task_status(_reconcile_task),
    }
    dead_tasks = [name for name, status in background_tasks.items() if not status["alive"]]

    ks = get_kill_switch()

    newest_cache_entry_age_seconds = None
    if _mft_pipeline is not None and _mft_pipeline.is_market_hours() and _live_cache is not None:
        cache_age_seconds, _ = _live_cache.ages(datetime.now(_ET))
        if cache_age_seconds:
            newest_cache_entry_age_seconds = min(cache_age_seconds.values())

    body = {
        "status": "ok" if not dead_tasks else "degraded",
        "model_versions": {
            "synthesis": settings.ARGUS_PORTFOLIO_MODEL,
            "sentiment": settings.ARGUS_SENTIMENT_MODEL,
            "fundamental": settings.ARGUS_FUNDAMENTAL_MODEL,
            "finbert": "ProsusAI/finbert",
        },
        "can_make_calls": can_make_calls,
        "governor_report": governor.get_usage_report(),
        "background_tasks": background_tasks,
        "kill_switch_halted": ks.is_halted if ks else False,
        "newest_cache_entry_age_seconds": newest_cache_entry_age_seconds,
    }

    if dead_tasks:
        raise HTTPException(503, f"Background task(s) not running: {', '.join(dead_tasks)}")
    return body


@app.get("/pipeline/status")
async def pipeline_status():
    """Reports the background MFT pipeline's live state without perturbing it.

    Intended for watching the unattended collector — buffer depth, session
    cache freshness, and the last automatic collection cycle's outcome — all
    without triggering a real analysis request.

    Raises:
        HTTPException 503: If the pipeline hasn't started yet.
    """
    if _mft_pipeline is None:
        raise HTTPException(503, "Pipeline not yet initialized.")

    buffer_depth = await asyncio.to_thread(_mft_pipeline.buffer.row_counts)

    cache_age_seconds, bar_age_seconds = (
        _live_cache.ages(datetime.now(_ET)) if _live_cache is not None else ({}, {})
    )

    return {
        "tracked_tickers": list(_mft_pipeline.tickers),
        "buffer_depth": buffer_depth,
        "cache_age_seconds": cache_age_seconds,
        "bar_age_seconds": bar_age_seconds,
        "is_market_hours": _mft_pipeline.is_market_hours(),
        "collector_enabled": settings.ARGUS_COLLECTOR_ENABLED,
        "last_collection_result": (
            None if _last_collection_result is None else asdict(_last_collection_result)
        ),
        "reconcile_enabled": settings.ARGUS_RECONCILE_ENABLED,
        "reconcile_hour_et": settings.ARGUS_RECONCILE_HOUR_ET,
        "macro_classifier_fitted": (
            _macro_status_agent.classifier.is_fitted if _macro_status_agent else False
        ),
    }


@app.post("/analyze", response_model=AnalysisResponse, dependencies=[Depends(require_api_key)])
async def analyze(req: AnalysisRequest):
    """Executes a real-time portfolio analysis pipeline across a target ticker universe.

    Checks system safety (kill switches), validates that the MFT live cache is fully
    populated for all requested tickers, then runs LangGraph arbitrated orchestration
    and returns target allocation distributions.

    The kill switch's drawdown gate is process-global, fixed at startup from
    settings.ARGUS_RISK_TOLERANCE (see argus/risk/kill_switch.py's class
    docstring) — a request's own risk_tolerance does not change which
    threshold guards it. A mismatch is logged, not rejected: this endpoint
    still serves the request under the configured threshold (KS-6).

    Args:
        req: AnalysisRequest with tickers, total_wealth, invest_pct, and risk_tolerance.

    Raises:
        HTTPException 401: If ARGUS_API_KEY is set and the X-API-Key header doesn't match.
        HTTPException 429: If the governor's rate budget is exhausted for a
            configured model (GOV-7), or another analysis (a concurrent
            /analyze call or the unattended collector) is already in progress
            (API-3) — retry after a short wait.
        HTTPException 503: If the graph hasn't finished initializing, the kill switch
            is triggered (before or after the graph run), VIX is above the blackout
            threshold, the market is currently closed, the MFT cache has not yet
            been populated for all requested tickers, the cache hasn't been
            refreshed recently (pipeline stalled), the cached bars are older
            than expected (stale data survived a restart), or a configured
            model has no rate-limit profile (a misconfiguration, not a
            transient condition).
        HTTPException 500: If the LangGraph execution fails or produces no allocation.
    """
    ks = get_kill_switch()
    if ks and ks.is_halted:
        raise HTTPException(503, "System halted. Kill switch triggered. Manual reset required.")
    if ks and not ks.new_positions_allowed:
        raise HTTPException(503, "New positions blocked. VIX above threshold.")
    if ks and req.risk_tolerance != ks.risk_tolerance:
        logger.warning(
            "[API] Request risk_tolerance=%s differs from the kill switch's configured "
            "risk_tolerance=%s; the configured threshold still governs this request.",
            req.risk_tolerance,
            ks.risk_tolerance,
        )

    # Daily bars would mismatch the resolution the technical agent expects, so a
    # closed market (checked first) means idle rather than a lower-resolution
    # fallback. Checking it first also confines every age check below to a
    # weekday 09:30-16:00 ET window, eliminating the weekend/holiday false positive.
    if _mft_pipeline is None or not _mft_pipeline.is_market_hours():
        raise HTTPException(
            503,
            "US equity market is currently closed. MFT pipeline is idle. "
            "Retry between 09:30 and 16:00 ET on a weekday.",
        )

    # Registered only once every earlier gate has cleared (API-12) — a
    # rejected request must not mutate pipeline state or spend a slot in the
    # tracked-universe cap
    _mft_pipeline.register_tickers(req.tickers)

    if _live_cache is None:
        raise HTTPException(503, "Live session cache not yet initialized.")

    # Admission answers only "how old is this" for each ticker; the ordering
    # against market hours above is what decides whether that age matters
    # right now (issue #78)
    admission = _live_cache.admit(req.tickers, datetime.now(_ET))

    if admission.absent:
        raise HTTPException(
            503,
            f"MFT live cache not yet populated for: {admission.absent}. "
            "The pipeline is warming up — retry after the next sweep (~5 min).",
        )
    if admission.stalled:
        raise HTTPException(
            503,
            f"MFT pipeline appears stalled for: {admission.stalled}. "
            "The live cache hasn't been refreshed recently — check /pipeline/status.",
        )
    if admission.stale:
        raise HTTPException(
            503,
            f"MFT cache data is stale for: {admission.stale}. "
            "The underlying candles are older than expected — check /pipeline/status.",
        )

    live_states = admission.admitted
    logger.info("[API] MFT cache hit for all %d tickers", len(live_states))

    state = ARGUSState(
        ticker=req.tickers[0],
        universe=req.tickers,
        total_wealth=req.total_wealth,
        invest_pct=req.invest_pct,
        risk_tolerance=req.risk_tolerance,
        backtest_mode=False,
        session_seed=None,
        as_of=None,
        price_history={},
        session_states=live_states,
        macro_context=None,
        technical_signals={},
        fundamental_signals={},
        sentiment_signals={},
        cultural_memory={"wisdom": [], "warnings": []},
        risk_assessments={},
        aggregated_signals={},
        portfolio_allocation=None,
        decisions=[],
        errors=[],
    )

    if _graph is None:
        raise HTTPException(503, "Agent graph not yet initialized.")

    config = {"configurable": {"thread_id": str(uuid4())}}

    # No await between this check and the acquire below, so nothing can slip
    # in between them (API-3) — Semaphore.acquire returns synchronously when
    # unlocked, and a later refactor that inserts an await here breaks that.
    if _analyze_semaphore.locked():
        raise HTTPException(
            429,
            "Another analysis is already in progress. Retry shortly.",
            headers={"Retry-After": str(SYSTEM.analyze_retry_after_seconds)},
        )

    try:
        async with _analyze_semaphore:
            final_state = await asyncio.wait_for(
                asyncio.to_thread(_graph.invoke, state, config),
                timeout=settings.ARGUS_ANALYZE_DEADLINE_SECONDS,
            )
    except asyncio.TimeoutError:
        # The underlying thread can't be cancelled and keeps running (GOV-12
        # notes async job submission as future work), but the server no longer
        # holds the connection open past a proxy's own timeout with no signal.
        logger.error(
            "[API] Graph run exceeded the %ss deadline", settings.ARGUS_ANALYZE_DEADLINE_SECONDS
        )
        raise HTTPException(
            504,
            f"Analysis did not complete within {settings.ARGUS_ANALYZE_DEADLINE_SECONDS}s. "
            "It may still be running and consuming governor quota in the background.",
        )
    except RateLimitExceeded as e:
        logger.warning("[API] Governor rate limit exhausted: %s", e)
        raise HTTPException(429, f"Rate limit exhausted: {e}")
    except UnregisteredModel as e:
        logger.error("[API] Model configuration error: %s", e)
        raise HTTPException(503, f"Model configuration error: {e}")
    except Exception as e:
        ref = uuid4()
        logger.error("[API] Graph error (ref %s): %s", ref, e, exc_info=True)
        raise HTTPException(500, f"Agent graph error (ref {ref})")

    # /analyze is the other entry point into the graph besides the unattended
    # collector (RE-11) — without this, decisions.jsonl (the only source
    # reconcile_decisions reads) never sees a request served through this
    # endpoint, and an API-only deployment reconciles nothing
    append_decisions_jsonl(
        final_state.get("decisions") or [], f"{settings.ARGUS_DATA_DIR}/decisions.jsonl"
    )

    # Re-checked rather than trusted from before the (potentially many-second)
    # graph run: the kill switch is process-global and the reconcile loop can
    # trip it mid-run (API-8)
    ks = get_kill_switch()
    if ks and ks.is_halted:
        raise HTTPException(
            503, "System halted during analysis. Kill switch triggered. Manual reset required."
        )

    allocation = final_state.get("portfolio_allocation")
    if not allocation:
        raise HTTPException(500, "Portfolio allocation failed.")

    macro_context = final_state.get("macro_context")
    macro_regime = macro_context.macro_regime.value if macro_context else "unknown"
    vix_level = macro_context.vix_level if macro_context else 0.0

    return AnalysisResponse(
        session_id=allocation.session_id,
        portfolio=[p.model_dump() for p in allocation.portfolio],
        cash_reserve_pct=allocation.cash_reserve_pct,
        macro_regime=macro_regime,
        vix_level=vix_level,
        governor_report=governor.get_usage_report(),
        timestamp=datetime.now().isoformat(),  # noqa: DTZ005
        errors=final_state.get("errors") or [],
    )


@app.get("/memory/stats")
async def get_memory_stats():
    """Retrieves high-level metadata and diagnostic stats from the vector DB memory vault."""
    return get_cultural_memory().summary_stats()


@app.get("/governor/report")
async def get_governor_report():
    """Provides a detailed usage count of today's API provider calls and tokens."""
    return governor.get_usage_report()


@app.post("/kill-switch/reset", dependencies=[Depends(require_api_key)])
async def reset_kill_switch(new_inception_value: float = Query(gt=1000)):
    """Resets the safety monitoring daemon to a new inception capital base.

    Also deletes any persisted halt-dump files (see KillSwitch.reset), so a
    subsequent restart can't silently re-apply a halt this call just resolved.
    Rebases the persisted paper-equity curve to new_inception_value in the same
    operation (KS-15) — otherwise the next reconcile pass would call
    update_portfolio_value with the stale, already-drawn-down equity and
    re-halt within check_interval_seconds.

    Args:
        new_inception_value: New portfolio value to use as the drawdown base
            (USD); must exceed 1000, matching AnalysisRequest.total_wealth's floor.

    Returns:
        Dict with ``status`` key on success.

    Raises:
        HTTPException 401: If ARGUS_API_KEY is set and the X-API-Key header doesn't match.
        HTTPException 404: If the kill switch has not been initialized.
    """
    ks = get_kill_switch()
    if ks:
        book_path = f"{settings.ARGUS_DATA_DIR}/paper_equity.json"
        book = paper_book.load(book_path)
        book.rebase(new_inception_value)
        paper_book.save(book, book_path)
        ks.reset(new_inception_value)
        return {"status": "Reset successful"}
    raise HTTPException(404, "Kill switch not initialized")


@app.get("/kill-switch/status")
async def kill_switch_status():
    """Reports the kill switch's current gate state, so an operator can see why the
    system halted without reading logs.

    Returns:
        Dict mirroring KillSwitchStatus's fields, plus the configured risk_tolerance.

    Raises:
        HTTPException 404: If the kill switch has not been initialized.
    """
    ks = get_kill_switch()
    if ks is None:
        raise HTTPException(404, "Kill switch not initialized")
    status = ks.status
    return {
        "halted": status.halted,
        "new_positions_blocked": status.new_positions_blocked,
        "reason": status.reason,
        "triggered_at": status.triggered_at.isoformat() if status.triggered_at else None,
        "realized_drawdown": status.realized_drawdown,
        "current_vix": status.current_vix,
        "risk_tolerance": ks.risk_tolerance,
    }
