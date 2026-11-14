"""
api/main.py

FastAPI gateway service for the ARGUS multi-agent decision system.

Exposes REST API endpoints to orchestrate execution graphs and audit system
safety/governor statistics. There is no live /backtest endpoint; a
multi-year walk-forward backtest isn't offered, so scripts/replay_backtest.py
replays recorded fixtures through this same graph over the ~60-day window
that is genuinely available.

Responsibilities:
  - Build the app and its routes
  - Route analysis requests to the LangGraph execution pipeline, enforcing
    the kill-switch, market-hours and live-cache admission gates first
  - Append every /analyze decision to the same decisions.jsonl log the
    unattended collector writes to (see argus/orchestration/collector.py),
    so reconciliation sees both entry points into the graph
  - Expose health, memory, governor, and kill-switch management endpoints

Not responsible for:
  - Request/response models (see api/models.py)
  - Startup checks and the single-process lock (see api/guards.py)
  - Startup/shutdown, the background loops' lifetimes and the shared runtime
    state (see api/lifespan.py)
  - The collector and reconcile loops themselves (see
    argus/orchestration/scheduler.py)
  - Agent logic (see argus/agents/)
"""

import asyncio
import logging
from dataclasses import asdict
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import argus
from api.lifespan import lifespan, runtime
from api.models import AnalysisRequest, AnalysisResponse
from argus.agents.sentiment import FINBERT_MODEL
from argus.config import settings
from argus.governor import RateLimitExceeded, UnregisteredModel, governor
from argus.infra.clock import now_local, now_market
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.checkpoint import new_run_config
from argus.orchestration.decision_log import append_decisions_jsonl
from argus.orchestration.state import ARGUSState, initial_state
from argus.params import SYSTEM
from argus.risk import paper_book
from argus.risk.kill_switch import KillSwitch, get_kill_switch
from argus.schemas.requests import MIN_TOTAL_WEALTH

logger = logging.getLogger(__name__)

app = FastAPI(title="ARGUS API", version=argus.__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.ARGUS_CORS_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
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
    if settings.ARGUS_API_KEY and x_api_key != settings.ARGUS_API_KEY:
        raise HTTPException(401, "Missing or invalid X-API-Key header.")


def _background_task_status(task: asyncio.Task | None) -> dict:
    """Reports a background task's liveness, distinguishing "never enabled" from "died".

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

    Reads background-task liveness, the kill switch's halt state, and cache
    freshness — not just in-memory governor counters — so a dead
    pipeline/collector/reconcile loop shows up here instead of staying
    invisible behind a green /health.

    Raises:
        HTTPException 503: If any background task (MFT pipeline, collector,
            or reconciliation loop) has stopped running.
    """
    try:
        can_make_calls = governor.get_remaining_capacity(settings.ARGUS_PORTFOLIO_MODEL) > 0
    except Exception:
        logger.exception("/health governor capacity check failed")
        can_make_calls = False

    background_tasks = {
        "mft_pipeline": _background_task_status(runtime.pipeline_task),
        "collector": _background_task_status(runtime.collector_task),
        "reconcile": _background_task_status(runtime.reconcile_task),
    }
    dead_tasks = [name for name, status in background_tasks.items() if not status["alive"]]
    if dead_tasks:
        raise HTTPException(503, f"Background task(s) not running: {', '.join(dead_tasks)}")

    ks = get_kill_switch()

    newest_cache_entry_age_seconds = None
    pipeline = runtime.pipeline
    if pipeline is not None and pipeline.is_market_hours() and runtime.live_cache is not None:
        cache_age_seconds, _ = runtime.live_cache.ages(now_market())
        if cache_age_seconds:
            newest_cache_entry_age_seconds = min(cache_age_seconds.values())

    return {
        "status": "ok",
        "model_versions": {
            "synthesis": settings.ARGUS_PORTFOLIO_MODEL,
            "sentiment": settings.ARGUS_SENTIMENT_MODEL,
            "fundamental": settings.ARGUS_FUNDAMENTAL_MODEL,
            "finbert": FINBERT_MODEL,
        },
        "can_make_calls": can_make_calls,
        "governor_report": governor.get_usage_report(),
        "background_tasks": background_tasks,
        "kill_switch_halted": ks.is_halted if ks else False,
        "newest_cache_entry_age_seconds": newest_cache_entry_age_seconds,
    }


@app.get("/pipeline/status")
async def pipeline_status():
    """Reports the background MFT pipeline's live state without perturbing it.

    Intended for watching the unattended collector — buffer depth, session
    cache freshness, and the last automatic collection cycle's outcome — all
    without triggering a real analysis request.

    Raises:
        HTTPException 503: If the pipeline hasn't started yet.
    """
    pipeline = runtime.pipeline
    if pipeline is None:
        raise HTTPException(503, "Pipeline not yet initialized.")

    buffer_depth = await asyncio.to_thread(pipeline.buffer.row_counts)

    cache_age_seconds, bar_age_seconds = (
        runtime.live_cache.ages(now_market()) if runtime.live_cache is not None else ({}, {})
    )

    return {
        "tracked_tickers": list(pipeline.tickers),
        "buffer_depth": buffer_depth,
        "cache_age_seconds": cache_age_seconds,
        "bar_age_seconds": bar_age_seconds,
        "is_market_hours": pipeline.is_market_hours(),
        "collector_enabled": settings.ARGUS_COLLECTOR_ENABLED,
        "last_collection_result": (
            None
            if runtime.last_collection_result is None
            else asdict(runtime.last_collection_result)
        ),
        "reconcile_enabled": settings.ARGUS_RECONCILE_ENABLED,
        "reconcile_hour_et": settings.ARGUS_RECONCILE_HOUR_ET,
        "macro_classifier_fitted": runtime.macro_classifier_fitted,
    }


def _raise_if_halted(phase: str = "") -> KillSwitch | None:
    """Raises 503 if the kill switch has halted the system; otherwise returns the switch.

    Args:
        phase: Appended to the message to say when the halt was seen, e.g.
            " during analysis" for the re-check after a graph run.

    Returns:
        The kill switch, or None if none has been initialized.

    Raises:
        HTTPException 503: If the kill switch is halted.
    """
    ks = get_kill_switch()
    if ks and ks.is_halted:
        raise HTTPException(
            503, f"System halted{phase}. Kill switch triggered. Manual reset required."
        )
    return ks


def _admit_live_session(req: AnalysisRequest) -> dict[str, dict]:
    """Applies every pre-graph gate and returns the live feature dict per requested ticker.

    The kill switch's drawdown gate is process-global, fixed at startup from
    settings.ARGUS_RISK_TOLERANCE (see argus/risk/kill_switch.py's class
    docstring) — a request's own risk_tolerance does not change which
    threshold guards it. A mismatch is logged, not rejected.

    Args:
        req: The validated request.

    Returns:
        Ticker → live technical feature dict, one entry per requested ticker.

    Raises:
        HTTPException 503: If the system is halted or blocked on VIX, the market
            is closed, or the live cache is missing, stalled or stale for any
            requested ticker.
    """
    ks = _raise_if_halted()
    if ks:
        if not ks.new_positions_allowed:
            raise HTTPException(503, "New positions blocked. VIX above threshold.")
        if req.risk_tolerance != ks.risk_tolerance:
            logger.warning(
                "Request risk_tolerance=%s differs from the kill switch's configured "
                "risk_tolerance=%s; the configured threshold still governs this request.",
                req.risk_tolerance,
                ks.risk_tolerance,
            )

    # Daily bars would mismatch the resolution the technical agent expects, and
    # checking this first confines every age check below to a weekday
    # 09:30-16:00 ET window, eliminating the weekend/holiday false positive
    pipeline = runtime.pipeline
    if pipeline is None or not pipeline.is_market_hours():
        raise HTTPException(
            503,
            "US equity market is currently closed. MFT pipeline is idle. "
            "Retry between 09:30 and 16:00 ET on a weekday.",
        )

    # Registered only once every earlier gate has cleared — a rejected
    # request must not mutate pipeline state or spend a slot in the
    # tracked-universe cap
    pipeline.register_tickers(req.tickers)

    if runtime.live_cache is None:
        raise HTTPException(503, "Live session cache not yet initialized.")

    # Admission answers only "how old is this" for each ticker; the ordering
    # against market hours above is what decides whether that age matters
    # right now
    admission = runtime.live_cache.admit(req.tickers, now_market())

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

    logger.info("MFT cache hit for all %d tickers", len(admission.admitted))
    return admission.admitted


async def _run_graph(state: ARGUSState) -> dict:
    """Invokes the shared graph once, under the analyze lock and the request deadline.

    Args:
        state: Initial graph state for this request.

    Returns:
        The graph's final state.

    Raises:
        HTTPException 429: If another analysis holds the lock, or the governor's
            rate budget is exhausted.
        HTTPException 503: If the graph isn't built yet, or a configured model has
            no rate-limit profile (a misconfiguration, not a transient condition).
        HTTPException 504: If the run exceeds ARGUS_ANALYZE_DEADLINE_SECONDS.
        HTTPException 500: For any other graph failure, with a correlation ref
            instead of the internal exception text.
    """
    graph = runtime.graph
    if graph is None:
        raise HTTPException(503, "Agent graph not yet initialized.")

    config = new_run_config("analyze")

    # No await between this check and the acquire below, so nothing can slip
    # in between them — Semaphore.acquire returns synchronously when
    # unlocked, and a later refactor that inserts an await here breaks that.
    if runtime.analyze_lock.locked():
        raise HTTPException(
            429,
            "Another analysis is already in progress. Retry shortly.",
            headers={"Retry-After": str(SYSTEM.analyze_retry_after_seconds)},
        )

    try:
        async with runtime.analyze_lock:
            return await asyncio.wait_for(
                asyncio.to_thread(graph.invoke, state, config),
                timeout=settings.ARGUS_ANALYZE_DEADLINE_SECONDS,
            )
    except TimeoutError:
        # The underlying thread can't be cancelled and keeps running — async
        # job submission is a possible future improvement — but the server no
        # longer holds the connection open past a proxy's own timeout with no signal.
        logger.error("Graph run exceeded the %ss deadline", settings.ARGUS_ANALYZE_DEADLINE_SECONDS)
        raise HTTPException(
            504,
            f"Analysis did not complete within {settings.ARGUS_ANALYZE_DEADLINE_SECONDS}s. "
            "It may still be running and consuming governor quota in the background.",
        ) from None
    except RateLimitExceeded as e:
        logger.warning("Governor rate limit exhausted: %s", e)
        raise HTTPException(429, f"Rate limit exhausted: {e}") from e
    except UnregisteredModel as e:
        logger.error("Model configuration error: %s", e)
        raise HTTPException(503, f"Model configuration error: {e}") from e
    except Exception as e:
        ref = uuid4()
        logger.error("Graph error (ref %s): %s", ref, e, exc_info=True)
        raise HTTPException(500, f"Agent graph error (ref {ref})") from e


def _build_response(final_state: dict) -> AnalysisResponse:
    """Turns a finished graph state into the /analyze response.

    Args:
        final_state: The graph's final state.

    Returns:
        The allocation and diagnostics.

    Raises:
        HTTPException 503: If the portfolio agent produced no allocation. Refusing
            to answer is correct here — see PortfolioManagerAgent.allocate's
            degrade-not-fabricate contract — so this is an upstream LLM outage,
            not a bug in this server; 503 says so rather than the generic 500.
    """
    allocation = final_state.get("portfolio_allocation")
    if not allocation:
        raise HTTPException(503, "Portfolio allocation failed.")

    macro_context = final_state.get("macro_context")
    return AnalysisResponse(
        session_id=allocation.session_id,
        portfolio=allocation.portfolio,
        cash_reserve_pct=allocation.cash_reserve_pct,
        macro_regime=macro_context.macro_regime.value if macro_context else "unknown",
        vix_level=macro_context.vix_level if macro_context else None,
        governor_report=governor.get_usage_report(),
        timestamp=now_local().isoformat(),
        errors=final_state.get("errors") or [],
    )


@app.post("/analyze", response_model=AnalysisResponse, dependencies=[Depends(require_api_key)])
async def analyze(req: AnalysisRequest):
    """Executes a real-time portfolio analysis pipeline across a target ticker universe.

    Checks system safety (kill switches), validates that the MFT live cache is fully
    populated for all requested tickers, then runs LangGraph arbitrated orchestration
    and returns target allocation distributions.

    Args:
        req: AnalysisRequest with tickers, total_wealth, invest_pct, and risk_tolerance.

    Raises:
        HTTPException 401: If ARGUS_API_KEY is set and the X-API-Key header doesn't match.
        HTTPException 429: If the governor's rate budget is exhausted for a
            configured model, or another analysis (a concurrent /analyze
            call or the unattended collector) is already in progress —
            retry after a short wait.
        HTTPException 503: If the graph hasn't finished initializing, the kill switch
            is triggered (before or after the graph run), VIX is above the blackout
            threshold, the market is currently closed, the MFT cache has not yet
            been populated for all requested tickers, the cache hasn't been
            refreshed recently (pipeline stalled), the cached bars are older
            than expected (stale data survived a restart), a configured model
            has no rate-limit profile (a misconfiguration, not a transient
            condition), or the portfolio agent's LLM call failed and correctly
            degraded to no allocation rather than fabricating one.
        HTTPException 504: If the graph run exceeds the analysis deadline.
        HTTPException 500: If the LangGraph execution fails for a reason other
            than the LLM call above.
    """
    live_states = _admit_live_session(req)
    final_state = await _run_graph(
        initial_state(
            req.tickers,
            session_states=live_states,
            total_wealth=req.total_wealth,
            invest_pct=req.invest_pct,
            risk_tolerance=req.risk_tolerance,
        )
    )

    # /analyze is the other entry point into the graph besides the unattended
    # collector — without this, decisions.jsonl (the only source
    # reconcile_decisions reads) never sees a request served through this
    # endpoint, and an API-only deployment reconciles nothing. Off the event
    # loop because it takes a file lock and fsyncs.
    await asyncio.to_thread(
        append_decisions_jsonl,
        final_state.get("decisions") or [],
        str(settings.data_path("decisions.jsonl")),
    )

    # Re-checked rather than trusted from before the (potentially many-second)
    # graph run: the kill switch is process-global and the reconcile loop can
    # trip it mid-run
    _raise_if_halted(" during analysis")

    return _build_response(final_state)


@app.get("/memory/stats")
async def get_memory_stats():
    """Retrieves high-level metadata and diagnostic stats from the vector DB memory vault."""
    return get_cultural_memory().summary_stats()


@app.get("/governor/report")
async def get_governor_report():
    """Provides a detailed usage count of today's API provider calls and tokens."""
    return governor.get_usage_report()


@app.post("/kill-switch/reset", dependencies=[Depends(require_api_key)])
async def reset_kill_switch(new_inception_value: float = Query(gt=MIN_TOTAL_WEALTH)):
    """Resets the safety monitoring daemon to a new inception capital base.

    Also deletes any persisted halt-dump files (see KillSwitch.reset), so a
    subsequent restart can't silently re-apply a halt this call just resolved.
    Rebases the persisted paper-equity curve to new_inception_value in the same
    operation — otherwise the next reconcile pass would call
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
        HTTPException 500: If the persisted paper-equity file exists but is
            corrupt. Rebasing a book paper_book.load() couldn't actually read
            would silently drop its runs_applied, letting a run already
            reflected in the old (unreadable) equity get double-applied by
            the next reconcile pass — so this is refused rather than papered
            over. Move the corrupt file aside and retry.
    """
    ks = get_kill_switch()
    if ks is None:
        raise HTTPException(404, "Kill switch not initialized")

    book_path = str(settings.data_path("paper_equity.json"))
    try:
        book = paper_book.load(book_path)
    except Exception as exc:
        raise HTTPException(
            500,
            f"{book_path} exists but could not be read ({exc}); move it aside and retry "
            "— rebasing over unreadable state risks double-applying an already-applied run.",
        ) from exc
    book.rebase(new_inception_value)
    paper_book.save(book, book_path)
    ks.reset(new_inception_value)
    return {"status": "Reset successful"}


@app.get("/kill-switch/status")
async def kill_switch_status():
    """Reports the kill switch's current gate state.

    Lets an operator see why the system halted without reading logs.

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
        "halt_persisted": status.halt_persisted,
        "risk_tolerance": ks.risk_tolerance,
    }
