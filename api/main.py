"""HTTP routes for portfolio analysis, diagnostics, and safety controls."""

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
    """Gate mutating endpoints (/analyze, /kill-switch/reset) behind a shared secret."""
    if settings.ARGUS_API_KEY and x_api_key != settings.ARGUS_API_KEY:
        raise HTTPException(401, "Missing or invalid X-API-Key header.")


def _background_task_status(task: asyncio.Task | None) -> dict:
    """Report a background task's liveness, distinguishing "never enabled" from "died"."""
    if task is None:
        return {"enabled": False, "alive": True}
    if not task.done():
        return {"enabled": True, "alive": True}
    if task.cancelled():
        error = "task was canceled"
    else:
        exc = task.exception()
        error = str(exc) if exc else "task exited unexpectedly"
    return {"enabled": True, "alive": False, "error": error}


@app.get("/health")
async def health():
    """Report task liveness, safety state, and cache freshness; return 503 for a dead background task."""
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
    """Report the background MFT pipeline's live state without perturbing it."""
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
            None if runtime.last_collection_result is None else asdict(runtime.last_collection_result)
        ),
        "reconcile_enabled": settings.ARGUS_RECONCILE_ENABLED,
        "reconcile_hour_et": settings.ARGUS_RECONCILE_HOUR_ET,
        "macro_classifier_fitted": runtime.macro_classifier_fitted,
    }


def _raise_if_halted(phase: str = "") -> KillSwitch | None:
    """Raise 503 if the kill switch has halted the system; otherwise returns the switch."""
    ks = get_kill_switch()
    if ks and ks.is_halted:
        raise HTTPException(503, f"System halted{phase}. Kill switch triggered. Manual reset required.")
    return ks


def _admit_live_session(req: AnalysisRequest) -> dict[str, dict]:
    """Apply every pre-graph gate and return the live feature dict per requested ticker."""
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

    # Check market hours before freshness; daily bars cannot substitute for intraday technical features.
    pipeline = runtime.pipeline
    if pipeline is None or not pipeline.is_market_hours():
        raise HTTPException(
            503,
            "US equity market is currently closed. MFT pipeline is idle. "
            "Retry between 09:30 and 16:00 ET on a weekday.",
        )

    # Register tickers only after the earlier gates pass, so rejected requests consume no tracking slots.
    pipeline.register_tickers(req.tickers)

    if runtime.live_cache is None:
        raise HTTPException(503, "Live session cache not yet initialized.")

    # Market hours determine when the cached publication and bar ages matter.
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
    """Invoke the shared graph once, under the analyze lock and the request deadline."""
    graph = runtime.graph
    if graph is None:
        raise HTTPException(503, "Agent graph not yet initialized.")

    config = new_run_config("analyze")

    # Keep the lock check and acquire adjacent; an intervening await would allow another request to enter.
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
        # The worker thread continues after timeout; only the HTTP wait is bounded.
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
    """Turn a finished graph state into the /analyze response."""
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
    """Execute a real-time portfolio analysis pipeline across a target ticker universe."""
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

    # Persist API decisions alongside collector decisions; the file lock and fsync must run off the event loop.
    await asyncio.to_thread(
        append_decisions_jsonl,
        final_state.get("decisions") or [],
        str(settings.data_path("decisions.jsonl")),
    )

    # Recheck safety because the independent monitor can halt the system during graph execution.
    _raise_if_halted(" during analysis")

    return _build_response(final_state)


@app.get("/memory/stats")
async def get_memory_stats():
    """Retrieve high-level metadata and diagnostic stats from the vector DB memory vault."""
    return get_cultural_memory().summary_stats()


@app.get("/governor/report")
async def get_governor_report():
    """Provide a detailed usage count of today's API provider calls and tokens."""
    return governor.get_usage_report()


@app.post("/kill-switch/reset", dependencies=[Depends(require_api_key)])
async def reset_kill_switch(new_inception_value: float = Query(gt=MIN_TOTAL_WEALTH)):
    """Reset the safety monitoring daemon to a new inception capital base."""
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
    """Report the kill switch's current gate state."""
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
