# ruff: noqa: E402
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
  - Enforce kill-switch and VIX blackout checks before every allocation request
  - Expose health, memory, governor, and kill-switch management endpoints
  - Host the MFT pipeline as a background asyncio task and maintain a live
    session state cache consumed by /analyze
  - Optionally run the unattended collector and daily reconciliation loops
    (ARGUS_COLLECTOR_ENABLED / ARGUS_RECONCILE_ENABLED) so the system keeps
    accumulating decisions and outcomes without a human calling /analyze

Not responsible for:
  - Agent logic (see argus/agents/)
  - Signal computation (see argus/orchestration/)
  - Execution order routing
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Literal, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# .env must be loaded before any LangChain/Groq imports that read env vars
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from argus.agents.macro import MacroStatisticalAgent
from argus.config import settings
from argus.data.pipeline import MFTDataPipeline
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.collector import CollectionResult, run_collection_cycle
from argus.orchestration.governor import governor
from argus.orchestration.graph import build_graph
from argus.orchestration.reconciliation import load_decisions_from_jsonl, reconcile_decisions
from argus.orchestration.state import ARGUSState
from argus.params import RECONCILIATION
from argus.risk.kill_switch import get_kill_switch, initialize_kill_switch
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger("argus.api")

_ET = ZoneInfo("America/New_York")

# Entries older than _SESSION_STATE_TTL_SECONDS are treated as missing so a prior
# session's stale intraday data is never injected silently
_SESSION_STATE_TTL_SECONDS = 2100  # 35 min = default MFT_DECISION_INTERVAL_SECONDS + 5 min buffer
_live_session_cache: dict[str, tuple[dict, datetime]] = {}

# Initialized during lifespan startup; tickers are registered dynamically per
# /analyze request in addition to the ARGUS_UNIVERSE seed
_mft_pipeline: MFTDataPipeline | None = None
_pipeline_task: asyncio.Task | None = None
_collector_task: asyncio.Task | None = None
_reconcile_task: asyncio.Task | None = None
_last_collection_result: CollectionResult | None = None

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
    """Receives compressed technical feature dicts from the MFT pipeline every 30 minutes.

    Updates the module-level live cache so the next /analyze call picks up
    fresh intraday indicators without re-fetching historical data. Each entry
    stores a (state_dict, updated_at) tuple for TTL-based staleness detection.

    Args:
        session_states: Mapping of ticker → technical feature dict from MFTDataPipeline.
    """
    now = datetime.now()
    for ticker, state in session_states.items():
        _live_session_cache[ticker] = (state, now)
    logger.info("[MFT] Live session cache updated: %d ticker(s)", len(_live_session_cache))


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
            )
            logger.info("[Collector] cycle result: %s", _last_collection_result)
        except Exception:
            logger.exception("[Collector] cycle failed")
        await asyncio.sleep(settings.ARGUS_COLLECTOR_INTERVAL_SECONDS)


async def _reconcile_loop() -> None:
    """Runs reconcile_decisions once a day at settings.ARGUS_RECONCILE_HOUR_ET."""
    while True:
        now_et = datetime.now(_ET)
        target = now_et.replace(
            hour=settings.ARGUS_RECONCILE_HOUR_ET, minute=0, second=0, microsecond=0
        )
        if target <= now_et:
            target += timedelta(days=1)
        await asyncio.sleep((target - now_et).total_seconds())

        try:
            decisions_log = f"{settings.ARGUS_DATA_DIR}/decisions.jsonl"
            decisions = load_decisions_from_jsonl(decisions_log)
            stored = reconcile_decisions(
                decisions,
                market_data=LiveMarketDataProvider(),
                cultural=get_cultural_memory(),
                horizon_days=RECONCILIATION.horizon_days,
            )
            logger.info("[Reconcile] stored %d/%d outcome(s)", stored, len(decisions))
        except Exception:
            logger.exception("[Reconcile] cycle failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Starts background tasks on startup and stops them cleanly on shutdown."""
    global _mft_pipeline, _pipeline_task, _collector_task, _reconcile_task, _graph, _macro_status_agent

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

    # Placeholder base; /kill-switch/reset re-bases once the real session is known
    initialize_kill_switch(
        risk_tolerance=settings.ARGUS_RISK_TOLERANCE, portfolio_value=settings.ARGUS_TOTAL_WEALTH
    )
    logger.info("[Startup] Kill switch initialized.")

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

    if _mft_pipeline is not None:
        _mft_pipeline.buffer.close()
    logger.info("[Shutdown] Complete.")


app = FastAPI(title="ARGUS API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.ARGUS_CORS_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    """Request payload defining parameters for stock universe valuation and rebalancing."""

    tickers: list[str] = Field(min_length=1, max_length=20)
    total_wealth: float = Field(gt=1000)
    invest_pct: float = Field(gt=0.05, le=0.95)
    risk_tolerance: Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"] = "MODERATE"


class AnalysisResponse(BaseModel):
    """Synthesized portfolio allocation recommendation and system diagnostics."""

    session_id: str
    portfolio: list[dict]
    cash_reserve_pct: float
    expected_sharpe: Optional[float]
    macro_regime: str
    vix_level: float
    governor_report: dict
    timestamp: str
    errors: list[str] = []


@app.get("/health")
async def health():
    """Validates connectivity to model backends, API quotas, and system diagnostics."""
    try:
        llama_cap = governor.get_remaining_capacity("llama-3.3-70b-versatile")
        can_make_calls = llama_cap > 0
    except Exception:
        can_make_calls = False

    return {
        "status": "ok",
        "model_versions": {
            "synthesis": "llama-3.3-70b-versatile",
            "sentiment": "llama-3.1-8b-instant",
            "fundamental": "llama-3.3-70b-versatile",
            "finbert": "ProsusAI/finbert",
        },
        "can_make_calls": can_make_calls,
        "governor_report": governor.get_usage_report(),
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
    if _mft_pipeline is None:
        raise HTTPException(503, "Pipeline not yet initialized.")

    buffer_depth: dict[str, int] = {}
    for ticker in _mft_pipeline.buffer.get_all_tickers():
        df = _mft_pipeline.buffer.get_candles(ticker)
        buffer_depth[ticker] = 0 if df is None else len(df)

    session_cache_age_seconds = {
        ticker: (datetime.now() - updated_at).total_seconds()
        for ticker, (_, updated_at) in _live_session_cache.items()
    }

    return {
        "tracked_tickers": list(_mft_pipeline.tickers),
        "buffer_depth": buffer_depth,
        "session_cache_age_seconds": session_cache_age_seconds,
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


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    """Executes a real-time portfolio analysis pipeline across a target ticker universe.

    Checks system safety (kill switches), validates that the MFT live cache is fully
    populated for all requested tickers, then runs LangGraph arbitrated orchestration
    and returns target allocation distributions.

    Args:
        req: AnalysisRequest with tickers, total_wealth, invest_pct, and risk_tolerance.

    Raises:
        HTTPException 503: If the kill switch is triggered, VIX is above the blackout
            threshold, the market is currently closed, or the MFT cache has not yet
            been populated for all requested tickers.
        HTTPException 500: If the LangGraph execution fails or produces no allocation.
    """
    ks = get_kill_switch()
    if ks and ks.is_halted:
        raise HTTPException(503, "System halted. Kill switch triggered. Manual reset required.")
    if ks and not ks.new_positions_allowed:
        raise HTTPException(503, "New positions blocked. VIX above threshold.")

    if _mft_pipeline is not None:
        _mft_pipeline.register_tickers(req.tickers)

    # Daily bars would mismatch the resolution the technical agent expects
    now = datetime.now()
    missing_from_cache = [
        t for t in req.tickers
        if t not in _live_session_cache
        or (now - _live_session_cache[t][1]).total_seconds() > _SESSION_STATE_TTL_SECONDS
    ]
    if missing_from_cache:
        if _mft_pipeline is None or not _mft_pipeline.is_market_hours():
            raise HTTPException(
                503,
                "US equity market is currently closed. MFT pipeline is idle. "
                "Retry between 09:30 and 16:00 ET on a weekday.",
            )
        raise HTTPException(
            503,
            f"MFT live cache not yet populated for: {missing_from_cache}. "
            "The pipeline is warming up — retry after the next session cycle (~30 min after market open).",
        )

    live_states = {t: _live_session_cache[t][0] for t in req.tickers}
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

    config = {"configurable": {"thread_id": str(uuid4())}}

    try:
        final_state = await asyncio.to_thread(_graph.invoke, state, config)
    except Exception as e:
        logger.error("[API] Graph error: %s", e)
        raise HTTPException(500, f"Agent graph error: {str(e)}")

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
        expected_sharpe=allocation.expected_sharpe,
        macro_regime=macro_regime,
        vix_level=vix_level,
        governor_report=governor.get_usage_report(),
        timestamp=datetime.now().isoformat(),
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


@app.post("/kill-switch/reset")
async def reset_kill_switch(new_inception_value: float):
    """Resets the safety monitoring daemon to a new inception capital base.

    Args:
        new_inception_value: New portfolio value to use as the drawdown base (USD).

    Returns:
        Dict with ``status`` key on success.

    Raises:
        HTTPException 404: If the kill switch has not been initialized.
    """
    ks = get_kill_switch()
    if ks:
        ks.reset(new_inception_value)
        return {"status": "Reset successful"}
    raise HTTPException(404, "Kill switch not initialized")
