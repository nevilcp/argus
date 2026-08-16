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
import sys
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Literal, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# .env must be loaded before any LangChain/Groq imports that read env vars
load_dotenv()

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from argus.agents.macro import MacroStatisticalAgent
from argus.config import settings
from argus.data.pipeline import MFTDataPipeline, max_bar_age_seconds, session_state_ttl_seconds
from argus.data.tickers import TICKER_PATTERN
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.collector import CollectionResult, run_collection_cycle
from argus.orchestration.governor import REGISTERED_MODELS, RateLimitExceeded, UnregisteredModel, governor
from argus.orchestration.graph import build_graph
from argus.orchestration.reconciliation import load_decisions_from_jsonl, reconcile_decisions
from argus.orchestration.state import ARGUSState
from argus.params import RECONCILIATION
from argus.risk import paper_book
from argus.risk.kill_switch import get_kill_switch, initialize_kill_switch
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger("argus.api")

_ET = ZoneInfo("America/New_York")

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


def _reconcile_once() -> None:
    """Runs one reconciliation pass: outcome backfill, paper-book update, kill-switch sync.

    Synchronous by design — `_reconcile_loop` runs this via `asyncio.to_thread`
    since it performs per-ticker yfinance fetches directly, which would
    otherwise block the event loop for the whole app for the duration of a run.
    """
    decisions_log = f"{settings.ARGUS_DATA_DIR}/decisions.jsonl"
    decisions = load_decisions_from_jsonl(decisions_log)
    market_data = LiveMarketDataProvider()
    stored = reconcile_decisions(
        decisions,
        market_data=market_data,
        cultural=get_cultural_memory(),
        horizon_days=RECONCILIATION.horizon_days,
    )
    logger.info("[Reconcile] stored %d/%d outcome(s)", stored, len(decisions))

    try:
        book_path = f"{settings.ARGUS_DATA_DIR}/paper_equity.json"
        book = paper_book.load(book_path)
        for run_timestamp, run_return in paper_book.compute_run_returns(
            decisions, market_data, RECONCILIATION.horizon_days
        ):
            book.apply_run(run_timestamp, run_return)
        paper_book.save(book, book_path)

        ks = get_kill_switch()
        if ks is not None:
            ks.update_portfolio_value(book.equity)
        logger.info(
            "[Reconcile] paper equity=$%.2f (drawdown=%.1f%%)",
            book.equity,
            book.drawdown_from_peak() * 100,
        )
    except Exception:
        logger.exception("[Reconcile] paper-book update failed")


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
            await asyncio.to_thread(_reconcile_once)
        except Exception:
            logger.exception("[Reconcile] cycle failed")


def _assert_single_worker() -> None:
    """Fails fast if launched with more than one uvicorn worker.

    The rate governor and kill switch are in-process singletons (see
    argus/orchestration/governor.py). A second worker process would run its
    own ignorant copy of both, silently doubling real Groq usage against the
    same account and gating risk independently rather than jointly — the
    Actions collector (scripts/reconcile_outcomes.py) is already a second,
    independent governor against that same key, and this assertion exists so
    a deployment doesn't add a third by accident. See Dockerfile.api's CMD.

    Raises:
        RuntimeError: If ``--workers`` appears in argv with a value other than "1".
    """
    argv = sys.argv
    if "--workers" in argv:
        idx = argv.index("--workers")
        if idx + 1 < len(argv) and argv[idx + 1] != "1":
            raise RuntimeError(
                f"ARGUS must run with --workers 1 (in-process governor/kill-switch "
                f"singletons); got --workers {argv[idx + 1]}"
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
    global _mft_pipeline, _pipeline_task, _collector_task, _reconcile_task, _graph, _macro_status_agent

    _assert_single_worker()
    _assert_registered_models()

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
        _mft_pipeline.buffer.close()
    logger.info("[Shutdown] Complete.")


app = FastAPI(title="ARGUS API", version="1.0.0", lifespan=lifespan)
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
        llama_cap = governor.get_remaining_capacity(settings.ARGUS_PORTFOLIO_MODEL)
        can_make_calls = llama_cap > 0
    except Exception:
        can_make_calls = False

    return {
        "status": "ok",
        "model_versions": {
            "synthesis": settings.ARGUS_PORTFOLIO_MODEL,
            "sentiment": settings.ARGUS_SENTIMENT_MODEL,
            "fundamental": settings.ARGUS_FUNDAMENTAL_MODEL,
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

    buffer_depth = await asyncio.to_thread(_mft_pipeline.buffer.row_counts)

    now = datetime.now()
    now_et = datetime.now(_ET)
    cache_age_seconds = {
        ticker: (now - updated_at).total_seconds()
        for ticker, (_, updated_at) in _live_session_cache.items()
    }
    bar_age_seconds = {
        ticker: (now_et - datetime.fromisoformat(state["timestamp"])).total_seconds()
        for ticker, (state, _) in _live_session_cache.items()
    }

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
            configured model (GOV-7) — retry after a short wait.
        HTTPException 503: If the kill switch is triggered, VIX is above the blackout
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

    if _mft_pipeline is not None:
        _mft_pipeline.register_tickers(req.tickers)

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

    absent = [t for t in req.tickers if t not in _live_session_cache]
    if absent:
        raise HTTPException(
            503,
            f"MFT live cache not yet populated for: {absent}. "
            "The pipeline is warming up — retry after the next session cycle (~30 min after market open).",
        )

    now = datetime.now()
    stalled = [
        t for t in req.tickers
        if (now - _live_session_cache[t][1]).total_seconds() > session_state_ttl_seconds()
    ]
    if stalled:
        raise HTTPException(
            503,
            f"MFT pipeline appears stalled for: {stalled}. "
            "The live cache hasn't been refreshed recently — check /pipeline/status.",
        )

    # Catches what a write-time TTL can't: a restart that republishes old candles
    # under a fresh write timestamp (MFT-1)
    now_et = datetime.now(_ET)
    max_bar_age = max_bar_age_seconds(_mft_pipeline.interval_minutes)
    stale = [
        t for t in req.tickers
        if (now_et - datetime.fromisoformat(_live_session_cache[t][0]["timestamp"])).total_seconds()
        > max_bar_age
    ]
    if stale:
        raise HTTPException(
            503,
            f"MFT cache data is stale for: {stale}. "
            "The underlying candles are older than expected — check /pipeline/status.",
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


@app.post("/kill-switch/reset", dependencies=[Depends(require_api_key)])
async def reset_kill_switch(new_inception_value: float = Query(gt=1000)):
    """Resets the safety monitoring daemon to a new inception capital base.

    Also deletes any persisted halt-dump files (see KillSwitch.reset), so a
    subsequent restart can't silently re-apply a halt this call just resolved.

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
