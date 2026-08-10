# ruff: noqa: E402
"""
api/main.py

FastAPI gateway service for the ARGUS multi-agent decision system.

Exposes REST API endpoints to orchestrate execution graphs, invoke walk-forward
backtesting simulations, and audit system safety/governor statistics.

Responsibilities:
  - Route analysis and backtesting requests to the LangGraph execution pipeline
  - Enforce kill-switch and VIX blackout checks before every allocation request
  - Expose health, memory, governor, and kill-switch management endpoints
  - Host the MFT pipeline as a background asyncio task and maintain a live
    session state cache consumed by /analyze

Not responsible for:
  - Agent logic (see argus/agents/)
  - Signal computation (see argus/orchestration/)
  - Execution order routing
"""

import asyncio
import logging
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from dotenv import load_dotenv

# .env must be loaded before any LangChain/Groq imports that read env vars
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from argus.agents.macro import MacroStatisticalAgent
from argus.backtesting.engine import run_backtest
from argus.config import settings
from argus.data.pipeline import MFTDataPipeline
from argus.memory.cultural import cultural_memory
from argus.orchestration.governor import governor
from argus.orchestration.graph import graph
from argus.orchestration.state import ARGUSState
from argus.risk.kill_switch import get_kill_switch

logger = logging.getLogger("argus.api")

app = FastAPI(title="ARGUS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
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


class BacktestRequest(BaseModel):
    """Request payload parameters for executing historical walk-forward backtesting."""

    tickers: list[str]
    start_date: str = "2021-01-04"
    end_date: str = "2024-12-31"
    initial_cash: float = 100_000.0
    risk_tolerance: str = "MODERATE"
    run_bias_audit: bool = True


_backtest_jobs: dict[str, dict] = {}

# Live 5-minute intraday session states keyed by ticker, populated by the MFT
# background loop. Read by /analyze and injected into ARGUSState before graph
# execution. Values are (state_dict, updated_at) tuples; entries older than
# _SESSION_STATE_TTL_SECONDS are treated as missing to prevent stale intraday
# data from a prior trading session from being injected silently.
_SESSION_STATE_TTL_SECONDS = 2100  # 35 min = one _SESSION_INTERVAL + 5 min buffer
_live_session_cache: dict[str, tuple[dict, datetime]] = {}

# Singleton MFT pipeline shared across the server lifetime. Initialized empty;
# tickers are registered dynamically per /analyze request.
_mft_pipeline: MFTDataPipeline | None = None


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


@app.on_event("startup")
async def startup_event():
    """Pre-fits macro models and launches the MFT pipeline background task."""
    global _mft_pipeline

    logger.info("[Startup] Fitting Macro Statistical Agent on historical data...")
    macro_agent = MacroStatisticalAgent()
    try:
        macro_agent.fit_on_history()
        logger.info("[Startup] Macro Agent fitted successfully.")
    except Exception as e:
        logger.warning(
            "[Startup] Macro Agent fit failed (possibly due to missing API keys): %s", e
        )

    # Start the MFT pipeline as a non-blocking background task.
    # The pipeline begins with an empty ticker list; tickers are registered
    # on the first /analyze call so the fetch loop tracks only requested symbols.
    _mft_pipeline = MFTDataPipeline(tickers=[])
    asyncio.create_task(_mft_pipeline.start(on_session_ready=_mft_session_callback))
    logger.info("[Startup] MFT pipeline background task launched.")

    logger.info("[Startup] Governor initialized: %s", governor.get_usage_report())


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

    # Register requested tickers with the running MFT pipeline so they are
    # fetched on the next intraday cycle (within _FETCH_INTERVAL seconds).
    if _mft_pipeline is not None:
        _mft_pipeline.register_tickers(req.tickers)

    # Reject if any ticker is missing from the cache or its entry has expired.
    # Falling back to daily-compressed indicators would produce a mismatched signal
    # resolution relative to what the TechnicalStatisticalAgent is calibrated for.
    now = datetime.now()
    missing_from_cache = [
        t for t in req.tickers
        if t not in _live_session_cache
        or (now - _live_session_cache[t][1]).total_seconds() > _SESSION_STATE_TTL_SECONDS
    ]
    if missing_from_cache:
        if _mft_pipeline is None or not _mft_pipeline._is_market_hours():
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
        final_state = await asyncio.to_thread(graph.invoke, state, config)
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
    )


@app.post("/backtest")
async def run_backtest_endpoint(req: BacktestRequest):
    """Executes a synchronous walk-forward historical backtesting simulation.

    Args:
        req: BacktestRequest with tickers, date range, initial_cash, and risk_tolerance.

    Returns:
        Dict with job_id, status, and results from the backtest engine.

    Raises:
        HTTPException 500: If the backtest engine raises an exception.
    """
    job_id = str(uuid4())
    try:
        results = await asyncio.to_thread(
            run_backtest,
            universe=req.tickers,
            start=req.start_date,
            end=req.end_date,
            initial_cash=req.initial_cash,
            invest_pct=0.80,
            risk_tolerance=req.risk_tolerance,
        )
        return {"job_id": job_id, "status": "COMPLETED", "results": results}
    except Exception as e:
        logger.error("[API] Backtest failed: %s", e)
        raise HTTPException(500, f"Backtest failed: {str(e)}")


@app.get("/backtest/{job_id}")
async def get_backtest_result(job_id: str):
    """Deprecated endpoint; historical simulation results are returned synchronously.

    Raises:
        HTTPException 404: Always; async polling is no longer supported.
    """
    raise HTTPException(
        404, "Job ID polling is no longer supported; POST /backtest is now synchronous."
    )


@app.get("/memory/stats")
async def get_memory_stats():
    """Retrieves high-level metadata and diagnostic stats from the vector DB memory vault."""
    return cultural_memory.summary_stats()


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
