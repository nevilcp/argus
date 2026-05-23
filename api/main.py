"""
api/main.py
===========
FastAPI entrypoint exposing the ARGUS v2 multi-agent graph as HTTP endpoints.
"""

import asyncio
import logging
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from argus.agents.macro import MacroStatisticalAgent
from argus.backtesting.engine import run_backtest
from argus.config import settings
from argus.memory.cultural import cultural_memory
from argus.orchestration.governor import governor
from argus.orchestration.graph import graph
from argus.orchestration.state import ARGUSState
from argus.risk.kill_switch import get_kill_switch

try:
    from langfuse.callback import CallbackHandler
except ImportError:
    CallbackHandler = None


logger = logging.getLogger("argus.api")

app = FastAPI(title="ARGUS v2 API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=20)
    total_wealth: float = Field(gt=1000)
    invest_pct: float = Field(gt=0.05, le=0.95)
    risk_tolerance: Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"] = "MODERATE"

class AnalysisResponse(BaseModel):
    session_id: str
    portfolio: list[dict]
    cash_reserve_pct: float
    expected_sharpe: Optional[float]
    macro_regime: str
    vix_level: float
    governor_report: dict
    timestamp: str

class BacktestRequest(BaseModel):
    tickers: list[str]
    start_date: str = "2021-01-04"
    end_date: str = "2024-12-31"
    initial_cash: float = 100_000.0
    risk_tolerance: str = "MODERATE"
    run_bias_audit: bool = True

_backtest_jobs: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Startup Event
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("[Startup] Fitting Macro Statistical Agent on historical data...")
    macro_agent = MacroStatisticalAgent()
    try:
        macro_agent.fit_on_history()
        logger.info("[Startup] Macro Agent fitted successfully.")
    except Exception as e:
        logger.warning(f"[Startup] Macro Agent fit failed (possibly due to missing API keys): {e}")
        
    logger.info(f"[Startup] Governor initialized: {governor.get_usage_report()}")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
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
            "fundamental": "gemini-3.1-flash-lite",
            "finbert": "ProsusAI/finbert"
        },
        "can_make_calls": can_make_calls,
        "governor_report": governor.get_usage_report()
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    ks = get_kill_switch()
    if ks and ks.is_halted:
        raise HTTPException(503, "System halted. Kill switch triggered. Manual reset required.")
    if ks and not ks.new_positions_allowed:
        raise HTTPException(503, "New positions blocked. VIX above threshold.")

    state = ARGUSState(
        ticker=req.tickers[0],
        total_wealth=req.total_wealth,
        invest_pct=req.invest_pct,
        risk_tolerance=req.risk_tolerance,
        universe=req.tickers,
        backtest_mode=False,
        session_seed=None,
        session_states={},
        price_history={},
        technical_signals={},
        macro_context=None,
        fundamental_signals={},
        sentiment_signals={},
        risk_assessments={},
        aggregated_signals={},
        cultural_wisdom=[],
        cultural_warnings=[],
        portfolio_allocation=None,
        decisions=[],
        errors=[],
    )

    config = {"configurable": {"thread_id": str(uuid4())}}
    
    if CallbackHandler and settings.langfuse_secret_key and settings.langfuse_public_key:
        langfuse_handler = CallbackHandler(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host
        )
        config["callbacks"] = [langfuse_handler]

    
    try:
        final_state = await asyncio.to_thread(graph.invoke, state, config)
    except Exception as e:
        logger.error(f"[API] Graph error: {e}")
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
    job_id = str(uuid4())
    try:
        results = await asyncio.to_thread(
            run_backtest,
            universe=req.tickers,
            start=req.start_date,
            end=req.end_date,
            initial_cash=req.initial_cash,
            invest_pct=0.80,
            risk_tolerance=req.risk_tolerance
        )
        return {"job_id": job_id, "status": "COMPLETED", "results": results}
    except Exception as e:
        logger.error(f"[API] Backtest failed: {e}")
        raise HTTPException(500, f"Backtest failed: {str(e)}")


@app.get("/backtest/{job_id}")
async def get_backtest_result(job_id: str):
    raise HTTPException(404, "Job ID polling is no longer supported; POST /backtest is now synchronous.")


@app.get("/memory/stats")
async def get_memory_stats():
    return cultural_memory.summary_stats()


@app.get("/governor/report")
async def get_governor_report():
    return governor.get_usage_report()


@app.post("/kill-switch/reset")
async def reset_kill_switch(new_inception_value: float):
    ks = get_kill_switch()
    if ks:
        ks.reset(new_inception_value)
        return {"status": "Reset successful"}
    raise HTTPException(404, "Kill switch not initialized")
