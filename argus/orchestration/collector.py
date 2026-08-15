"""
argus/orchestration/collector.py

Unattended collection cycle shared by the always-on API process and the
scheduled GitHub Actions runner.

A single function, run_collection_cycle(), is the one place that knows how to
turn "sweep the MFT pipeline once, then run the graph" into logged decisions.
Both api/main.py's background collector loop and scripts/collect_session.py
call it, so the two deployment paths can't drift apart from each other.

Responsibilities:
  - Run one MFT fetch-and-compress sweep via MFTDataPipeline.run_once()
  - Invoke the caller-supplied compiled LangGraph DAG over whatever tickers
    the sweep actually populated
  - Append every resulting ARGUSDecision to a JSONL decision log

Not responsible for:
  - Scheduling (see api/main.py's collector loop and scripts/collect_session.py)
  - Reconciling logged decisions into outcomes (see orchestration/reconciliation.py)
  - Market-hours/holiday awareness beyond what MFTDataPipeline already checks
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from argus.data.pipeline import MFTDataPipeline
from argus.orchestration.state import ARGUSState
from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger("argus.collector")


@dataclass
class CollectionResult:
    """Outcome of one run_collection_cycle() call."""

    ran: bool
    """False when the cycle skipped the graph entirely (no session data yet)."""

    reason: str
    """Human-readable explanation — e.g. "market closed", "collected"."""

    tickers_with_session_data: list[str] = field(default_factory=list)
    decisions_logged: int = 0
    macro_regime: str | None = None
    errors: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def _append_decisions_jsonl(decisions: list[ARGUSDecision], path: str) -> int:
    """Appends each decision to a JSONL log, one compact JSON object per line.

    Args:
        decisions: Decisions produced by this cycle's graph invocation.
        path: Destination file; parent directories are created if missing.

    Returns:
        Count of decisions written.
    """
    if not decisions:
        return 0

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for decision in decisions:
            f.write(decision.model_dump_json() + "\n")
    return len(decisions)


async def run_collection_cycle(
    universe: list[str],
    total_wealth: float,
    invest_pct: float,
    risk_tolerance: str,
    *,
    pipeline: MFTDataPipeline,
    compiled_graph,
    decisions_log_path: str = "data/decisions.jsonl",
) -> CollectionResult:
    """Runs one unattended fetch → analyze → log cycle for the given universe.

    Skips the graph entirely — rather than invoking it against a stale or
    empty buffer — when the MFT sweep produces no usable session data, which
    happens outside market hours or on a market holiday
    (MFTDataPipeline._is_market_hours() only knows weekday/weekend).

    Args:
        universe: Tickers to analyze this cycle.
        total_wealth: Total investable capital passed into ARGUSState.
        invest_pct: Fraction of total_wealth to deploy.
        risk_tolerance: 'CONSERVATIVE', 'MODERATE', or 'AGGRESSIVE'.
        pipeline: Already-constructed MFTDataPipeline to sweep. Passed in
            (rather than built here) so a long-lived caller — the API
            process's background loop — reuses its warm buffer across cycles
            instead of starting cold every time.
        compiled_graph: Already-built graph (build_graph()) to invoke. Passed
            in for the same reason as ``pipeline``: a long-lived caller builds
            the six agents (LLM clients, the macro HMM classifier load) once
            rather than reconstructing them every cycle. Each ``.invoke()``
            still opens its own checkpoint connection — see
            graph.py's ``_CheckpointedGraph``.
        decisions_log_path: JSONL file each cycle's decisions are appended to.

    Returns:
        A CollectionResult summarizing what happened.
    """
    session_states = await pipeline.run_once()

    tickers_with_data = [t for t in universe if t in session_states]
    if not tickers_with_data:
        reason = "market closed" if not pipeline.is_market_hours() else "buffer not yet warm"
        logger.info("run_collection_cycle: skipping graph invocation — %s", reason)
        return CollectionResult(ran=False, reason=reason)

    state: ARGUSState = ARGUSState(
        ticker=tickers_with_data[0],
        universe=tickers_with_data,
        total_wealth=total_wealth,
        invest_pct=invest_pct,
        risk_tolerance=risk_tolerance,
        backtest_mode=False,
        session_seed=None,
        as_of=None,
        price_history={},
        session_states=session_states,
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

    config = {"configurable": {"thread_id": f"collector-{datetime.now().strftime('%Y%m%d%H%M%S')}"}}

    logger.info(
        "run_collection_cycle: invoking graph for %d/%d ticker(s) with session data",
        len(tickers_with_data),
        len(universe),
    )
    # SqliteSaver checkpoints synchronously (see graph.py's module docstring), so
    # invoke() blocks — run it off the event loop rather than stalling this coroutine
    final_state = await asyncio.to_thread(compiled_graph.invoke, state, config)

    decisions: list[ARGUSDecision] = final_state.get("decisions") or []
    logged = _append_decisions_jsonl(decisions, decisions_log_path)
    errors = final_state.get("errors") or []

    macro_context = final_state.get("macro_context")
    macro_regime = macro_context.macro_regime.value if macro_context else None

    logger.info(
        "run_collection_cycle: logged %d decision(s), macro_regime=%s", logged, macro_regime
    )
    return CollectionResult(
        ran=True,
        reason="collected",
        tickers_with_session_data=tickers_with_data,
        decisions_logged=logged,
        macro_regime=macro_regime,
        errors=errors,
    )
