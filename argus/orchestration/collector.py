"""Unattended fetch, analyze, and log cycles shared by API and scheduled runners."""

from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum

from argus.config import settings
from argus.data.pipeline import MFTDataPipeline
from argus.infra.clock import now_local
from argus.orchestration.checkpoint import (
    new_run_config,
    prune_checkpoints,
    retention_cutoff,
)
from argus.orchestration.decision_log import append_decisions_jsonl
from argus.orchestration.state import initial_state
from argus.params import COLLECTOR
from argus.risk.kill_switch import get_kill_switch
from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger(__name__)

# Inspect every decision for missing inputs so cycle diagnostics identify degraded upstream signals.
_DECISION_SIGNAL_FIELDS = ("technical", "macro", "fundamental", "sentiment", "risk", "allocation")


class CycleOutcome(StrEnum):
    """Collection result distinguished as no-op, usable success, or degraded output."""

    NO_OP = "no_op"
    SUCCESS = "success"
    DEGRADED = "degraded"


@dataclass
class CollectionResult:
    """Outcome of one run_collection_cycle() call."""

    ran: bool
    reason: str
    outcome: CycleOutcome = CycleOutcome.NO_OP
    tickers_with_session_data: list[str] = field(default_factory=list)
    decisions_logged: int = 0
    decisions_with_allocation: int = 0
    degraded_inputs: dict[str, int] = field(default_factory=dict)
    macro_regime: str | None = None
    errors: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: now_local().isoformat())


def _summarize_degraded_inputs(decisions: list[ARGUSDecision]) -> dict[str, int]:
    """Count missing signal fields across the completed decisions."""
    counts = {name: sum(1 for d in decisions if getattr(d, name) is None) for name in _DECISION_SIGNAL_FIELDS}
    return {name: count for name, count in counts.items() if count > 0}


def _prune_stale_checkpoints(checkpoint_db_path: str) -> None:
    """Prune prunes checkpoint threads older than the reconciliation retention window."""
    try:
        pruned = prune_checkpoints(checkpoint_db_path, retention_cutoff())
        if pruned:
            logger.info("pruned %d stale checkpoint thread(s)", pruned)
    except Exception:
        logger.exception("checkpoint pruning failed")


async def run_collection_cycle(
    universe: list[str],
    total_wealth: float,
    invest_pct: float,
    risk_tolerance: str,
    *,
    pipeline: MFTDataPipeline,
    compiled_graph,
    decisions_log_path: str | None = None,
    checkpoint_db_path: str | None = None,
    analyze_lock: asyncio.Semaphore | None = None,
) -> CollectionResult:
    """Run one gated fetch, analysis, and log cycle, skipping empty or unusable session data."""
    if checkpoint_db_path is not None:
        _prune_stale_checkpoints(checkpoint_db_path)

    ks = get_kill_switch()
    if ks is not None and ks.is_halted:
        reason = f"halted: {ks.status.reason}"
        logger.warning("skipping graph invocation - %s", reason)
        return CollectionResult(ran=False, reason=reason)

    session_states = await pipeline.run_once()

    tickers_with_data = [t for t in universe if t in session_states]
    if not tickers_with_data:
        reason = "market closed" if not pipeline.is_market_hours() else "buffer not yet warm"
        logger.info("skipping graph invocation - %s", reason)
        return CollectionResult(ran=False, reason=reason)

    if analyze_lock is not None and analyze_lock.locked():
        reason = "analysis already in progress"
        logger.info("skipping graph invocation - %s", reason)
        return CollectionResult(ran=False, reason=reason)

    state = initial_state(
        tickers_with_data,
        session_states=session_states,
        total_wealth=total_wealth,
        invest_pct=invest_pct,
        risk_tolerance=risk_tolerance,
    )

    config = new_run_config("collector")

    logger.info(
        "invoking graph for %d/%d ticker(s) with session data",
        len(tickers_with_data),
        len(universe),
    )
    # Acquire without an intervening await, then run synchronous SQLite checkpointing off the event loop.
    async with analyze_lock if analyze_lock is not None else nullcontext():
        final_state = await asyncio.to_thread(compiled_graph.invoke, state, config)

    if decisions_log_path is None:
        decisions_log_path = str(settings.data_path("decisions.jsonl"))
    return _summarize_cycle(final_state, tickers_with_data, decisions_log_path)


def _summarize_cycle(final_state: dict, tickers_with_data: list[str], decisions_log_path: str) -> CollectionResult:
    """Log the cycle's decisions and summarizes the graph's final state."""
    decisions: list[ARGUSDecision] = final_state.get("decisions") or []
    logged = append_decisions_jsonl(decisions, decisions_log_path)
    errors = final_state.get("errors") or []

    macro_context = final_state.get("macro_context")
    macro_regime = macro_context.macro_regime.value if macro_context else None

    decisions_with_allocation = sum(1 for d in decisions if d.allocation is not None)
    degraded_inputs = _summarize_degraded_inputs(decisions)
    outcome = (
        CycleOutcome.SUCCESS
        if decisions_with_allocation >= COLLECTOR.min_decisions_with_allocation
        else CycleOutcome.DEGRADED
    )

    logger.info(
        "logged %d decision(s) (%d with allocation), macro_regime=%s, outcome=%s",
        logged,
        decisions_with_allocation,
        macro_regime,
        outcome.value,
    )
    return CollectionResult(
        ran=True,
        reason="collected",
        outcome=outcome,
        tickers_with_session_data=tickers_with_data,
        decisions_logged=logged,
        decisions_with_allocation=decisions_with_allocation,
        degraded_inputs=degraded_inputs,
        macro_regime=macro_regime,
        errors=errors,
    )


async def run_collection_cycle_from_settings(
    *,
    pipeline: MFTDataPipeline,
    compiled_graph,
    universe: list[str] | None = None,
    total_wealth: float | None = None,
    invest_pct: float | None = None,
    risk_tolerance: str | None = None,
    decisions_log_path: str | None = None,
    checkpoint_db_path: str | None = None,
    analyze_lock: asyncio.Semaphore | None = None,
) -> CollectionResult:
    """Run one cycle with every unspecified argument taken from the environment settings."""
    return await run_collection_cycle(
        universe=list(settings.ARGUS_UNIVERSE if universe is None else universe),
        total_wealth=settings.ARGUS_TOTAL_WEALTH if total_wealth is None else total_wealth,
        invest_pct=settings.ARGUS_INVEST_PCT if invest_pct is None else invest_pct,
        risk_tolerance=settings.ARGUS_RISK_TOLERANCE if risk_tolerance is None else risk_tolerance,
        pipeline=pipeline,
        compiled_graph=compiled_graph,
        decisions_log_path=decisions_log_path or str(settings.data_path("decisions.jsonl")),
        checkpoint_db_path=checkpoint_db_path or str(settings.data_path("argus_graph.db")),
        analyze_lock=analyze_lock,
    )
