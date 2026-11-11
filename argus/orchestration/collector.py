"""
argus/orchestration/collector.py

Unattended collection cycle shared by the always-on API process and the
scheduled GitHub Actions runner.

A single function, run_collection_cycle(), is the one place that knows how to
turn "sweep the MFT pipeline once, then run the graph" into logged decisions.
Both the API's background collector loop (orchestration/scheduler.py) and
scripts/collect_session.py call it (via run_collection_cycle_from_settings), so
the two deployment paths can't drift apart from each other.

Responsibilities:
  - Refuse to run while the process-local kill switch is halted
  - Run one MFT fetch-and-compress sweep via MFTDataPipeline.run_once()
  - Invoke the caller-supplied compiled LangGraph DAG over whatever tickers
    the sweep actually populated
  - Append every resulting ARGUSDecision to a JSONL decision log
    (decision_log.append_decisions_jsonl is also called directly by the
    API's /analyze route, the loop's other entry point into the graph, so
    the log reflects both)
  - Classify the cycle's outcome (CycleOutcome: no-op, success, or
    degraded) so a run that produced only unusable decisions is
    distinguishable from one that produced real ones
  - Prune stale LangGraph checkpoint threads from the checkpoint database
    before returning (see _prune_stale_checkpoints), so whatever a caller
    publishes next — the scheduled collector force-pushes the whole data
    directory to argus-data every cycle — is already bounded (issue #99)

Not responsible for:
  - Scheduling (see orchestration/scheduler.py and scripts/collect_session.py)
  - Reconciling logged decisions into outcomes (see orchestration/reconcile.py)
  - Market-hours/holiday awareness beyond what MFTDataPipeline already checks
"""

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

# Decision fields whose absence marks a degraded input; checked on every
# decision the graph produced so a cycle summary can say which upstream
# signal(s) went missing rather than just "some things were null".
_DECISION_SIGNAL_FIELDS = ("technical", "macro", "fundamental", "sentiment", "risk", "allocation")


class CycleOutcome(StrEnum):
    """What a completed run_collection_cycle() call actually achieved.

    The three outcomes issue #91 asks the collector to tell apart — market
    closed (NO_OP), a normal run (SUCCESS), and a run that produced only
    decisions with no usable allocation (DEGRADED) — look identical from a
    green exit code otherwise.
    """

    NO_OP = "no_op"
    SUCCESS = "success"
    DEGRADED = "degraded"


@dataclass
class CollectionResult:
    """Outcome of one run_collection_cycle() call.

    Attributes:
        ran: False when the cycle skipped the graph entirely (no session
            data yet).
        reason: Human-readable explanation, e.g. "market closed", "collected".
        outcome: NO_OP when `ran` is False; otherwise SUCCESS or DEGRADED
            depending on whether decisions_with_allocation clears
            params.COLLECTOR.min_decisions_with_allocation.
        tickers_with_session_data: Tickers the MFT sweep actually populated
            session data for this cycle.
        decisions_logged: Count of decisions appended to the JSONL log.
        decisions_with_allocation: Of those, how many carry a real
            (non-null) allocation — the ones actually worth reconciling.
        degraded_inputs: Counts, per signal name, of decisions produced
            without that signal (e.g. {"fundamental": 20} means every
            decision this cycle came back with fundamental=None). Only
            signals that degraded at least once are included.
        macro_regime: The macro agent's regime classification for this
            cycle, or None if the graph didn't run or produced no macro
            context.
        errors: Error messages accumulated during the graph invocation.
        timestamp: ISO-formatted time this result was constructed.
    """

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
    """Counts, per signal name, how many of this cycle's decisions came back without it.

    Args:
        decisions: Decisions produced by this cycle's graph invocation.

    Returns:
        {signal_name: count}, omitting any signal that degraded zero times.
    """
    counts = {
        name: sum(1 for d in decisions if getattr(d, name) is None)
        for name in _DECISION_SIGNAL_FIELDS
    }
    return {name: count for name, count in counts.items() if count > 0}


def _prune_stale_checkpoints(checkpoint_db_path: str) -> None:
    """Best-effort prunes checkpoint threads older than the reconciliation retention window.

    Uses retention_cutoff() — the same window
    reconcile.py's own daily pass prunes to — so a checkpoint thread
    survives exactly as long under either path. Called on every collector
    cycle (issue #99) so the state each cycle publishes to argus-data is
    already bounded, rather than relying solely on the once-daily reconcile
    job to catch up after the fact.

    Failures are logged and swallowed: pruning is housekeeping independent of
    this cycle's actual result, and must never turn a successful collection
    into a failed one.
    """
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
    """Runs one unattended fetch → analyze → log cycle for the given universe.

    Refuses to run while this process's kill switch is halted — the
    unattended collector is one of the two entry paths into the graph
    (/analyze is the other, already gated in api/main.py), and a halted
    system must stop feeding new decisions into decisions.jsonl just as it
    stops serving /analyze. A process with no kill switch initialized
    (get_kill_switch() returns None — e.g. scripts/collect_session.py's
    standalone GitHub Actions runner) is ungated; that cross-process gap
    is documented, not solved, here.

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
            checkpoint.py's ``CheckpointedGraph``.
        decisions_log_path: JSONL file each cycle's decisions are appended to. None
            (default) resolves to ``<ARGUS_DATA_DIR>/decisions.jsonl`` at call time,
            so tests/conftest.py's per-test ARGUS_DATA_DIR override also isolates a
            call site that omits this argument.
        checkpoint_db_path: LangGraph checkpoint database this cycle's
            compiled_graph checkpoints to. When given, stale checkpoint
            threads are pruned (see _prune_stale_checkpoints) before this
            call returns, regardless of whether the graph ran this cycle —
            so whatever a caller publishes afterward (the scheduled
            collector force-pushes the whole data directory to argus-data
            every tick) is already bounded. None skips pruning entirely.
        analyze_lock: The same semaphore api/main.py's `/analyze` guards its
            graph invocation with — the collector is a third caller of
            the same graph and governor, so it shares the slot rather than
            invoking concurrently with a live `/analyze` run. It skips rather
            than waits: an unattended cycle has nowhere to be on time, so
            blocking it against a slow foreground request buys nothing.
            None for a standalone caller with no `/analyze` to coordinate
            with (scripts/collect_session.py's one-shot Actions runner).

    Returns:
        A CollectionResult summarizing what happened.
    """
    if checkpoint_db_path is not None:
        _prune_stale_checkpoints(checkpoint_db_path)

    ks = get_kill_switch()
    if ks is not None and ks.is_halted:
        reason = f"halted: {ks.status.reason}"
        logger.warning("skipping graph invocation — %s", reason)
        return CollectionResult(ran=False, reason=reason)

    session_states = await pipeline.run_once()

    tickers_with_data = [t for t in universe if t in session_states]
    if not tickers_with_data:
        reason = "market closed" if not pipeline.is_market_hours() else "buffer not yet warm"
        logger.info("skipping graph invocation — %s", reason)
        return CollectionResult(ran=False, reason=reason)

    if analyze_lock is not None and analyze_lock.locked():
        reason = "analysis already in progress"
        logger.info("skipping graph invocation — %s", reason)
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
    # SqliteSaver checkpoints synchronously (see graph.py's module docstring), so
    # invoke() blocks — run it off the event loop rather than stalling this coroutine.
    # No await between the locked() check above and this acquire, so a concurrent
    # /analyze can't slip in between the check and the hold.
    async with analyze_lock if analyze_lock is not None else nullcontext():
        final_state = await asyncio.to_thread(compiled_graph.invoke, state, config)

    if decisions_log_path is None:
        decisions_log_path = str(settings.data_path("decisions.jsonl"))
    return _summarize_cycle(final_state, tickers_with_data, decisions_log_path)


def _summarize_cycle(
    final_state: dict, tickers_with_data: list[str], decisions_log_path: str
) -> CollectionResult:
    """Logs the cycle's decisions and summarizes the graph's final state.

    Args:
        final_state: The state returned by the compiled graph.
        tickers_with_data: Tickers that had session data this cycle.
        decisions_log_path: JSONL file the decisions are appended to.

    Returns:
        A CollectionResult with ``ran=True``.
    """
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
    """Runs one cycle with every unspecified argument taken from the environment settings.

    The always-on API loop passes only its shared pipeline, graph and lock;
    scripts/collect_session.py passes whichever CLI overrides were given. Both
    therefore resolve the universe, capital, risk tolerance and store paths the
    same way, instead of each re-listing the settings.

    Args:
        pipeline: See run_collection_cycle.
        compiled_graph: See run_collection_cycle.
        universe: Tickers to analyze; defaults to ARGUS_UNIVERSE.
        total_wealth: Defaults to ARGUS_TOTAL_WEALTH.
        invest_pct: Defaults to ARGUS_INVEST_PCT.
        risk_tolerance: Defaults to ARGUS_RISK_TOLERANCE.
        decisions_log_path: Defaults to ``<ARGUS_DATA_DIR>/decisions.jsonl``.
        checkpoint_db_path: Defaults to ``<ARGUS_DATA_DIR>/argus_graph.db``.
        analyze_lock: See run_collection_cycle.

    Returns:
        A CollectionResult summarizing what happened.
    """
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
