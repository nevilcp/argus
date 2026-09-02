"""
argus/orchestration/collector.py

Unattended collection cycle shared by the always-on API process and the
scheduled GitHub Actions runner.

A single function, run_collection_cycle(), is the one place that knows how to
turn "sweep the MFT pipeline once, then run the graph" into logged decisions.
Both api/main.py's background collector loop and scripts/collect_session.py
call it, so the two deployment paths can't drift apart from each other.

Responsibilities:
  - Refuse to run while the process-local kill switch is halted
  - Run one MFT fetch-and-compress sweep via MFTDataPipeline.run_once()
  - Invoke the caller-supplied compiled LangGraph DAG over whatever tickers
    the sweep actually populated
  - Append every resulting ARGUSDecision to a JSONL decision log
    (append_decisions_jsonl is also called directly by api/main.py's
    /analyze, the loop's other entry point into the graph, so the log
    reflects both)
  - Classify the cycle's outcome (CycleOutcome: no-op, success, or
    degraded) so a run that produced only unusable decisions is
    distinguishable from one that produced real ones

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
from enum import Enum
from pathlib import Path

from argus.config import settings
from argus.data.pipeline import MFTDataPipeline
from argus.orchestration.state import ARGUSState
from argus.params import COLLECTOR
from argus.risk.kill_switch import get_kill_switch
from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger("argus.collector")

# Decision fields whose absence marks a degraded input; checked on every
# decision the graph produced so a cycle summary can say which upstream
# signal(s) went missing rather than just "some things were null".
_DECISION_SIGNAL_FIELDS = ("technical", "macro", "fundamental", "sentiment", "risk", "allocation")


class CycleOutcome(str, Enum):
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
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())  # noqa: DTZ005


def append_decisions_jsonl(decisions: list[ARGUSDecision], path: str) -> int:
    """Appends each decision to a JSONL log, one compact JSON object per line.

    Shared by run_collection_cycle and api/main.py's /analyze — the two
    entry points into the graph — so decisions.jsonl (the source
    reconciliation reads, see orchestration/reconciliation.py) reflects
    both instead of only the unattended collector's.

    Each decision is stamped with the running process's image tag (issue #95)
    before it's written, so a logged decision can be traced back to the build
    that produced it regardless of which docker tag was pulled to run it.

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
            stamped = decision.model_copy(update={"image_tag": settings.ARGUS_IMAGE_TAG})
            f.write(stamped.model_dump_json() + "\n")
    return len(decisions)


def _summarize_degraded_inputs(decisions: list[ARGUSDecision]) -> dict[str, int]:
    """Counts, per signal name, how many of this cycle's decisions came back without it.

    Args:
        decisions: Decisions produced by this cycle's graph invocation.

    Returns:
        {signal_name: count}, omitting any signal that degraded zero times.
    """
    counts = {
        name: sum(1 for d in decisions if getattr(d, name) is None) for name in _DECISION_SIGNAL_FIELDS
    }
    return {name: count for name, count in counts.items() if count > 0}


async def run_collection_cycle(
    universe: list[str],
    total_wealth: float,
    invest_pct: float,
    risk_tolerance: str,
    *,
    pipeline: MFTDataPipeline,
    compiled_graph,
    decisions_log_path: str = "data/decisions.jsonl",
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
            graph.py's ``_CheckpointedGraph``.
        decisions_log_path: JSONL file each cycle's decisions are appended to.
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
    ks = get_kill_switch()
    if ks is not None and ks.is_halted:
        reason = f"halted: {ks.status.reason}"
        logger.warning("run_collection_cycle: skipping graph invocation — %s", reason)
        return CollectionResult(ran=False, reason=reason)

    session_states = await pipeline.run_once()

    tickers_with_data = [t for t in universe if t in session_states]
    if not tickers_with_data:
        reason = "market closed" if not pipeline.is_market_hours() else "buffer not yet warm"
        logger.info("run_collection_cycle: skipping graph invocation — %s", reason)
        return CollectionResult(ran=False, reason=reason)

    if analyze_lock is not None and analyze_lock.locked():
        reason = "analysis already in progress"
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

    thread_id = f"collector-{datetime.now().strftime('%Y%m%d%H%M%S')}"  # noqa: DTZ005
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "run_collection_cycle: invoking graph for %d/%d ticker(s) with session data",
        len(tickers_with_data),
        len(universe),
    )
    # SqliteSaver checkpoints synchronously (see graph.py's module docstring), so
    # invoke() blocks — run it off the event loop rather than stalling this coroutine.
    # No await between the locked() check above and this acquire, so a concurrent
    # /analyze can't slip in between the check and the hold.
    if analyze_lock is not None:
        async with analyze_lock:
            final_state = await asyncio.to_thread(compiled_graph.invoke, state, config)
    else:
        final_state = await asyncio.to_thread(compiled_graph.invoke, state, config)

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
        "run_collection_cycle: logged %d decision(s) (%d with allocation), "
        "macro_regime=%s, outcome=%s",
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
