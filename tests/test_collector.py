"""Tests for collector.py's `analyze_lock` parameter and cycle-health reporting.

The unattended collector shares /analyze's concurrency slot rather than
invoking the graph alongside a live /analyze run, and skips rather than
waits when the slot is already held.

The kill-switch gate is not covered here; see tests/test_kill_switch.py's
"Collector gate" section for that.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest import mock

import pytest

from argus.orchestration.collector import CycleOutcome, run_collection_cycle
from argus.params import COLLECTOR
from argus.schemas.signals import ARGUSDecision, PositionAllocation

_SESSION_STATES = {"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}}


def _decision(ticker: str = "AAPL", *, with_allocation: bool = False) -> ARGUSDecision:
    """Builds a minimal ARGUSDecision, optionally carrying a real allocation."""
    allocation = None
    if with_allocation:
        allocation = PositionAllocation(
            ticker=ticker,
            allocation_pct=0.05,
            allocation_usd=5_000.0,
            stop_loss=90.0,
            thesis="test thesis",
            composite_conviction=0.5,
            time_horizon="30 days",
        )
    return ARGUSDecision(ticker=ticker, session_timestamp=datetime(2024, 1, 2), allocation=allocation)


def _graph() -> mock.Mock:
    """Builds a fake compiled graph whose invoke() returns a minimal final_state dict."""
    graph = mock.Mock()
    graph.invoke = mock.Mock(return_value={"decisions": [], "errors": [], "macro_context": None})
    return graph


async def _run_cycle(graph: mock.Mock, **kwargs: Any):
    """Runs one cycle over a one-ticker universe against a fake pipeline.

    Args:
        graph: The fake compiled graph the cycle should invoke.
        **kwargs: Passed straight through, for the analyze_lock each test varies.

    Returns:
        The CollectionResult the cycle produced.
    """
    pipeline = mock.Mock()
    pipeline.run_once = mock.AsyncMock(return_value=_SESSION_STATES)
    return await run_collection_cycle(
        universe=["AAPL"],
        total_wealth=100_000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
        pipeline=pipeline,
        compiled_graph=graph,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_run_collection_cycle_skips_when_analyze_lock_is_held():
    """A held analyze_lock makes the collector skip rather than invoke the graph."""
    lock = asyncio.Semaphore(1)
    await lock.acquire()
    graph = _graph()
    graph.invoke.side_effect = AssertionError("must not run while /analyze holds the lock")

    result = await _run_cycle(graph, analyze_lock=lock)

    assert result.ran is False
    assert "already in progress" in result.reason
    graph.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_run_collection_cycle_runs_when_analyze_lock_is_free():
    """A free analyze_lock lets the collector acquire it and invoke the graph normally."""
    lock = asyncio.Semaphore(1)
    graph = _graph()

    result = await _run_cycle(graph, analyze_lock=lock)

    assert result.ran is True
    graph.invoke.assert_called_once()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_run_collection_cycle_runs_unguarded_when_analyze_lock_is_none():
    """No analyze_lock (scripts/collect_session.py's standalone caller) runs unguarded."""
    graph = _graph()

    result = await _run_cycle(graph)

    assert result.ran is True
    graph.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_run_collection_cycle_reports_no_op_when_skipped():
    """A cycle that never invokes the graph (market closed) is a NO_OP, not degraded."""
    lock = asyncio.Semaphore(1)
    await lock.acquire()
    graph = _graph()

    result = await _run_cycle(graph, analyze_lock=lock)

    assert result.ran is False
    assert result.outcome is CycleOutcome.NO_OP


@pytest.mark.asyncio
async def test_run_collection_cycle_reports_success_when_allocations_clear_threshold():
    """Enough decisions carry a real allocation: the cycle is SUCCESS, not DEGRADED."""
    decisions = [_decision(with_allocation=True) for _ in range(COLLECTOR.min_decisions_with_allocation)]
    graph = _graph()
    graph.invoke.return_value = {"decisions": decisions, "errors": [], "macro_context": None}

    result = await _run_cycle(graph)

    assert result.outcome is CycleOutcome.SUCCESS
    assert result.decisions_with_allocation == len(decisions)
    assert "allocation" not in result.degraded_inputs


@pytest.mark.asyncio
async def test_run_collection_cycle_reports_degraded_when_no_decision_has_allocation():
    """Twenty decisions, all null allocation: issue #91's "looks like a success" case."""
    decisions = [_decision(with_allocation=False) for _ in range(20)]
    graph = _graph()
    graph.invoke.return_value = {"decisions": decisions, "errors": [], "macro_context": None}

    result = await _run_cycle(graph)

    assert result.ran is True
    assert result.outcome is CycleOutcome.DEGRADED
    assert result.decisions_with_allocation == 0
    assert result.decisions_logged == 20
    assert result.degraded_inputs["allocation"] == 20
    assert result.degraded_inputs["fundamental"] == 20
    assert result.degraded_inputs["sentiment"] == 20
    assert result.degraded_inputs["risk"] == 20
