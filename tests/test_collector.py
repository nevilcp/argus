"""Tests for collector.py's `analyze_lock` parameter.

The unattended collector shares /analyze's concurrency slot rather than
invoking the graph alongside a live /analyze run, and skips rather than
waits when the slot is already held.

The kill-switch gate is not covered here; see tests/test_kill_switch.py's
"Collector gate" section for that.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import mock

import pytest

from argus.orchestration.collector import run_collection_cycle

_SESSION_STATES = {"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}}


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
