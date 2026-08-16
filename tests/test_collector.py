"""
tests/test_collector.py

Tests for argus/orchestration/collector.py's `analyze_lock` parameter (PR5b,
API-3): the unattended collector shares /analyze's concurrency slot rather
than invoking the graph alongside a live /analyze run, and skips rather than
waits when the slot is already held.

Not responsible for:
  - The kill-switch gate (KS-2), covered in tests/test_kill_switch.py's
    "Collector gate" section
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from argus.orchestration.collector import run_collection_cycle


def _pipeline(session_states: dict) -> mock.Mock:
    """Builds a fake MFTDataPipeline whose run_once() returns the given session states."""
    pipeline = mock.Mock()
    pipeline.run_once = mock.AsyncMock(return_value=session_states)
    return pipeline


def _graph() -> mock.Mock:
    """Builds a fake compiled graph whose invoke() returns a minimal final_state dict."""
    graph = mock.Mock()
    graph.invoke = mock.Mock(return_value={"decisions": [], "errors": [], "macro_context": None})
    return graph


@pytest.mark.asyncio
async def test_run_collection_cycle_skips_when_analyze_lock_is_held():
    """A held analyze_lock makes the collector skip rather than invoke the graph."""
    lock = asyncio.Semaphore(1)
    await lock.acquire()
    graph = _graph()
    graph.invoke.side_effect = AssertionError("must not run while /analyze holds the lock")

    result = await run_collection_cycle(
        universe=["AAPL"],
        total_wealth=100_000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
        pipeline=_pipeline({"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}}),
        compiled_graph=graph,
        analyze_lock=lock,
    )

    assert result.ran is False
    assert "already in progress" in result.reason
    graph.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_run_collection_cycle_runs_when_analyze_lock_is_free():
    """A free analyze_lock lets the collector acquire it and invoke the graph normally."""
    lock = asyncio.Semaphore(1)
    graph = _graph()

    result = await run_collection_cycle(
        universe=["AAPL"],
        total_wealth=100_000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
        pipeline=_pipeline({"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}}),
        compiled_graph=graph,
        analyze_lock=lock,
    )

    assert result.ran is True
    graph.invoke.assert_called_once()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_run_collection_cycle_runs_unguarded_when_analyze_lock_is_none():
    """No analyze_lock (scripts/collect_session.py's standalone caller) runs unguarded."""
    graph = _graph()

    result = await run_collection_cycle(
        universe=["AAPL"],
        total_wealth=100_000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
        pipeline=_pipeline({"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}}),
        compiled_graph=graph,
    )

    assert result.ran is True
    graph.invoke.assert_called_once()
