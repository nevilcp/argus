"""
tests/test_kill_switch_collector_gate.py

Tests for the kill-switch gate in argus/orchestration/collector.py:
run_collection_cycle refusing to run while halted.

fetch_vix is mocked throughout; _check() is driven directly, so no test sleeps
on a real check interval.
"""

from __future__ import annotations

from unittest import mock

import argus.risk.kill_switch as kill_switch_module
from argus.orchestration.collector import run_collection_cycle
from tests.helpers.kill_switch import make_ks


async def test_collector_skips_when_kill_switch_halted():
    """run_collection_cycle refuses to run while this process's kill switch is halted."""
    ks = make_ks("MODERATE")
    ks._halted.set()
    ks._halt_reason = "test halt"
    kill_switch_module._kill_switch = ks

    pipeline = mock.Mock()
    pipeline.run_once = mock.AsyncMock(side_effect=AssertionError("must not run while halted"))

    result = await run_collection_cycle(
        universe=["AAPL"],
        total_wealth=100_000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
        pipeline=pipeline,
        compiled_graph=mock.Mock(),
    )

    assert result.ran is False
    assert "halted" in result.reason
    pipeline.run_once.assert_not_called()


async def test_collector_runs_when_no_kill_switch_initialized():
    """A process with no kill switch initialized (get_kill_switch() is None) is ungated."""
    assert kill_switch_module.get_kill_switch() is None

    pipeline = mock.Mock()
    pipeline.run_once = mock.AsyncMock(return_value={})

    result = await run_collection_cycle(
        universe=["AAPL"],
        total_wealth=100_000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
        pipeline=pipeline,
        compiled_graph=mock.Mock(),
    )

    pipeline.run_once.assert_awaited_once()
    assert result.ran is False  # no session data for AAPL in the empty {} session_states
    assert "halted" not in result.reason
