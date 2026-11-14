"""Collector admission under the process kill switch."""

from __future__ import annotations

from unittest import mock

import argus.risk.kill_switch as kill_switch_module
from argus.orchestration.collector import run_collection_cycle
from tests.helpers.kill_switch import make_ks


async def test_collector_skips_when_kill_switch_halted():
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
    assert result.ran is False
    assert "halted" not in result.reason
