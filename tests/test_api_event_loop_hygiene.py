"""The API must never block the event loop on reads or reconciliation.

These exercise route/loop functions directly via FastAPI's TestClient or as
bare coroutines, mirroring test_api_freshness.py's pattern of setting
`runtime.pipeline` and friends directly on api.lifespan.runtime rather than
running the app's lifespan.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from datetime import UTC, datetime
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from api.lifespan import runtime
from argus.orchestration import scheduler
from argus.orchestration.collector import CollectionResult

_ET = ZoneInfo("America/New_York")

pytestmark = pytest.mark.usefixtures("api_state")


def test_pipeline_status_reports_buffer_depth_without_materializing_frames(monkeypatch, client):
    """/pipeline/status reads buffer depth via row_counts(), not by materializing each
    ticker's frame."""
    fake_pipeline = mock.Mock()
    fake_pipeline.tickers = ["AAPL"]
    fake_pipeline.is_market_hours.return_value = True
    fake_pipeline.buffer.row_counts.return_value = {"AAPL": 42}
    fake_pipeline.buffer.get_candles.side_effect = AssertionError("must not materialize frames")
    monkeypatch.setattr(runtime, "pipeline", fake_pipeline)

    response = client.get("/pipeline/status")

    assert response.status_code == 200
    assert response.json()["buffer_depth"] == {"AAPL": 42}
    fake_pipeline.buffer.get_candles.assert_not_called()
    fake_pipeline.buffer.row_counts.assert_called_once()


def test_reconcile_once_is_a_plain_sync_function():
    """`reconcile_once` is synchronous, so `reconcile_loop` can run it via `asyncio.to_thread`."""
    assert not inspect.iscoroutinefunction(scheduler.reconcile_once)


def test_seconds_until_next_reconcile_is_dst_safe():
    """The sleep duration must reflect real elapsed time across a DST jump.

    `target - now_et` alone is wrong here: both operands are built from the
    same ZoneInfo("America/New_York") instance, so aware-datetime
    subtraction compares wall-clock fields directly rather than each side's
    own UTC offset — silently off by exactly the DST shift. 2026-03-08 is a
    US spring-forward date (clocks jump 2:00 AM -> 3:00 AM), so the true gap
    from 19:00 EST the day before to 18:00 EDT the next day is 22 hours, not
    the 23 naive wall-clock hours between the two timestamps.
    """
    now_et = datetime(2026, 3, 7, 19, 0, tzinfo=_ET)  # after 18:00, so target rolls to tomorrow

    seconds = scheduler.seconds_until_next_reconcile(now_et, hour=18)

    assert seconds == 22 * 3600


def test_seconds_until_next_reconcile_matches_utc_diff_on_a_normal_day():
    """Outside a DST transition, the DST-safe computation still matches a plain UTC diff."""
    now_et = datetime(2026, 6, 1, 9, 0, tzinfo=_ET)
    target_et = datetime(2026, 6, 1, 18, 0, tzinfo=_ET)

    seconds = scheduler.seconds_until_next_reconcile(now_et, hour=18)

    expected = (target_et.astimezone(UTC) - now_et.astimezone(UTC)).total_seconds()
    assert seconds == expected == 9 * 3600


async def test_reconcile_loop_runs_its_cycle_body_via_to_thread(monkeypatch):
    """`reconcile_loop` offloads its per-cycle body through asyncio.to_thread, not inline."""
    real_sleep = asyncio.sleep
    calls = []

    def fake_reconcile_once():
        calls.append(True)

    async def fake_sleep(_seconds):
        await real_sleep(0)

    async def fake_to_thread(fn, *args, **kwargs):
        await real_sleep(0)
        return fn(*args, **kwargs)

    monkeypatch.setattr(scheduler, "reconcile_once", fake_reconcile_once)
    monkeypatch.setattr(scheduler.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(scheduler.asyncio, "to_thread", fake_to_thread)

    task = asyncio.create_task(scheduler.reconcile_loop())
    try:
        for _ in range(200):
            if calls:
                break
            await real_sleep(0.005)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert calls, "expected reconcile_once to run via to_thread"


async def test_collector_loop_reports_each_result_and_survives_a_failed_cycle(monkeypatch):
    """A cycle that raises is logged and skipped; the next one still runs and is reported."""
    real_sleep = asyncio.sleep
    first, second = CollectionResult(ran=False, reason="a"), CollectionResult(ran=False, reason="b")
    cycle = mock.AsyncMock(side_effect=[RuntimeError("boom"), first, second])
    reported: list[CollectionResult] = []

    async def fake_sleep(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(scheduler, "run_collection_cycle_from_settings", cycle)
    monkeypatch.setattr(scheduler.asyncio, "sleep", fake_sleep)
    lock = asyncio.Semaphore(1)
    pipeline, graph = mock.Mock(), mock.Mock()

    task = asyncio.create_task(scheduler.collector_loop(pipeline, graph, lock, reported.append))
    try:
        for _ in range(200):
            if len(reported) == 2:
                break
            await real_sleep(0.005)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert reported == [first, second]
    cycle.assert_called_with(pipeline=pipeline, compiled_graph=graph, analyze_lock=lock)
