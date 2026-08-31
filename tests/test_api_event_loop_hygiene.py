"""
tests/test_api_event_loop_hygiene.py

Tests for PR4 (MFT-6/7/13, API-4): api/main.py must not block the event loop
on buffer reads or the daily reconciliation cycle.

These exercise route/loop functions directly via FastAPI's TestClient or as
bare coroutines, mirroring test_api_freshness.py's pattern of setting
`_mft_pipeline` and friends directly on the api.main module rather than
running the app's lifespan.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from datetime import datetime, timezone
from unittest import mock
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from argus.data.live_session_cache import LiveSessionCache

_ET = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Installs a fresh live session cache for each test."""
    monkeypatch.setattr(api_main, "_live_cache", LiveSessionCache(interval_minutes=1))


@pytest.fixture
def client():
    """A TestClient over api.main.app without running its lifespan startup."""
    return TestClient(api_main.app)


def test_pipeline_status_reports_buffer_depth_without_materializing_frames(monkeypatch, client):
    """/pipeline/status reads buffer depth via row_counts(), not by materializing each ticker's frame."""
    fake_pipeline = mock.Mock()
    fake_pipeline.tickers = ["AAPL"]
    fake_pipeline.is_market_hours.return_value = True
    fake_pipeline.buffer.row_counts.return_value = {"AAPL": 42}
    fake_pipeline.buffer.get_candles.side_effect = AssertionError("must not materialize frames")
    monkeypatch.setattr(api_main, "_mft_pipeline", fake_pipeline)

    response = client.get("/pipeline/status")

    assert response.status_code == 200
    assert response.json()["buffer_depth"] == {"AAPL": 42}
    fake_pipeline.buffer.get_candles.assert_not_called()
    fake_pipeline.buffer.row_counts.assert_called_once()


def test_reconcile_once_is_a_plain_sync_function():
    """`_reconcile_once` is synchronous, so `_reconcile_loop` can run it via `asyncio.to_thread`."""
    assert not inspect.iscoroutinefunction(api_main._reconcile_once)


def test_seconds_until_next_reconcile_is_dst_safe():
    """API-13 regression: the sleep duration must reflect real elapsed time across a DST jump.

    `target - now_et` alone is wrong here: both operands are built from the
    same ZoneInfo("America/New_York") instance, so aware-datetime
    subtraction compares wall-clock fields directly rather than each side's
    own UTC offset — silently off by exactly the DST shift. 2026-03-08 is a
    US spring-forward date (clocks jump 2:00 AM -> 3:00 AM), so the true gap
    from 19:00 EST the day before to 18:00 EDT the next day is 22 hours, not
    the 23 naive wall-clock hours between the two timestamps.
    """
    now_et = datetime(2026, 3, 7, 19, 0, tzinfo=_ET)  # after 18:00, so target rolls to tomorrow

    seconds = api_main._seconds_until_next_reconcile(now_et, hour=18)

    assert seconds == 22 * 3600


def test_seconds_until_next_reconcile_matches_utc_diff_on_a_normal_day():
    """Outside a DST transition, the DST-safe computation still matches a plain UTC diff."""
    now_et = datetime(2026, 6, 1, 9, 0, tzinfo=_ET)
    target_et = datetime(2026, 6, 1, 18, 0, tzinfo=_ET)

    seconds = api_main._seconds_until_next_reconcile(now_et, hour=18)

    expected = (target_et.astimezone(timezone.utc) - now_et.astimezone(timezone.utc)).total_seconds()
    assert seconds == expected == 9 * 3600


@pytest.mark.asyncio
async def test_reconcile_loop_runs_its_cycle_body_via_to_thread(monkeypatch):
    """`_reconcile_loop` offloads its per-cycle body through asyncio.to_thread, not inline."""
    real_sleep = asyncio.sleep
    calls = []

    def fake_reconcile_once():
        calls.append(True)

    async def fake_sleep(_seconds):
        await real_sleep(0)

    async def fake_to_thread(fn, *args, **kwargs):
        await real_sleep(0)
        return fn(*args, **kwargs)

    monkeypatch.setattr(api_main, "_reconcile_once", fake_reconcile_once)
    monkeypatch.setattr(api_main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(api_main.asyncio, "to_thread", fake_to_thread)

    task = asyncio.create_task(api_main._reconcile_loop())
    try:
        for _ in range(200):
            if calls:
                break
            await real_sleep(0.005)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert calls, "expected _reconcile_once to run via to_thread"
