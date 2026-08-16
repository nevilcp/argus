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
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Clears the live session cache around each test."""
    monkeypatch.setattr(api_main, "_live_session_cache", {})
    yield


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
