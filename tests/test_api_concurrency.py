"""
tests/test_api_concurrency.py

Tests for PR5b's /analyze concurrency and process-safety additions:
  - API-3: a semaphore of 1 around the graph invocation, 429 + Retry-After
    when a run is already in progress, released even when the graph raises
  - API-8: the kill switch is re-checked after the graph returns, not just
    before it
  - the graph-not-yet-initialized guard (`_graph is None` -> 503)
  - API-9: `_mft_session_callback` evicts live-cache entries for tickers no
    longer tracked

These exercise the route function and callback directly via FastAPI's
TestClient / a bare coroutine call rather than going through the app's
lifespan, so no MFT pipeline, collector, or reconcile loop needs to start.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from argus.risk.kill_switch import KillSwitch

_PAYLOAD = {"tickers": ["AAPL"], "total_wealth": 100_000, "invest_pct": 0.5}


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Clears the kill-switch singleton and live session cache around each test."""
    kill_switch_module._kill_switch = None
    monkeypatch.setattr(api_main, "_live_session_cache", {})
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")
    monkeypatch.setattr(api_main, "_analyze_semaphore", asyncio.Semaphore(1))
    yield
    kill_switch_module._kill_switch = None


@pytest.fixture
def client():
    """A TestClient over api.main.app without running its lifespan startup."""
    return TestClient(api_main.app)


def _pipeline(monkeypatch, *, market_hours: bool, interval_minutes: int = 1):
    """Installs a fake MFTDataPipeline as the module-level `_mft_pipeline`."""
    fake = mock.Mock()
    fake.is_market_hours.return_value = market_hours
    fake.interval_minutes = interval_minutes
    monkeypatch.setattr(api_main, "_mft_pipeline", fake)
    return fake


def _seed_cache(ticker: str, *, bar_age_seconds: float, write_age_seconds: float = 0.0):
    """Populates `_live_session_cache` with a state whose bar and write times are both controlled."""
    bar_ts = datetime.now(api_main._ET) - timedelta(seconds=bar_age_seconds)
    write_ts = datetime.now() - timedelta(seconds=write_age_seconds)
    api_main._live_session_cache[ticker] = ({"timestamp": bar_ts.isoformat()}, write_ts)


def _ready(client, monkeypatch):
    """Clears every freshness gate ahead of /analyze so a request reaches the graph."""
    _pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=5, write_age_seconds=5)


# ---------------------------------------------------------------------------
# API-3: semaphore
# ---------------------------------------------------------------------------


def test_analyze_rejects_a_concurrent_run_with_429(client, monkeypatch):
    """A held semaphore is reported as 429 with a Retry-After header, not run against the graph."""
    _ready(client, monkeypatch)
    fake_graph = mock.Mock()
    fake_graph.invoke.side_effect = AssertionError("must not run while another analysis holds the slot")
    monkeypatch.setattr(api_main, "_graph", fake_graph)

    held = mock.Mock()
    held.locked.return_value = True
    monkeypatch.setattr(api_main, "_analyze_semaphore", held)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 429
    assert response.headers["retry-after"]
    fake_graph.invoke.assert_not_called()


def test_analyze_releases_the_semaphore_when_the_graph_raises(client, monkeypatch):
    """The slot is released on a graph exception, not just on success — easy to wedge permanently."""
    _ready(client, monkeypatch)
    fake_graph = mock.Mock()
    fake_graph.invoke.side_effect = RuntimeError("boom")
    monkeypatch.setattr(api_main, "_graph", fake_graph)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 500
    assert not api_main._analyze_semaphore.locked()


# ---------------------------------------------------------------------------
# Graph-not-initialized guard
# ---------------------------------------------------------------------------


def test_analyze_rejects_when_graph_not_yet_initialized(client, monkeypatch):
    """A None _graph (before lifespan startup finishes) is a 503, not an AttributeError."""
    _ready(client, monkeypatch)
    monkeypatch.setattr(api_main, "_graph", None)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    assert "graph" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# API-8: kill switch re-checked after the graph returns
# ---------------------------------------------------------------------------


def test_analyze_rejects_when_kill_switch_trips_during_the_graph_run(client, monkeypatch):
    """A halt that lands while the graph is running is caught after invoke() returns, not missed."""
    _ready(client, monkeypatch)
    ks = KillSwitch("MODERATE")
    kill_switch_module._kill_switch = ks

    def _trip_and_return(*args, **kwargs):
        ks._halted.set()
        return {"portfolio_allocation": None, "errors": []}

    fake_graph = mock.Mock()
    fake_graph.invoke.side_effect = _trip_and_return
    monkeypatch.setattr(api_main, "_graph", fake_graph)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    assert "halted" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# API-9: live cache eviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mft_session_callback_evicts_untracked_tickers(monkeypatch):
    """A ticker that dropped out of the tracked universe is dropped from the live cache too."""
    fake_pipeline = mock.Mock()
    fake_pipeline.tickers = ["AAPL"]
    monkeypatch.setattr(api_main, "_mft_pipeline", fake_pipeline)
    api_main._live_session_cache["MSFT"] = ({"timestamp": "stale"}, datetime.now())

    await api_main._mft_session_callback({"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}})

    assert "MSFT" not in api_main._live_session_cache
    assert "AAPL" in api_main._live_session_cache
