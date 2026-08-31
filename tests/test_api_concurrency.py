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
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from argus.data.live_session_cache import LiveSessionCache
from argus.risk.kill_switch import KillSwitch
from argus.schemas.signals import ARGUSDecision

_PAYLOAD = {"tickers": ["AAPL"], "total_wealth": 100_000, "invest_pct": 0.5}


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Clears the kill-switch singleton and installs a fresh live session cache around each test."""
    kill_switch_module._kill_switch = None
    monkeypatch.setattr(api_main, "_live_cache", LiveSessionCache(interval_minutes=1))
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")
    monkeypatch.setattr(api_main, "_analyze_semaphore", asyncio.Semaphore(1))
    yield
    kill_switch_module._kill_switch = None


@pytest.fixture
def client():
    """A TestClient over api.main.app without running its lifespan startup."""
    return TestClient(api_main.app)


def _ready(monkeypatch):
    """Clears every freshness gate ahead of /analyze so a request reaches the graph.

    Installs an in-market-hours pipeline and publishes an AAPL state whose bar
    and write times are both recent enough to clear the staleness gates.
    """
    fake_pipeline = mock.Mock(interval_minutes=1)
    fake_pipeline.is_market_hours.return_value = True
    monkeypatch.setattr(api_main, "_mft_pipeline", fake_pipeline)

    recent = datetime.now(api_main._ET) - timedelta(seconds=5)
    api_main._live_cache.publish({"AAPL": {"timestamp": recent.isoformat()}}, ["AAPL"], now=recent)


def _install_graph(monkeypatch, **invoke_behavior):
    """Installs a fake graph as the module-level `_graph`, configuring its invoke() call.

    Args:
        invoke_behavior: Mock keyword arguments for `invoke` (`side_effect`, `return_value`).

    Returns:
        The installed fake graph.
    """
    fake_graph = mock.Mock()
    fake_graph.invoke.configure_mock(**invoke_behavior)
    monkeypatch.setattr(api_main, "_graph", fake_graph)
    return fake_graph


def _graph_result(decisions: list) -> dict:
    """A successful graph return value carrying `decisions` and a minimal allocation."""
    allocation = mock.Mock(
        session_id="sess-1", portfolio=[], cash_reserve_pct=0.5, expected_sharpe=1.0
    )
    return {
        "decisions": decisions,
        "errors": [],
        "portfolio_allocation": allocation,
        "macro_context": None,
    }


# ---------------------------------------------------------------------------
# API-3: semaphore
# ---------------------------------------------------------------------------


def test_analyze_rejects_a_concurrent_run_with_429(client, monkeypatch):
    """A held semaphore is reported as 429 with a Retry-After header, not run against the graph."""
    _ready(monkeypatch)
    fake_graph = _install_graph(
        monkeypatch,
        side_effect=AssertionError("must not run while another analysis holds the slot"),
    )

    held = mock.Mock()
    held.locked.return_value = True
    monkeypatch.setattr(api_main, "_analyze_semaphore", held)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 429
    assert response.headers["retry-after"]
    fake_graph.invoke.assert_not_called()


def test_analyze_releases_the_semaphore_when_the_graph_raises(client, monkeypatch):
    """The slot is released on a graph exception, not just on success — easy to wedge permanently."""
    _ready(monkeypatch)
    _install_graph(monkeypatch, side_effect=RuntimeError("boom"))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 500
    assert not api_main._analyze_semaphore.locked()


# ---------------------------------------------------------------------------
# GOV-12: request deadline
# ---------------------------------------------------------------------------


def test_analyze_returns_504_when_the_graph_exceeds_its_deadline(client, monkeypatch):
    """A graph run that outlives ARGUS_ANALYZE_DEADLINE_SECONDS is bounded with a
    504, not left open past whatever timeout the caller's own proxy already gave up at."""
    _ready(monkeypatch)
    monkeypatch.setattr(api_main.settings, "ARGUS_ANALYZE_DEADLINE_SECONDS", 0.05)
    _install_graph(monkeypatch, side_effect=lambda *a, **kw: time.sleep(0.5))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 504
    assert not api_main._analyze_semaphore.locked()


# ---------------------------------------------------------------------------
# Graph-not-initialized guard
# ---------------------------------------------------------------------------


def test_analyze_rejects_when_graph_not_yet_initialized(client, monkeypatch):
    """A None _graph (before lifespan startup finishes) is a 503, not an AttributeError."""
    _ready(monkeypatch)
    monkeypatch.setattr(api_main, "_graph", None)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    assert "graph" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# API-8: kill switch re-checked after the graph returns
# ---------------------------------------------------------------------------


def test_analyze_rejects_when_kill_switch_trips_during_the_graph_run(client, monkeypatch):
    """A halt that lands while the graph is running is caught after invoke() returns, not missed."""
    _ready(monkeypatch)
    ks = KillSwitch("MODERATE")
    kill_switch_module._kill_switch = ks

    def _trip_and_return(*args, **kwargs):
        ks._halted.set()
        return {"portfolio_allocation": None, "errors": []}

    _install_graph(monkeypatch, side_effect=_trip_and_return)

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
    api_main._live_cache.publish({"MSFT": {"timestamp": "stale"}}, ["MSFT"])

    await api_main._mft_session_callback({"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}})

    result = api_main._live_cache.admit(["MSFT", "AAPL"], datetime.now(api_main._ET))
    assert "MSFT" in result.absent
    assert "AAPL" not in result.absent


# ---------------------------------------------------------------------------
# RE-11: /analyze decisions reach decisions.jsonl
# ---------------------------------------------------------------------------


def test_analyze_appends_decisions_to_decisions_jsonl(client, monkeypatch):
    """A successful /analyze run appends its decisions to the log the collector also writes to.

    Without this, decisions.jsonl — the only source reconcile_decisions
    reads — never sees a decision made through /analyze, and an API-only
    deployment reconciles nothing.
    """
    _ready(monkeypatch)
    decision = ARGUSDecision(ticker="AAPL", session_timestamp=datetime.now())
    _install_graph(monkeypatch, return_value=_graph_result([decision]))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 200
    log_path = Path(api_main.settings.ARGUS_DATA_DIR) / "decisions.jsonl"
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    assert ARGUSDecision.model_validate_json(lines[0]).decision_id == decision.decision_id


def test_analyze_writes_nothing_when_the_graph_produces_no_decisions(client, monkeypatch):
    """A run with an empty decisions list leaves decisions.jsonl unwritten, not an empty file."""
    _ready(monkeypatch)
    _install_graph(monkeypatch, return_value=_graph_result([]))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 200
    log_path = Path(api_main.settings.ARGUS_DATA_DIR) / "decisions.jsonl"
    assert not log_path.exists()
