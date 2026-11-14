"""API concurrency, deadline, safety recheck, and cache eviction regressions."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from api import lifespan
from api.lifespan import runtime
from argus.infra.clock import now_market
from argus.risk.kill_switch import KillSwitch
from argus.schemas.signals import ARGUSDecision
from tests.helpers.api import api_pipeline
from tests.helpers.builders import position_allocation

pytestmark = pytest.mark.usefixtures("api_state")

_PAYLOAD = {"tickers": ["AAPL"], "total_wealth": 100_000, "invest_pct": 0.5}


@pytest.fixture(autouse=True)
def _reset_analyze_semaphore(monkeypatch):
    """Reset the analyze semaphore for each test."""
    monkeypatch.setattr(runtime, "analyze_lock", asyncio.Semaphore(1))


def _ready(monkeypatch):
    """Clear every freshness gate ahead of /analyze so a request reaches the graph."""
    api_pipeline(monkeypatch)

    recent = now_market() - timedelta(seconds=5)
    runtime.live_cache.publish({"AAPL": {"timestamp": recent.isoformat()}}, ["AAPL"], now=recent)


def _install_graph(monkeypatch, **invoke_behavior):
    """Install a fake graph as `runtime.graph` and configure invoke()."""
    fake_graph = mock.Mock()
    fake_graph.invoke.configure_mock(**invoke_behavior)
    monkeypatch.setattr(runtime, "graph", fake_graph)
    return fake_graph


def _graph_result(decisions: list) -> dict:
    """Build a successful graph result with decisions and a minimal allocation."""
    allocation = mock.Mock(session_id="sess-1", portfolio=[], cash_reserve_pct=0.5, expected_sharpe=1.0)
    return {
        "decisions": decisions,
        "errors": [],
        "portfolio_allocation": allocation,
        "macro_context": None,
    }


def test_analyze_rejects_a_concurrent_run_with_429(client, monkeypatch):
    _ready(monkeypatch)
    fake_graph = _install_graph(
        monkeypatch,
        side_effect=AssertionError("must not run while another analysis holds the slot"),
    )

    held = mock.Mock()
    held.locked.return_value = True
    monkeypatch.setattr(runtime, "analyze_lock", held)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 429
    assert response.headers["retry-after"]
    fake_graph.invoke.assert_not_called()


def test_analyze_releases_the_semaphore_when_the_graph_raises(client, monkeypatch):
    _ready(monkeypatch)
    _install_graph(monkeypatch, side_effect=RuntimeError("boom"))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 500
    assert not runtime.analyze_lock.locked()


def test_analyze_returns_504_when_the_graph_exceeds_its_deadline(client, monkeypatch):
    _ready(monkeypatch)
    monkeypatch.setattr(api_main.settings, "ARGUS_ANALYZE_DEADLINE_SECONDS", 0.05)
    _install_graph(monkeypatch, side_effect=lambda *a, **kw: time.sleep(0.5))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 504
    assert not runtime.analyze_lock.locked()


def test_analyze_rejects_when_graph_not_yet_initialized(client, monkeypatch):
    _ready(monkeypatch)
    monkeypatch.setattr(runtime, "graph", None)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    assert "graph" in response.json()["detail"].lower()


def test_analyze_rejects_when_kill_switch_trips_during_the_graph_run(client, monkeypatch):
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


async def test_mft_session_callback_evicts_untracked_tickers(monkeypatch):
    fake_pipeline = mock.Mock()
    fake_pipeline.tickers = ["AAPL"]
    monkeypatch.setattr(runtime, "pipeline", fake_pipeline)
    runtime.live_cache.publish({"MSFT": {"timestamp": "stale"}}, ["MSFT"])

    await lifespan._mft_session_callback({"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}})

    result = runtime.live_cache.admit(["MSFT", "AAPL"], now_market())
    assert "MSFT" in result.absent
    assert "AAPL" not in result.absent


def test_analyze_appends_decisions_to_decisions_jsonl(client, monkeypatch):
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
    _ready(monkeypatch)
    _install_graph(monkeypatch, return_value=_graph_result([]))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 200
    log_path = Path(api_main.settings.ARGUS_DATA_DIR) / "decisions.jsonl"
    assert not log_path.exists()


def test_analyze_returns_each_position_as_its_allocation_model(client, monkeypatch):
    _ready(monkeypatch)
    position = position_allocation("AAPL", 0.1, advisor_note="Hold.")
    result = _graph_result([])
    result["portfolio_allocation"].portfolio = [position]
    _install_graph(monkeypatch, return_value=result)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["portfolio"] == [position.model_dump(mode="json")]
