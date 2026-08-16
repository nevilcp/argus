"""
tests/test_api_freshness.py

Tests for /analyze's freshness gate added in PR2 (MFT-1, API-5): the gate
must reject on bar age, not just cache-write age, and must check market
hours before either age check so bar-age is never evaluated outside a
weekday 09:30-16:00 ET session.

These exercise the route function directly via FastAPI's TestClient rather
than going through the app's lifespan, so no MFT pipeline, collector, or
reconcile loop needs to start. `_mft_pipeline` and `_live_session_cache`
are set directly on the api.main module instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module

_PAYLOAD = {"tickers": ["AAPL"], "total_wealth": 100_000, "invest_pct": 0.5}


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Clears the kill-switch singleton and live session cache around each test."""
    kill_switch_module._kill_switch = None
    monkeypatch.setattr(api_main, "_live_session_cache", {})
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")
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


def test_analyze_rejects_when_market_closed(client, monkeypatch):
    """Market-closed is checked before any cache lookup, regardless of what's cached."""
    _pipeline(monkeypatch, market_hours=False)
    response = client.post("/analyze", json=_PAYLOAD)
    assert response.status_code == 503
    assert "closed" in response.json()["detail"].lower()


def test_analyze_rejects_absent_ticker_as_warming_up(client, monkeypatch):
    """A ticker never seen by the pipeline is reported as warming up, not stalled or stale."""
    _pipeline(monkeypatch, market_hours=True)
    response = client.post("/analyze", json=_PAYLOAD)
    assert response.status_code == 503
    assert "warming up" in response.json()["detail"].lower()


def test_analyze_rejects_stalled_write_before_checking_bar_age(client, monkeypatch):
    """A cache entry that stopped being refreshed is reported as stalled, even with a fresh bar."""
    _pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=5, write_age_seconds=10_000)
    response = client.post("/analyze", json=_PAYLOAD)
    assert response.status_code == 503
    assert "stalled" in response.json()["detail"].lower()


def test_analyze_rejects_stale_bar_reproduced_case(client, monkeypatch):
    """The reproduced MFT-1 case: a bar 9 days old, written just now, is rejected as stale."""
    _pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=9 * 24 * 3600, write_age_seconds=5)
    response = client.post("/analyze", json=_PAYLOAD)
    assert response.status_code == 503
    assert "stale" in response.json()["detail"].lower()


def test_analyze_rejects_stale_bar_on_a_weekday_holiday(client, monkeypatch):
    """is_market_hours only checks weekday, so a holiday's stale republished bar must fail-close."""
    _pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=2 * 24 * 3600, write_age_seconds=5)
    response = client.post("/analyze", json=_PAYLOAD)
    assert response.status_code == 503
    assert "stale" in response.json()["detail"].lower()


def test_analyze_market_closed_outranks_bar_age(client, monkeypatch):
    """A hopelessly stale bar still surfaces as market-closed, not stale, outside session hours."""
    _pipeline(monkeypatch, market_hours=False)
    _seed_cache("AAPL", bar_age_seconds=9 * 24 * 3600, write_age_seconds=9 * 24 * 3600)
    response = client.post("/analyze", json=_PAYLOAD)
    assert response.status_code == 503
    assert "closed" in response.json()["detail"].lower()


def test_analyze_passes_freshness_gate_with_a_current_bar(client, monkeypatch):
    """A fresh write and a fresh bar clear every freshness check and reach the graph."""
    _pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=5, write_age_seconds=5)
    fake_graph = mock.Mock()
    fake_graph.invoke.side_effect = RuntimeError("sentinel: freshness gate cleared")
    monkeypatch.setattr(api_main, "_graph", fake_graph)

    response = client.post("/analyze", json=_PAYLOAD)

    # A 500 (rather than any of the 503 freshness variants) proves every gate
    # above was cleared and the graph itself was reached; the exception text
    # is redacted behind a correlation ref (API-7, see test_api_validation.py)
    # so it isn't asserted here.
    assert response.status_code == 500
    assert fake_graph.invoke.called
