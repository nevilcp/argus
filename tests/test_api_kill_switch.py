"""
tests/test_api_kill_switch.py

Tests for the kill-switch-facing surface added to api/main.py in PR2: the
ARGUS_API_KEY auth dependency (KS-9), GET /kill-switch/status (KS-8), and
POST /kill-switch/reset's input validation (KS-9).

These exercise the route functions directly via FastAPI's TestClient rather
than going through the app's lifespan, so no MFT pipeline, collector, or
reconcile loop needs to start.
"""

from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from argus.risk.kill_switch import KillSwitch


@pytest.fixture(autouse=True)
def _reset_kill_switch_singleton():
    """Clears the module-level KillSwitch singleton before and after each test."""
    kill_switch_module._kill_switch = None
    yield
    kill_switch_module._kill_switch = None


@pytest.fixture
def client():
    """A TestClient over api.main.app without running its lifespan startup."""
    return TestClient(api_main.app)


def test_analyze_rejects_without_api_key_when_configured(client, monkeypatch):
    """require_api_key blocks /analyze if ARGUS_API_KEY is set and no header is sent."""
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "secret123")
    response = client.post(
        "/analyze",
        json={"tickers": ["AAPL"], "total_wealth": 100_000, "invest_pct": 0.5},
    )
    assert response.status_code == 401


def test_analyze_allows_without_api_key_when_unset(client, monkeypatch):
    """A blank ARGUS_API_KEY (the default) disables the auth check entirely."""
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")
    # No kill switch, no pipeline: falls through to the 503 pipeline-not-initialized
    # path rather than a 401, proving auth wasn't the blocker
    response = client.post(
        "/analyze",
        json={"tickers": ["AAPL"], "total_wealth": 100_000, "invest_pct": 0.5},
    )
    assert response.status_code != 401


def test_kill_switch_reset_requires_api_key_when_configured(client, monkeypatch):
    """require_api_key also gates /kill-switch/reset."""
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "secret123")
    response = client.post("/kill-switch/reset", params={"new_inception_value": 50_000})
    assert response.status_code == 401


def test_kill_switch_reset_accepts_matching_api_key(client, monkeypatch):
    """The correct X-API-Key header clears the auth gate."""
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "secret123")
    kill_switch_module._kill_switch = KillSwitch("MODERATE")
    response = client.post(
        "/kill-switch/reset",
        params={"new_inception_value": 50_000},
        headers={"X-API-Key": "secret123"},
    )
    assert response.status_code == 200


def test_kill_switch_reset_rejects_value_at_or_below_floor(client, monkeypatch):
    """KS-9: new_inception_value must exceed 1000, matching AnalysisRequest.total_wealth."""
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")
    kill_switch_module._kill_switch = KillSwitch("MODERATE")
    response = client.post("/kill-switch/reset", params={"new_inception_value": 500})
    assert response.status_code == 422


def test_kill_switch_status_404_when_uninitialized(client):
    """GET /kill-switch/status reports 404 before the singleton exists."""
    response = client.get("/kill-switch/status")
    assert response.status_code == 404


def test_kill_switch_status_reports_gate_state(client):
    """GET /kill-switch/status surfaces the halt state and last-observed VIX (KS-8)."""
    ks = KillSwitch("MODERATE")
    ks._portfolio_inception_value = 100_000.0
    ks._current_portfolio_value = 100_000.0
    ks._high_water_mark = 100_000.0
    with mock.patch("argus.data.fetchers.fetch_vix", return_value=18.0):
        ks._check()
    kill_switch_module._kill_switch = ks

    response = client.get("/kill-switch/status")

    assert response.status_code == 200
    body = response.json()
    assert body["halted"] is False
    assert body["current_vix"] == pytest.approx(18.0)
    assert body["risk_tolerance"] == "MODERATE"
