"""
tests/test_api_freshness.py

Route-level tests for /analyze's freshness gate (issue #78). The four
freshness clauses themselves — absent/stalled/stale/malformed/naive/
one-full-interval-old — now live at the LiveSessionCache seam (see
tests/test_live_session_cache.py) and are verified there with an explicit
clock, no HTTP client required.

What's genuinely a route concern and stays here: market hours outranking
every age check, each admission group mapping to its own response text, and
ticker registration happening only after the earlier gates pass — none of
that is LiveSessionCache's to know about.

These exercise the route function directly via FastAPI's TestClient rather
than going through the app's lifespan, so no MFT pipeline, collector, or
reconcile loop needs to start. `_mft_pipeline` and `_live_cache` are set
directly on the api.main module instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from argus.data.live_session_cache import AdmissionResult, LiveSessionCache

_PAYLOAD = {"tickers": ["AAPL"], "total_wealth": 100_000, "invest_pct": 0.5}


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Clears the kill-switch singleton and installs a fresh live session cache around each test."""
    kill_switch_module._kill_switch = None
    monkeypatch.setattr(api_main, "_live_cache", LiveSessionCache(interval_minutes=1))
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


def test_analyze_market_closed_outranks_bar_age(client, monkeypatch):
    """A hopelessly stale bar still surfaces as market-closed, not stale, outside session hours (issue #78)."""
    _pipeline(monkeypatch, market_hours=False)
    now = datetime.now(api_main._ET) - timedelta(days=9)
    api_main._live_cache.publish({"AAPL": {"timestamp": now.isoformat()}}, ["AAPL"], now=now)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    assert "closed" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    "admission_field, expected_substring",
    [
        ("absent", "warming up"),
        ("stalled", "stalled"),
        ("stale", "stale"),
    ],
)
def test_analyze_maps_each_admission_group_to_its_own_response_text(
    client, monkeypatch, admission_field, expected_substring
):
    """Each of LiveSessionCache.admit()'s three rejection groups produces distinct response wording."""
    _pipeline(monkeypatch, market_hours=True)
    result = AdmissionResult(**{admission_field: ["AAPL"]})
    monkeypatch.setattr(api_main._live_cache, "admit", mock.Mock(return_value=result))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    assert expected_substring in response.json()["detail"].lower()


def test_analyze_market_closed_skips_ticker_registration(client, monkeypatch):
    """A rejected-by-market-hours request must not mutate pipeline state (API-12)."""
    fake_pipeline = _pipeline(monkeypatch, market_hours=False)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    fake_pipeline.register_tickers.assert_not_called()
