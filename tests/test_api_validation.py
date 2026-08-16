"""
tests/test_api_validation.py

Tests for PR5a: AnalysisRequest's ticker validation, normalization, and
dedupe (API-1, API-6 gateway half), and /analyze's redaction of internal
exception details behind a correlation ref (API-7).

These exercise the route function directly via FastAPI's TestClient rather
than going through the app's lifespan, so no MFT pipeline, collector, or
reconcile loop needs to start.
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


@pytest.mark.parametrize(
    "ticker",
    [
        "",
        "A" * 500,
        "'; DROP TABLE ohlcv; --",
        "../../etc/passwd",
        "AAPLLL",
        ".B",
        "AAPL.",
        "^GSPC",
        "BTC-USD",
    ],
)
def test_analyze_rejects_malformed_ticker_with_422(client, ticker):
    """A hostile or malformed ticker is rejected by pydantic before any handler runs."""
    response = client.post(
        "/analyze",
        json={"tickers": [ticker], "total_wealth": 100_000, "invest_pct": 0.5},
    )
    assert response.status_code == 422


def test_analyze_accepts_share_class_tickers(client, monkeypatch):
    """Dotted and hyphenated share-class symbols pass validation and reach the pipeline."""
    fake_pipeline = _pipeline(monkeypatch, market_hours=False)
    response = client.post(
        "/analyze",
        json={"tickers": ["BRK.B", "BRK-B"], "total_wealth": 100_000, "invest_pct": 0.5},
    )
    assert response.status_code == 503
    fake_pipeline.register_tickers.assert_called_once_with(["BRK.B", "BRK-B"])


def test_analyze_upcases_and_dedupes_tickers(client, monkeypatch):
    """Lowercase, whitespace-padded, and repeated tickers collapse to one upper-cased entry each."""
    fake_pipeline = _pipeline(monkeypatch, market_hours=False)
    response = client.post(
        "/analyze",
        json={"tickers": [" aapl ", "AAPL", "msft"], "total_wealth": 100_000, "invest_pct": 0.5},
    )
    assert response.status_code == 503
    fake_pipeline.register_tickers.assert_called_once_with(["AAPL", "MSFT"])


def test_analyze_redacts_internal_exception_details(client, monkeypatch):
    """A raw graph exception never reaches the client; only a correlation ref does (API-7)."""
    _pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=5, write_age_seconds=5)
    fake_graph = mock.Mock()
    fake_graph.invoke.side_effect = RuntimeError("api_key=sk-secret123 at /etc/passwd")
    monkeypatch.setattr(api_main, "_graph", fake_graph)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "sk-secret123" not in detail
    assert "/etc/passwd" not in detail
    assert detail.startswith("Agent graph error (ref ")
