"""Tests for AnalysisRequest's ticker validation and /analyze's error redaction.

Covers ticker validation, normalization, and dedupe on AnalysisRequest, and
/analyze's redaction of internal exception details behind a correlation ref.

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


def _payload(*tickers: str) -> dict:
    """An /analyze request body that varies only in the tickers it asks for."""
    return {"tickers": list(tickers), "total_wealth": 100_000, "invest_pct": 0.5}


@pytest.fixture(autouse=True)
def _reset_singletons(_fresh_live_cache, _no_api_key):
    """Kill switch, live cache, and API key are all reset around each test via conftest fixtures."""


@pytest.fixture
def client():
    """A TestClient over api.main.app without running its lifespan startup."""
    return TestClient(api_main.app)


def _pipeline(monkeypatch, *, market_hours: bool):
    """Installs a fake MFTDataPipeline as the module-level `_mft_pipeline`."""
    fake = mock.Mock()
    fake.is_market_hours.return_value = market_hours
    fake.interval_minutes = 1
    monkeypatch.setattr(api_main, "_mft_pipeline", fake)
    return fake


def _seed_cache(ticker: str, *, bar_age_seconds: float, write_age_seconds: float = 0.0):
    """Populates `_live_cache` with a state whose bar and write times are both controlled."""
    now = datetime.now(api_main._ET)
    bar_ts = now - timedelta(seconds=bar_age_seconds)
    write_ts = now - timedelta(seconds=write_age_seconds)
    api_main._live_cache.publish(
        {ticker: {"timestamp": bar_ts.isoformat()}}, [ticker], now=write_ts
    )


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
    response = client.post("/analyze", json=_payload(ticker))
    assert response.status_code == 422


def test_analyze_accepts_share_class_tickers(client, monkeypatch):
    """Dotted/hyphenated share-class tickers validate and reach the pipeline.

    Market is open here (rather than closed) because register_tickers now only
    runs once the market-hours gate clears — a rejected request must not
    mutate pipeline state.
    """
    fake_pipeline = _pipeline(monkeypatch, market_hours=True)
    response = client.post("/analyze", json=_payload("BRK.B", "BRK-B"))
    assert response.status_code == 503
    fake_pipeline.register_tickers.assert_called_once_with(["BRK.B", "BRK-B"])


def test_analyze_upcases_and_dedupes_tickers(client, monkeypatch):
    """Lowercase, whitespace-padded, and repeated tickers collapse to one upper-cased entry each."""
    fake_pipeline = _pipeline(monkeypatch, market_hours=True)
    response = client.post("/analyze", json=_payload(" aapl ", "AAPL", "msft"))
    assert response.status_code == 503
    fake_pipeline.register_tickers.assert_called_once_with(["AAPL", "MSFT"])


def test_analyze_reports_a_missing_allocation_as_503_not_500(client, monkeypatch):
    """A degraded (no-allocation) graph run is an upstream LLM outage, not a server bug."""
    _pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=5, write_age_seconds=5)
    fake_graph = mock.Mock()
    fake_graph.invoke.return_value = {"decisions": [], "portfolio_allocation": None}
    monkeypatch.setattr(api_main, "_graph", fake_graph)

    response = client.post("/analyze", json=_payload("AAPL"))

    assert response.status_code == 503
    assert response.json()["detail"] == "Portfolio allocation failed."


def test_analyze_redacts_internal_exception_details(client, monkeypatch):
    """A raw graph exception never reaches the client; only a correlation ref does."""
    _pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=5, write_age_seconds=5)
    fake_graph = mock.Mock()
    fake_graph.invoke.side_effect = RuntimeError("api_key=sk-secret123 at /etc/passwd")
    monkeypatch.setattr(api_main, "_graph", fake_graph)

    response = client.post("/analyze", json=_payload("AAPL"))

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "sk-secret123" not in detail
    assert "/etc/passwd" not in detail
    assert detail.startswith("Agent graph error (ref ")
