"""Tests for AnalysisRequest's ticker validation and /analyze's error redaction."""

from __future__ import annotations

from datetime import timedelta
from unittest import mock

import pytest

from api.lifespan import runtime
from argus.infra.clock import now_market
from tests.helpers.api import api_pipeline

pytestmark = pytest.mark.usefixtures("api_state")


def _payload(*tickers: str) -> dict:
    """Build an analysis request for the specified tickers."""
    return {"tickers": list(tickers), "total_wealth": 100_000, "invest_pct": 0.5}


def _seed_cache(ticker: str, *, bar_age_seconds: float, write_age_seconds: float = 0.0):
    """Populate `_live_cache` with a state whose bar and write times are both controlled."""
    now = now_market()
    bar_ts = now - timedelta(seconds=bar_age_seconds)
    write_ts = now - timedelta(seconds=write_age_seconds)
    runtime.live_cache.publish({ticker: {"timestamp": bar_ts.isoformat()}}, [ticker], now=write_ts)


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
    response = client.post("/analyze", json=_payload(ticker))
    assert response.status_code == 422


def test_analyze_accepts_share_class_tickers(client, monkeypatch):
    fake_pipeline = api_pipeline(monkeypatch, market_hours=True)
    response = client.post("/analyze", json=_payload("BRK.B", "BRK-B"))
    assert response.status_code == 503
    fake_pipeline.register_tickers.assert_called_once_with(["BRK.B", "BRK-B"])


def test_analyze_upcases_and_dedupes_tickers(client, monkeypatch):
    fake_pipeline = api_pipeline(monkeypatch, market_hours=True)
    response = client.post("/analyze", json=_payload(" aapl ", "AAPL", "msft"))
    assert response.status_code == 503
    fake_pipeline.register_tickers.assert_called_once_with(["AAPL", "MSFT"])


def test_analyze_reports_a_missing_allocation_as_503_not_500(client, monkeypatch):
    api_pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=5, write_age_seconds=5)
    fake_graph = mock.Mock()
    fake_graph.invoke.return_value = {"decisions": [], "portfolio_allocation": None}
    monkeypatch.setattr(runtime, "graph", fake_graph)

    response = client.post("/analyze", json=_payload("AAPL"))

    assert response.status_code == 503
    assert response.json()["detail"] == "Portfolio allocation failed."


def test_analyze_redacts_internal_exception_details(client, monkeypatch):
    api_pipeline(monkeypatch, market_hours=True)
    _seed_cache("AAPL", bar_age_seconds=5, write_age_seconds=5)
    fake_graph = mock.Mock()
    fake_graph.invoke.side_effect = RuntimeError("api_key=sk-secret123 at /etc/passwd")
    monkeypatch.setattr(runtime, "graph", fake_graph)

    response = client.post("/analyze", json=_payload("AAPL"))

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "sk-secret123" not in detail
    assert "/etc/passwd" not in detail
    assert detail.startswith("Agent graph error (ref ")
