"""Route-level tests for /analyze's freshness gate."""

from __future__ import annotations

from datetime import timedelta
from unittest import mock

import pytest

from api.lifespan import runtime
from argus.data.live_session_cache import AdmissionResult
from argus.infra.clock import now_market
from tests.helpers.api import api_pipeline

_PAYLOAD = {"tickers": ["AAPL"], "total_wealth": 100_000, "invest_pct": 0.5}

pytestmark = pytest.mark.usefixtures("api_state")


def test_analyze_market_closed_outranks_bar_age(client, monkeypatch):
    api_pipeline(monkeypatch, market_hours=False)
    now = now_market() - timedelta(days=9)
    runtime.live_cache.publish({"AAPL": {"timestamp": now.isoformat()}}, ["AAPL"], now=now)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    assert "closed" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    ("admission_field", "expected_substring"),
    [
        ("absent", "warming up"),
        ("stalled", "stalled"),
        ("stale", "stale"),
    ],
)
def test_analyze_maps_each_admission_group_to_its_own_response_text(
    client, monkeypatch, admission_field, expected_substring
):
    api_pipeline(monkeypatch, market_hours=True)
    result = AdmissionResult(**{admission_field: ["AAPL"]})
    monkeypatch.setattr(runtime.live_cache, "admit", mock.Mock(return_value=result))

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    assert expected_substring in response.json()["detail"].lower()


def test_analyze_market_closed_skips_ticker_registration(client, monkeypatch):
    fake_pipeline = api_pipeline(monkeypatch, market_hours=False)

    response = client.post("/analyze", json=_PAYLOAD)

    assert response.status_code == 503
    fake_pipeline.register_tickers.assert_not_called()
