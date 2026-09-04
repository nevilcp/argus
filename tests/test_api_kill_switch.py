"""Tests for the kill-switch-facing surface on api/main.py.

Covers the ARGUS_API_KEY auth dependency, GET /kill-switch/status, and
POST /kill-switch/reset's input validation.

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
from argus.risk import paper_book
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


def test_kill_switch_reset_rebases_the_persisted_paper_book(client, monkeypatch, tmp_path):
    """Reset rebases paper_equity.json, not just the in-memory kill switch.

    Otherwise the next reconcile pass would feed the stale, drawn-down equity
    back in and re-halt within check_interval_seconds.
    """
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")
    kill_switch_module._kill_switch = KillSwitch("MODERATE")
    book_path = tmp_path / "paper_equity.json"  # ARGUS_DATA_DIR is this tmp_path (conftest)
    drawn_down = paper_book.PaperBook(equity=80_000.0, high_water_mark=100_000.0)
    paper_book.save(drawn_down, str(book_path))

    response = client.post("/kill-switch/reset", params={"new_inception_value": 150_000})

    assert response.status_code == 200
    rebased = paper_book.load(str(book_path))
    assert rebased.equity == pytest.approx(150_000.0)
    assert rebased.high_water_mark == pytest.approx(150_000.0)
    assert rebased.rebased_at is not None


def test_kill_switch_reset_refuses_to_rebase_over_a_corrupt_paper_book(
    client, monkeypatch, tmp_path
):
    """A corrupt paper_equity.json fails the reset with a clear 500, not a silent fresh rebase.

    Falling back to a fresh book here would drop the old file's runs_applied,
    letting an already-applied run get double-compounded by the next
    reconcile pass — the same hazard load()'s hard-failure behavior guards
    against elsewhere.
    """
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")
    kill_switch_module._kill_switch = KillSwitch("MODERATE")
    book_path = tmp_path / "paper_equity.json"  # ARGUS_DATA_DIR is this tmp_path (conftest)
    book_path.write_text('{"equity": 80000.0, "high_wat')

    response = client.post("/kill-switch/reset", params={"new_inception_value": 150_000})

    assert response.status_code == 500
    assert book_path.read_text() == '{"equity": 80000.0, "high_wat'


def test_kill_switch_reset_rejects_value_at_or_below_floor(client, monkeypatch):
    """new_inception_value must exceed 1000, matching AnalysisRequest.total_wealth."""
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")
    kill_switch_module._kill_switch = KillSwitch("MODERATE")
    response = client.post("/kill-switch/reset", params={"new_inception_value": 500})
    assert response.status_code == 422


def test_kill_switch_status_404_when_uninitialized(client):
    """GET /kill-switch/status reports 404 before the singleton exists."""
    response = client.get("/kill-switch/status")
    assert response.status_code == 404


def test_kill_switch_status_reports_gate_state(client):
    """GET /kill-switch/status surfaces the halt state and last-observed VIX."""
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
