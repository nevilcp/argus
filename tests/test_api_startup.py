"""
tests/test_api_startup.py

Tests for api/main.py's PR3 boot-time assertions: the single-worker invariant
(GOV-8) and the configured-model registration check (GOV-11). Both run pure
sys.argv/settings checks, so they're exercised directly rather than through
the app's lifespan.

Also covers PR5b's API-2 additions: `_assert_single_worker`'s extension to
`--workers=N` and `WEB_CONCURRENCY`, and `_acquire_process_lock`'s real
cross-process guard (an exclusive flock on `${ARGUS_DATA_DIR}/argus.lock`).

Also covers PR 1 of the release-remediation issue: `_configure_logging`
(OBS-1) and `/health`'s background-task liveness reporting (OBS-2).
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
from datetime import datetime
from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from argus.risk.kill_switch import KillSwitch


def test_assert_single_worker_allows_default_argv(monkeypatch):
    """No --workers flag at all (uvicorn's own default of 1) passes."""
    monkeypatch.setattr(api_main.sys, "argv", ["uvicorn", "api.main:app"])
    api_main._assert_single_worker()


def test_assert_single_worker_allows_explicit_one(monkeypatch):
    """--workers 1 passes, matching Dockerfile.api's CMD."""
    monkeypatch.setattr(api_main.sys, "argv", ["uvicorn", "api.main:app", "--workers", "1"])
    api_main._assert_single_worker()


def test_assert_single_worker_rejects_multiple(monkeypatch):
    """--workers N for N != 1 raises rather than silently running two governor singletons."""
    monkeypatch.setattr(api_main.sys, "argv", ["uvicorn", "api.main:app", "--workers", "4"])
    with pytest.raises(RuntimeError):
        api_main._assert_single_worker()


def test_assert_registered_models_passes_for_defaults(monkeypatch):
    """The default ARGUS_*_MODEL settings are all registered rate-limit profiles."""
    api_main._assert_registered_models()


def test_assert_registered_models_rejects_unregistered_model(monkeypatch):
    """A typo'd or unsupported model in settings fails at boot rather than at first call."""
    monkeypatch.setattr(api_main.settings, "ARGUS_SENTIMENT_MODEL", "not-a-real-model")
    with pytest.raises(api_main.UnregisteredModel):
        api_main._assert_registered_models()


def test_assert_single_worker_allows_explicit_one_equals_form(monkeypatch):
    """--workers=1 passes, the equals-sign spelling of the space-separated form."""
    monkeypatch.setattr(api_main.sys, "argv", ["uvicorn", "api.main:app", "--workers=1"])
    api_main._assert_single_worker()


def test_assert_single_worker_rejects_multiple_equals_form(monkeypatch):
    """--workers=4 raises just like the space-separated --workers 4 does."""
    monkeypatch.setattr(api_main.sys, "argv", ["uvicorn", "api.main:app", "--workers=4"])
    with pytest.raises(RuntimeError):
        api_main._assert_single_worker()


def test_assert_single_worker_allows_web_concurrency_one(monkeypatch):
    """WEB_CONCURRENCY=1 in the environment passes."""
    monkeypatch.setattr(api_main.sys, "argv", ["uvicorn", "api.main:app"])
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    api_main._assert_single_worker()


def test_assert_single_worker_rejects_web_concurrency_above_one(monkeypatch):
    """WEB_CONCURRENCY=4 is a fourth bypass of the single-worker invariant argv alone can't see."""
    monkeypatch.setattr(api_main.sys, "argv", ["uvicorn", "api.main:app"])
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with pytest.raises(RuntimeError):
        api_main._assert_single_worker()


def test_acquire_process_lock_succeeds_and_creates_the_lock_file(monkeypatch, tmp_path):
    """A clean data directory gets an exclusive lock file created under it."""
    monkeypatch.setattr(api_main.settings, "ARGUS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api_main, "_process_lock_file", None)

    api_main._acquire_process_lock()

    assert (tmp_path / "argus.lock").exists()
    assert api_main._process_lock_file is not None
    fcntl.flock(api_main._process_lock_file, fcntl.LOCK_UN)
    api_main._process_lock_file.close()
    api_main._process_lock_file = None


def test_acquire_process_lock_rejects_a_second_holder(monkeypatch, tmp_path):
    """A second process (simulated: a second fd) can't also acquire the same lock file."""
    monkeypatch.setattr(api_main.settings, "ARGUS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(api_main, "_process_lock_file", None)

    lock_path = tmp_path / "argus.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    other_fd = open(lock_path, "w")
    fcntl.flock(other_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        with pytest.raises(RuntimeError):
            api_main._acquire_process_lock()
    finally:
        fcntl.flock(other_fd, fcntl.LOCK_UN)
        other_fd.close()
        api_main._process_lock_file = None


@pytest.fixture(autouse=True)
def _reset_argus_logger():
    """Clears the "argus" logger's handlers around each test in this module."""
    argus_logger = logging.getLogger("argus")
    original_handlers = list(argus_logger.handlers)
    original_level = argus_logger.level
    original_propagate = argus_logger.propagate
    argus_logger.handlers = []
    yield
    argus_logger.handlers = original_handlers
    argus_logger.level = original_level
    argus_logger.propagate = original_propagate


def test_configure_logging_attaches_a_handler_at_the_configured_level(monkeypatch):
    """_configure_logging (OBS-1) makes ARGUS_LOG_LEVEL take effect on the argus logger tree."""
    monkeypatch.setattr(api_main.settings, "ARGUS_LOG_LEVEL", "DEBUG")

    api_main._configure_logging()

    argus_logger = logging.getLogger("argus")
    assert len(argus_logger.handlers) == 1
    assert argus_logger.level == logging.DEBUG
    assert argus_logger.propagate is False


def test_configure_logging_is_idempotent():
    """Calling _configure_logging twice (e.g. across repeated lifespans) doesn't stack handlers."""
    api_main._configure_logging()
    api_main._configure_logging()

    assert len(logging.getLogger("argus").handlers) == 1


@pytest.fixture
def health_client():
    """A TestClient over api.main.app without running its lifespan startup."""
    return TestClient(api_main.app)


@pytest.fixture(autouse=True)
def _reset_health_globals(monkeypatch):
    """Clears the module-level state /health reads, so tests don't leak into each other."""
    monkeypatch.setattr(api_main, "_pipeline_task", None)
    monkeypatch.setattr(api_main, "_collector_task", None)
    monkeypatch.setattr(api_main, "_reconcile_task", None)
    monkeypatch.setattr(api_main, "_mft_pipeline", None)
    monkeypatch.setattr(api_main, "_live_session_cache", {})
    kill_switch_module._kill_switch = None
    yield
    kill_switch_module._kill_switch = None


def test_health_reports_ok_when_no_background_tasks_are_configured(health_client):
    """Loops that were never started (e.g. collector/reconcile disabled) aren't a failure (OBS-2)."""
    response = health_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["background_tasks"] == {
        "mft_pipeline": {"enabled": False, "alive": True},
        "collector": {"enabled": False, "alive": True},
        "reconcile": {"enabled": False, "alive": True},
    }
    assert body["kill_switch_halted"] is False
    assert body["newest_cache_entry_age_seconds"] is None


@pytest.mark.asyncio
async def test_health_returns_503_naming_a_crashed_background_task(monkeypatch):
    """A task that died with an exception makes /health a 503 naming it (OBS-2)."""

    async def _boom():
        raise RuntimeError("pipeline exploded")

    task = asyncio.create_task(_boom())
    with contextlib.suppress(RuntimeError):
        await task
    monkeypatch.setattr(api_main, "_pipeline_task", task)

    with pytest.raises(HTTPException) as exc_info:
        await api_main.health()

    assert exc_info.value.status_code == 503
    assert "mft_pipeline" in exc_info.value.detail


@pytest.mark.asyncio
async def test_health_returns_503_naming_a_cancelled_background_task(monkeypatch):
    """A task that stopped via cancellation (not just an exception) is also reported dead."""

    async def _spin():
        await asyncio.sleep(10)

    task = asyncio.create_task(_spin())
    await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    monkeypatch.setattr(api_main, "_collector_task", task)

    with pytest.raises(HTTPException) as exc_info:
        await api_main.health()

    assert exc_info.value.status_code == 503
    assert "collector" in exc_info.value.detail


def test_health_reports_kill_switch_halted_state(health_client):
    """kill_switch_halted mirrors the KillSwitch singleton's is_halted flag."""
    ks = KillSwitch("MODERATE")
    ks._portfolio_inception_value = 100_000.0
    ks._current_portfolio_value = 50_000.0
    ks._high_water_mark = 100_000.0
    with mock.patch("argus.data.fetchers.fetch_vix", return_value=18.0):
        ks._check()
    kill_switch_module._kill_switch = ks

    response = health_client.get("/health")

    assert response.json()["kill_switch_halted"] is True


def test_health_reports_newest_cache_entry_age_during_market_hours(monkeypatch, health_client):
    """newest_cache_entry_age_seconds surfaces the freshest live-cache entry's age."""
    fake_pipeline = mock.Mock()
    fake_pipeline.is_market_hours.return_value = True
    monkeypatch.setattr(api_main, "_mft_pipeline", fake_pipeline)
    monkeypatch.setattr(
        api_main,
        "_live_session_cache",
        {"AAPL": ({}, datetime.now())},
    )

    response = health_client.get("/health")

    age = response.json()["newest_cache_entry_age_seconds"]
    assert age is not None
    assert age < 5
