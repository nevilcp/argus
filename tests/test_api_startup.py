"""Tests for the API's boot-time guards (api/guards.py), startup wiring and liveness reporting."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import fcntl
import logging
from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from api import guards, lifespan
from api.lifespan import runtime
from argus.config import settings
from argus.data.live_session_cache import LiveSessionCache
from argus.governor import UnregisteredModel
from argus.infra.log import LOG_ROOTS, configure_logging
from argus.risk.kill_switch import KillSwitch


def _set_uvicorn_argv(monkeypatch, *flags: str) -> None:
    """Point sys.argv at a uvicorn invocation carrying the given extra flags."""
    monkeypatch.setattr(guards.sys, "argv", ["uvicorn", "api.main:app", *flags])


def test_assert_single_worker_allows_default_argv(monkeypatch):
    _set_uvicorn_argv(monkeypatch)
    guards.assert_single_worker()


def test_assert_single_worker_allows_explicit_one(monkeypatch):
    _set_uvicorn_argv(monkeypatch, "--workers", "1")
    guards.assert_single_worker()


def test_assert_single_worker_rejects_multiple(monkeypatch):
    _set_uvicorn_argv(monkeypatch, "--workers", "4")
    with pytest.raises(RuntimeError):
        guards.assert_single_worker()


def test_assert_registered_models_passes_for_defaults():
    guards.assert_registered_models()


def test_assert_registered_models_rejects_unregistered_model(monkeypatch):
    monkeypatch.setattr(api_main.settings, "ARGUS_SENTIMENT_MODEL", "not-a-real-model")
    with pytest.raises(UnregisteredModel):
        guards.assert_registered_models()


def test_assert_single_worker_allows_explicit_one_equals_form(monkeypatch):
    _set_uvicorn_argv(monkeypatch, "--workers=1")
    guards.assert_single_worker()


def test_assert_single_worker_rejects_multiple_equals_form(monkeypatch):
    _set_uvicorn_argv(monkeypatch, "--workers=4")
    with pytest.raises(RuntimeError):
        guards.assert_single_worker()


def test_assert_single_worker_allows_web_concurrency_one(monkeypatch):
    _set_uvicorn_argv(monkeypatch)
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    guards.assert_single_worker()


def test_assert_single_worker_rejects_web_concurrency_above_one(monkeypatch):
    _set_uvicorn_argv(monkeypatch)
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with pytest.raises(RuntimeError):
        guards.assert_single_worker()


def test_acquire_process_lock_succeeds_and_creates_the_lock_file(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main.settings, "ARGUS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(guards, "_process_lock_file", None)

    guards.acquire_process_lock()

    assert (tmp_path / "argus.lock").exists()
    assert guards._process_lock_file is not None
    fcntl.flock(guards._process_lock_file, fcntl.LOCK_UN)
    guards._process_lock_file.close()
    guards._process_lock_file = None


def test_acquire_process_lock_rejects_a_second_holder(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main.settings, "ARGUS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(guards, "_process_lock_file", None)

    lock_path = tmp_path / "argus.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as other_fd:
        fcntl.flock(other_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            with pytest.raises(RuntimeError):
                guards.acquire_process_lock()
        finally:
            fcntl.flock(other_fd, fcntl.LOCK_UN)
            guards._process_lock_file = None


@pytest.mark.parametrize("root", LOG_ROOTS)
def test_configure_logging_attaches_a_handler_at_the_configured_level(monkeypatch, root):
    monkeypatch.setattr(settings, "ARGUS_LOG_LEVEL", "DEBUG")

    configure_logging()

    tree = logging.getLogger(root)
    assert len(tree.handlers) == 1
    assert tree.level == logging.DEBUG
    assert tree.propagate is False


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()

    assert all(len(logging.getLogger(root).handlers) == 1 for root in LOG_ROOTS)


def test_configure_logging_formats_records_with_module_and_function():
    configure_logging()

    handler = logging.getLogger("argus").handlers[0]
    record = logging.LogRecord("argus.data.cache", logging.INFO, __file__, 1, "hello", None, None, func="put")

    assert "argus.data.cache.put: hello" in handler.format(record)


def test_warns_on_permissive_cors_and_blank_api_key(monkeypatch, caplog):
    monkeypatch.setattr(api_main.settings, "ARGUS_CORS_ORIGINS", ["*"])
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")

    with caplog.at_level(logging.WARNING, logger="api.guards"):
        guards.warn_on_permissive_security_defaults()

    messages = [r.message for r in caplog.records]
    assert any("ARGUS_CORS_ORIGINS" in m for m in messages)
    assert any("ARGUS_API_KEY" in m for m in messages)


def test_no_warning_once_cors_and_api_key_are_configured(monkeypatch, caplog):
    monkeypatch.setattr(api_main.settings, "ARGUS_CORS_ORIGINS", ["https://example.com"])
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "a-real-secret")

    with caplog.at_level(logging.WARNING, logger="api.guards"):
        guards.warn_on_permissive_security_defaults()

    assert caplog.records == []


@pytest.fixture(autouse=True)
def _reset_health_globals(monkeypatch, api_state):
    """Clear the module-level state /health reads, so tests don't leak into each other."""
    monkeypatch.setattr(runtime, "pipeline_task", None)
    monkeypatch.setattr(runtime, "collector_task", None)
    monkeypatch.setattr(runtime, "reconcile_task", None)
    monkeypatch.setattr(runtime, "pipeline", None)


def test_health_reports_ok_when_no_background_tasks_are_configured(client):
    response = client.get("/health")

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


async def test_health_returns_503_naming_a_crashed_background_task(monkeypatch):

    async def _boom():
        raise RuntimeError("pipeline exploded")

    task = asyncio.create_task(_boom())
    with contextlib.suppress(RuntimeError):
        await task
    monkeypatch.setattr(runtime, "pipeline_task", task)

    with pytest.raises(HTTPException) as exc_info:
        await api_main.health()

    assert exc_info.value.status_code == 503
    assert "mft_pipeline" in exc_info.value.detail


async def test_health_returns_503_naming_a_canceled_background_task(monkeypatch):

    async def _spin():
        await asyncio.sleep(10)

    task = asyncio.create_task(_spin())
    await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    monkeypatch.setattr(runtime, "collector_task", task)

    with pytest.raises(HTTPException) as exc_info:
        await api_main.health()

    assert exc_info.value.status_code == 503
    assert "collector" in exc_info.value.detail


def test_health_reports_kill_switch_halted_state(client):
    ks = KillSwitch("MODERATE")
    ks._portfolio_inception_value = 100_000.0
    ks._current_portfolio_value = 50_000.0
    ks._high_water_mark = 100_000.0
    with mock.patch("argus.data.fetchers.fetch_vix", return_value=18.0):
        ks._check()
    kill_switch_module._kill_switch = ks

    response = client.get("/health")

    assert response.json()["kill_switch_halted"] is True


def test_health_reports_newest_cache_entry_age_during_market_hours(monkeypatch, client):
    fake_pipeline = mock.Mock()
    fake_pipeline.is_market_hours.return_value = True
    monkeypatch.setattr(runtime, "pipeline", fake_pipeline)
    cache = LiveSessionCache(interval_minutes=1)
    cache.publish({"AAPL": {}}, ["AAPL"])
    monkeypatch.setattr(runtime, "live_cache", cache)

    response = client.get("/health")

    age = response.json()["newest_cache_entry_age_seconds"]
    assert age is not None
    assert age < 5


@pytest.fixture
def fake_startup(monkeypatch):
    """Replace the lifespan's network- and model-touching dependencies with fakes."""
    for field in dataclasses.fields(runtime):
        monkeypatch.setattr(runtime, field.name, getattr(runtime, field.name))

    async def idle(**_kwargs):
        await asyncio.Event().wait()

    pipeline = mock.Mock(tickers=["AAPL"], interval_minutes=5)
    pipeline.start = idle
    pipeline.stop = mock.AsyncMock()
    pipeline.close_buffer = mock.AsyncMock()
    pipeline.is_market_hours.return_value = False
    pipeline.buffer.row_counts.return_value = {"AAPL": 3}
    monkeypatch.setattr(lifespan, "MFTDataPipeline", lambda **_kwargs: pipeline)
    monkeypatch.setattr(lifespan, "build_graph", lambda **_kwargs: mock.Mock())
    monkeypatch.setattr(lifespan.RegimeClassifier, "load", classmethod(lambda cls, path: mock.Mock(is_fitted=True)))
    monkeypatch.setattr(settings, "ARGUS_COLLECTOR_ENABLED", False)
    monkeypatch.setattr(settings, "ARGUS_RECONCILE_ENABLED", False)
    monkeypatch.setattr(settings, "ARGUS_UNIVERSE", ["AAPL"])
    return pipeline


def test_lifespan_brings_the_app_up_and_tears_it_down(fake_startup):
    with TestClient(api_main.app) as app_client:
        assert runtime.graph is not None
        assert runtime.live_cache is not None
        assert kill_switch_module.get_kill_switch() is not None

        status = app_client.get("/pipeline/status").json()
        assert status["tracked_tickers"] == ["AAPL"]
        assert status["buffer_depth"] == {"AAPL": 3}
        assert status["macro_classifier_fitted"] is True
        assert app_client.get("/health").status_code == 200
        assert guards._process_lock_file is not None
        lock_file = guards._process_lock_file

    fake_startup.stop.assert_awaited_once()
    fake_startup.close_buffer.assert_awaited_once()
    assert runtime.pipeline_task.cancelled()
    assert lock_file.closed
