"""
tests/test_api_startup.py

Tests for api/main.py's PR3 boot-time assertions: the single-worker invariant
(GOV-8) and the configured-model registration check (GOV-11). Both run pure
sys.argv/settings checks, so they're exercised directly rather than through
the app's lifespan.

Also covers PR5b's API-2 additions: `_assert_single_worker`'s extension to
`--workers=N` and `WEB_CONCURRENCY`, and `_acquire_process_lock`'s real
cross-process guard (an exclusive flock on `${ARGUS_DATA_DIR}/argus.lock`).
"""

from __future__ import annotations

import fcntl

import pytest

import api.main as api_main


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
