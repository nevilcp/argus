"""
tests/test_api_startup.py

Tests for api/main.py's PR3 boot-time assertions: the single-worker invariant
(GOV-8) and the configured-model registration check (GOV-11). Both run pure
sys.argv/settings checks, so they're exercised directly rather than through
the app's lifespan.
"""

from __future__ import annotations

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
