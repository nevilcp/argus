"""
tests/conftest.py

Shared pytest fixtures.

Responsibilities:
  - Provide an opt-in network guard a test can request to prove it never
    falls through to a real socket call
  - Isolate every test from the production candle buffer
  - Restore the logger trees `configure_logging` mutates
  - Reset the kill-switch singleton and api.main's live-session-cache /
    API-key singletons that several api.main test modules each used to
    reset independently

Not responsible for:
  - Test data (see tests/fixtures/)
  - Blocking network globally — tests/test_integration.py's TestEndToEnd
    class intentionally exercises live yfinance/FRED calls, and CI relies
    on that; this fixture is opt-in per test, not autouse
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Iterator

import pytest

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from argus.config import settings
from argus.data.live_session_cache import LiveSessionCache
from argus.infra.log import LOG_ROOTS
from tests.helpers.paths import FIXTURES_DIR

GOLDEN_DIR = FIXTURES_DIR / "golden"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Registers --update-golden, which rewrites the committed behaviour snapshots.

    Only a PR that deliberately changes behaviour may use it, and its
    description must show the resulting snapshot diff.
    """
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite tests/fixtures/golden/* instead of asserting against them.",
    )


@pytest.fixture
def assert_golden(request: pytest.FixtureRequest) -> Callable[[str, str], None]:
    """Returns a checker that compares text to a committed snapshot, or rewrites it.

    Args:
        request: Pytest's request, used to read --update-golden.

    Returns:
        A function taking (snapshot file name under tests/fixtures/golden/, actual text).
    """
    update = request.config.getoption("--update-golden")

    def _check(name: str, actual: str) -> None:
        path = GOLDEN_DIR / name
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(actual)
            return
        assert path.exists(), f"{path} missing; generate it with `pytest --update-golden`"
        assert actual == path.read_text(), (
            f"output differs from {path.name}; if the change is intended, "
            "re-run with --update-golden and show the diff in the PR"
        )

    return _check


@pytest.fixture(autouse=True)
def _reset_kill_switch_singleton() -> Iterator[None]:
    """Clears the module-level KillSwitch singleton before and after each test."""
    kill_switch_module._kill_switch = None
    yield
    kill_switch_module._kill_switch = None


@pytest.fixture(autouse=True)
def _reset_logging_trees() -> Iterator[None]:
    """Clears the logger trees `configure_logging` touches, before and after each test.

    `configure_logging` sets ``propagate = False``, which would otherwise stop
    every later test's ``caplog`` from seeing records once any test (the API
    lifespan, a script's ``main()``) had called it.
    """
    trees = [logging.getLogger(name) for name in LOG_ROOTS]
    saved = [(list(t.handlers), t.level, t.propagate) for t in trees]
    for tree in trees:
        tree.handlers = []
    yield
    for tree, (handlers, level, propagate) in zip(trees, saved, strict=True):
        tree.handlers, tree.level, tree.propagate = handlers, level, propagate


@pytest.fixture
def _fresh_live_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installs a fresh LiveSessionCache as api.main's module-level singleton."""
    monkeypatch.setattr(api_main, "_live_cache", LiveSessionCache(interval_minutes=1))


@pytest.fixture
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disables the ARGUS_API_KEY auth gate for the duration of a test."""
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points ARGUS_DATA_DIR at a per-test tmp_path so no test can reach the production buffer.

    Any call site that builds ``MFTDataPipeline`` (or otherwise resolves a
    path from ``settings.ARGUS_DATA_DIR``) without an explicit override lands
    here instead of ``data/ohlcv_buffer.db``, which a local `pytest` run
    would otherwise insert into and prune.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        monkeypatch: Pytest's monkeypatch fixture.
    """
    monkeypatch.setattr(settings, "ARGUS_DATA_DIR", str(tmp_path))


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Makes any outbound socket connection raise, for the duration of the requesting test.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.

    Yields:
        None.
    """

    def _blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Network access attempted in a test requesting block_network")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    return
