"""
tests/conftest.py

Shared pytest fixtures.

Responsibilities:
  - Skip `live_network`-marked tests when Yahoo Finance is unreachable
  - Isolate every test from the production candle buffer
  - Restore the logger trees `configure_logging` mutates
  - Reset the kill-switch singleton, and (opt-in via `api_state`) api.main's
    live-session-cache / API-key singletons
  - Provide the lifespan-free API `client`

Not responsible for:
  - Test data (see tests/fixtures/)
  - Blocking network — tests/test_integration.py's TestEndToEnd class
    intentionally exercises live yfinance/FRED calls, and CI relies on that
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

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
def api_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resets api.main's live-cache and API-key singletons for the duration of a test.

    The kill switch is reset separately by the autouse fixture above. Request
    this through ``pytestmark = pytest.mark.usefixtures("api_state")``.
    """
    monkeypatch.setattr(api_main, "_live_cache", LiveSessionCache(interval_minutes=1))
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")


@pytest.fixture
def client() -> TestClient:
    """A TestClient over api.main.app without running its lifespan startup."""
    return TestClient(api_main.app)


@pytest.fixture(scope="session")
def live_network_skip_reason() -> str | None:
    """Returns why live-network tests should skip, or None if Yahoo Finance is reachable.

    Session-scoped and only built when a ``live_network`` test asks for it, so
    one run pays the reachability probe's timeout at most once, and a run that
    selects no such test opens no socket.
    """
    try:
        with socket.create_connection(("query1.finance.yahoo.com", 443), timeout=3.0):
            return None
    except OSError as exc:
        return f"no live network access to Yahoo Finance ({exc})"


@pytest.fixture(autouse=True)
def _skip_without_live_network(request: pytest.FixtureRequest) -> None:
    """Skips a ``live_network``-marked test, with a visible reason, when the network is down."""
    if request.node.get_closest_marker("live_network") is None:
        return
    reason = request.getfixturevalue("live_network_skip_reason")
    if reason is not None:
        pytest.skip(reason)


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
