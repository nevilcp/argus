"""Shared pytest fixtures."""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import argus.risk.kill_switch as kill_switch_module
from api.lifespan import runtime
from argus.config import settings
from argus.data.live_session_cache import LiveSessionCache
from argus.infra.log import LOG_ROOTS
from tests.helpers.paths import FIXTURES_DIR

GOLDEN_DIR = FIXTURES_DIR / "golden"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --update-golden, which rewrites the committed behavior snapshots."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite tests/fixtures/golden/* instead of asserting against them.",
    )


@pytest.fixture
def assert_golden(request: pytest.FixtureRequest) -> Callable[[str, str], None]:
    """Return a checker that compares text to a committed snapshot, or rewrites it."""
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
    """Clear the module-level KillSwitch singleton before and after each test."""
    kill_switch_module._kill_switch = None
    yield
    kill_switch_module._kill_switch = None


@pytest.fixture(autouse=True)
def _reset_logging_trees() -> Iterator[None]:
    """Clear the logger trees `configure_logging` touches, before and after each test."""
    trees = [logging.getLogger(name) for name in LOG_ROOTS]
    saved = [(list(t.handlers), t.level, t.propagate) for t in trees]
    for tree in trees:
        tree.handlers = []
    yield
    for tree, (handlers, level, propagate) in zip(trees, saved, strict=True):
        tree.handlers, tree.level, tree.propagate = handlers, level, propagate


@pytest.fixture
def api_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the API runtime's live cache and the API-key setting for the duration of a test."""
    monkeypatch.setattr(runtime, "live_cache", LiveSessionCache(interval_minutes=1))
    monkeypatch.setattr(api_main.settings, "ARGUS_API_KEY", "")


@pytest.fixture
def client() -> TestClient:
    """Return an API test client without starting background loops."""
    return TestClient(api_main.app)


@pytest.fixture(scope="session")
def live_network_skip_reason() -> str | None:
    """Return why live-network tests should skip, or None if Yahoo Finance is reachable."""
    try:
        with socket.create_connection(("query1.finance.yahoo.com", 443), timeout=3.0):
            return None
    except OSError as exc:
        return f"no live network access to Yahoo Finance ({exc})"


@pytest.fixture(autouse=True)
def _skip_without_live_network(request: pytest.FixtureRequest) -> None:
    """Skip a `live_network`-marked test, with a visible reason, when the network is down."""
    if request.node.get_closest_marker("live_network") is None:
        return
    reason = request.getfixturevalue("live_network_skip_reason")
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ARGUS_DATA_DIR at a per-test tmp_path so no test can reach the production buffer."""
    monkeypatch.setattr(settings, "ARGUS_DATA_DIR", str(tmp_path))
