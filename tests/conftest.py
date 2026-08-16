"""
tests/conftest.py

Shared pytest fixtures.

Responsibilities:
  - Provide an opt-in network guard a test can request to prove it never
    falls through to a real socket call
  - Isolate every test from the production candle buffer

Not responsible for:
  - Test data (see tests/fixtures/)
  - Blocking network globally — tests/test_integration.py's TestEndToEnd
    class intentionally exercises live yfinance/FRED calls, and CI relies
    on that; this fixture is opt-in per test, not autouse
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

from argus.config import settings


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
    yield
