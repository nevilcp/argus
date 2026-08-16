"""
tests/conftest.py

Shared pytest fixtures.

Responsibilities:
  - Provide an opt-in network guard a test can request to prove it never
    falls through to a real socket call

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
