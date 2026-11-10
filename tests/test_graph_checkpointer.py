"""Tests for orchestration/checkpoint.py's _open_checkpointer failure handling."""

from __future__ import annotations

import logging
from unittest import mock

import argus.orchestration.checkpoint as checkpoint_module


def test_open_checkpointer_failure_returns_none_logs_and_closes_the_connection(
    tmp_path, monkeypatch, caplog
):
    """A failure after the connection opens degrades to None without leaking the connection."""
    conn = mock.Mock()
    monkeypatch.setattr(checkpoint_module.sqlite3, "connect", mock.Mock(return_value=conn))
    monkeypatch.setattr(
        checkpoint_module, "SqliteSaver", mock.Mock(side_effect=RuntimeError("bad schema"))
    )

    with caplog.at_level(logging.WARNING, logger="argus.orchestration.checkpoint"):
        result = checkpoint_module._open_checkpointer(str(tmp_path / "graph.db"))

    assert result is None
    conn.close.assert_called_once_with()
    assert any("bad schema" in (r.exc_text or "") for r in caplog.records)


def test_open_checkpointer_failure_before_connecting_returns_none_and_logs(
    tmp_path, monkeypatch, caplog
):
    """A failure to open the connection at all is logged rather than swallowed silently."""
    monkeypatch.setattr(
        checkpoint_module.sqlite3, "connect", mock.Mock(side_effect=OSError("disk is read-only"))
    )

    with caplog.at_level(logging.WARNING, logger="argus.orchestration.checkpoint"):
        result = checkpoint_module._open_checkpointer(str(tmp_path / "graph.db"))

    assert result is None
    assert any("disk is read-only" in (r.exc_text or "") for r in caplog.records)
