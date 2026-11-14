"""Tests for orchestration/checkpoint.py: checkpointer failure handling and run configs."""

from __future__ import annotations

import logging
from unittest import mock

import argus.orchestration.checkpoint as checkpoint_module


def test_open_checkpointer_failure_returns_none_logs_and_closes_the_connection(tmp_path, monkeypatch, caplog):
    conn = mock.Mock()
    monkeypatch.setattr(checkpoint_module.sqlite3, "connect", mock.Mock(return_value=conn))
    monkeypatch.setattr(checkpoint_module, "SqliteSaver", mock.Mock(side_effect=RuntimeError("bad schema")))

    with caplog.at_level(logging.WARNING, logger="argus.orchestration.checkpoint"):
        result = checkpoint_module._open_checkpointer(str(tmp_path / "graph.db"))

    assert result is None
    conn.close.assert_called_once_with()
    assert any("bad schema" in (r.exc_text or "") for r in caplog.records)


def test_open_checkpointer_failure_before_connecting_returns_none_and_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(checkpoint_module.sqlite3, "connect", mock.Mock(side_effect=OSError("disk is read-only")))

    with caplog.at_level(logging.WARNING, logger="argus.orchestration.checkpoint"):
        result = checkpoint_module._open_checkpointer(str(tmp_path / "graph.db"))

    assert result is None
    assert any("disk is read-only" in (r.exc_text or "") for r in caplog.records)


def test_new_run_config_gives_each_run_a_unique_prefixed_thread_id():
    first = checkpoint_module.new_run_config("analyze")["configurable"]["thread_id"]
    second = checkpoint_module.new_run_config("analyze")["configurable"]["thread_id"]

    assert first.startswith("analyze-")
    assert first != second
