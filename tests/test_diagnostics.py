"""Tests argus/infra/diagnostics.py's Diagnostics collector."""

import logging

from argus.infra.diagnostics import Diagnostics


def test_record_appends_the_message_and_logs_it_at_the_given_level(caplog):
    """Each note is kept in order and logged at warning by default, error on request."""
    diagnostics = Diagnostics()

    with caplog.at_level(logging.WARNING, logger="argus.infra.diagnostics"):
        diagnostics.record("first")
        diagnostics.record("second", "error")

    assert diagnostics.messages == ["first", "second"]
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, "first"),
        (logging.ERROR, "second"),
    ]


def test_each_collector_starts_empty():
    """Two collectors never share a message list."""
    a, b = Diagnostics(), Diagnostics()
    a.record("only in a")

    assert b.messages == []
