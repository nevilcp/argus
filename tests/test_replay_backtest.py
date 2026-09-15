"""Tests for scripts/replay_backtest.py's argument parsing.

Before this, sys.argv[1:] was consumed directly as session directories, so
any flag (e.g. --help, a typo) became a Path and blew up with a raw
traceback instead of a usage error.
"""

from pathlib import Path
from unittest import mock

import pytest

from scripts.replay_backtest import DEFAULT_SESSION_DIR, main


def test_unrecognized_flag_reports_a_usage_error_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["replay_backtest.py", "--not-a-real-flag"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_no_args_defaults_to_the_fixtures_dir(monkeypatch):
    monkeypatch.setattr("sys.argv", ["replay_backtest.py"])
    with mock.patch("scripts.replay_backtest.replay_sessions", return_value=[]) as fake:
        main()

    fake.assert_called_once_with([DEFAULT_SESSION_DIR])


def test_positional_args_are_parsed_as_session_dirs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["replay_backtest.py", "dir_a", "dir_b"])
    with mock.patch("scripts.replay_backtest.replay_sessions", return_value=[]) as fake:
        main()

    fake.assert_called_once_with([Path("dir_a"), Path("dir_b")])
