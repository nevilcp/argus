"""Tests for scripts/remediate_macro_regime_tags.py's default store location."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from argus.config import settings
from scripts import remediate_macro_regime_tags


def test_persist_dir_defaults_to_the_stores_real_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no --persist-dir the script targets the directory the app writes to, not ./chroma_db."""
    monkeypatch.setattr(settings, "ARGUS_CHROMA_DIR", "")
    monkeypatch.setattr(sys, "argv", ["remediate_macro_regime_tags"])
    monkeypatch.chdir(tmp_path)

    remediate_macro_regime_tags.main()

    expected = Path(settings.ARGUS_DATA_DIR) / "chroma_db" / "chroma.sqlite3"
    assert str(expected) in capsys.readouterr().out
