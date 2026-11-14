"""Crash-safe file writes."""

from __future__ import annotations

from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via a sibling tmp file and an atomic replace."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
