"""Crash-safe file writes."""

from __future__ import annotations

from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Writes `text` to `path` via a sibling tmp file and an atomic replace.

    A reader (or a crash mid-write) sees either the old contents or the new,
    never a truncated file. The parent directory must already exist.

    Args:
        path: Destination file.
        text: Full new contents, UTF-8.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
