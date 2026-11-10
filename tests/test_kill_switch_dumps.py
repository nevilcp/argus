"""
tests/test_kill_switch_dumps.py

Tests for argus/risk/kill_switch.py's halt-dump housekeeping: reset()
deleting dumps, and pruning and sorting them.

fetch_vix is mocked throughout; _check() is driven directly, so no test sleeps
on a real check interval.
"""

from __future__ import annotations

import json
from datetime import datetime

import argus.risk.kill_switch as kill_switch_module
from tests.helpers.kill_switch import check_at_vix, make_ks


def test_reset_deletes_persisted_halt_dumps(tmp_path, monkeypatch):
    """reset() removes halt-dump files so a restart can't resurrect a resolved halt."""
    monkeypatch.chdir(tmp_path)
    ks = make_ks("MODERATE", inception=100_000.0)
    ks.update_portfolio_value(80_000.0)
    check_at_vix(ks, 15.0)
    assert ks.is_halted
    assert kill_switch_module._find_latest_halt_file() is not None

    ks.reset(120_000.0)

    assert kill_switch_module._find_latest_halt_file() is None


# ---------------------------------------------------------------------------
# Halt-dump pruning and content-based sort
# ---------------------------------------------------------------------------


def _write_halt_dump(runs_dir, filename: str, halt_time: datetime) -> None:
    """Writes a minimal halt-dump JSON file with the given halt_time content."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    dump = {"halt_time": halt_time.isoformat(), "reason": "test"}
    (runs_dir / filename).write_text(json.dumps(dump))


def test_list_halt_dumps_sorts_by_halt_time_content_not_filename(tmp_path, monkeypatch):
    """Dumps are ranked by their own halt_time field, not the filename's embedded timestamp."""
    monkeypatch.chdir(tmp_path)
    runs_dir = tmp_path / "runs"
    # Filename order (alphabetical) is deliberately the reverse of halt_time order
    _write_halt_dump(runs_dir, "argus_halt_20260101_000000.json", datetime(2026, 6, 1))
    _write_halt_dump(runs_dir, "argus_halt_20260601_000000.json", datetime(2026, 1, 1))

    dumps = kill_switch_module._list_halt_dumps(str(runs_dir))

    assert [d.name for d in dumps] == [
        "argus_halt_20260601_000000.json",
        "argus_halt_20260101_000000.json",
    ]


def test_prune_halt_dumps_keeps_only_the_most_recent_n(tmp_path, monkeypatch):
    """_prune_halt_dumps deletes all but KILL_SWITCH.max_halt_dumps_retained newest dumps."""
    monkeypatch.chdir(tmp_path)
    runs_dir = tmp_path / "runs"
    keep = kill_switch_module.KILL_SWITCH.max_halt_dumps_retained
    for i in range(keep + 3):
        _write_halt_dump(runs_dir, f"argus_halt_dump_{i}.json", datetime(2026, 1, 1 + i))

    kill_switch_module._prune_halt_dumps(str(runs_dir))

    remaining = kill_switch_module._list_halt_dumps(str(runs_dir))
    assert len(remaining) == keep
