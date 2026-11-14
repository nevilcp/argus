"""Tests for orchestration/checkpoint.py's retention: prune_checkpoints."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from argus.orchestration.checkpoint import prune_checkpoints, retention_cutoff
from argus.params import RECONCILIATION
from tests.helpers.builders import (
    matured_decision,
)
from tests.helpers.decision_store import put_checkpoint


def test_prune_checkpoints_deletes_only_threads_older_than_cutoff(tmp_path):
    db_path = str(tmp_path / "argus_graph.db")
    put_checkpoint(db_path, "stale-thread", datetime(2026, 1, 1))
    put_checkpoint(db_path, "fresh-thread", datetime(2026, 1, 20))

    deleted = prune_checkpoints(db_path, cutoff=datetime(2026, 1, 10))

    assert deleted == 1
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpoint_threads = {row[0] for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()}
    conn.close()
    assert checkpoint_threads == {"fresh-thread"}


def test_prune_checkpoints_missing_db_is_a_noop():
    assert prune_checkpoints("/nonexistent/argus_graph.db", cutoff=datetime(2026, 1, 1)) == 0


def test_prune_checkpoints_nothing_stale_deletes_nothing(tmp_path):
    db_path = str(tmp_path / "argus_graph.db")
    put_checkpoint(db_path, "fresh-thread", datetime.now())

    deleted = prune_checkpoints(db_path, cutoff=datetime.now() - timedelta(days=30))

    assert deleted == 0


def test_prune_checkpoints_normalizes_a_non_utc_offset_before_comparing(tmp_path):
    db_path = str(tmp_path / "argus_graph.db")
    non_utc_ts = datetime(2026, 1, 1, 20, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    put_checkpoint(db_path, "offset-thread", non_utc_ts)

    deleted = prune_checkpoints(db_path, cutoff=datetime(2026, 1, 2))

    assert deleted == 0


def test_checkpoint_pruning_stabilizes_the_published_file_size_across_successive_cycles(
    tmp_path,
):
    db_path = str(tmp_path / "argus_graph.db")
    retention_days = RECONCILIATION.horizon_days + RECONCILIATION.retention_margin_days
    sim_now = datetime(2026, 1, 1)
    payload = [matured_decision(f"TICK{i}", sim_now) for i in range(20)]

    sizes = []
    total_cycles = 3 * retention_days
    for cycle in range(total_cycles):
        put_checkpoint(db_path, f"collector-{cycle:04d}", sim_now, decisions=payload)
        prune_checkpoints(db_path, cutoff=sim_now - timedelta(days=retention_days))
        sizes.append(Path(db_path).stat().st_size)
        sim_now += timedelta(days=1)

    assert sizes[retention_days] > sizes[0]

    # After retention fills, each new checkpoint thread replaces one expired thread instead of growing the store.
    plateau = sizes[retention_days + 2 :]
    assert plateau, "test needs more cycles than the retention window to reach a plateau"
    assert all(size == plateau[0] for size in plateau)


def test_retention_cutoff_defaults_to_the_declared_horizon_plus_the_margin():
    window = timedelta(days=RECONCILIATION.horizon_days + RECONCILIATION.retention_margin_days)

    before = datetime.now()
    cutoff = retention_cutoff()
    after = datetime.now()

    assert before - window <= cutoff <= after - window


def test_retention_cutoff_widens_with_a_longer_horizon():
    default = retention_cutoff()
    longer = retention_cutoff(RECONCILIATION.horizon_days + 10)

    assert abs((default - longer) - timedelta(days=10)) < timedelta(seconds=5)
