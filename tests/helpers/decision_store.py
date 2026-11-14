"""Decision writers for JSONL logs and graph checkpoints."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite import SqliteSaver

from argus.orchestration.checkpoint import build_checkpoint_serde
from argus.schemas.signals import ARGUSDecision


def write_decisions_log(path: Path, *decisions: ARGUSDecision) -> Path:
    """Write `decisions` as one JSON object per line at `path` and return that path."""
    with open(path, "w", encoding="utf-8") as f:
        for decision in decisions:
            f.write(decision.model_dump_json() + "\n")
    return path


def put_checkpoint(db_path: str, thread_id: str, ts: datetime, decisions: list[ARGUSDecision] | None = None) -> None:
    """Write one checkpoint for `thread_id` stamped with `ts`."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint: Checkpoint = {
        "v": 1,
        "ts": ts.isoformat(),
        "id": thread_id,
        "channel_values": {} if decisions is None else {"decisions": decisions},
        "channel_versions": {} if decisions is None else {"decisions": "1"},
        "versions_seen": {},
        "updated_channels": None,
    }
    metadata: CheckpointMetadata = {"source": "input", "step": 1}

    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        SqliteSaver(conn, serde=build_checkpoint_serde()).put(config, checkpoint, metadata, {})
    finally:
        conn.close()
