"""Graph checkpoint serialization, isolated connections, and retention."""

import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

from argus.infra.clock import as_naive_timestamp, now_local
from argus.orchestration.state import ARGUSState
from argus.params import RECONCILIATION

logger = logging.getLogger(__name__)

# Allowlist every nested signal model so checkpoint deserialization preserves validated model types.
_CHECKPOINT_MODEL_CLASSES = (
    "MacroContext",
    "TechnicalSignal",
    "FundamentalSignal",
    "SentimentSignal",
    "RiskAssessment",
    "AggregatedSignal",
    "PositionAllocation",
    "PortfolioAllocation",
    "ARGUSDecision",
)


def build_checkpoint_serde() -> JsonPlusSerializer:
    """Return the JsonPlusSerializer used to read/write ARGUSState checkpoints."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[("argus.schemas.signals", cls) for cls in _CHECKPOINT_MODEL_CLASSES]
    )


def _open_checkpointer(checkpoint_db_path: str) -> SqliteSaver | None:
    """Open a fresh SqliteSaver connection, or None if the path can't be opened."""
    conn: sqlite3.Connection | None = None
    try:
        if checkpoint_db_path != ":memory:":
            Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        return SqliteSaver(conn, serde=build_checkpoint_serde())
    except Exception:
        logger.warning(
            "could not open %s; running without a checkpointer",
            checkpoint_db_path,
            exc_info=True,
        )
        if conn is not None:
            conn.close()
        return None


class CheckpointedGraph:
    """Graph wrapper with a fresh SQLite connection and compilation per invocation."""

    def __init__(self, builder: StateGraph, checkpoint_db_path: str):
        self._builder = builder
        self._checkpoint_db_path = checkpoint_db_path

    def invoke(self, state: ARGUSState, config: RunnableConfig | None = None) -> dict:
        """Compile with a fresh checkpointer and invoke the graph once."""
        checkpointer = _open_checkpointer(self._checkpoint_db_path)
        compiled = self._builder.compile(checkpointer=checkpointer)
        try:
            return compiled.invoke(state, config)
        finally:
            if checkpointer is not None:
                checkpointer.conn.close()


def new_run_config(prefix: str) -> RunnableConfig:
    """Build the run config for one graph invocation, under a fresh unique thread_id."""
    thread_id = f"{prefix}-{now_local():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
    return {"configurable": {"thread_id": thread_id}}


@contextmanager
def open_saver(db_path: str) -> Iterator[SqliteSaver]:
    """Open a SqliteSaver over the checkpoint database, closing its connection on exit."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=build_checkpoint_serde())
    finally:
        conn.close()


def newest_checkpoint_per_thread(saver: SqliteSaver) -> Iterator[tuple[str, Checkpoint]]:
    """Yield each thread's newest checkpoint exactly once, as (thread_id, checkpoint)."""
    seen_threads: set[str] = set()
    for tup in saver.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        if thread_id in seen_threads:
            continue
        seen_threads.add(thread_id)
        yield thread_id, tup.checkpoint


def retention_cutoff(horizon_days: int = RECONCILIATION.horizon_days) -> datetime:
    """Return the common cutoff for retained decisions, checkpoints, and snapshots."""
    return now_local() - timedelta(days=horizon_days + RECONCILIATION.retention_margin_days)


def prune_checkpoints(db_path: str, cutoff: datetime) -> int:
    """Delete every checkpoint thread whose most recent checkpoint predates cutoff, then VACUUMs."""
    if not Path(db_path).exists():
        return 0

    cutoff_naive = as_naive_timestamp(cutoff)
    with open_saver(db_path) as saver:
        stale_threads: list[str] = []
        for thread_id, checkpoint in newest_checkpoint_per_thread(saver):
            ts_raw = checkpoint.get("ts")
            if ts_raw is None:
                continue
            if as_naive_timestamp(datetime.fromisoformat(ts_raw)) < cutoff_naive:
                stale_threads.append(thread_id)

        if not stale_threads:
            return 0

        params = [(thread_id,) for thread_id in stale_threads]
        conn = saver.conn
        cur = conn.cursor()
        cur.executemany("DELETE FROM checkpoints WHERE thread_id = ?", params)
        cur.executemany("DELETE FROM writes WHERE thread_id = ?", params)
        conn.commit()
        conn.execute("VACUUM")

    logger.info(
        "deleted %d thread(s) from %s (cutoff=%s)",
        len(stale_threads),
        db_path,
        cutoff_naive.isoformat(),
    )
    return len(stale_threads)
