"""
argus/orchestration/checkpoint.py

Everything that touches the LangGraph checkpoint database (argus_graph.db), so
the graph, the collector and reconciliation share one serializer, one way to
open the file, and one place that reaches into LangGraph's table layout.

Responsibilities:
  - build_checkpoint_serde: the msgpack allowlist for ARGUSState's nested models
  - CheckpointedGraph: compiles the DAG with a fresh connection on every invoke()
  - new_run_config: the run config (a fresh unique thread_id) every entry point invokes with
  - open_saver: a SqliteSaver over an existing database, closed on exit
  - newest_checkpoint_per_thread: each thread's most recent checkpoint
  - prune_checkpoints / retention_cutoff: bound the database

Not responsible for:
  - Reading decisions out of checkpoints (see orchestration/decision_log.py)
  - Building the DAG itself (see orchestration/graph.py)

Dependencies:
  - langgraph (SqliteSaver, JsonPlusSerializer)
  - argus.orchestration.state (ARGUSState)
"""

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

# Every schemas.signals model nested in ARGUSState must be listed here or the
# checkpointer's msgpack deserializer degrades it to a plain dict
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
    """Returns the JsonPlusSerializer used to read/write ARGUSState checkpoints."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("argus.schemas.signals", cls) for cls in _CHECKPOINT_MODEL_CLASSES
        ]
    )


def _open_checkpointer(checkpoint_db_path: str) -> SqliteSaver | None:
    """Opens a fresh SqliteSaver connection, or None if the path can't be opened.

    Args:
        checkpoint_db_path: SQLite file to checkpoint to.

    Returns:
        A new SqliteSaver, or None on any I/O failure (a checkpointer-less
        compile still runs the graph, just without resumability).
    """
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
    """Compiles the DAG fresh, with its own SQLite connection, on every invoke().

    StateGraph.compile() does no I/O beyond opening the checkpoint connection;
    the expensive part — constructing the six agents (LLM clients, the macro
    HMM classifier load from disk) — happens once in build_graph() and is
    captured in the node closures this wraps. Concurrent callers (parallel
    /analyze requests, the unattended collector loop) therefore each get their
    own sqlite3.Connection instead of contending on one shared checkpointer.
    """

    def __init__(self, builder: StateGraph, checkpoint_db_path: str):
        """Stores the built (but not compiled) graph and its checkpoint path."""
        self._builder = builder
        self._checkpoint_db_path = checkpoint_db_path

    def invoke(self, state: ARGUSState, config: RunnableConfig | None = None) -> dict:
        """Compiles with a fresh checkpointer and invokes the graph once.

        Args:
            state: Initial ARGUSState for this session.
            config: LangGraph run config, e.g. ``{"configurable": {"thread_id": ...}}``.

        Returns:
            The final ARGUSState dict produced by the run.
        """
        checkpointer = _open_checkpointer(self._checkpoint_db_path)
        compiled = self._builder.compile(checkpointer=checkpointer)
        try:
            return compiled.invoke(state, config)
        finally:
            if checkpointer is not None:
                checkpointer.conn.close()


def new_run_config(prefix: str) -> RunnableConfig:
    """Builds the run config for one graph invocation, under a fresh unique thread_id.

    A timestamp alone is only second-resolution, so two runs starting within
    the same second would share a thread_id; the uuid suffix makes it unique
    regardless of process restarts.

    Args:
        prefix: Names the entry point (e.g. "analyze", "collector", "replay"),
            so a checkpoint row can be traced back to what started it.

    Returns:
        ``{"configurable": {"thread_id": ...}}`` for ``graph.invoke``.
    """
    thread_id = f"{prefix}-{now_local():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
    return {"configurable": {"thread_id": thread_id}}


@contextmanager
def open_saver(db_path: str) -> Iterator[SqliteSaver]:
    """Opens a SqliteSaver over the checkpoint database, closing its connection on exit."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=build_checkpoint_serde())
    finally:
        conn.close()


def newest_checkpoint_per_thread(saver: SqliteSaver) -> Iterator[tuple[str, Checkpoint]]:
    """Yields each thread's newest checkpoint exactly once, as (thread_id, checkpoint).

    SqliteSaver.list() yields checkpoints newest-first, so a thread's first tuple
    is its most recent — and so most complete — one.
    """
    seen_threads: set[str] = set()
    for tup in saver.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        if thread_id in seen_threads:
            continue
        seen_threads.add(thread_id)
        yield thread_id, tup.checkpoint


def retention_cutoff(horizon_days: int = RECONCILIATION.horizon_days) -> datetime:
    """The cutoff before which a reconciled decision, checkpoint or snapshot may be pruned.

    One definition for every store that is bounded on the reconciliation window:
    collector.py's per-cycle pruning calls it with the default horizon, and
    reconcile.py's run_reconciliation_pass with the horizon it was given, so
    the prune paths can't silently drift onto different windows.

    Args:
        horizon_days: Calendar days after a decision's session_timestamp at
            which its outcome is scored.

    Returns:
        The naive local-time cutoff, ``horizon_days`` plus the retention margin ago.
    """
    return now_local() - timedelta(days=horizon_days + RECONCILIATION.retention_margin_days)


def prune_checkpoints(db_path: str, cutoff: datetime) -> int:
    """Deletes every checkpoint thread whose most recent checkpoint predates cutoff, then VACUUMs.

    Each build_graph() invocation checkpoints under a fresh, never-reused
    thread_id (see build_graph()'s callers) and nothing resumes a thread by
    id, so a thread's checkpoints exist only as load_decisions_from_checkpoints'
    read path — a fallback now that decisions.jsonl is the durable
    record both /analyze and the collector write to. Retention, not removal:
    the checkpoint DB stays available as that fallback for anything within the
    reconciliation window, it just no longer grows without bound.

    The DELETEs name LangGraph's ``checkpoints`` and ``writes`` tables
    directly: SqliteSaver has no public API for removing a thread, so this
    is the one place that depends on that layout.

    Args:
        db_path: Path to the SQLite file build_graph() checkpoints to.
        cutoff: Threads whose newest checkpoint timestamp is before this are
            deleted. Should be tz-naive, matching the naive timestamps
            LangGraph checkpoint metadata is compared against here.

    Returns:
        Count of threads deleted. 0 if the database doesn't exist yet.
    """
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
