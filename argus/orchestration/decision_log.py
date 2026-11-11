"""
argus/orchestration/decision_log.py

The durable record of completed decisions: the decisions.jsonl log and the
LangGraph checkpoint database it falls back to.

Responsibilities:
  - decisions_log_lock: serialise appenders and the compactor on one log
  - append_decisions_jsonl: add a session's decisions to the log, stamped with
    the running image tag
  - load_decisions_from_jsonl / load_decisions_from_checkpoints: read
    ARGUSDecision objects back out of either source
  - compact_decisions_jsonl: bound the log so it does not grow forever
    (checkpoint pruning lives in orchestration/checkpoint.py)

Not responsible for:
  - Deciding which decisions to reconcile (see orchestration/reconcile.py)
  - Opening or pruning the checkpoint database (see orchestration/checkpoint.py)

Dependencies:
  - argus.orchestration.checkpoint (open_saver, newest_checkpoint_per_thread)
  - argus.outcomes (needs_reconciliation)
"""

from __future__ import annotations

import fcntl
import logging
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from argus.config import settings
from argus.infra.clock import as_naive_timestamp
from argus.infra.files import atomic_write_text
from argus.orchestration.checkpoint import newest_checkpoint_per_thread, open_saver
from argus.outcomes import needs_reconciliation
from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger(__name__)


@contextmanager
def decisions_log_lock(path: str | Path) -> Iterator[None]:
    """Holds an exclusive lock on a decisions.jsonl log for the duration of the block.

    Appenders (the collector and /analyze) and the compactor all rewrite or extend
    the same file, from different threads and, with the scheduled collector,
    different processes. An append that landed between compaction's read and its
    replace would otherwise be overwritten. The lock is an flock on a sidecar
    ``<path>.lock`` file, the same pattern as newsapi_budget.NewsApiBudget; flock is
    scoped to the open file description, so it serializes threads and processes alike.
    Released when the sidecar file is closed on exit.

    Args:
        path: The decisions.jsonl path whose sidecar lock to take.
    """
    log_path = Path(path)
    lock_path = log_path.with_suffix(log_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def append_decisions_jsonl(decisions: list[ARGUSDecision], path: str) -> int:
    """Appends each decision to a JSONL log, one compact JSON object per line.

    Shared by run_collection_cycle and api/main.py's /analyze — the two
    entry points into the graph — so decisions.jsonl (the source
    reconciliation reads, see orchestration/reconcile.py) reflects
    both instead of only the unattended collector's.

    Each decision is stamped with the running process's image tag (issue #95)
    before it's written, so a logged decision can be traced back to the build
    that produced it regardless of which docker tag was pulled to run it.

    Args:
        decisions: Decisions produced by this cycle's graph invocation.
        path: Destination file; parent directories are created if missing.

    Returns:
        Count of decisions written.
    """
    if not decisions:
        return 0

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Held so a concurrent compaction (compact_decisions_jsonl below) can't
    # read the log, then replace it over this append.
    with decisions_log_lock(log_path), open(log_path, "a", encoding="utf-8") as f:
        for decision in decisions:
            stamped = decision.model_copy(update={"image_tag": settings.ARGUS_IMAGE_TAG})
            f.write(stamped.model_dump_json() + "\n")
    return len(decisions)


def load_decisions_from_checkpoints(db_path: str = "argus_graph.db") -> list[ARGUSDecision]:
    """Reads every session's decisions back out of the LangGraph checkpoint database.

    Each session runs under its own thread_id (see build_graph()'s callers),
    and SqliteSaver.list() yields checkpoints newest-first; this keeps each
    thread's first (= most recent, most complete) checkpoint's `decisions`
    channel value and skips the rest, reading the checkpoint the graph
    already writes rather than maintaining a dedicated decision archive.

    Args:
        db_path: Path to the SQLite file build_graph() checkpoints to.

    Returns:
        Every ARGUSDecision found across all threads. Empty list if the
        database doesn't exist yet or holds no checkpoints.
    """
    if not Path(db_path).exists():
        # sqlite3.connect would otherwise create an empty database file here
        return []

    decisions: list[ARGUSDecision] = []
    with open_saver(db_path) as saver:
        for _thread_id, checkpoint in newest_checkpoint_per_thread(saver):
            channel_values = checkpoint.get("channel_values", {})
            decisions.extend(channel_values.get("decisions") or [])
    return decisions


def _read_decisions_jsonl(path: str) -> tuple[list[ARGUSDecision], list[str]]:
    """Reads a JSONL decision log, separating parsed decisions from lines that fail to parse.

    Args:
        path: Path to a JSONL file of ARGUSDecision.model_dump_json() lines.

    Returns:
        (decisions, malformed_lines): every parsed ARGUSDecision, and the raw text of
        each non-blank line that failed to parse. Both empty if the file doesn't exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return [], []

    decisions: list[ARGUSDecision] = []
    malformed: list[str] = []
    with open(file_path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                decisions.append(ARGUSDecision.model_validate_json(line))
            except Exception as exc:
                logger.warning(
                    "skipping malformed line %d in %s: %s",
                    line_no,
                    path,
                    exc,
                )
                malformed.append(line)
    return decisions, malformed


def load_decisions_from_jsonl(path: str) -> list[ARGUSDecision]:
    """Reads decisions back out of a JSONL decision log, one ARGUSDecision per line.

    The lighter-weight counterpart to load_decisions_from_checkpoints, for
    deployments that don't carry the full LangGraph checkpoint database
    around — the unattended collector (argus/orchestration/collector.py)
    appends each decision here instead, so the state a scheduled run has to
    persist stays small.

    Args:
        path: Path to a JSONL file of ARGUSDecision.model_dump_json() lines.

    Returns:
        Every ARGUSDecision found in the file. Empty list if the file
        doesn't exist yet. A line that fails to parse is logged and skipped
        rather than aborting the whole read.
    """
    return _read_decisions_jsonl(path)[0]


@dataclass
class CompactionResult:
    """Outcome of one compact_decisions_jsonl() call.

    Attributes:
        retained: Count of decisions written back to the log.
        retired_unresolved: Count of decisions dropped for aging past
            RECONCILIATION.unresolved_retirement_days without ever having
            received an outcome — distinct from `retained`'s complement,
            which also includes decisions dropped as routine, already-resolved
            cleanup.
        malformed_preserved: Count of lines that failed to parse and were written
            back verbatim, so a compaction never deletes evidence it cannot read.
    """

    retained: int
    retired_unresolved: int = 0
    malformed_preserved: int = 0


def compact_decisions_jsonl(
    path: str,
    cutoff: datetime,
    unresolved_cutoff: datetime,
    resolved_ids: Collection[str],
) -> CompactionResult:
    """Rewrites a decisions.jsonl log, dropping resolved sessions older than cutoff.

    A decision is resolved once it has a stored outcome (its decision_id is in
    resolved_ids) or could never produce one (see _needs_reconciliation) —
    anything else has no recorded outcome yet and is not eligible for this
    routine prune, however old it is: a reconcile pass that never ran, or
    never found a matured price, must not cost it the evidence it would
    otherwise have produced. Such a decision is instead retired once it also
    passes the much wider `unresolved_cutoff`, at which point it is declared
    permanently unresolvable and dropped anyway, counted separately from a
    routine, resolved prune.

    Meant to run right after a reconcile_decisions() pass over the same log
    (see orchestration/scheduler.py's reconcile_once and scripts/reconcile_outcomes.py),
    with resolved_ids reflecting that same pass's outcome stores.

    Args:
        path: decisions.jsonl path.
        cutoff: Resolved decisions with session_timestamp before this are
            dropped. Should be tz-naive, matching the naive timestamps this
            log stores.
        unresolved_cutoff: Unresolved decisions with session_timestamp before
            this are retired anyway. Should be tz-naive and at or before
            cutoff (i.e. further in the past), so an unresolved decision
            always gets at least as long as a resolved one.
        resolved_ids: decision_id values that already have a stored outcome.

    Returns:
        A CompactionResult. Every field is 0 (and no file written) if the
        file doesn't exist yet.
    """
    file_path = Path(path)
    if not file_path.exists():
        return CompactionResult(retained=0)

    cutoff_naive = as_naive_timestamp(cutoff)
    unresolved_cutoff_naive = as_naive_timestamp(unresolved_cutoff)

    # Read through replace under one lock: an append landing between the read and the
    # replace would otherwise be overwritten by the rewrite below.
    with decisions_log_lock(path):
        decisions, malformed = _read_decisions_jsonl(path)

        kept: list[ARGUSDecision] = []
        retired_unresolved = 0
        for decision in decisions:
            session_ts = as_naive_timestamp(decision.session_timestamp)
            if session_ts >= cutoff_naive:
                kept.append(decision)
                continue

            resolved = decision.decision_id in resolved_ids or not needs_reconciliation(decision)
            if resolved:
                continue
            if session_ts >= unresolved_cutoff_naive:
                kept.append(decision)
            else:
                retired_unresolved += 1

        result = CompactionResult(
            retained=len(kept),
            retired_unresolved=retired_unresolved,
            malformed_preserved=len(malformed),
        )
        if len(kept) == len(decisions):
            return result

        # Lines that failed to parse are carried over verbatim: dropping them would
        # silently and permanently delete records this code merely cannot read.
        lines = [d.model_dump_json() for d in kept] + malformed
        atomic_write_text(file_path, "".join(line + "\n" for line in lines))

    logger.info(
        "kept %d/%d decision(s) in %s (cutoff=%s, retired %d "
        "unresolved past %s, preserved %d malformed line(s))",
        len(kept),
        len(decisions),
        path,
        cutoff_naive.isoformat(),
        retired_unresolved,
        unresolved_cutoff_naive.isoformat(),
        len(malformed),
    )
    return result
