"""Locked decision JSONL persistence and checkpoint fallback loading."""

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
    """Hold a sidecar file lock across decision-log append or compaction."""
    log_path = Path(path)
    lock_path = log_path.with_suffix(log_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def append_decisions_jsonl(decisions: list[ARGUSDecision], path: str) -> int:
    """Append each decision to a JSONL log, one compact JSON object per line."""
    if not decisions:
        return 0

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Hold the sidecar lock so compaction cannot replace the log over this append.
    with decisions_log_lock(log_path), open(log_path, "a", encoding="utf-8") as f:
        for decision in decisions:
            stamped = decision.model_copy(update={"image_tag": settings.ARGUS_IMAGE_TAG})
            f.write(stamped.model_dump_json() + "\n")
    return len(decisions)


def load_decisions_from_checkpoints(db_path: str = "argus_graph.db") -> list[ARGUSDecision]:
    """Read every session's decisions back out of the LangGraph checkpoint database."""
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
    """Read a JSONL decision log, separating parsed decisions from lines that fail to parse."""
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
    """Read decisions back out of a JSONL decision log, one ARGUSDecision per line."""
    return _read_decisions_jsonl(path)[0]


@dataclass
class CompactionResult:
    """Outcome of one compact_decisions_jsonl() call."""

    retained: int
    retired_unresolved: int = 0
    malformed_preserved: int = 0


def compact_decisions_jsonl(
    path: str,
    cutoff: datetime,
    unresolved_cutoff: datetime,
    resolved_ids: Collection[str],
) -> CompactionResult:
    """Compact resolved history while retaining unresolved decisions until their longer retirement cutoff."""
    file_path = Path(path)
    if not file_path.exists():
        return CompactionResult(retained=0)

    cutoff_naive = as_naive_timestamp(cutoff)
    unresolved_cutoff_naive = as_naive_timestamp(unresolved_cutoff)

    # Lock the entire read and replace cycle so concurrent appends cannot be overwritten.
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

        # Keep unreadable lines verbatim so parser failures cannot permanently delete evidence.
        lines = [d.model_dump_json() for d in kept] + malformed
        atomic_write_text(file_path, "".join(line + "\n" for line in lines))

    logger.info(
        "kept %d/%d decision(s) in %s (cutoff=%s, retired %d unresolved past %s, preserved %d malformed line(s))",
        len(kept),
        len(decisions),
        path,
        cutoff_naive.isoformat(),
        retired_unresolved,
        unresolved_cutoff_naive.isoformat(),
        len(malformed),
    )
    return result
