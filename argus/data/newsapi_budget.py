"""
argus/data/newsapi_budget.py

NewsAPI's free-tier request budget, enforced across threads and processes.

Responsibilities:
  - Count requests against a 100/day cap with UTC-day rollover
  - Persist the count under ARGUS_DATA_DIR so the API server and the
    collector's once-per-tick container share one ledger

Not responsible for:
  - Making the NewsAPI request (see data/fetchers.py)
"""

from __future__ import annotations

import fcntl
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

from argus.config import settings
from argus.infra.files import atomic_write_text

logger = logging.getLogger(__name__)

NEWSAPI_DAILY_LIMIT = 100


class NewsApiBudget:
    """Tracks NewsAPI's free-tier 100 request/day budget with UTC-day rollover.

    A 20-ticker sweep spends 20 of the 100 — a 5-run/day ceiling, tighter than
    anything Groq imposes — so this is checked before every request rather
    than discovered from a 426/429 response.

    Persisted under ARGUS_DATA_DIR and read fresh on every reservation rather
    than cached in memory: the long-lived API server and the Actions
    collector's once-per-tick container both read-modify-write the same
    file, so a process-local cache of "today's count" would go stale the
    moment the other process wrote. An flock on a sidecar lock file
    serializes that read-modify-write cycle across processes (tmp-then-replace
    alone only makes each individual write atomic, not the cycle around it).
    Any disk error during that cycle is treated as a failed reservation
    rather than raised, so a full or read-only volume degrades the caller
    instead of crashing it.
    """

    def __init__(self, daily_limit: int = NEWSAPI_DAILY_LIMIT) -> None:
        """Initializes a budget tracker backed by ARGUS_DATA_DIR.

        Args:
            daily_limit: Maximum requests permitted per UTC day.
        """
        self._daily_limit = daily_limit

    def _persist_path(self) -> Path:
        """Returns the counter's path, read lazily so it honors the current ARGUS_DATA_DIR."""
        return settings.data_path("newsapi_budget.json")

    @staticmethod
    def _read_count(path: Path, today: date) -> int | None:
        """Returns today's persisted request count.

        Returns:
            The count; 0 if the file is absent or holds a previous day's count;
            None if its content is corrupt, which the caller must not mistake for
            an empty budget.
        """
        try:
            raw = json.loads(path.read_text())
            if date.fromisoformat(raw["date"]) == today:
                return int(raw["requests_today"])
        except FileNotFoundError:
            pass
        except KeyError, TypeError, ValueError:  # JSONDecodeError is a ValueError
            return None
        return 0

    @staticmethod
    def _write_count(path: Path, today: date, requests_today: int) -> None:
        """Writes the count atomically."""
        atomic_write_text(
            path, json.dumps({"date": today.isoformat(), "requests_today": requests_today})
        )

    def _reserve_under_file_lock(self) -> bool:
        """Reads, checks, and (if room remains) increments the persisted count.

        Holds an flock on a sidecar file for the whole read-modify-write
        cycle, so a concurrent thread or process can't interleave its own
        cycle in between — the source of the undercounting a Python-level
        Lock alone can't stop, since flock is scoped to the open file
        description and so already serializes both. The lock is released by
        closing `lock_file` on the `with` block's exit; no explicit unlock,
        since one that raised after a successful write would discard the
        `True` below and report a reservation that had, in fact, persisted.
        """
        path = self._persist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            today = datetime.now(UTC).date()
            requests_today = self._read_count(path, today)
            if requests_today is None:
                # Treating corrupt content as an empty budget would let a bad write
                # reset the day's count. Mark today as spent instead; tomorrow's date
                # makes the file stale, so the budget heals on its own.
                logger.error(
                    "%s is corrupt; blocking NewsAPI requests for the rest of the UTC day",
                    path,
                )
                self._write_count(path, today, self._daily_limit)
                return False
            if requests_today >= self._daily_limit:
                return False
            self._write_count(path, today, requests_today + 1)
            return True

    def try_reserve(self) -> bool:
        """Reserves one request against today's UTC budget if any remains.

        Returns:
            True if a request was reserved, False if today's budget is
            spent or the reservation could not be safely persisted.
        """
        try:
            return self._reserve_under_file_lock()
        except OSError as exc:
            logger.warning("disk error — %s", exc)
            return False
