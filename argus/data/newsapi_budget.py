"""Persistent NewsAPI request reservations across threads and processes."""

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
    """UTC-day NewsAPI budget with locked reservations and fail-closed persistence."""

    def __init__(self, daily_limit: int = NEWSAPI_DAILY_LIMIT) -> None:
        self._daily_limit = daily_limit

    def _persist_path(self) -> Path:
        """Return the counter's path, read lazily so it honors the current ARGUS_DATA_DIR."""
        return settings.data_path("newsapi_budget.json")

    @staticmethod
    def _read_count(path: Path, today: date) -> int | None:
        """Return today's persisted request count."""
        try:
            raw = json.loads(path.read_text())
            if date.fromisoformat(raw["date"]) == today:
                return int(raw["requests_today"])
        except FileNotFoundError:
            pass
        except KeyError, TypeError, ValueError:
            return None
        return 0

    @staticmethod
    def _write_count(path: Path, today: date, requests_today: int) -> None:
        """Write the count atomically."""
        atomic_write_text(path, json.dumps({"date": today.isoformat(), "requests_today": requests_today}))

    def _reserve_under_file_lock(self) -> bool:
        """Reserve one request under a file lock when the daily budget allows it."""
        path = self._persist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            today = datetime.now(UTC).date()
            requests_today = self._read_count(path, today)
            if requests_today is None:
                # Treat a corrupt ledger as spent for today; resetting it would bypass the daily request budget.
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
        """Reserve one request against today's UTC budget if any remains."""
        try:
            return self._reserve_under_file_lock()
        except OSError as exc:
            logger.warning("disk error - %s", exc)
            return False
