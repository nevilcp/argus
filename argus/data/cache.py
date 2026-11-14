"""Bounded TTL caches and SQLite intraday and daily candle stores."""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable, Hashable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from argus.config import settings
from argus.infra.clock import MARKET_TZ, now_local
from argus.params import SYSTEM

logger = logging.getLogger(__name__)


class TTLCache[K: Hashable, V]:
    """Capacity-bounded in-memory cache with lazy expiry."""

    def __init__(
        self,
        ttl: timedelta,
        clock: Callable[[], datetime] = now_local,
        *,
        max_entries: int,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._ttl = ttl
        self._clock = clock
        self._max_entries = max_entries
        self._entries: dict[K, tuple[V, datetime]] = {}
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """Return the cached value for key, or None if absent or expired."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            value, cached_at = entry
            if self._clock() - cached_at >= self._ttl:
                del self._entries[key]
                return None
            return value

    def set(self, key: K, value: V) -> None:
        """Store value under key, timestamped at the current clock time."""
        with self._lock:
            now = self._clock()
            expired = [k for k, (_, cached_at) in self._entries.items() if now - cached_at >= self._ttl]
            for k in expired:
                del self._entries[k]

            if key not in self._entries and len(self._entries) >= self._max_entries:
                oldest_key = min(self._entries, key=lambda k: self._entries[k][1])
                del self._entries[oldest_key]

            self._entries[key] = (value, now)


_CREATE_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker    TEXT NOT NULL,
    interval  TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    PRIMARY KEY (ticker, timestamp)
)
"""

_INSERT_OHLCV = """
INSERT OR REPLACE INTO ohlcv (ticker, interval, timestamp, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_PRUNE_OHLCV = """
DELETE FROM ohlcv
WHERE ticker = ?
  AND timestamp NOT IN (
      SELECT timestamp FROM ohlcv
      WHERE ticker = ?
      ORDER BY timestamp DESC
      LIMIT ?
  )
"""

_SELECT_CANDLES = """
SELECT timestamp, open, high, low, close, volume
FROM ohlcv
WHERE ticker = ? AND interval = ?
ORDER BY timestamp ASC
"""

_SELECT_TICKERS = """
SELECT DISTINCT ticker FROM ohlcv WHERE interval = ?
"""

_SELECT_ROW_COUNTS = """
SELECT ticker, COUNT(*) FROM ohlcv WHERE interval = ? GROUP BY ticker
"""

_SELECT_ROW_COUNT = """
SELECT COUNT(*) FROM ohlcv WHERE ticker = ? AND interval = ?
"""

_MIN_CANDLE_ROWS = 14


def _canonical_timestamp(ts: object) -> str:
    """Canonicalize a candle timestamp to UTC ISO-8601 for storage."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(MARKET_TZ.key)
    return stamp.tz_convert(UTC).isoformat()


class OHLCVBuffer:
    """Locked SQLite intraday candle buffer with WAL readers."""

    def __init__(
        self,
        db_path: str = ":memory:",
        buffer_size: int = SYSTEM.default_buffer_size,
        *,
        interval: str,
    ) -> None:
        self._db_path = db_path
        self._buffer_size = buffer_size
        self._interval = interval
        self._lock = threading.Lock()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate_schema()
        self._purge_stale_interval()
        logger.info(
            "OHLCVBuffer initialized (db=%s, buffer_size=%d, interval=%s)",
            db_path,
            buffer_size,
            interval,
        )

    def _migrate_schema(self) -> None:
        """Drop and recreates `ohlcv` once, adding the per-row `interval` column."""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            self._conn.execute("DROP TABLE IF EXISTS ohlcv")
            self._conn.execute("PRAGMA user_version = 1")
        self._conn.execute(_CREATE_OHLCV)
        self._conn.commit()

    def _purge_stale_interval(self) -> None:
        """Discard rows written under a different interval than this buffer's configured one."""
        cursor = self._conn.execute("DELETE FROM ohlcv WHERE interval != ?", (self._interval,))
        self._conn.commit()
        if cursor.rowcount:
            logger.warning(
                "interval is now %s, discarded %d row(s) written under a different interval",
                self._interval,
                cursor.rowcount,
            )

    def insert_candle(self, ticker: str, candle: dict) -> None:
        """Insert a single candlestick entry and prunes history past the buffer limit."""
        self.insert_candles(ticker, [candle])

    def insert_candles(self, ticker: str, candles: list[dict]) -> None:
        """Bulk-insert candlestick entries in one transaction and prunes once."""
        if not candles:
            return

        ticker = ticker.upper()
        # Offset untimestamped candles within one clock sample so coarse clocks cannot collapse batch primary keys.
        now = datetime.now(UTC)
        missing_count = 0
        rows = []
        for candle in candles:
            ts = candle.get("timestamp")
            if ts is None:
                ts = now + timedelta(microseconds=missing_count)
                missing_count += 1
            rows.append(
                (
                    ticker,
                    self._interval,
                    _canonical_timestamp(ts),
                    candle.get("open"),
                    candle.get("high"),
                    candle.get("low"),
                    candle.get("close"),
                    candle.get("volume"),
                )
            )

        with self._lock:
            self._conn.executemany(_INSERT_OHLCV, rows)
            self._conn.execute(_PRUNE_OHLCV, (ticker, ticker, self._buffer_size))
            self._conn.commit()

        logger.debug("%s <- %d candle(s)", ticker, len(rows))

    def get_candles(self, ticker: str) -> pd.DataFrame | None:
        """Retrieve buffered candles as a pandas DataFrame, tz-converted to ET."""
        ticker = ticker.upper()
        with self._lock:
            rows = self._conn.execute(_SELECT_CANDLES, (ticker, self._interval)).fetchall()

        if len(rows) < _MIN_CANDLE_ROWS:
            logger.debug(
                "%s has only %d rows (min %d)",
                ticker,
                len(rows),
                _MIN_CANDLE_ROWS,
            )
            return None

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Parse mixed timestamp offsets as UTC, then convert to Eastern before wall-clock resampling and VWAP.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df = df.tz_convert(MARKET_TZ.key)
        df = df.astype(float)
        logger.debug("%s -> %d rows", ticker, len(df))
        return df

    def get_all_tickers(self) -> list[str]:
        """Return a list of distinct tickers currently cached in the buffer."""
        with self._lock:
            rows = self._conn.execute(_SELECT_TICKERS, (self._interval,)).fetchall()
        return [r[0] for r in rows]

    def prune_untracked(self, tracked_tickers: set[str]) -> int:
        """Delete every row for a ticker outside the given tracked set."""
        if not tracked_tickers:
            return 0
        tracked = {t.upper() for t in tracked_tickers}
        placeholders = ",".join("?" for _ in tracked)
        with self._lock:
            cursor = self._conn.execute(f"DELETE FROM ohlcv WHERE ticker NOT IN ({placeholders})", tuple(tracked))
            self._conn.commit()
        if cursor.rowcount:
            logger.info("discarded %d row(s)", cursor.rowcount)
        return cursor.rowcount

    def row_counts(self) -> dict[str, int]:
        """Count stored candles without loading frames or applying the indicator-readiness floor."""
        with self._lock:
            rows = self._conn.execute(_SELECT_ROW_COUNTS, (self._interval,)).fetchall()
        return dict(rows)

    def row_count(self, ticker: str) -> int:
        """Return one ticker's row count, without counting every other ticker's rows."""
        with self._lock:
            row = self._conn.execute(_SELECT_ROW_COUNT, (ticker.upper(), self._interval)).fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
        logger.info("OHLCVBuffer closed (db=%s)", self._db_path)


_CREATE_DAILY_OHLCV = """
CREATE TABLE IF NOT EXISTS daily_ohlcv (
    ticker        TEXT NOT NULL,
    trading_date  TEXT NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    volume        REAL,
    PRIMARY KEY (ticker, trading_date)
)
"""

_CREATE_DAILY_REFRESH = """
CREATE TABLE IF NOT EXISTS daily_ohlcv_refresh (
    ticker        TEXT PRIMARY KEY,
    refreshed_on  TEXT NOT NULL
)
"""

_INSERT_DAILY_OHLCV = """
INSERT OR REPLACE INTO daily_ohlcv (ticker, trading_date, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_UPSERT_DAILY_REFRESH = """
INSERT INTO daily_ohlcv_refresh (ticker, refreshed_on) VALUES (?, ?)
ON CONFLICT(ticker) DO UPDATE SET refreshed_on = excluded.refreshed_on
"""

_SELECT_DAILY_REFRESH = "SELECT refreshed_on FROM daily_ohlcv_refresh WHERE ticker = ?"

_DELETE_STALE_DAILY_OHLCV = "DELETE FROM daily_ohlcv WHERE ticker = ? AND trading_date < ?"

_SELECT_DAILY_OHLCV = """
SELECT trading_date, open, high, low, close, volume
FROM daily_ohlcv
WHERE ticker = ?
ORDER BY trading_date ASC
"""


class DailyBarCache:
    """Daily candle cache with UTC refresh markers and bounded history."""

    def __init__(self, db_path: str | None = None, *, max_tickers: int = SYSTEM.max_tracked_tickers + 1) -> None:
        if max_tickers < 1:
            raise ValueError("max_tickers must be positive")
        self._max_tickers = max_tickers
        self._db_path = db_path if db_path is not None else str(settings.data_path("daily_bars_cache.db"))
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_DAILY_OHLCV)
        self._conn.execute(_CREATE_DAILY_REFRESH)
        self._conn.commit()
        logger.info("DailyBarCache initialized (db=%s)", db_path)

    def get(self, ticker: str) -> pd.DataFrame | None:
        """Return a ticker's cached daily bars if refreshed today (UTC), else None."""
        today = datetime.now(UTC).date().isoformat()
        with self._lock:
            refreshed = self._conn.execute(_SELECT_DAILY_REFRESH, (ticker,)).fetchone()
            if refreshed is None or refreshed[0] != today:
                return None
            rows = self._conn.execute(_SELECT_DAILY_OHLCV, (ticker,)).fetchall()

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["trading_date", "open", "high", "low", "close", "volume"])
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        df.set_index("trading_date", inplace=True)
        df.index.name = None
        logger.debug("%s -> %d cached rows", ticker, len(df))
        return df

    def put(self, ticker: str, df: pd.DataFrame) -> None:
        """Store a ticker's freshly fetched daily bars and marks it refreshed for today."""
        today = datetime.now(UTC).date().isoformat()
        rows = [
            (
                ticker,
                idx.date().isoformat(),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume"),
            )
            for idx, row in df.iterrows()
        ]

        with self._lock:
            self._conn.executemany(_INSERT_DAILY_OHLCV, rows)
            if not df.empty:
                earliest = df.index.min().date().isoformat()
                self._conn.execute(_DELETE_STALE_DAILY_OHLCV, (ticker, earliest))
            self._conn.execute(_UPSERT_DAILY_REFRESH, (ticker, today))
            retained = "SELECT ticker FROM daily_ohlcv_refresh ORDER BY refreshed_on DESC, rowid DESC LIMIT ?"
            self._conn.execute(f"DELETE FROM daily_ohlcv WHERE ticker NOT IN ({retained})", (self._max_tickers,))
            self._conn.execute(
                f"DELETE FROM daily_ohlcv_refresh WHERE ticker NOT IN ({retained})",
                (self._max_tickers,),
            )
            self._conn.commit()

        logger.debug("%s <- %d rows, refreshed_on=%s", ticker, len(rows), today)

    def prune_untracked(self, tracked_tickers: set[str]) -> int:
        """Remove bars and refresh markers outside the current daily-fetch universe."""
        tracked = tuple(t.upper() for t in tracked_tickers)
        placeholders = ",".join("?" for _ in tracked)
        clause = f"WHERE ticker NOT IN ({placeholders})" if tracked else ""
        with self._lock:
            cursor = self._conn.execute(f"DELETE FROM daily_ohlcv {clause}", tracked)
            self._conn.execute(f"DELETE FROM daily_ohlcv_refresh {clause}", tracked)
            self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
        logger.info("DailyBarCache closed (db=%s)", self._db_path)
