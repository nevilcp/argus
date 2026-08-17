"""
argus/data/cache.py

SQLite-backed caches for OHLCV price candles: a rolling intraday buffer and a
per-trading-day cache for daily bars.

Responsibilities:
  - Buffer rolling intraday OHLCV candles with a configurable max-rows limit
  - Cache daily OHLCV bars keyed by (ticker, trading_date), so a ticker
    refreshed earlier the same UTC day is served from disk rather than refetched

Not responsible for:
  - Fetching raw data (see data/fetchers.py)
  - Semantic vector storage or decision archiving — the LangGraph checkpoint
    (argus_graph.db, see orchestration/graph.py) and ChromaDB
    (memory/cultural.py) cover this; a third, unused archive (formerly
    DecisionLogger, here) was deleted rather than wired up.

Dependencies:
  - sqlite3 (stdlib)
  - pandas
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("argus.cache")

_ET = "America/New_York"

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


def _canonical_timestamp(ts: object) -> str:
    """Canonicalizes a candle timestamp to UTC ISO-8601 for storage.

    Naive input is interpreted as ET — every naive timestamp reaching this
    buffer comes from a US session.

    Args:
        ts: datetime, pandas Timestamp, or ISO-8601 string; naive or aware.

    Returns:
        UTC ISO-8601 string.
    """
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(_ET)
    return stamp.tz_convert(timezone.utc).isoformat()


class OHLCVBuffer:
    """Rolling in-memory candle cache backed by SQLite to support real-time indicators.

    WAL mode is enabled to allow concurrent readers while the buffer is being written
    by the async fetch loop. Thread safety is enforced by a single module-level lock.
    """

    def __init__(self, db_path: str = ":memory:", buffer_size: int = 78, *, interval: str) -> None:
        """Opens (or creates) the SQLite-backed candle buffer.

        Args:
            db_path: SQLite connection path, or ':memory:' for an ephemeral buffer.
            buffer_size: Max candles retained per ticker; older rows are pruned on insert.
            interval: Candle resolution this buffer holds (e.g. "1m"). Rows
                written under a different interval are purged on open —
                mixed-resolution rows would silently corrupt resampled
                indicators.
        """
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
            "OHLCVBuffer initialised (db=%s, buffer_size=%d, interval=%s)",
            db_path,
            buffer_size,
            interval,
        )

    def _migrate_schema(self) -> None:
        """Drops and recreates `ohlcv` once, adding the per-row `interval` column.

        `user_version` 0→1 marks the migration as applied so it never re-runs
        against a buffer already on the new schema.
        """
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            self._conn.execute("DROP TABLE IF EXISTS ohlcv")
            self._conn.execute("PRAGMA user_version = 1")
        self._conn.execute(_CREATE_OHLCV)
        self._conn.commit()

    def _purge_stale_interval(self) -> None:
        """Discards rows written under a different interval than this buffer's configured one."""
        cursor = self._conn.execute("DELETE FROM ohlcv WHERE interval != ?", (self._interval,))
        self._conn.commit()
        if cursor.rowcount:
            logger.warning(
                "OHLCVBuffer: interval is now %s, discarded %d row(s) written under a "
                "different interval",
                self._interval,
                cursor.rowcount,
            )

    def insert_candle(self, ticker: str, candle: dict) -> None:
        """Inserts a single candlestick entry and prunes historical values exceeding the buffer limit.

        Args:
            ticker: Equity ticker symbol; case-insensitive, normalized to upper.
            candle: Dict with keys timestamp, open, high, low, close, volume.
                timestamp may be naive (interpreted as ET) or tz-aware; stored
                as canonical UTC ISO-8601.
        """
        self.insert_candles(ticker, [candle])

    def insert_candles(self, ticker: str, candles: list[dict]) -> None:
        """Bulk-inserts candlestick entries in one transaction and prunes once.

        A row-by-row insert with a commit per row measured ~655ms/ticker for a
        full 2-day fetch; one `executemany` and one commit for the whole batch
        is 406x faster and the natural shape for `_fetch_one_ticker`'s
        per-sweep bulk load.

        Args:
            ticker: Equity ticker symbol; case-insensitive, normalized to upper.
            candles: Dicts with keys timestamp, open, high, low, close, volume.
                Each timestamp may be naive (interpreted as ET) or tz-aware;
                stored as canonical UTC ISO-8601.
        """
        if not candles:
            return

        ticker = ticker.upper()
        rows = []
        for candle in candles:
            ts = candle.get("timestamp")
            if ts is None:
                ts = datetime.now(timezone.utc)
            ts = _canonical_timestamp(ts)
            rows.append((
                ticker,
                self._interval,
                ts,
                candle.get("open"),
                candle.get("high"),
                candle.get("low"),
                candle.get("close"),
                candle.get("volume"),
            ))

        with self._lock:
            self._conn.executemany(_INSERT_OHLCV, rows)
            self._conn.execute(_PRUNE_OHLCV, (ticker, ticker, self._buffer_size))
            self._conn.commit()

        logger.debug("OHLCVBuffer.insert_candles: %s ← %d candle(s)", ticker, len(rows))

    def get_candles(self, ticker: str) -> Optional[pd.DataFrame]:
        """Retrieves buffered candles as a pandas DataFrame, tz-converted to ET.

        Returns None when fewer than 14 rows exist, as most indicators require at
        least that many periods to produce meaningful values.

        Args:
            ticker: Equity ticker symbol; case-insensitive.

        Returns:
            DataFrame with float OHLCV columns and a tz-aware (ET) datetime
            index, sorted ascending, or None.
        """
        ticker = ticker.upper()
        with self._lock:
            rows = self._conn.execute(_SELECT_CANDLES, (ticker, self._interval)).fetchall()

        if len(rows) < 14:
            logger.debug(
                "OHLCVBuffer.get_candles: %s has only %d rows (min 14)", ticker, len(rows)
            )
            return None

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # utc=True + format="ISO8601" is load-bearing: utc=True alone still raises when naive
        # and aware strings coexist, and tz_convert must run before any indicator math since
        # resampling/VWAP bin by wall clock
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df = df.tz_convert(_ET)
        df = df.astype(float)
        logger.debug("OHLCVBuffer.get_candles: %s → %d rows", ticker, len(df))
        return df

    def get_all_tickers(self) -> list[str]:
        """Returns a list of distinct tickers currently cached in the buffer.

        Returns:
            List of ticker strings.
        """
        with self._lock:
            rows = self._conn.execute(_SELECT_TICKERS, (self._interval,)).fetchall()
        return [r[0] for r in rows]

    def prune_untracked(self, tracked_tickers: set[str]) -> int:
        """Deletes every row for a ticker outside the given tracked set (API-9/API-10).

        Without this, a ticker fetched once (e.g. a one-off `/analyze` request)
        keeps being swept and compressed forever, since `compress_all` used to
        iterate every ticker ever inserted rather than the tracked universe.

        Args:
            tracked_tickers: Tickers that should be retained; empty is treated
                as "unknown universe" and skipped rather than wiping the buffer.

        Returns:
            Row count deleted.
        """
        if not tracked_tickers:
            return 0
        tracked = {t.upper() for t in tracked_tickers}
        placeholders = ",".join("?" for _ in tracked)
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM ohlcv WHERE ticker NOT IN ({placeholders})", tuple(tracked)
            )
            self._conn.commit()
        if cursor.rowcount:
            logger.info("OHLCVBuffer.prune_untracked: discarded %d row(s)", cursor.rowcount)
        return cursor.rowcount

    def row_counts(self) -> dict[str, int]:
        """Returns each tracked ticker's row count without materializing its candle frame.

        A `GROUP BY COUNT(*)` rather than one `get_candles()` call per ticker —
        the latter builds and tz-converts a full DataFrame just to call `len()`
        on it, and its own 14-row floor would report 0 for a ticker holding
        fewer rows than that even though the row count is real.

        Returns:
            Mapping of ticker → row count, for this buffer's configured interval.
        """
        with self._lock:
            rows = self._conn.execute(_SELECT_ROW_COUNTS, (self._interval,)).fetchall()
        return {ticker: count for ticker, count in rows}

    def close(self) -> None:
        """Closes the underlying database connection."""
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
    """Disk-backed cache of daily OHLCV bars keyed by (ticker, trading_date).

    Daily bars only change once per trading day, so a ticker already refreshed
    today never needs a second network round-trip today no matter how many
    times a sweep runs. WAL mode allows concurrent readers/writers across the
    ThreadPoolExecutor workers in fetchers.fetch_multiple_daily.
    """

    def __init__(self, db_path: str = "daily_bars_cache.db") -> None:
        """Opens (or creates) the SQLite-backed daily bar cache.

        Args:
            db_path: SQLite connection path. Defaults to a process-relative
                file so the cache survives process restarts within the same
                UTC day; pass ':memory:' for an ephemeral cache in tests.
        """
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_DAILY_OHLCV)
        self._conn.execute(_CREATE_DAILY_REFRESH)
        self._conn.commit()
        logger.info("DailyBarCache initialised (db=%s)", db_path)

    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        """Returns a ticker's cached daily bars if refreshed today (UTC), else None.

        Args:
            ticker: Equity ticker symbol.

        Returns:
            Sorted DataFrame with lowercase OHLCV columns and a datetime index,
            or None if the ticker hasn't been refreshed since UTC midnight.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            refreshed = self._conn.execute(_SELECT_DAILY_REFRESH, (ticker,)).fetchone()
            if refreshed is None or refreshed[0] != today:
                return None
            rows = self._conn.execute(_SELECT_DAILY_OHLCV, (ticker,)).fetchall()

        if not rows:
            return None

        df = pd.DataFrame(
            rows, columns=["trading_date", "open", "high", "low", "close", "volume"]
        )
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        df.set_index("trading_date", inplace=True)
        df.index.name = None
        logger.debug("DailyBarCache.get: %s → %d cached rows", ticker, len(df))
        return df

    def put(self, ticker: str, df: pd.DataFrame) -> None:
        """Stores a ticker's freshly fetched daily bars and marks it refreshed for today.

        Also prunes any rows older than the fetched window (RE-14) — each
        `fetch_ohlcv_daily(ticker, period=...)` call re-supplies a full
        trailing window, but INSERT OR REPLACE never removed the prior
        window's now-stale rows, so daily_ohlcv grew by one row per ticker
        per day forever.

        Args:
            ticker: Equity ticker symbol.
            df: DataFrame with lowercase OHLCV columns and a datetime index,
                as returned by fetchers.fetch_ohlcv_daily.
        """
        today = datetime.now(timezone.utc).date().isoformat()
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
            self._conn.commit()

        logger.debug("DailyBarCache.put: %s ← %d rows, refreshed_on=%s", ticker, len(rows), today)

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._conn.close()
        logger.info("DailyBarCache closed (db=%s)", self._db_path)
