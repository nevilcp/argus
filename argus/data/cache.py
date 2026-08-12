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
from typing import Optional

import pandas as pd

logger = logging.getLogger("argus.cache")

_CREATE_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker    TEXT NOT NULL,
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
INSERT OR REPLACE INTO ohlcv (ticker, timestamp, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?)
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
WHERE ticker = ?
ORDER BY timestamp ASC
"""

_SELECT_TICKERS = """
SELECT DISTINCT ticker FROM ohlcv
"""


class OHLCVBuffer:
    """Rolling in-memory candle cache backed by SQLite to support real-time indicators.

    WAL mode is enabled to allow concurrent readers while the buffer is being written
    by the async fetch loop. Thread safety is enforced by a single module-level lock.
    """

    def __init__(self, db_path: str = ":memory:", buffer_size: int = 78) -> None:
        """Opens (or creates) the SQLite-backed candle buffer.

        Args:
            db_path: SQLite connection path, or ':memory:' for an ephemeral buffer.
            buffer_size: Max candles retained per ticker; older rows are pruned on insert.
        """
        self._db_path = db_path
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_OHLCV)
        self._conn.commit()
        logger.info("OHLCVBuffer initialised (db=%s, buffer_size=%d)", db_path, buffer_size)

    def insert_candle(self, ticker: str, candle: dict) -> None:
        """Inserts a single candlestick entry and prunes historical values exceeding the buffer limit.

        Args:
            ticker: Equity ticker symbol.
            candle: Dict with keys timestamp, open, high, low, close, volume.
        """
        ts = candle.get("timestamp")
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        elif ts is None:
            ts = datetime.now(timezone.utc).isoformat()

        row = (
            ticker,
            str(ts),
            candle.get("open"),
            candle.get("high"),
            candle.get("low"),
            candle.get("close"),
            candle.get("volume"),
        )

        with self._lock:
            self._conn.execute(_INSERT_OHLCV, row)
            self._conn.execute(_PRUNE_OHLCV, (ticker, ticker, self._buffer_size))
            self._conn.commit()

        logger.debug("OHLCVBuffer.insert_candle: %s @ %s", ticker, ts)

    def get_candles(self, ticker: str) -> Optional[pd.DataFrame]:
        """Retrieves buffered candles as a pandas DataFrame indexed by timestamp.

        Returns None when fewer than 14 rows exist, as most indicators require at
        least that many periods to produce meaningful values.

        Args:
            ticker: Equity ticker symbol.

        Returns:
            DataFrame with float OHLCV columns and a datetime index, or None.
        """
        with self._lock:
            rows = self._conn.execute(_SELECT_CANDLES, (ticker,)).fetchall()

        if len(rows) < 14:
            logger.debug(
                "OHLCVBuffer.get_candles: %s has only %d rows (min 14)", ticker, len(rows)
            )
            return None

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        logger.debug("OHLCVBuffer.get_candles: %s → %d rows", ticker, len(df))
        return df

    def get_all_tickers(self) -> list[str]:
        """Returns a list of distinct tickers currently cached in the buffer.

        Returns:
            List of ticker strings.
        """
        with self._lock:
            rows = self._conn.execute(_SELECT_TICKERS).fetchall()
        return [r[0] for r in rows]

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
            self._conn.execute(_UPSERT_DAILY_REFRESH, (ticker, today))
            self._conn.commit()

        logger.debug("DailyBarCache.put: %s ← %d rows, refreshed_on=%s", ticker, len(rows), today)

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._conn.close()
        logger.info("DailyBarCache closed (db=%s)", self._db_path)
