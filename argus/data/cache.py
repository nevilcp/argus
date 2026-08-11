"""
argus/data/cache.py

SQLite-backed rolling cache for intraday price candles.

Responsibilities:
  - Buffer rolling intraday OHLCV candles with a configurable max-rows limit

Not responsible for:
  - Fetching raw data (see data/fetchers.py)
  - Semantic vector storage or decision archiving — the LangGraph checkpoint
    (argus_graph.db, see orchestration/graph.py) and ChromaDB
    (memory/cultural.py) cover this; see docs/adr/0010 for why a third,
    unused archive (formerly DecisionLogger, here) was deleted rather than
    wired up.

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
