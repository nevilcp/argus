"""
argus/data/cache.py

SQLite-backed persistence layer for caching price candles and trading decisions.

Responsibilities:
  - Buffer rolling intraday OHLCV candles with a configurable max-rows limit
  - Archive completed ARGUSDecision records for post-session auditing

Not responsible for:
  - Fetching raw data (see data/fetchers.py)
  - Semantic vector storage (see memory/cultural.py)

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

from argus.schemas.signals import ARGUSDecision

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


_CREATE_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id       TEXT PRIMARY KEY,
    ticker            TEXT,
    session_timestamp TEXT,
    signal            TEXT,
    conviction        REAL,
    allocation_pct    REAL,
    allocation_usd    REAL,
    stop_loss         REAL,
    macro_regime      TEXT,
    var_99            REAL,
    total_api_calls   INTEGER,
    full_json         TEXT
)
"""

_INSERT_DECISION = """
INSERT OR REPLACE INTO decisions (
    decision_id, ticker, session_timestamp, signal, conviction,
    allocation_pct, allocation_usd, stop_loss, macro_regime,
    var_99, total_api_calls, full_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_RECENT = """
SELECT * FROM decisions
WHERE ticker = ?
ORDER BY session_timestamp DESC
LIMIT ?
"""


class DecisionLogger:
    """Archives completed ARGUSDecision models to an SQLite database file for auditability.

    Flattens scalar decision fields into indexed columns while also persisting
    the complete JSON blob, enabling both tabular queries and full object reconstruction.
    """

    def __init__(self, db_path: str = "argus_decisions.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_DECISIONS)
        self._conn.commit()
        logger.info("DecisionLogger initialised (db=%s)", db_path)

    def log(self, decision: ARGUSDecision) -> None:
        """Persists a flattened decision profile and its complete JSON representation.

        Args:
            decision: Validated ARGUSDecision to archive.
        """
        signal = decision.aggregated.signal.value if decision.aggregated else None
        conviction = decision.aggregated.conviction if decision.aggregated else None

        alloc_pct = decision.allocation.allocation_pct if decision.allocation else None
        alloc_usd = decision.allocation.allocation_usd if decision.allocation else None
        stop_loss = decision.allocation.stop_loss if decision.allocation else None

        macro_regime = decision.macro.macro_regime.value if decision.macro else None
        var_99 = decision.risk.var_99 if decision.risk else None

        row = (
            decision.decision_id,
            decision.ticker,
            decision.session_timestamp.isoformat(),
            signal,
            conviction,
            alloc_pct,
            alloc_usd,
            stop_loss,
            macro_regime,
            var_99,
            decision.total_api_calls,
            decision.model_dump_json(),
        )

        with self._lock:
            self._conn.execute(_INSERT_DECISION, row)
            self._conn.commit()

        logger.info(
            "DecisionLogger.log: %s [%s] signal=%s conviction=%.2f",
            decision.ticker,
            decision.decision_id,
            signal,
            conviction or 0.0,
        )

    def get_recent(self, ticker: str, n: int = 20) -> list[dict]:
        """Retrieves the n most recent decision records matching a symbol.

        Args:
            ticker: Equity ticker symbol.
            n: Maximum number of records to return (default 20).

        Returns:
            List of dicts matching the decisions table schema, ordered by session_timestamp desc.
        """
        with self._lock:
            rows = self._conn.execute(_SELECT_RECENT, (ticker, n)).fetchall()
        result = [dict(r) for r in rows]
        logger.debug("DecisionLogger.get_recent: %s → %d records", ticker, len(result))
        return result

    def close(self) -> None:
        """Closes the database connection."""
        self._conn.close()
        logger.info("DecisionLogger closed (db=%s)", self._db_path)
