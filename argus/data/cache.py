"""
argus/data/cache.py
===================
SQLite-backed persistence layer for ARGUS v2.

Two classes are provided:

OHLCVBuffer
    In-memory (or file-backed) SQLite store for the rolling 78-candle MFT
    window.  Designed for high-frequency writes from the data pipeline; uses
    an in-memory DB by default for maximum throughput.

DecisionLogger
    Persistent SQLite store (``argus_decisions.db``) that archives every
    completed ``ARGUSDecision`` as both a flattened summary row and a full
    JSON blob.  Provides the audit trail required by the Governor and the
    Streamlit memory browser.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger("argus.cache")

# ──────────────────────────────────────────────────────────────────────────────
# OHLCVBuffer
# ──────────────────────────────────────────────────────────────────────────────

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
    """
    Rolling OHLCV buffer backed by SQLite.

    Keeps the last ``buffer_size`` candles per ticker.  When the buffer is
    full, the oldest rows are pruned after each insert.  An in-memory SQLite
    database (``:memory:``) is used by default; pass a file path for
    persistence across process restarts.

    Thread-safety
    -------------
    A ``threading.Lock`` serialises all writes.  Multiple threads in the
    pipeline fetch loop may call :meth:`insert_candle` concurrently without
    data corruption.

    Parameters
    ----------
    db_path:
        SQLite connection string.  Use ``":memory:"`` (default) for speed or
        a file path for warm-start persistence.
    buffer_size:
        Maximum number of candles to retain per ticker (default 78 ≈ one
        full 6.5-hour trading session at 5-minute resolution).
    """

    def __init__(self, db_path: str = ":memory:", buffer_size: int = 78) -> None:
        self._db_path    = db_path
        self._buffer_size = buffer_size
        self._lock       = threading.Lock()
        self._conn       = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_OHLCV)
        self._conn.commit()
        logger.info("OHLCVBuffer initialised (db=%s, buffer_size=%d)", db_path, buffer_size)

    # ── Writes ────────────────────────────────────────────────────────────────

    def insert_candle(self, ticker: str, candle: dict) -> None:
        """
        Insert a single 5-minute candle for *ticker* and prune old rows.

        Parameters
        ----------
        ticker:
            Equity symbol.
        candle:
            Dict with keys ``timestamp`` (ISO-8601 str or datetime),
            ``open``, ``high``, ``low``, ``close``, ``volume``.
        """
        ts = candle.get("timestamp")
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        elif ts is None:
            ts = datetime.now(timezone.utc).isoformat()

        row = (
            ticker, str(ts),
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

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_candles(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Return buffered candles for *ticker* as a DataFrame.

        Returns ``None`` when fewer than 14 rows are available (insufficient
        for the shortest indicator window — RSI-14).

        Returns
        -------
        pd.DataFrame or None
            Columns: ``open, high, low, close, volume``.
            Index: ``DatetimeIndex`` sorted ascending.
        """
        with self._lock:
            rows = self._conn.execute(_SELECT_CANDLES, (ticker,)).fetchall()

        if len(rows) < 14:
            logger.debug(
                "OHLCVBuffer.get_candles: %s has only %d rows (min 14)", ticker, len(rows)
            )
            return None

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        logger.debug("OHLCVBuffer.get_candles: %s → %d rows", ticker, len(df))
        return df

    def get_all_tickers(self) -> list[str]:
        """Return a list of distinct tickers that have data in the buffer."""
        with self._lock:
            rows = self._conn.execute(_SELECT_TICKERS).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
        logger.info("OHLCVBuffer closed (db=%s)", self._db_path)


# ──────────────────────────────────────────────────────────────────────────────
# DecisionLogger
# ──────────────────────────────────────────────────────────────────────────────

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
    """
    Persistent decision archive backed by a SQLite file database.

    Each completed :class:`~argus.schemas.signals.ARGUSDecision` is stored
    as a flattened summary row (for fast SQL queries) together with the full
    Pydantic JSON blob (for complete audit trail / memory retrieval).

    Parameters
    ----------
    db_path:
        Path to the SQLite database file (default ``"argus_decisions.db"``).
        The file is created if it does not exist.
    """

    def __init__(self, db_path: str = "argus_decisions.db") -> None:
        self._db_path = db_path
        self._lock    = threading.Lock()
        self._conn    = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_DECISIONS)
        self._conn.commit()
        logger.info("DecisionLogger initialised (db=%s)", db_path)

    # ── Writes ────────────────────────────────────────────────────────────────

    def log(self, decision: ARGUSDecision) -> None:
        """
        Persist *decision* to the database.

        Extracts commonly-queried scalar fields into dedicated columns and
        serialises the full Pydantic model to JSON for the ``full_json``
        column.

        Parameters
        ----------
        decision:
            A completed :class:`~argus.schemas.signals.ARGUSDecision` instance.
        """
        # ── Flatten scalar fields ──────────────────────────────────────────
        signal     = decision.aggregated.signal.value if decision.aggregated else None
        conviction = decision.aggregated.conviction   if decision.aggregated else None

        alloc_pct  = decision.allocation.allocation_pct if decision.allocation else None
        alloc_usd  = decision.allocation.allocation_usd if decision.allocation else None
        stop_loss  = decision.allocation.stop_loss      if decision.allocation else None

        macro_regime = decision.macro.macro_regime.value if decision.macro else None
        var_99       = decision.risk.var_99              if decision.risk   else None

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
            decision.ticker, decision.decision_id, signal, conviction or 0.0,
        )

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_recent(self, ticker: str, n: int = 20) -> list[dict]:
        """
        Return the *n* most recent decision records for *ticker*.

        Parameters
        ----------
        ticker:
            Equity symbol.
        n:
            Maximum number of records to return (default 20).

        Returns
        -------
        list[dict]
            Each element is a flat dict of all columns including ``full_json``.
        """
        with self._lock:
            rows = self._conn.execute(_SELECT_RECENT, (ticker, n)).fetchall()
        result = [dict(r) for r in rows]
        logger.debug("DecisionLogger.get_recent: %s → %d records", ticker, len(result))
        return result

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
        logger.info("DecisionLogger closed (db=%s)", self._db_path)
