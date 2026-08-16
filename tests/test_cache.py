"""
Tests for argus/data/cache.py: DailyBarCache, the per-(ticker, trading_date)
disk cache that lets fetch_multiple_daily skip the network on same-day
re-runs, and OHLCVBuffer, the rolling intraday candle buffer.
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from argus.data.cache import DailyBarCache, OHLCVBuffer
from tests.helpers.candles import dst_straddling_candles


def _bars(dates: list[str]) -> pd.DataFrame:
    """Builds a minimal OHLCV DataFrame with one row per given date.

    Args:
        dates: ISO date strings, one per row.

    Returns:
        DataFrame with lowercase OHLCV columns and a datetime index.
    """
    n = len(dates)
    return pd.DataFrame(
        {
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [float(i) for i in range(n)],
            "volume": [100.0] * n,
        },
        index=pd.to_datetime(dates),
    )


@pytest.fixture
def cache():
    """A fresh in-memory DailyBarCache."""
    return DailyBarCache(db_path=":memory:")


def test_get_returns_none_for_uncached_ticker(cache):
    """A ticker never stored has no cached bars."""
    assert cache.get("AAPL") is None


def test_put_then_get_round_trips(cache):
    """A stored ticker's bars come back sorted with the right values, refreshed today."""
    df = _bars(["2026-08-10", "2026-08-11", "2026-08-12"])
    cache.put("AAPL", df)

    result = cache.get("AAPL")

    assert result is not None
    assert len(result) == 3
    assert list(result["close"]) == [0.0, 1.0, 2.0]


def test_get_returns_none_when_refreshed_before_today(cache):
    """A ticker refreshed on a prior UTC day is treated as stale, not served."""
    cache.put("AAPL", _bars(["2026-08-11"]))

    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    cache._conn.execute(
        "UPDATE daily_ohlcv_refresh SET refreshed_on = ? WHERE ticker = ?", (yesterday, "AAPL")
    )
    cache._conn.commit()

    assert cache.get("AAPL") is None


def test_put_overwrites_prior_rows_for_same_ticker(cache):
    """A second put() for the same ticker replaces (not appends to) its cached bars."""
    cache.put("AAPL", _bars(["2026-08-11"]))
    cache.put("AAPL", _bars(["2026-08-11", "2026-08-12"]))

    result = cache.get("AAPL")
    assert len(result) == 2


def test_get_is_isolated_per_ticker(cache):
    """Caching one ticker doesn't make an unrelated ticker appear cached."""
    cache.put("AAPL", _bars(["2026-08-12"]))
    assert cache.get("MSFT") is None


def test_dst_straddling_insert_returns_a_monotonic_et_frame():
    """A buffer fed DST fall-back candles returns them sorted, ET-indexed, not a raise."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=20, interval="1m")

    # Padding bars ahead of the transition so the buffer clears get_candles' 14-row floor —
    # n_per_side beyond 6 would overlap pre/post UTC instants in the shared DST helper itself
    padding = pd.date_range("2025-11-02 04:00:00", periods=4, freq="1min", tz="UTC")
    for i, ts in enumerate(padding):
        buffer.insert_candle(
            "AAPL", {"timestamp": ts, "open": i, "high": i + 1, "low": i, "close": i, "volume": 1000}
        )

    candles = dst_straddling_candles(n_per_side=6)
    for idx, row in candles.iterrows():
        buffer.insert_candle("AAPL", {"timestamp": idx, **row.to_dict()})

    result = buffer.get_candles("AAPL")

    assert result is not None
    assert len(result) == len(padding) + len(candles)
    assert result.index.is_monotonic_increasing
    assert str(result.index.tz) == "America/New_York"


def test_naive_et_and_aware_utc_inputs_for_one_instant_store_identically():
    """The same instant, given as naive ET, aware ET, and aware UTC, canonicalizes to one row."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=20, interval="1m")
    aware_et = pd.Timestamp("2024-01-02T09:30:00", tz="America/New_York")

    buffer.insert_candle("AAPL", {"timestamp": "2024-01-02T09:30:00", "close": 1.0})
    buffer.insert_candle("AAPL", {"timestamp": aware_et, "close": 2.0})
    buffer.insert_candle("AAPL", {"timestamp": aware_et.tz_convert("UTC"), "close": 3.0})

    rows = buffer._conn.execute("SELECT close FROM ohlcv WHERE ticker = 'AAPL'").fetchall()
    assert rows == [(3.0,)]


def test_ticker_case_collapses_to_one_key_space():
    """A lowercase and uppercase ticker write into the same row space."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=20, interval="1m")
    buffer.insert_candle("aapl", {"timestamp": "2024-01-02T09:30:00", "close": 1.0})
    buffer.insert_candle("AAPL", {"timestamp": "2024-01-02T09:31:00", "close": 2.0})

    assert buffer.get_all_tickers() == ["AAPL"]


def test_schema_migration_from_v0_drops_and_recreates(tmp_path):
    """A pre-PR1 buffer file (no interval column, user_version 0) migrates without raising."""
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE ohlcv (
            ticker TEXT NOT NULL, timestamp TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (ticker, timestamp)
        )
        """
    )
    conn.execute("INSERT INTO ohlcv VALUES ('AAPL', '2024-01-02T09:30:00', 1, 1, 1, 1, 100)")
    conn.commit()
    conn.close()

    buffer = OHLCVBuffer(db_path=db_path, buffer_size=20, interval="1m")

    assert buffer.get_all_tickers() == []
    assert buffer._conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_interval_change_purges_stale_rows(tmp_path):
    """Re-opening a buffer file under a different interval discards the old interval's rows."""
    db_path = str(tmp_path / "buffer.db")
    old = OHLCVBuffer(db_path=db_path, buffer_size=20, interval="1m")
    old.insert_candle("AAPL", {"timestamp": "2024-01-02T09:30:00", "close": 1.0})
    old.close()

    new = OHLCVBuffer(db_path=db_path, buffer_size=20, interval="5m")

    assert new._conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0] == 0


class _CommitCountingConnection:
    """Forwards every call to a real sqlite3.Connection except commit(), which it also counts.

    sqlite3.Connection is an immutable C type — neither its class nor its
    instances accept a patched `commit` attribute — so counting calls to it
    requires wrapping the connection rather than monkeypatching it directly.
    """

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real
        self.commits = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def commit(self) -> None:
        self.commits += 1
        self._real.commit()


def test_insert_candles_bulk_uses_a_single_commit():
    """Bulk-inserting N candles commits once, not once per row."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=50, interval="1m")
    proxy = _CommitCountingConnection(buffer._conn)
    buffer._conn = proxy

    candles = [{"timestamp": f"2024-01-02T09:{i:02d}:00", "close": float(i)} for i in range(20)]
    buffer.insert_candles("AAPL", candles)

    assert proxy.commits == 1
    df = buffer.get_candles("AAPL")
    assert len(df) == 20
    assert df["close"].iloc[-1] == 19.0


def test_insert_candles_is_a_noop_for_an_empty_list():
    """Bulk-inserting an empty batch touches nothing."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=20, interval="1m")
    buffer.insert_candles("AAPL", [])
    assert buffer.get_all_tickers() == []


def test_insert_candle_delegates_to_insert_candles(monkeypatch):
    """The single-candle path is a thin wrapper over the bulk path."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=20, interval="1m")
    received = []
    monkeypatch.setattr(
        buffer, "insert_candles", lambda ticker, candles: received.append((ticker, candles))
    )

    buffer.insert_candle("AAPL", {"timestamp": "2024-01-02T09:30:00", "close": 1.0})

    assert received == [("AAPL", [{"timestamp": "2024-01-02T09:30:00", "close": 1.0}])]


def test_row_counts_returns_per_ticker_counts_without_the_get_candles_floor():
    """row_counts() reports real per-ticker counts, including below get_candles' own 14-row floor."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=50, interval="1m")
    for i in range(5):
        buffer.insert_candle("AAPL", {"timestamp": f"2024-01-02T09:{i:02d}:00", "close": float(i)})
    for i in range(20):
        buffer.insert_candle("MSFT", {"timestamp": f"2024-01-02T09:{i:02d}:00", "close": float(i)})

    assert buffer.row_counts() == {"AAPL": 5, "MSFT": 20}


def test_prune_untracked_deletes_rows_outside_the_given_set():
    """prune_untracked removes every row for a ticker not in the tracked set (API-9/API-10)."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=50, interval="1m")
    buffer.insert_candle("AAPL", {"timestamp": "2024-01-02T09:30:00", "close": 1.0})
    buffer.insert_candle("ORPHAN", {"timestamp": "2024-01-02T09:30:00", "close": 1.0})

    deleted = buffer.prune_untracked({"AAPL"})

    assert deleted == 1
    assert buffer.get_all_tickers() == ["AAPL"]


def test_prune_untracked_is_case_insensitive():
    """A lowercase tracked-set entry still matches the upper-cased stored ticker."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=50, interval="1m")
    buffer.insert_candle("AAPL", {"timestamp": "2024-01-02T09:30:00", "close": 1.0})

    deleted = buffer.prune_untracked({"aapl"})

    assert deleted == 0
    assert buffer.get_all_tickers() == ["AAPL"]


def test_prune_untracked_skips_an_empty_tracked_set():
    """An empty tracked set is treated as 'unknown universe' rather than wiping the buffer."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=50, interval="1m")
    buffer.insert_candle("AAPL", {"timestamp": "2024-01-02T09:30:00", "close": 1.0})

    deleted = buffer.prune_untracked(set())

    assert deleted == 0
    assert buffer.get_all_tickers() == ["AAPL"]


def test_interval_change_logs_the_discarded_count(tmp_path, caplog):
    """The interval-change purge logs how many stale rows it discarded, at WARNING."""
    db_path = str(tmp_path / "buffer.db")
    old = OHLCVBuffer(db_path=db_path, buffer_size=20, interval="1m")
    for i in range(3):
        old.insert_candle("AAPL", {"timestamp": f"2024-01-02T09:3{i}:00", "close": float(i)})
    old.close()

    with caplog.at_level(logging.WARNING, logger="argus.cache"):
        OHLCVBuffer(db_path=db_path, buffer_size=20, interval="5m")

    assert any("discarded 3" in record.message for record in caplog.records)
