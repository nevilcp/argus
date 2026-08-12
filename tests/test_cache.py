"""
Tests for argus/data/cache.py's DailyBarCache — the per-(ticker, trading_date)
disk cache that lets fetch_multiple_daily skip the network on same-day re-runs.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from argus.data.cache import DailyBarCache


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
