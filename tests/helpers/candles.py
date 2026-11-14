"""Shared OHLCV candle builders for pipeline, cache and fetcher tests."""

from __future__ import annotations

import pandas as pd

ET = "America/New_York"


def _ohlcv_frame(index: pd.DatetimeIndex, volume: int = 1000) -> pd.DataFrame:
    """Build an OHLCV frame over `index` whose closes rise 0, 1, 2, ... bar by bar."""
    closes = list(range(len(index)))
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": closes,
            "close": closes,
            "volume": [volume] * len(index),
        },
        index=index,
    )


def ohlcv_frame(
    n: int,
    *,
    start: str = "2024-01-02 09:30",
    freq: str = "1min",
    tz: str | None = None,
    volume: int = 1000,
) -> pd.DataFrame:
    """Build an intraday OHLCV frame of `n` bars whose closes rise 0, 1, 2, ... bar by bar."""
    return _ohlcv_frame(pd.date_range(start, periods=n, freq=freq, tz=tz), volume)


def daily_bars(dates: list[str]) -> pd.DataFrame:
    """Build a minimal daily OHLCV frame with one row per given date."""
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


def dst_straddling_candles(n_per_side: int = 6, freq: str = "1min") -> pd.DataFrame:
    """Build an ET intraday frame straddling the 2025-11-02 fall-back DST transition."""
    # 01:54-01:59 EDT (UTC-4), the six minutes immediately before the transition
    pre_utc = pd.date_range("2025-11-02 05:54:00", periods=n_per_side, freq=freq, tz="UTC")
    # 01:00-01:05 EST (UTC-5), the same wall-clock minutes again, one hour later in UTC
    post_utc = pd.date_range("2025-11-02 06:00:00", periods=n_per_side, freq=freq, tz="UTC")
    return _ohlcv_frame(pre_utc.append(post_utc).tz_convert(ET))
