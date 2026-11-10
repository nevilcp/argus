"""
tests/helpers/candles.py

Shared OHLCV candle builders for pipeline, cache and fetcher tests.

Responsibilities:
  - Build an intraday OHLCV frame of N bars, tz-naive or tz-aware
  - Build a daily OHLCV frame over explicit dates
  - Build a frame straddling the 2025-11-02 DST fall-back transition

Not responsible for:
  - Buffer construction or insertion (see argus/data/cache.py)
  - Indicator computation (see argus/data/pipeline.py)
"""

from __future__ import annotations

import pandas as pd

ET = "America/New_York"


def _ohlcv_frame(index: pd.DatetimeIndex, volume: int = 1000) -> pd.DataFrame:
    """Builds an OHLCV frame over `index` whose closes rise 0, 1, 2, ... bar by bar."""
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
    """Builds an intraday OHLCV frame of `n` bars whose closes rise 0, 1, 2, ... bar by bar.

    Args:
        n: Number of bars.
        start: Local start timestamp, interpreted in `tz` when one is given.
        freq: Pandas frequency string for bar spacing.
        tz: Timezone of the index (e.g. `ET`); None builds a tz-naive index.
        volume: Volume of every bar.

    Returns:
        DataFrame with open/high/low/close/volume columns and a DatetimeIndex.
    """
    return _ohlcv_frame(pd.date_range(start, periods=n, freq=freq, tz=tz), volume)


def daily_bars(dates: list[str]) -> pd.DataFrame:
    """Builds a minimal daily OHLCV frame with one row per given date.

    Args:
        dates: ISO date strings, one per row.

    Returns:
        DataFrame with lowercase OHLCV columns and a tz-naive datetime index.
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


def dst_straddling_candles(n_per_side: int = 6, freq: str = "1min") -> pd.DataFrame:
    """Builds an ET intraday frame straddling the 2025-11-02 fall-back DST transition.

    The 01:00-01:59 ET wall-clock hour occurs twice that day (once at UTC-4,
    once at UTC-5). Built from UTC instants either side of the transition so
    the result is unambiguous and its UTC instants are monotonic, despite the
    repeated local wall-clock hour.

    Args:
        n_per_side: Number of bars on each side of the transition.
        freq: Pandas frequency string for bar spacing.

    Returns:
        DataFrame of `2 * n_per_side` bars, tz-aware (America/New_York).
    """
    # 01:54-01:59 EDT (UTC-4), the six minutes immediately before the transition
    pre_utc = pd.date_range("2025-11-02 05:54:00", periods=n_per_side, freq=freq, tz="UTC")
    # 01:00-01:05 EST (UTC-5), the same wall-clock minutes again, one hour later in UTC
    post_utc = pd.date_range("2025-11-02 06:00:00", periods=n_per_side, freq=freq, tz="UTC")
    return _ohlcv_frame(pre_utc.append(post_utc).tz_convert(ET))
