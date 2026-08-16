"""
tests/helpers/candles.py

Shared tz-aware OHLCV candle builders for pipeline and cache tests.

Responsibilities:
  - Build a tz-aware America/New_York intraday OHLCV frame of N bars
  - Build a frame straddling the 2025-11-02 DST fall-back transition

Not responsible for:
  - Buffer construction or insertion (see argus/data/cache.py)
  - Indicator computation (see argus/data/pipeline.py)
"""

from __future__ import annotations

import pandas as pd

_ET = "America/New_York"


def et_intraday_candles(
    n: int, start: str = "2024-01-02 09:30", freq: str = "1min"
) -> pd.DataFrame:
    """Builds a tz-aware ET intraday OHLCV frame of monotonically increasing bars.

    Args:
        n: Number of bars.
        start: Naive local start timestamp, interpreted as ET.
        freq: Pandas frequency string for bar spacing.

    Returns:
        DataFrame with open/high/low/close/volume columns and a tz-aware
        (America/New_York) DatetimeIndex.
    """
    dates = pd.date_range(start, periods=n, freq=freq, tz=_ET)
    closes = list(range(n))
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": closes,
            "close": closes,
            "volume": [1000] * n,
        },
        index=dates,
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
    dates = pre_utc.append(post_utc).tz_convert(_ET)
    n = 2 * n_per_side
    closes = list(range(n))
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": closes,
            "close": closes,
            "volume": [1000] * n,
        },
        index=dates,
    )
