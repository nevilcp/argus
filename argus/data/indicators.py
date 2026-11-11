"""
argus/data/indicators.py

Pure compression of a ticker's buffered candles into the technical feature
dict the agents read.

Responsibilities:
  - Resample raw candles to the indicator resolution
  - Compute RSI, MACD histogram, Bollinger %B, ATR, ADX, VWAP distance, volume
    ratio and the 30-minute / 1-day momentum, each None when it cannot be
    measured or is not finite — never a fabricated default
  - Say how many raw and resampled bars a ticker needs before that is worth
    attempting

Not responsible for:
  - Fetching or buffering candles (see data/fetchers.py, data/cache.py)
  - Deciding which tickers to compress or when (see data/pipeline.py)

Dependencies:
  - pandas_ta
"""

from __future__ import annotations

import importlib.metadata  # noqa: F401 — pandas_ta.maps uses this without importing it
import logging
import math

import pandas as pd
import pandas_ta as ta

from argus.params import SYSTEM

logger = logging.getLogger(__name__)

# 6.5h regular session (9:30-16:00 ET) in minutes
_TRADING_MINUTES_PER_DAY = 390

# RSI/MACD/BB/ATR/ADX/VWAP/volume-ratio are computed on bars resampled to this
# resolution regardless of the raw fetch interval — RSI-14 on 1m bars is a
# 14-minute lookback, too twitchy for a decision cadence measured in tens of minutes
_INDICATOR_RESAMPLE_MINUTES = 5


# MACD(12, 26, 9). A non-NaN histogram first appears at bar slow + signal - 1
# (pandas-ta's measured warmup; test_indicators pins the derivation to it), so
# the readiness floor follows the parameters rather than a copied literal.
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
MACD_WARMUP_BARS = _MACD_SLOW + _MACD_SIGNAL - 1


def bars_per_day(interval_minutes: int) -> int:
    """Returns how many candles a full trading session yields at the given interval."""
    return max(1, _TRADING_MINUTES_PER_DAY // interval_minutes)


def required_raw_bars(interval_minutes: int) -> int:
    """Cheap pre-filter: minimum raw bars before a ticker is worth resampling.

    Not authoritative — gaps make raw-bar count a lower bound on the
    resampled count, not an equality, so `compress_candles` re-checks the
    resampled length itself before publishing.

    Args:
        interval_minutes: Native candle resolution in minutes.

    Returns:
        The larger of the indicator-resample floor and one full trading
        day's worth of bars (momentum_1d's lookback).
    """
    indicator_floor = MACD_WARMUP_BARS * math.ceil(_INDICATOR_RESAMPLE_MINUTES / interval_minutes)
    momentum_floor = bars_per_day(interval_minutes) + 1
    return max(indicator_floor, momentum_floor)


def _finite_or_none(value: float | None) -> float | None:
    """Returns `value` unless it's None, NaN, or infinite.

    Args:
        value: A computed indicator value, or None.

    Returns:
        `value` unchanged, or None if it isn't a finite number.
    """
    if value is None or not math.isfinite(value):
        return None
    return value


def _rounded(value: float | None, digits: int) -> float | None:
    """Rounds a computed indicator value, passing None through untouched.

    Args:
        value: An indicator value, or None where it couldn't be computed.
        digits: Decimal places to round to.

    Returns:
        The rounded value, or None.
    """
    return None if value is None else round(value, digits)


def _last_series_value(series: pd.Series | None) -> float | None:
    """Reads the final value of a pandas_ta result series.

    Args:
        series: A pandas_ta result, or None when it produced nothing.

    Returns:
        The last value as a float, or None if `series` is None or empty. May be
        NaN — callers that need a finite result pass it through `_finite_or_none`.
    """
    if series is None or series.empty:
        return None
    return float(series.iloc[-1])


def _last_column_value(
    df: pd.DataFrame | None, prefix: str, missing_label: str | None = None
) -> float | None:
    """Reads the final value of the first `df` column whose name starts with `prefix`.

    pandas_ta names its output columns after the parameters they were computed
    with (e.g. MACDh_12_26_9), so a prefix match is the only stable way to
    locate one across pandas_ta versions.

    Args:
        df: A pandas_ta result frame, or None when it produced nothing.
        prefix: Upper-case column-name prefix identifying the wanted series.
        missing_label: Indicator name to log at WARNING when `df` has rows but
            carries no matching column; omitted for indicators that stay quiet.

    Returns:
        The last value as a float, or None if `df` is empty or has no match.
    """
    if df is None or df.empty:
        return None
    matches = [c for c in df.columns if c.upper().startswith(prefix)]
    if not matches:
        if missing_label:
            logger.warning("%s column not found in %s", missing_label, list(df.columns))
        return None
    return float(df[matches[0]].iloc[-1])


def _atr_pct(ind_df: pd.DataFrame, close_last: float) -> float | None:
    """Computes ATR-14 as a fraction of the latest close.

    Args:
        ind_df: Resampled OHLCV frame at `_INDICATOR_RESAMPLE_MINUTES`.
        close_last: Latest raw close, the denominator.

    Returns:
        ATR as a fraction of price, or None if ATR or the close is unusable.
    """
    atr = _last_series_value(ta.atr(ind_df["high"], ind_df["low"], ind_df["close"], length=14))
    if atr is None or not close_last:
        return None
    return _finite_or_none(atr / close_last)


def _vwap_distance(ind_df: pd.DataFrame, close_last: float) -> float | None:
    """Computes the latest close's signed distance from session VWAP, as a fraction.

    Args:
        ind_df: Resampled OHLCV frame at `_INDICATOR_RESAMPLE_MINUTES`.
        close_last: Latest raw close.

    Returns:
        `(close - vwap) / vwap`, or None when VWAP can't be computed.
    """
    try:
        vwap_series = ta.vwap(ind_df["high"], ind_df["low"], ind_df["close"], ind_df["volume"])
        vwap = _finite_or_none(_last_series_value(vwap_series))
    except Exception:
        # pandas_ta raises on zero-volume days; leave unset rather than fabricate 0.0
        return None
    if not vwap or not close_last:
        return None
    return _finite_or_none((close_last - vwap) / vwap)


def _volume_ratio(volume_s: pd.Series) -> float | None:
    """Compares the latest bar's volume against its recent mean.

    Args:
        volume_s: Resampled volume series, ascending.

    Returns:
        Latest volume over its rolling 20-bar mean (the full-series mean below
        20 bars), or None if the mean is zero or the ratio isn't finite.
    """
    mean = (
        float(volume_s.rolling(20).mean().iloc[-1])
        if len(volume_s) >= 20
        else float(volume_s.mean())
    )
    # A falsy check here only catches an exact 0.0 — a near-zero mean is still
    # truthy and would divide through to an extreme outlier, not a missing value.
    if not math.isfinite(mean) or abs(mean) < SYSTEM.min_volume_mean_for_ratio:
        return None
    return _finite_or_none(float(volume_s.iloc[-1]) / mean)


def _return_since(close_s: pd.Series, delta: pd.Timedelta, *, same_session: bool) -> float | None:
    """Looks up a return from the closest bar at least `delta` before the last one.

    Locating by elapsed time rather than a fixed bar count keeps the lookup
    correct under any candle interval and immune to buffer gaps or eviction,
    where a bar-count clamp is not.

    Args:
        close_s: Ascending-index close price series, ET-aware.
        delta: Minimum lookback duration from the last bar's timestamp.
        same_session: True requires the located bar share the last bar's ET
            session date (momentum_30m); False requires it precede that date
            (momentum_1d).

    Returns:
        `(last_close / located_close) - 1.0`, or None if no bar satisfies
        the session constraint or the located close is zero.
    """
    last_ts = close_s.index[-1]
    target_ts = last_ts - delta
    pos = close_s.index.searchsorted(target_ts, side="right") - 1
    if pos < 0:
        return None

    located_date = close_s.index[pos].date()
    last_date = last_ts.date()
    if same_session and located_date != last_date:
        return None
    if not same_session and located_date >= last_date:
        return None

    located_close = float(close_s.iloc[pos])
    if located_close == 0:
        return None
    return (float(close_s.iloc[-1]) / located_close) - 1.0


def _resample_ohlcv(df: pd.DataFrame, interval_minutes: int, target_minutes: int) -> pd.DataFrame:
    """Resamples raw OHLCV candles to a coarser resolution.

    Args:
        df: Raw candles at `interval_minutes` resolution, indexed by timestamp.
        interval_minutes: Native resolution of `df`.
        target_minutes: Desired output resolution.

    Returns:
        Resampled OHLCV DataFrame, or `df` unchanged if it's already at or
        coarser than `target_minutes`.
    """
    if target_minutes <= interval_minutes:
        return df
    resampled = df.resample(f"{target_minutes}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return resampled.dropna(subset=["close"])


def compress_candles(df: pd.DataFrame, interval_minutes: int) -> dict | None:
    """Calculates technical indicators on a candle frame using pandas-ta.

    Args:
        df: Raw OHLCV DataFrame at `interval_minutes` resolution, float columns,
            ET-aware datetime index.
        interval_minutes: Native candle resolution of `df`, in minutes.

    Returns:
        None if the resampled frame doesn't clear `MACD_WARMUP_BARS` — too
        little data for a trustworthy MACD histogram. Otherwise a dict
        with keys: rsi_14, macd_histogram, bb_percent_b, atr_pct, adx_14,
        vwap_distance, volume_ratio, momentum_30m, momentum_1d, close,
        timestamp. Every field but close/timestamp is None when pandas_ta
        can't compute it or the result isn't finite — never a fabricated
        default.
    """
    ind_df = _resample_ohlcv(df, interval_minutes, _INDICATOR_RESAMPLE_MINUTES)
    if len(ind_df) < MACD_WARMUP_BARS:
        return None

    close_last = float(df["close"].iloc[-1])
    close_s = df["close"]

    rsi_14 = _last_series_value(ta.rsi(ind_df["close"], length=14))
    macd_hist = _finite_or_none(
        _last_column_value(
            ta.macd(ind_df["close"], fast=_MACD_FAST, slow=_MACD_SLOW, signal=_MACD_SIGNAL),
            "MACDH_",
            "MACD histogram",
        )
    )
    bb_pct_b = _last_column_value(ta.bbands(ind_df["close"], length=20, std=2), "BBP_", "BB %B")
    adx_14 = _last_column_value(
        ta.adx(ind_df["high"], ind_df["low"], ind_df["close"], length=14), "ADX_"
    )

    return {
        "rsi_14": _rounded(rsi_14, 4),
        "macd_histogram": _rounded(macd_hist, 6),
        "bb_percent_b": _rounded(bb_pct_b, 4),
        "atr_pct": _rounded(_atr_pct(ind_df, close_last), 6),
        "adx_14": _rounded(adx_14, 4),
        "vwap_distance": _rounded(_vwap_distance(ind_df, close_last), 6),
        "volume_ratio": _rounded(_volume_ratio(ind_df["volume"]), 4),
        "momentum_30m": _rounded(
            _return_since(close_s, pd.Timedelta(minutes=30), same_session=True), 6
        ),
        "momentum_1d": _rounded(
            _return_since(close_s, pd.Timedelta(days=1), same_session=False), 6
        ),
        "close": round(close_last, 4),
        "timestamp": df.index[-1].isoformat(),
    }
