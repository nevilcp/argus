"""Intraday candle compression into validated technical features."""

from __future__ import annotations

import importlib.metadata  # noqa: F401 - pandas_ta.maps uses this without importing it
import logging
import math

import pandas as pd
import pandas_ta as ta

from argus.params import SYSTEM

logger = logging.getLogger(__name__)

_TRADING_MINUTES_PER_DAY = 390

# Compute indicators on resampled five-minute bars so raw fetch resolution cannot change their horizon.
_INDICATOR_RESAMPLE_MINUTES = 5


# Derive readiness from the first non-NaN MACD histogram at slow + signal - 1 bars.
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
MACD_WARMUP_BARS = _MACD_SLOW + _MACD_SIGNAL - 1


def bars_per_day(interval_minutes: int) -> int:
    """Return how many candles a full trading session yields at the given interval."""
    return max(1, _TRADING_MINUTES_PER_DAY // interval_minutes)


def required_raw_bars(interval_minutes: int) -> int:
    """Return the minimum raw-bar count worth resampling; gaps still require a resampled readiness check."""
    indicator_floor = MACD_WARMUP_BARS * math.ceil(_INDICATOR_RESAMPLE_MINUTES / interval_minutes)
    momentum_floor = bars_per_day(interval_minutes) + 1
    return max(indicator_floor, momentum_floor)


def _finite_or_none(value: float | None) -> float | None:
    """Return `value` unless it's None, NaN, or infinite."""
    if value is None or not math.isfinite(value):
        return None
    return value


def _rounded(value: float | None, digits: int) -> float | None:
    """Round a computed indicator value, passing None through untouched."""
    return None if value is None else round(value, digits)


def _last_series_value(series: pd.Series | None) -> float | None:
    """Read the final value of a pandas_ta result series."""
    if series is None or series.empty:
        return None
    return float(series.iloc[-1])


def _last_column_value(df: pd.DataFrame | None, prefix: str, missing_label: str | None = None) -> float | None:
    """Read the final value of the first `df` column whose name starts with `prefix`."""
    if df is None or df.empty:
        return None
    matches = [c for c in df.columns if c.upper().startswith(prefix)]
    if not matches:
        if missing_label:
            logger.warning("%s column not found in %s", missing_label, list(df.columns))
        return None
    return float(df[matches[0]].iloc[-1])


def _atr_pct(ind_df: pd.DataFrame, close_last: float) -> float | None:
    """Compute ATR-14 as a fraction of the latest close."""
    atr = _last_series_value(ta.atr(ind_df["high"], ind_df["low"], ind_df["close"], length=14))
    if atr is None or not close_last:
        return None
    return _finite_or_none(atr / close_last)


def _vwap_distance(ind_df: pd.DataFrame, close_last: float) -> float | None:
    """Compute the latest close's signed distance from session VWAP, as a fraction."""
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
    """Compare the latest bar's volume against its recent mean."""
    mean = float(volume_s.rolling(20).mean().iloc[-1]) if len(volume_s) >= 20 else float(volume_s.mean())
    # Reject near-zero volume means before division can produce an extreme outlier.
    if not math.isfinite(mean) or abs(mean) < SYSTEM.min_volume_mean_for_ratio:
        return None
    return _finite_or_none(float(volume_s.iloc[-1]) / mean)


def _return_since(close_s: pd.Series, delta: pd.Timedelta, *, same_session: bool) -> float | None:
    """Look up a return from the closest bar at least `delta` before the last one."""
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
    """Resample raw OHLCV candles to a coarser resolution."""
    if target_minutes <= interval_minutes:
        return df
    resampled = df.resample(f"{target_minutes}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return resampled.dropna(subset=["close"])


def compress_candles(df: pd.DataFrame, interval_minutes: int) -> dict | None:
    """Calculate technical indicators on a candle frame using pandas-ta."""
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
    adx_14 = _last_column_value(ta.adx(ind_df["high"], ind_df["low"], ind_df["close"], length=14), "ADX_")

    return {
        "rsi_14": _rounded(rsi_14, 4),
        "macd_histogram": _rounded(macd_hist, 6),
        "bb_percent_b": _rounded(bb_pct_b, 4),
        "atr_pct": _rounded(_atr_pct(ind_df, close_last), 6),
        "adx_14": _rounded(adx_14, 4),
        "vwap_distance": _rounded(_vwap_distance(ind_df, close_last), 6),
        "volume_ratio": _rounded(_volume_ratio(ind_df["volume"]), 4),
        "momentum_30m": _rounded(_return_since(close_s, pd.Timedelta(minutes=30), same_session=True), 6),
        "momentum_1d": _rounded(_return_since(close_s, pd.Timedelta(days=1), same_session=False), 6),
        "close": round(close_last, 4),
        "timestamp": df.index[-1].isoformat(),
    }
