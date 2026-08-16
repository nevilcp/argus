"""
argus/data/pipeline.py

Mid-Frequency Trading (MFT) data pipeline for real-time asset ingestion.

Responsibilities:
  - Asynchronously fetch intraday candles, at a configurable resolution, across
    the target ticker universe
  - Buffer candles in OHLCVBuffer and compress them into technical feature dicts,
    resampling to a coarser resolution for indicators that need it
  - Trigger downstream agent decision cycles at configurable intervals

Not responsible for:
  - Market data source selection (see data/fetchers.py)
  - Persistent agent state (see orchestration/state.py)
  - Execution order routing

Dependencies:
  - asyncio
  - pandas_ta
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata  # noqa: F401 — pandas_ta.maps uses this without importing it
import logging
import math
import re
import time
from collections.abc import Callable
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_ta as ta

from argus.config import settings
from argus.data.cache import OHLCVBuffer
from argus.data.fetchers import fetch_ohlcv_intraday
from argus.params import SYSTEM
from argus.schemas.signals import missing_session_state_keys

logger = logging.getLogger("argus.pipeline")

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)

# 6.5h regular session (9:30-16:00 ET) in minutes
_TRADING_MINUTES_PER_DAY = 390

# yf.download's chart endpoint returns the whole series in one request regardless
# of interval granularity, so this costs nothing extra against any rate limit
_FETCH_PERIOD_DAYS = SYSTEM.candle_buffer_days_retained
_FETCH_PERIOD = f"{_FETCH_PERIOD_DAYS}d"

# RSI/MACD/BB/ATR/ADX/VWAP/volume-ratio are computed on bars resampled to this
# resolution regardless of the raw fetch interval — RSI-14 on 1m bars is a
# 14-minute lookback, too twitchy for a decision cadence measured in tens of minutes
_INDICATOR_RESAMPLE_MINUTES = 5

_FETCH_INTERVAL = 300

_INTERVAL_RE = re.compile(r"^(\d+)(m|h)$")


def _parse_interval_minutes(interval: str) -> int:
    """Converts a yfinance interval string (e.g. "1m", "30m", "1h") to minutes.

    Args:
        interval: yfinance-style interval string.

    Returns:
        Interval length in minutes.

    Raises:
        ValueError: If the string doesn't match the supported "<n>m" / "<n>h" shape.
    """
    match = _INTERVAL_RE.match(interval)
    if not match:
        raise ValueError(f"unsupported interval string: {interval!r}")
    value, unit = match.groups()
    return int(value) * (60 if unit == "h" else 1)


def _bars_per_day(interval_minutes: int) -> int:
    """Returns how many candles a full trading session yields at the given interval."""
    return max(1, _TRADING_MINUTES_PER_DAY // interval_minutes)


def _derive_buffer_size(interval_minutes: int) -> int:
    """Sizes the rolling buffer to hold `_FETCH_PERIOD_DAYS` of candles at the given interval.

    One extra day and one extra bar of slack beyond a bare `_FETCH_PERIOD_DAYS`
    fit so a routine session doesn't evict day-1's open before momentum_1d's
    lookback (see `_return_since`) can reach it.

    Args:
        interval_minutes: Native candle resolution in minutes.

    Returns:
        Buffer row capacity per ticker.
    """
    return (_bars_per_day(interval_minutes) + 1) * (_FETCH_PERIOD_DAYS + 1)


# MACD(12, 26, 9)'s measured pandas-ta warmup floor for a non-NaN histogram:
# slow(26) + signal(9) - 1. Pinned as a literal rather than derived from the
# call site so a pandas-ta upgrade that shifts this fails a test loudly
# instead of silently changing what counts as "ready".
_MACD_WARMUP_BARS = 34


def _required_indicator_bars() -> int:
    """Returns the minimum resampled-bar count for a non-NaN MACD histogram."""
    return _MACD_WARMUP_BARS


def _required_raw_bars(interval_minutes: int) -> int:
    """Cheap pre-filter: minimum raw bars before a ticker is worth resampling.

    Not authoritative — gaps make raw-bar count a lower bound on the
    resampled count, not an equality, so `_compress_candles` re-checks the
    resampled length itself before publishing.

    Args:
        interval_minutes: Native candle resolution in minutes.

    Returns:
        The larger of the indicator-resample floor and one full trading
        day's worth of bars (momentum_1d's lookback).
    """
    indicator_floor = _required_indicator_bars() * math.ceil(
        _INDICATOR_RESAMPLE_MINUTES / interval_minutes
    )
    momentum_floor = _bars_per_day(interval_minutes) + 1
    return max(indicator_floor, momentum_floor)


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    """Returns `value` unless it's None, NaN, or infinite.

    Args:
        value: A computed indicator value, or None.

    Returns:
        `value` unchanged, or None if it isn't a finite number.
    """
    if value is None or not math.isfinite(value):
        return None
    return value


def _return_since(
    close_s: pd.Series, delta: pd.Timedelta, *, same_session: bool
) -> Optional[float]:
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


def session_state_ttl_seconds() -> int:
    """Derives how old a live-cache entry's write time may be before api/main.py treats it as missing.

    Owned here rather than in api/main.py because the budget is built from
    `_FETCH_INTERVAL`, which this module owns; api/ is deliberately outside
    `mypy argus/`'s scope.

    Returns:
        Budget in seconds: the decision cadence plus one fetch sweep plus a
        fixed jitter margin.
    """
    return settings.MFT_DECISION_INTERVAL_SECONDS + _FETCH_INTERVAL + SYSTEM.freshness_margin_seconds


def max_bar_age_seconds(interval_minutes: int) -> int:
    """Derives how old a session state's own bar timestamp may be before it counts as stale.

    Distinct from `session_state_ttl_seconds`: that budget catches a cache
    entry that stopped being refreshed, this one catches a fresh write that
    republishes an old bar (e.g. a restart replaying the persistent buffer).

    Args:
        interval_minutes: Native candle resolution in minutes.

    Returns:
        Budget in seconds: one fetch sweep plus two candle intervals plus a
        fixed jitter margin.
    """
    return _FETCH_INTERVAL + 2 * interval_minutes * 60 + SYSTEM.freshness_margin_seconds


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


# Cold-start warmup poll cadence — much shorter than a real session interval so
# a fresh buffer doesn't sit idle for the whole interval before its first compress
_WARMUP_POLL_INTERVAL = 60


class MFTDataPipeline:
    """Asynchronous mid-frequency data pipeline coordinating candlestick updates and feature extraction.

    Runs two concurrent asyncio loops: one that periodically fetches intraday
    candles for all tracked tickers, and one that fires a decision-cycle callback
    at a longer interval. Both loops only execute during US equity market hours.
    """

    def __init__(
        self,
        tickers: list[str],
        interval: str | None = None,
        db_path: str | None = None,
    ) -> None:
        """Creates the pipeline with a candle buffer for the given universe.

        Args:
            tickers: Initial ticker universe to track.
            interval: yfinance interval string overriding `settings.MFT_CANDLE_INTERVAL`.
                Exposed mainly for tests; production call sites rely on the default.
            db_path: SQLite path for the candle buffer. Defaults to a file
                under ``settings.ARGUS_DATA_DIR`` rather than ``:memory:`` so
                a process restart resumes with a warm buffer instead of the
                cold start a fresh buffer would otherwise incur.
        """
        self.tickers = tickers
        self.interval = interval or settings.MFT_CANDLE_INTERVAL
        self.interval_minutes = _parse_interval_minutes(self.interval)
        self.session_interval_seconds = settings.MFT_DECISION_INTERVAL_SECONDS
        buffer_size = settings.CANDLE_BUFFER_SIZE or _derive_buffer_size(self.interval_minutes)
        if db_path is None:
            data_dir = Path(settings.ARGUS_DATA_DIR)
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "ohlcv_buffer.db")
        self.buffer = OHLCVBuffer(db_path=db_path, buffer_size=buffer_size, interval=self.interval)
        self.running = False
        self._stop_event = asyncio.Event()
        logger.info(
            "MFTDataPipeline initialised: %d tickers, interval=%s, buffer_size=%d, buffer=%s",
            len(tickers),
            self.interval,
            buffer_size,
            db_path,
        )

    async def start(self, on_session_ready: Callable) -> None:
        """Starts concurrent fetch and session cycles, blocking until stopped.

        Args:
            on_session_ready: Async callback invoked with compressed session states
                after each session interval elapses during market hours.
        """
        self.running = True
        logger.info("MFTDataPipeline.start: launching fetch and session loops")
        tasks = [
            asyncio.create_task(self._fetch_loop()),
            asyncio.create_task(self._session_loop(on_session_ready)),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            # Either task ending (return or raise) must not orphan the other —
            # cancel and await both so a dead fetch loop can't leave the session
            # loop sweeping into a closed buffer, or vice versa
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _wait_or_stop(self, timeout: float) -> None:
        """Sleeps up to `timeout` seconds, waking immediately if `stop()` is called.

        Args:
            timeout: Max seconds to wait.
        """
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)

    def register_tickers(self, tickers: list[str]) -> None:
        """Adds new tickers to the live tracking universe without restarting the pipeline.

        Safe to call while the pipeline is running. The ``_fetch_loop`` reads
        ``self.tickers`` on every iteration, so new entries are picked up within
        ``_FETCH_INTERVAL`` seconds.

        Args:
            tickers: Ticker symbols to add. Duplicates are silently ignored.
        """
        new = [t for t in tickers if t not in self.tickers]
        if new:
            self.tickers.extend(new)
            logger.info(
                "MFTDataPipeline.register_tickers: added %d ticker(s): %s",
                len(new),
                new,
            )

    async def stop(self) -> None:
        """Signals active loops to stop execution."""
        self.running = False
        self._stop_event.set()
        logger.info("MFTDataPipeline.stop: shutdown requested")

    async def run_once(self) -> dict[str, dict]:
        """Runs a single fetch-and-compress cycle without the background loops.

        Used by the unattended collector (``argus/orchestration/collector.py``)
        to force one sweep on demand rather than waiting on ``start()``'s
        periodic loops. Outside market hours this skips the fetch and just
        compresses whatever the buffer already holds.

        Returns:
            Mapping of ticker → technical feature dict. Tickers that don't
            clear ``compress_all``'s readiness floor are omitted.
        """
        if self._is_market_hours():
            await self._sweep_once()
        else:
            logger.debug("run_once: outside market hours — compressing existing buffer only")
        return await asyncio.to_thread(self.compress_all)

    async def _sweep_once(self) -> None:
        """Fetches the latest candlesticks for every tracked ticker, one pass.

        Pauses ``SYSTEM.min_inter_request_seconds`` between requests — a fixed
        floor against yfinance rate limiting, not a spread computed to fill
        ``_FETCH_INTERVAL``; that framing was wrong in principle (pacing exists
        for rate limiting, not to occupy the cadence budget) and meaningless for
        a one-shot ``run_once()`` sweep. Iterates over a snapshot of
        ``self.tickers`` so that concurrent ``register_tickers`` calls cannot corrupt iteration.
        """
        tickers_snapshot = list(self.tickers)
        logger.debug("_sweep_once: starting universe sweep (%d tickers)", len(tickers_snapshot))
        last = len(tickers_snapshot) - 1
        for i, ticker in enumerate(tickers_snapshot):
            await self._fetch_one_ticker(ticker)
            if i < last:
                await asyncio.sleep(SYSTEM.min_inter_request_seconds)

    async def _fetch_loop(self) -> None:
        """Periodically downloads the latest intraday candlesticks for the tracking universe.

        Sleeps only the remainder of ``_FETCH_INTERVAL`` after each sweep, not a
        full interval on top of however long the sweep itself took — otherwise
        the real cadence drifts to sweep_duration + _FETCH_INTERVAL instead of
        the intended _FETCH_INTERVAL.
        """
        while self.running:
            started = time.monotonic()
            try:
                if self._is_market_hours():
                    await self._sweep_once()
                else:
                    logger.debug("_fetch_loop: outside market hours — idle")
            except Exception as exc:
                logger.error(
                    "_fetch_loop: sweep failed — %s: %s", type(exc).__name__, exc
                )

            elapsed = time.monotonic() - started
            await self._wait_or_stop(max(0.0, _FETCH_INTERVAL - elapsed))

    def _buffer_warm(self) -> bool:
        """Checks whether any tracked ticker already has an indicator-ready buffer depth.

        Returns:
            True once at least one ticker holds >= 14 rows (``OHLCVBuffer.get_candles``'
            own floor), letting callers skip waiting a full session interval on a
            cold start.
        """
        for ticker in self.buffer.get_all_tickers():
            try:
                if self.buffer.get_candles(ticker) is not None:
                    return True
            except Exception as exc:
                logger.warning(
                    "_buffer_warm: %s check failed — %s: %s", ticker, type(exc).__name__, exc
                )
        return False

    async def _session_loop(self, callback: Callable) -> None:
        """Periodically compiles buffered candlesticks and triggers downstream agent actions.

        Args:
            callback: Async callable receiving the ``session_states`` dict.
        """
        # A fresh buffer sleeping a full session interval before its first
        # compress would leave the live cache empty for that whole stretch for
        # no reason; poll faster until there's actually something to compress
        while self.running and not self._buffer_warm():
            await self._wait_or_stop(_WARMUP_POLL_INTERVAL)

        while self.running:
            try:
                if self._is_market_hours():
                    logger.info("_session_loop: triggering decision cycle")
                    session_states = await asyncio.to_thread(self.compress_all)
                    try:
                        await callback(session_states)
                    except Exception as exc:
                        logger.error(
                            "_session_loop: callback raised %s: %s",
                            type(exc).__name__,
                            exc,
                        )
            except Exception as exc:
                logger.error(
                    "_session_loop: iteration failed — %s: %s", type(exc).__name__, exc
                )
            await self._wait_or_stop(self.session_interval_seconds)

    async def _fetch_one_ticker(self, ticker: str) -> None:
        """Downloads and bulk-inserts all available candles for a specific symbol.

        Runs the network fetch and the buffer insert together inside one
        ``to_thread`` call — a `pd.DataFrame.iterrows()` insert loop run
        directly on the event loop measured ~655ms/ticker of blocking cost, on
        top of the network round-trip. Fetches `_FETCH_PERIOD` of candles at
        `self.interval` and inserts every row, warming the buffer to a full
        indicator-ready depth on the very first fetch cycle rather than only
        the latest candle. Duplicate timestamps are safely overwritten via the
        buffer's INSERT OR REPLACE contract.

        Args:
            ticker: Equity ticker symbol to fetch.
        """
        try:
            n_candles, latest_close = await asyncio.to_thread(self._fetch_and_insert, ticker)
            if n_candles == 0:
                logger.warning("_fetch_one_ticker: empty result for %s", ticker)
                return

            logger.debug(
                "_fetch_one_ticker: %s → %d candles inserted, latest close=%.2f",
                ticker,
                n_candles,
                latest_close,
            )

        except Exception as exc:
            logger.warning(
                "_fetch_one_ticker: %s failed — %s: %s",
                ticker,
                type(exc).__name__,
                exc,
            )

    def _fetch_and_insert(self, ticker: str) -> tuple[int, float]:
        """Synchronous fetch-and-bulk-insert for one ticker; run off the event loop.

        Args:
            ticker: Equity ticker symbol to fetch.

        Returns:
            Tuple of (candles inserted, latest close). (0, 0.0) on an empty fetch.
        """
        df: pd.DataFrame = fetch_ohlcv_intraday(ticker, self.interval, _FETCH_PERIOD)
        if df is None or df.empty:
            return 0, 0.0

        candles = [
            {
                "timestamp": idx.isoformat(),
                "open": float(row["open"]) if "open" in row.index else None,
                "high": float(row["high"]) if "high" in row.index else None,
                "low": float(row["low"]) if "low" in row.index else None,
                "close": float(row["close"]) if "close" in row.index else None,
                "volume": float(row["volume"]) if "volume" in row.index else None,
            }
            for idx, row in df.iterrows()
        ]
        self.buffer.insert_candles(ticker, candles)
        latest_close = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
        return len(df), latest_close

    def compress_all(self) -> dict[str, dict]:
        """Compresses cached candle metrics across all universe symbols into a feature dictionary.

        A ticker is only published once it clears both readiness checks:
        enough raw bars to be worth resampling (`_required_raw_bars`), and —
        the authoritative gate — enough resampled bars for a real MACD
        histogram, plus every required key present and finite
        (`missing_session_state_keys`). This is what makes "present in this
        dict" mean the same thing as "usable" to a caller that never learns
        what an indicator is.

        Returns:
            Mapping of ticker → technical feature dict (rsi_14, macd_histogram, etc.).
            Tickers below the readiness floor, or that failed compression,
            are omitted and logged.
        """
        states: dict[str, dict] = {}
        required_raw = _required_raw_bars(self.interval_minutes)
        for ticker in self.buffer.get_all_tickers():
            try:
                df = self.buffer.get_candles(ticker)
                if df is None or len(df) < required_raw:
                    continue
                state = self._compress_candles(df)
                if state is None:
                    continue
                missing = missing_session_state_keys(state)
                if missing:
                    logger.warning(
                        "compress_all: %s missing/non-finite required keys %s — omitting",
                        ticker,
                        missing,
                    )
                    continue
                states[ticker] = state
            except Exception as exc:
                logger.warning(
                    "compress_all: %s compression failed — %s: %s",
                    ticker,
                    type(exc).__name__,
                    exc,
                )
        logger.info("compress_all: %d tickers compressed", len(states))
        return states

    def _compress_candles(self, df: pd.DataFrame) -> Optional[dict]:
        """Calculates technical indicators on a DataFrame using pandas-ta.

        Args:
            df: Raw OHLCV DataFrame at `self.interval` resolution, float columns,
                datetime index.

        Returns:
            None if the resampled frame doesn't clear `_required_indicator_bars()`
            — too little data for a trustworthy MACD histogram. Otherwise a dict
            with keys: rsi_14, macd_histogram, bb_percent_b, atr_pct, adx_14,
            vwap_distance, volume_ratio, momentum_30m, momentum_1d, close,
            timestamp. Every field but close/timestamp is None when pandas_ta
            can't compute it or the result isn't finite — never a fabricated
            default.
        """
        ind_df = _resample_ohlcv(df, self.interval_minutes, _INDICATOR_RESAMPLE_MINUTES)
        if len(ind_df) < _required_indicator_bars():
            return None

        rsi_series = ta.rsi(ind_df["close"], length=14)
        rsi_14 = (
            float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None
        )

        macd_df = ta.macd(ind_df["close"], fast=12, slow=26, signal=9)
        macd_hist = None
        if macd_df is not None and not macd_df.empty:
            # pandas_ta names the histogram column with an 'h' suffix, e.g. MACDh_12_26_9
            hist_col = [c for c in macd_df.columns if c.upper().startswith("MACDH_")]
            if hist_col:
                macd_hist = _finite_or_none(float(macd_df[hist_col[0]].iloc[-1]))
            else:
                logger.warning("_compress_candles: MACD histogram column not found in %s", list(macd_df.columns))

        bb_df = ta.bbands(ind_df["close"], length=20, std=2)
        bb_pct_b = None
        if bb_df is not None and not bb_df.empty:
            # pandas_ta names the %B column with a 'BBP_' prefix, e.g. BBP_20_2.0
            pct_col = [c for c in bb_df.columns if c.upper().startswith("BBP_")]
            if pct_col:
                bb_pct_b = float(bb_df[pct_col[0]].iloc[-1])
            else:
                logger.warning("_compress_candles: BB %%B column not found in %s", list(bb_df.columns))

        atr_series = ta.atr(ind_df["high"], ind_df["low"], ind_df["close"], length=14)
        close_last = float(df["close"].iloc[-1])
        atr_pct = None
        if atr_series is not None and not atr_series.empty and close_last:
            atr_pct = _finite_or_none(float(atr_series.iloc[-1]) / close_last)

        adx_df = ta.adx(ind_df["high"], ind_df["low"], ind_df["close"], length=14)
        adx_14 = None
        if adx_df is not None and not adx_df.empty:
            adx_col = [c for c in adx_df.columns if c.upper().startswith("ADX_")]
            if adx_col:
                adx_14 = float(adx_df[adx_col[0]].iloc[-1])

        vwap_distance = None
        try:
            vwap_series = ta.vwap(ind_df["high"], ind_df["low"], ind_df["close"], ind_df["volume"])
            if vwap_series is not None and not vwap_series.empty:
                vwap_val = _finite_or_none(float(vwap_series.iloc[-1]))
                if vwap_val and close_last:
                    vwap_distance = _finite_or_none((close_last - vwap_val) / vwap_val)
        except Exception:
            # pandas_ta raises on zero-volume days; leave unset rather than fabricate 0.0
            pass

        vol_series = ind_df["volume"]
        vol_mean = (
            float(vol_series.rolling(20).mean().iloc[-1])
            if len(vol_series) >= 20
            else float(vol_series.mean())
        )
        volume_ratio = (
            _finite_or_none(float(vol_series.iloc[-1]) / vol_mean) if vol_mean else None
        )

        close_s = df["close"]
        momentum_30m = _return_since(close_s, pd.Timedelta(minutes=30), same_session=True)
        momentum_1d = _return_since(close_s, pd.Timedelta(days=1), same_session=False)

        return {
            "rsi_14": round(rsi_14, 4) if rsi_14 is not None else None,
            "macd_histogram": round(macd_hist, 6) if macd_hist is not None else None,
            "bb_percent_b": round(bb_pct_b, 4) if bb_pct_b is not None else None,
            "atr_pct": round(atr_pct, 6) if atr_pct is not None else None,
            "adx_14": round(adx_14, 4) if adx_14 is not None else None,
            "vwap_distance": round(vwap_distance, 6) if vwap_distance is not None else None,
            "volume_ratio": round(volume_ratio, 4) if volume_ratio is not None else None,
            "momentum_30m": round(momentum_30m, 6) if momentum_30m is not None else None,
            "momentum_1d": round(momentum_1d, 6) if momentum_1d is not None else None,
            "close": round(close_last, 4),
            "timestamp": df.index[-1].isoformat(),
        }

    def _is_market_hours(self) -> bool:
        """Checks if current Eastern Time falls within regular US stock market hours (09:30 - 16:00 ET).

        Returns:
            True during weekday trading hours in America/New_York timezone.
        """
        now_et = datetime.now(_ET)
        if now_et.weekday() >= 5:
            return False
        current = now_et.time().replace(second=0, microsecond=0)
        return _MARKET_OPEN <= current < _MARKET_CLOSE

    def is_market_hours(self) -> bool:
        """Public wrapper around ``_is_market_hours`` for callers outside this module.

        Returns:
            True during weekday US equity trading hours.
        """
        return self._is_market_hours()
