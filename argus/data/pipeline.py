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
import threading
import time
from collections.abc import Callable
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_ta as ta

from argus.config import settings
from argus.data.cache import OHLCVBuffer
from argus.data.fetchers import fetch_ohlcv_intraday
from argus.data.tickers import is_valid_ticker
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

# Once a ticker's buffer already holds an indicator-ready depth, a sweep only
# needs to catch the candles written since the last one — "1d" (yfinance's
# shortest supported intraday period) comfortably covers one _FETCH_INTERVAL
# gap, and insert_candles' INSERT OR REPLACE makes the overlap with what's
# already buffered a no-op rather than a duplicate
_STEADY_STATE_FETCH_PERIOD = "1d"

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


def _rounded(value: Optional[float], digits: int) -> Optional[float]:
    """Rounds a computed indicator value, passing None through untouched.

    Args:
        value: An indicator value, or None where it couldn't be computed.
        digits: Decimal places to round to.

    Returns:
        The rounded value, or None.
    """
    return None if value is None else round(value, digits)


def _last_series_value(series: Optional[pd.Series]) -> Optional[float]:
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
    df: Optional[pd.DataFrame], prefix: str, missing_label: Optional[str] = None
) -> Optional[float]:
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
            logger.warning(
                "_compress_candles: %s column not found in %s", missing_label, list(df.columns)
            )
        return None
    return float(df[matches[0]].iloc[-1])


def _atr_pct(ind_df: pd.DataFrame, close_last: float) -> Optional[float]:
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


def _vwap_distance(ind_df: pd.DataFrame, close_last: float) -> Optional[float]:
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


def _volume_ratio(volume_s: pd.Series) -> Optional[float]:
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
    if not mean:
        return None
    return _finite_or_none(float(volume_s.iloc[-1]) / mean)


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


_CANDLE_FIELDS = ("open", "high", "low", "close", "volume")


def _candle_from_row(timestamp: pd.Timestamp, row: pd.Series) -> dict:
    """Builds one OHLCVBuffer candle dict from a fetched frame's row.

    Args:
        timestamp: The row's index value.
        row: One row of a fetched OHLCV frame; a column absent from it lands as
            None rather than a fabricated default.

    Returns:
        Dict with a `timestamp` key plus every `_CANDLE_FIELDS` entry.
    """
    candle: dict = {"timestamp": timestamp.isoformat()}
    for field in _CANDLE_FIELDS:
        candle[field] = float(row[field]) if field in row.index else None
    return candle


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


class MFTDataPipeline:
    """Asynchronous pipeline coordinating candlestick updates and feature extraction across tickers.

    Runs a single asyncio loop that periodically fetches intraday candles for
    all tracked tickers, then republishes compressed session state from the
    same pass — publication used to sit on its own, much longer
    timer, so a live-cache entry was fresh for only a fraction of each cycle.
    The loop only runs during US equity market hours.
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
        self.interval = interval or settings.MFT_CANDLE_INTERVAL
        self.interval_minutes = _parse_interval_minutes(self.interval)
        buffer_size = settings.CANDLE_BUFFER_SIZE or _derive_buffer_size(self.interval_minutes)
        if db_path is None:
            data_dir = Path(settings.ARGUS_DATA_DIR)
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "ohlcv_buffer.db")
        self.buffer = OHLCVBuffer(db_path=db_path, buffer_size=buffer_size, interval=self.interval)
        self.running = False
        self._stop_event = asyncio.Event()
        # Held for the full body of every to_thread-run, buffer-touching worker
        # (_fetch_and_insert, compress_all) so close_buffer can wait out one that's
        # still running after its owning task was cancelled rather than joined
        self._buffer_op_lock = threading.Lock()
        # Routed through register_tickers rather than a bare assignment, so the
        # ticker-shape and universe-cap invariants apply uniformly here, to
        # run_collection_cycle, and to collect_session.py's --universe CLI arg
        # — not just to /analyze's already-validated request tickers
        self.tickers: list[str] = []
        self.last_requested_at: dict[str, datetime] = {}
        self.register_tickers(tickers)
        # The constructor's own universe is pinned against _evict_stale_tickers
        # — the unattended collector never calls register_tickers to
        # "touch" these, so a TTL applied uniformly would silently starve the
        # universe it exists to track
        self._pinned_tickers: frozenset[str] = frozenset(self.tickers)
        logger.info(
            "MFTDataPipeline initialised: %d tickers, interval=%s, buffer_size=%d, buffer=%s",
            len(self.tickers),
            self.interval,
            buffer_size,
            db_path,
        )

    async def start(self, on_session_ready: Callable) -> None:
        """Starts the fetch loop, blocking until stopped.

        Args:
            on_session_ready: Async callback invoked with compressed session states
                after each sweep during market hours.
        """
        self.running = True
        logger.info("MFTDataPipeline.start: launching fetch loop")
        await self._fetch_loop(on_session_ready)

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
        ``_FETCH_INTERVAL`` seconds. This is the single enforcement point for
        ticker-shape validation and the SYSTEM.max_tracked_tickers cap:
        __init__ routes its initial universe through here too.

        Args:
            tickers: Ticker symbols to add. Malformed symbols and duplicates
                are dropped; entries beyond the tracked-universe cap are
                dropped too, both logged at WARNING.
        """
        valid = [t for t in tickers if is_valid_ticker(t)]
        invalid = [t for t in tickers if not is_valid_ticker(t)]
        if invalid:
            logger.warning(
                "MFTDataPipeline.register_tickers: rejected malformed ticker(s): %s", invalid
            )

        new = [t for t in valid if t not in self.tickers]
        room = max(SYSTEM.max_tracked_tickers - len(self.tickers), 0)
        accepted, dropped = new[:room], new[room:]
        if dropped:
            logger.warning(
                "MFTDataPipeline.register_tickers: universe cap (%d) reached; dropping %d "
                "ticker(s): %s",
                SYSTEM.max_tracked_tickers,
                len(dropped),
                dropped,
            )
        if accepted:
            self.tickers.extend(accepted)
            logger.info(
                "MFTDataPipeline.register_tickers: added %d ticker(s): %s",
                len(accepted),
                accepted,
            )

        # Touches every already-tracked ticker too, not just newly-accepted
        # ones — a repeatedly-requested ticker must never look
        # unused to _evict_stale_tickers just because it was already tracked
        now = datetime.now(_ET)
        for ticker in valid:
            if ticker in self.tickers:
                self.last_requested_at[ticker] = now

    def _evict_stale_tickers(self) -> None:
        """Drops tracked tickers unused beyond SYSTEM.tracked_ticker_ttl_seconds.

        Without this, `register_tickers` only ever appends: repeated one-off
        `/analyze` requests for different tickers permanently saturate
        `SYSTEM.max_tracked_tickers` with no eviction and no new ticker ever
        admitted. Exempts `self._pinned_tickers` — the pipeline's own seed
        universe — since the unattended collector never calls
        `register_tickers` to refresh their `last_requested_at`.
        """
        cutoff = datetime.now(_ET) - timedelta(seconds=SYSTEM.tracked_ticker_ttl_seconds)
        stale = [
            t
            for t in self.tickers
            if t not in self._pinned_tickers and self.last_requested_at.get(t, cutoff) < cutoff
        ]
        if not stale:
            return
        for ticker in stale:
            self.tickers.remove(ticker)
            self.last_requested_at.pop(ticker, None)
        logger.info(
            "MFTDataPipeline._evict_stale_tickers: evicted %d unused ticker(s): %s",
            len(stale),
            stale,
        )

    async def stop(self) -> None:
        """Signals active loops to stop execution."""
        self.running = False
        self._stop_event.set()
        logger.info("MFTDataPipeline.stop: shutdown requested")

    async def close_buffer(self) -> None:
        """Closes the candle buffer, waiting out any still-running buffer worker first.

        Cancelling the fetch/collector loops' asyncio tasks unwinds their
        coroutines but can't interrupt a `to_thread` call already executing on
        a worker thread — that worker runs `_fetch_and_insert`/`compress_all`
        to completion regardless of the task's cancellation. Both methods hold
        `_buffer_op_lock` for their entire body, so blocking here until it's
        acquired guarantees any such orphaned worker has actually finished
        touching the buffer before its connection is closed. Callers must
        await both loops' tasks to completion first — with `self.running`
        already False by then, neither loop can start a *new* worker after
        this returns.
        """
        await asyncio.to_thread(self._buffer_op_lock.acquire)
        try:
            self.buffer.close()
        finally:
            self._buffer_op_lock.release()

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
        self._evict_stale_tickers()
        tickers_snapshot = list(self.tickers)
        logger.debug("_sweep_once: starting universe sweep (%d tickers)", len(tickers_snapshot))
        last = len(tickers_snapshot) - 1
        for i, ticker in enumerate(tickers_snapshot):
            await self._fetch_one_ticker(ticker)
            if i < last:
                await asyncio.sleep(SYSTEM.min_inter_request_seconds)

    async def _fetch_loop(self, on_session_ready: Callable) -> None:
        """Periodically downloads intraday candlesticks and republishes compressed session state.

        Publication is merged into this loop rather than living on
        a separate, much longer timer: the old split meant a live-cache entry
        was only fresh for a fraction of each publish cycle, since
        ``max_bar_age_seconds`` was sized off the fetch cadence while
        publication ran on a slower one. Sleeps only the remainder of
        ``_FETCH_INTERVAL`` after each pass, not a full interval on top of
        however long the pass itself took — otherwise the real cadence drifts
        to pass_duration + _FETCH_INTERVAL instead of the intended
        _FETCH_INTERVAL.

        Args:
            on_session_ready: Async callable receiving the ``session_states`` dict
                after each sweep during market hours.
        """
        while self.running:
            started = time.monotonic()
            try:
                if self._is_market_hours():
                    await self._sweep_once()
                    session_states = await asyncio.to_thread(self.compress_all)
                    try:
                        await on_session_ready(session_states)
                    except Exception as exc:
                        logger.error("_fetch_loop: callback raised %s: %s", type(exc).__name__, exc)
                else:
                    logger.debug("_fetch_loop: outside market hours — idle")
            except Exception as exc:
                logger.error("_fetch_loop: sweep failed — %s: %s", type(exc).__name__, exc)

            elapsed = time.monotonic() - started
            await self._wait_or_stop(max(0.0, _FETCH_INTERVAL - elapsed))

    def _buffer_warm(self) -> bool:
        """Checks whether any tracked ticker already has an indicator-ready buffer depth.

        Returns:
            True once at least one ticker holds >= 14 rows (``OHLCVBuffer.get_candles``'
            own floor).
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

        Fetches the full `_FETCH_PERIOD` only while the ticker's buffer is
        still cold (a new ticker, or the first sweep after a restart);
        otherwise fetches just `_STEADY_STATE_FETCH_PERIOD`'s worth.
        Re-downloading and re-upserting two full days of 1-minute candles
        every `_FETCH_INTERVAL` regardless of how warm the buffer already is
        multiplies out to ~15,600 row upserts per sweep at 20 tickers, almost
        all of them re-writing rows already on disk unchanged.

        Args:
            ticker: Equity ticker symbol to fetch.

        Returns:
            Tuple of (candles inserted, latest close). (0, 0.0) on an empty fetch.
        """
        with self._buffer_op_lock:
            is_cold = (
                self.buffer.row_counts().get(ticker, 0) < _required_raw_bars(self.interval_minutes)
            )
            period = _FETCH_PERIOD if is_cold else _STEADY_STATE_FETCH_PERIOD
            df: pd.DataFrame = fetch_ohlcv_intraday(ticker, self.interval, period)
            if df is None or df.empty:
                return 0, 0.0

            candles = [_candle_from_row(idx, row) for idx, row in df.iterrows()]
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

        Only iterates tickers still in `self.tickers` — the buffer
        can otherwise hold rows for a ticker fetched once by a past `/analyze`
        request and never tracked again. Also prunes the buffer of any such
        untracked rows (API-9's underlying leak), at this method's cadence.

        Returns:
            Mapping of ticker → technical feature dict (rsi_14, macd_histogram, etc.).
            Tickers below the readiness floor, or that failed compression,
            are omitted and logged.
        """
        states: dict[str, dict] = {}
        required_raw = _required_raw_bars(self.interval_minutes)
        with self._buffer_op_lock:
            tracked = set(self.tickers)
            for ticker in tracked & set(self.buffer.get_all_tickers()):
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
            self.buffer.prune_untracked(tracked)
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

        close_last = float(df["close"].iloc[-1])
        close_s = df["close"]

        rsi_14 = _last_series_value(ta.rsi(ind_df["close"], length=14))
        macd_hist = _finite_or_none(
            _last_column_value(
                ta.macd(ind_df["close"], fast=12, slow=26, signal=9),
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

    def _is_market_hours(self) -> bool:
        """Checks whether current Eastern Time is within regular US market hours (09:30-16:00 ET).

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
