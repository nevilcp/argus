"""
argus/data/pipeline.py

Mid-Frequency Trading (MFT) data pipeline for real-time asset ingestion.

Responsibilities:
  - Asynchronously fetch intraday candles, at a configurable resolution, across
    the target ticker universe
  - Buffer candles in OHLCVBuffer and compress them into technical feature dicts
    (the math is in data/indicators.py)
  - Trigger downstream agent decision cycles at configurable intervals

Not responsible for:
  - Market data source selection (the injected MarketDataProvider; see argus/seams.py)
  - Persistent agent state (see orchestration/state.py)
  - Execution order routing

Dependencies:
  - asyncio
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from datetime import time as dtime

import pandas as pd

from argus.config import settings
from argus.data.cache import OHLCVBuffer
from argus.data.indicators import bars_per_day, compress_candles, required_raw_bars
from argus.data.tickers import is_valid_ticker
from argus.infra.clock import now_market
from argus.params import SYSTEM
from argus.ports import MarketDataProvider
from argus.schemas.signals import missing_session_state_keys

logger = logging.getLogger(__name__)

_MARKET_OPEN = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)

# yf.download's chart endpoint returns the whole series in one request regardless
# of interval granularity, so this costs nothing extra against any rate limit
_FETCH_PERIOD_DAYS = SYSTEM.candle_buffer_days_retained
_FETCH_PERIOD = f"{_FETCH_PERIOD_DAYS}d"

# Once a ticker's buffer already holds an indicator-ready depth, a sweep only
# needs to catch the candles written since the last one — "1d" (yfinance's
# shortest supported intraday period) comfortably covers one mft_fetch_interval_seconds
# gap, and insert_candles' INSERT OR REPLACE makes the overlap with what's
# already buffered a no-op rather than a duplicate
_STEADY_STATE_FETCH_PERIOD = "1d"

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


def _derive_buffer_size(interval_minutes: int) -> int:
    """Sizes the rolling buffer to hold `_FETCH_PERIOD_DAYS` of candles at the given interval.

    One extra day and one extra bar of slack beyond a bare `_FETCH_PERIOD_DAYS`
    fit so a routine session doesn't evict day-1's open before momentum_1d's
    lookback (see data/indicators.py) can reach it.

    Args:
        interval_minutes: Native candle resolution in minutes.

    Returns:
        Buffer row capacity per ticker.
    """
    return (bars_per_day(interval_minutes) + 1) * (_FETCH_PERIOD_DAYS + 1)


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
        market_data: MarketDataProvider,
        interval: str | None = None,
        db_path: str | None = None,
    ) -> None:
        """Creates the pipeline with a candle buffer for the given universe.

        Args:
            tickers: Initial ticker universe to track.
            market_data: Where intraday candles come from. Injected rather than
                imported so the pipeline never reaches past the provider port.
            interval: yfinance interval string overriding `settings.MFT_CANDLE_INTERVAL`.
                Exposed mainly for tests; production call sites rely on the default.
            db_path: SQLite path for the candle buffer. Defaults to a file
                under ``settings.ARGUS_DATA_DIR`` rather than ``:memory:`` so
                a process restart resumes with a warm buffer instead of the
                cold start a fresh buffer would otherwise incur.
        """
        self.market_data = market_data
        self.interval = interval or settings.MFT_CANDLE_INTERVAL
        self.interval_minutes = _parse_interval_minutes(self.interval)
        buffer_size = settings.CANDLE_BUFFER_SIZE or _derive_buffer_size(self.interval_minutes)
        if db_path is None:
            db_file = settings.data_path("ohlcv_buffer.db")
            db_file.parent.mkdir(parents=True, exist_ok=True)
            db_path = str(db_file)
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
        # A stop() leaves the event set; left set, a restarted loop would skip its pacing waits
        self._stop_event.clear()
        logger.info("launching fetch loop")
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
        ``mft_fetch_interval_seconds`` seconds. This is the single enforcement point for
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
            logger.warning("rejected malformed ticker(s): %s", invalid)

        new = [t for t in valid if t not in self.tickers]
        room = max(SYSTEM.max_tracked_tickers - len(self.tickers), 0)
        accepted, dropped = new[:room], new[room:]
        if dropped:
            logger.warning(
                "universe cap (%d) reached; dropping %d ticker(s): %s",
                SYSTEM.max_tracked_tickers,
                len(dropped),
                dropped,
            )
        if accepted:
            self.tickers.extend(accepted)
            logger.info(
                "added %d ticker(s): %s",
                len(accepted),
                accepted,
            )

        # Touches every already-tracked ticker too, not just newly-accepted
        # ones — a repeatedly-requested ticker must never look
        # unused to _evict_stale_tickers just because it was already tracked
        now = now_market()
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
        cutoff = now_market() - timedelta(seconds=SYSTEM.tracked_ticker_ttl_seconds)
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
            "evicted %d unused ticker(s): %s",
            len(stale),
            stale,
        )

    async def stop(self) -> None:
        """Signals active loops to stop execution."""
        self.running = False
        self._stop_event.set()
        logger.info("shutdown requested")

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
            logger.debug("outside market hours — compressing existing buffer only")
        return await asyncio.to_thread(self.compress_all)

    async def _sweep_once(self) -> None:
        """Fetches the latest candlesticks for every tracked ticker, one pass.

        Pauses ``SYSTEM.min_inter_request_seconds`` between requests — a fixed
        floor against yfinance rate limiting, not a spread computed to fill
        ``mft_fetch_interval_seconds``; that framing was wrong in principle (pacing exists
        for rate limiting, not to occupy the cadence budget) and meaningless for
        a one-shot ``run_once()`` sweep. Iterates over a snapshot of
        ``self.tickers`` so that concurrent ``register_tickers`` calls cannot corrupt iteration.
        """
        self._evict_stale_tickers()
        tickers_snapshot = list(self.tickers)
        logger.debug("starting universe sweep (%d tickers)", len(tickers_snapshot))
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
        ``mft_fetch_interval_seconds`` after each pass, not a full interval on top of
        however long the pass itself took — otherwise the real cadence drifts
        to pass_duration + mft_fetch_interval_seconds instead of the intended
        mft_fetch_interval_seconds.

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
                        logger.error("callback raised %s: %s", type(exc).__name__, exc)
                else:
                    logger.debug("outside market hours — idle")
            except Exception as exc:
                logger.error("sweep failed — %s: %s", type(exc).__name__, exc)

            elapsed = time.monotonic() - started
            await self._wait_or_stop(max(0.0, SYSTEM.mft_fetch_interval_seconds - elapsed))

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
                logger.warning("empty result for %s", ticker)
                return

            logger.debug(
                "%s → %d candles inserted, latest close=%.2f",
                ticker,
                n_candles,
                latest_close,
            )

        except Exception as exc:
            logger.warning(
                "%s failed — %s: %s",
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
        every `mft_fetch_interval_seconds` regardless of how warm the buffer already is
        multiplies out to ~15,600 row upserts per sweep at 20 tickers, almost
        all of them re-writing rows already on disk unchanged.

        Args:
            ticker: Equity ticker symbol to fetch.

        Returns:
            Tuple of (candles inserted, latest close). (0, 0.0) on an empty fetch.
        """
        with self._buffer_op_lock:
            is_cold = self.buffer.row_count(ticker) < required_raw_bars(self.interval_minutes)
            period = _FETCH_PERIOD if is_cold else _STEADY_STATE_FETCH_PERIOD
            df: pd.DataFrame = self.market_data.ohlcv_intraday(ticker, self.interval, period)
            if df is None or df.empty:
                return 0, 0.0

            candles = [_candle_from_row(idx, row) for idx, row in df.iterrows()]
            self.buffer.insert_candles(ticker, candles)
            latest_close = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
            return len(df), latest_close

    def compress_all(self) -> dict[str, dict]:
        """Compresses cached candle metrics across all universe symbols into a feature dictionary.

        A ticker is only published once it clears both readiness checks:
        enough raw bars to be worth resampling (`required_raw_bars`), and —
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
        required_raw = required_raw_bars(self.interval_minutes)
        with self._buffer_op_lock:
            tracked = set(self.tickers)
            for ticker in tracked & set(self.buffer.get_all_tickers()):
                try:
                    df = self.buffer.get_candles(ticker)
                    if df is None or len(df) < required_raw:
                        continue
                    state = compress_candles(df, self.interval_minutes)
                    if state is None:
                        continue
                    missing = missing_session_state_keys(state)
                    if missing:
                        logger.warning(
                            "%s missing/non-finite required keys %s — omitting",
                            ticker,
                            missing,
                        )
                        continue
                    states[ticker] = state
                except Exception as exc:
                    logger.warning(
                        "%s compression failed — %s: %s",
                        ticker,
                        type(exc).__name__,
                        exc,
                    )
            self.buffer.prune_untracked(tracked)
        logger.info("%d tickers compressed", len(states))
        return states

    def _is_market_hours(self) -> bool:
        """Checks whether current Eastern Time is within regular US market hours (09:30-16:00 ET).

        Returns:
            True during weekday trading hours in America/New_York timezone.
        """
        now_et = now_market()
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
