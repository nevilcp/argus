"""Tracked intraday candle ingestion and technical feature publication."""

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

# Fetch the full raw series in one chart request; coarser indicators are resampled locally.
_FETCH_PERIOD_DAYS = SYSTEM.candle_buffer_days_retained
_FETCH_PERIOD = f"{_FETCH_PERIOD_DAYS}d"

# Fetch one day once warm; timestamp upserts make overlapping candles safe to replay.
_STEADY_STATE_FETCH_PERIOD = "1d"

_INTERVAL_RE = re.compile(r"^(\d+)(m|h)$")


def _parse_interval_minutes(interval: str) -> int:
    """Convert a yfinance interval string (e.g. "1m", "30m", "1h") to minutes."""
    match = _INTERVAL_RE.match(interval)
    if not match:
        raise ValueError(f"unsupported interval string: {interval!r}")
    value, unit = match.groups()
    return int(value) * (60 if unit == "h" else 1)


def _derive_buffer_size(interval_minutes: int) -> int:
    """Size the rolling buffer to hold `_FETCH_PERIOD_DAYS` of candles at the given interval."""
    return (bars_per_day(interval_minutes) + 1) * (_FETCH_PERIOD_DAYS + 1)


_CANDLE_FIELDS = ("open", "high", "low", "close", "volume")


def _candle_from_row(timestamp: pd.Timestamp, row: pd.Series) -> dict:
    """Build one OHLCVBuffer candle dict from a fetched frame's row."""
    candle: dict = {"timestamp": timestamp.isoformat()}
    for field in _CANDLE_FIELDS:
        candle[field] = float(row[field]) if field in row.index else None
    return candle


class MFTDataPipeline:
    """Tracked candle ingestion with feature publication on each completed market-hours sweep."""

    def __init__(
        self,
        tickers: list[str],
        market_data: MarketDataProvider,
        interval: str | None = None,
        db_path: str | None = None,
    ) -> None:
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
        # Hold this lock throughout buffer workers so shutdown can wait for work that survives task cancellation.
        self._buffer_op_lock = threading.Lock()
        # Apply ticker shape and capacity validation to constructor, collector, CLI, and HTTP entry points alike.
        self.tickers: list[str] = []
        self.last_requested_at: dict[str, datetime] = {}
        self.register_tickers(tickers)
        # Pin seed tickers because the collector does not refresh their last-requested timestamps.
        self._pinned_tickers: frozenset[str] = frozenset(self.tickers)
        logger.info(
            "MFTDataPipeline initialized: %d tickers, interval=%s, buffer_size=%d, buffer=%s",
            len(self.tickers),
            self.interval,
            buffer_size,
            db_path,
        )

    async def start(self, on_session_ready: Callable) -> None:
        """Start the fetch loop, blocking until stopped."""
        self.running = True
        self._stop_event.clear()
        logger.info("launching fetch loop")
        await self._fetch_loop(on_session_ready)

    async def _wait_or_stop(self, timeout: float) -> None:
        """Sleep up to `timeout` seconds, waking immediately if `stop()` is called."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)

    def register_tickers(self, tickers: list[str]) -> None:
        """Add new tickers to the live tracking universe without restarting the pipeline."""
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

        # Refresh existing tickers too so repeated requests cannot be mistaken for inactivity.
        now = now_market()
        for ticker in valid:
            if ticker in self.tickers:
                self.last_requested_at[ticker] = now

    def _evict_stale_tickers(self) -> None:
        """Drop tracked tickers unused beyond SYSTEM.tracked_ticker_ttl_seconds."""
        cutoff = now_market() - timedelta(seconds=SYSTEM.tracked_ticker_ttl_seconds)
        stale = [
            t for t in self.tickers if t not in self._pinned_tickers and self.last_requested_at.get(t, cutoff) < cutoff
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
        """Signal active loops to stop execution."""
        self.running = False
        self._stop_event.set()
        logger.info("shutdown requested")

    async def close_buffer(self) -> None:
        """Wait for orphaned buffer workers before closing SQLite after the loop tasks finish."""
        await asyncio.to_thread(self._buffer_op_lock.acquire)
        try:
            self.buffer.close()
        finally:
            self._buffer_op_lock.release()

    async def run_once(self) -> dict[str, dict]:
        """Run a single fetch-and-compress cycle without the background loops."""
        if self._is_market_hours():
            await self._sweep_once()
        else:
            logger.debug("outside market hours - compressing existing buffer only")
        return await asyncio.to_thread(self.compress_all)

    async def _sweep_once(self) -> None:
        """Fetch the latest candlesticks for every tracked ticker, one pass."""
        self._evict_stale_tickers()
        tickers_snapshot = list(self.tickers)
        logger.debug("starting universe sweep (%d tickers)", len(tickers_snapshot))
        last = len(tickers_snapshot) - 1
        for i, ticker in enumerate(tickers_snapshot):
            await self._fetch_one_ticker(ticker)
            if i < last:
                await asyncio.sleep(SYSTEM.min_inter_request_seconds)

    async def _fetch_loop(self, on_session_ready: Callable) -> None:
        """Fetch candles and publish features on each market-hours cadence."""
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
                    logger.debug("outside market hours - idle")
            except Exception as exc:
                logger.error("sweep failed - %s: %s", type(exc).__name__, exc)

            elapsed = time.monotonic() - started
            await self._wait_or_stop(max(0.0, SYSTEM.mft_fetch_interval_seconds - elapsed))

    async def _fetch_one_ticker(self, ticker: str) -> None:
        """Download and bulk-insert all available candles for a specific symbol."""
        try:
            n_candles, latest_close = await asyncio.to_thread(self._fetch_and_insert, ticker)
            if n_candles == 0:
                logger.warning("empty result for %s", ticker)
                return

            logger.debug(
                "%s -> %d candles inserted, latest close=%.2f",
                ticker,
                n_candles,
                latest_close,
            )

        except Exception as exc:
            logger.warning(
                "%s failed - %s: %s",
                ticker,
                type(exc).__name__,
                exc,
            )

    def _fetch_and_insert(self, ticker: str) -> tuple[int, float]:
        """Fetch a full window for cold buffers and a shorter overlapping window once ready."""
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
        """Compress cached candle metrics across all universe symbols into a feature dictionary."""
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
                            "%s missing/non-finite required keys %s - omitting",
                            ticker,
                            missing,
                        )
                        continue
                    states[ticker] = state
                except Exception as exc:
                    logger.warning(
                        "%s compression failed - %s: %s",
                        ticker,
                        type(exc).__name__,
                        exc,
                    )
            self.buffer.prune_untracked(tracked)
        logger.info("%d tickers compressed", len(states))
        return states

    def _is_market_hours(self) -> bool:
        """Check whether current Eastern Time is within regular US market hours (09:30-16:00 ET)."""
        now_et = now_market()
        if now_et.weekday() >= 5:
            return False
        current = now_et.time().replace(second=0, microsecond=0)
        return _MARKET_OPEN <= current < _MARKET_CLOSE

    def is_market_hours(self) -> bool:
        """Report whether the market-hours gate permits a fetch."""
        return self._is_market_hours()
