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
  - pandas_ta (lazy import; loaded only at indicator compute time)
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from argus.config import settings
from argus.data.cache import OHLCVBuffer
from argus.data.fetchers import fetch_ohlcv_intraday
from argus.params import SYSTEM

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

# Conservative, unmeasured estimate of one yfinance round-trip; used only to
# derive inter-ticker spacing so a sweep finishes within _FETCH_INTERVAL
_ESTIMATED_FETCH_SECONDS_PER_TICKER = 1.0

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

    Matches fetch_ohlcv_intraday's fetch period so a single sweep's candles
    always fit without evicting same-day history.

    Args:
        interval_minutes: Native candle resolution in minutes.

    Returns:
        Buffer row capacity per ticker.
    """
    return _bars_per_day(interval_minutes) * _FETCH_PERIOD_DAYS


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
        self.buffer = OHLCVBuffer(db_path=db_path, buffer_size=buffer_size)
        self.running = False
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
        await asyncio.gather(
            self._fetch_loop(),
            self._session_loop(on_session_ready),
        )

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
        logger.info("MFTDataPipeline.stop: shutdown requested")

    async def run_once(self) -> dict[str, dict]:
        """Runs a single fetch-and-compress cycle without the background loops.

        Used by the unattended collector (``argus/orchestration/collector.py``)
        to force one sweep on demand rather than waiting on ``start()``'s
        periodic loops. Outside market hours this skips the fetch and just
        compresses whatever the buffer already holds.

        Returns:
            Mapping of ticker → technical feature dict. Tickers with fewer
            than 14 buffered bars are omitted (see ``OHLCVBuffer.get_candles``).
        """
        if self._is_market_hours():
            await self._sweep_once()
        else:
            logger.debug("run_once: outside market hours — compressing existing buffer only")
        return self.compress_all()

    async def _sweep_once(self) -> None:
        """Fetches the latest candlesticks for every tracked ticker, one pass.

        Spaces requests across the sweep so it finishes within ``_FETCH_INTERVAL``
        rather than on a fixed per-batch schedule. Iterates over a snapshot of
        ``self.tickers`` so that concurrent ``register_tickers`` calls cannot corrupt iteration.
        """
        tickers_snapshot = list(self.tickers)
        logger.debug("_sweep_once: starting universe sweep (%d tickers)", len(tickers_snapshot))
        sleep_seconds = self._inter_request_sleep(len(tickers_snapshot))
        last = len(tickers_snapshot) - 1
        for i, ticker in enumerate(tickers_snapshot):
            await self._fetch_one_ticker(ticker)
            if i < last:
                await asyncio.sleep(sleep_seconds)

    async def _fetch_loop(self) -> None:
        """Periodically downloads the latest intraday candlesticks for the tracking universe."""
        while self.running:
            if self._is_market_hours():
                await self._sweep_once()
            else:
                logger.debug("_fetch_loop: outside market hours — idle")

            await asyncio.sleep(_FETCH_INTERVAL)

    def _inter_request_sleep(self, n_tickers: int) -> float:
        """Derives the per-ticker sleep so a sweep finishes within `_FETCH_INTERVAL`.

        Args:
            n_tickers: Number of tickers in the current sweep.

        Returns:
            Seconds to sleep between consecutive ticker fetches. 0.0 if there's
            nothing to space out, or if the universe is too large to fit even an
            unpaced sweep inside `_FETCH_INTERVAL` — in which case candles simply
            refresh less often than `_FETCH_INTERVAL` implies.
        """
        if n_tickers <= 1:
            return 0.0
        estimated_fetch_seconds = n_tickers * _ESTIMATED_FETCH_SECONDS_PER_TICKER
        slack = _FETCH_INTERVAL - estimated_fetch_seconds
        if slack <= 0:
            logger.warning(
                "_inter_request_sleep: %d tickers won't fit an estimated %.0fs fetch "
                "into the %ds cycle even unpaced; sweeping back-to-back",
                n_tickers,
                estimated_fetch_seconds,
                _FETCH_INTERVAL,
            )
            return 0.0
        return slack / n_tickers

    def _buffer_warm(self) -> bool:
        """Checks whether any tracked ticker already has an indicator-ready buffer depth.

        Returns:
            True once at least one ticker holds >= 14 rows (``OHLCVBuffer.get_candles``'
            own floor), letting callers skip waiting a full session interval on a
            cold start.
        """
        return any(self.buffer.get_candles(t) is not None for t in self.buffer.get_all_tickers())

    async def _session_loop(self, callback: Callable) -> None:
        """Periodically compiles buffered candlesticks and triggers downstream agent actions.

        Args:
            callback: Async callable receiving the ``session_states`` dict.
        """
        # A fresh buffer sleeping a full session interval before its first
        # compress would leave the live cache empty for that whole stretch for
        # no reason; poll faster until there's actually something to compress
        while self.running and not self._buffer_warm():
            await asyncio.sleep(_WARMUP_POLL_INTERVAL)

        while self.running:
            if self._is_market_hours():
                logger.info("_session_loop: triggering decision cycle")
                session_states = self.compress_all()
                try:
                    await callback(session_states)
                except Exception as exc:
                    logger.error(
                        "_session_loop: callback raised %s: %s",
                        type(exc).__name__,
                        exc,
                    )
            await asyncio.sleep(self.session_interval_seconds)

    async def _fetch_one_ticker(self, ticker: str) -> None:
        """Downloads and bulk-inserts all available candles for a specific symbol.

        Fetches `_FETCH_PERIOD` of candles at `self.interval` and inserts every row
        into the buffer. This warms the buffer to a full indicator-ready depth on
        the very first fetch cycle, eliminating the cold start that would result
        from inserting only the latest candle. Duplicate timestamps are safely
        overwritten via the buffer's INSERT OR REPLACE contract.

        Args:
            ticker: Equity ticker symbol to fetch.
        """
        try:
            df: pd.DataFrame = await asyncio.to_thread(
                fetch_ohlcv_intraday, ticker, self.interval, _FETCH_PERIOD
            )
            if df is None or df.empty:
                logger.warning("_fetch_one_ticker: empty result for %s", ticker)
                return

            for idx, row in df.iterrows():
                candle = {
                    "timestamp": idx.isoformat(),
                    "open": float(row["open"]) if "open" in row.index else None,
                    "high": float(row["high"]) if "high" in row.index else None,
                    "low": float(row["low"]) if "low" in row.index else None,
                    "close": float(row["close"]) if "close" in row.index else None,
                    "volume": float(row["volume"]) if "volume" in row.index else None,
                }
                self.buffer.insert_candle(ticker, candle)

            logger.debug(
                "_fetch_one_ticker: %s → %d candles inserted, latest close=%.2f",
                ticker,
                len(df),
                float(df["close"].iloc[-1]) if "close" in df.columns else 0.0,
            )

        except Exception as exc:
            logger.warning(
                "_fetch_one_ticker: %s failed — %s: %s",
                ticker,
                type(exc).__name__,
                exc,
            )

    def compress_all(self) -> dict[str, dict]:
        """Compresses cached candle metrics across all universe symbols into a feature dictionary.

        Returns:
            Mapping of ticker → technical feature dict (rsi_14, macd_histogram, etc.).
            Failed tickers are omitted and logged as warnings.
        """
        states: dict[str, dict] = {}
        for ticker in self.buffer.get_all_tickers():
            df = self.buffer.get_candles(ticker)
            if df is not None:
                try:
                    states[ticker] = self._compress_candles(df)
                except Exception as exc:
                    logger.warning(
                        "compress_all: %s compression failed — %s: %s",
                        ticker,
                        type(exc).__name__,
                        exc,
                    )
        logger.info("compress_all: %d tickers compressed", len(states))
        return states

    def _compress_candles(self, df: pd.DataFrame) -> dict:
        """Calculates technical indicators on a DataFrame using pandas-ta.

        Delayed import of pandas_ta avoids loading the heavy C extension on startup,
        which is particularly important in serverless and cold-start environments.

        Args:
            df: Raw OHLCV DataFrame at `self.interval` resolution, float columns,
                datetime index.

        Returns:
            Dict with keys: rsi_14, macd_histogram, bb_percent_b, atr_pct, adx_14,
            vwap_distance, volume_ratio, momentum_30m, momentum_1d, close, timestamp.
            rsi_14, bb_percent_b, atr_pct, adx_14, vwap_distance, and volume_ratio
            are None when pandas_ta cannot compute them; close and timestamp are
            always populated.
        """
        import pandas_ta as ta  # Lazy import to avoid heavy C extension load on startup

        ind_df = _resample_ohlcv(df, self.interval_minutes, _INDICATOR_RESAMPLE_MINUTES)

        rsi_series = ta.rsi(ind_df["close"], length=14)
        rsi_14 = (
            float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None
        )

        macd_df = ta.macd(ind_df["close"], fast=12, slow=26, signal=9)
        macd_hist = 0.0
        if macd_df is not None and not macd_df.empty:
            # pandas_ta names the histogram column with an 'h' suffix, e.g. MACDh_12_26_9
            hist_col = [c for c in macd_df.columns if c.upper().startswith("MACDH_")]
            if hist_col:
                macd_hist = float(macd_df[hist_col[0]].iloc[-1])
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
        atr_pct = (
            (float(atr_series.iloc[-1]) / close_last)
            if (atr_series is not None and not atr_series.empty and close_last)
            else None
        )

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
                vwap_val = float(vwap_series.iloc[-1])
                if vwap_val and close_last:
                    vwap_distance = (close_last - vwap_val) / vwap_val
        except Exception:
            # pandas_ta raises on zero-volume days; leave unset rather than fabricate 0.0
            pass

        vol_series = ind_df["volume"]
        vol_mean = (
            float(vol_series.rolling(20).mean().iloc[-1])
            if len(vol_series) >= 20
            else float(vol_series.mean())
        )
        volume_ratio = (float(vol_series.iloc[-1]) / vol_mean) if vol_mean else None

        close_s = df["close"]
        n = len(close_s)
        momentum_30m_bars = max(1, 30 // self.interval_minutes)
        momentum_1d_bars = _bars_per_day(self.interval_minutes)
        momentum_30m = (
            float(close_s.iloc[-1]) / float(close_s.iloc[max(-momentum_30m_bars, -n)])
        ) - 1.0
        momentum_1d = (
            float(close_s.iloc[-1]) / float(close_s.iloc[max(-momentum_1d_bars, -n)])
        ) - 1.0

        return {
            "rsi_14": round(rsi_14, 4) if rsi_14 is not None else None,
            "macd_histogram": round(macd_hist, 6),
            "bb_percent_b": round(bb_pct_b, 4) if bb_pct_b is not None else None,
            "atr_pct": round(atr_pct, 6) if atr_pct is not None else None,
            "adx_14": round(adx_14, 4) if adx_14 is not None else None,
            "vwap_distance": round(vwap_distance, 6) if vwap_distance is not None else None,
            "volume_ratio": round(volume_ratio, 4) if volume_ratio is not None else None,
            "momentum_30m": round(momentum_30m, 6),
            "momentum_1d": round(momentum_1d, 6),
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
