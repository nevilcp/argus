"""
argus/data/pipeline.py

Mid-Frequency Trading (MFT) data pipeline for real-time asset ingestion.

Responsibilities:
  - Asynchronously fetch 5-minute intraday candles across the target ticker universe
  - Buffer candles in OHLCVBuffer and compress them into technical feature dicts
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
from collections.abc import Callable
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

from argus.data.cache import OHLCVBuffer
from argus.data.fetchers import fetch_ohlcv_intraday

logger = logging.getLogger("argus.pipeline")

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)

# Tuned to stay under yfinance's 5 req/min informal rate ceiling for intraday data
_BATCH_SIZE = 4
_INTER_REQUEST_SLEEP = 13
_FETCH_INTERVAL = 300
_SESSION_INTERVAL = 1800


class MFTDataPipeline:
    """Asynchronous mid-frequency data pipeline coordinating candlestick updates and feature extraction.

    Runs two concurrent asyncio loops: one that periodically fetches intraday
    candles for all tracked tickers, and one that fires a decision-cycle callback
    at a longer interval. Both loops only execute during US equity market hours.
    """

    def __init__(
        self,
        tickers: list[str],
    ) -> None:
        """Creates the pipeline with an in-memory candle buffer for the given universe.

        Args:
            tickers: Initial ticker universe to track.
        """
        self.tickers = tickers
        self.buffer = OHLCVBuffer(db_path=":memory:", buffer_size=78)
        self.running = False
        logger.info("MFTDataPipeline initialised: %d tickers", len(tickers))

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

    async def _fetch_loop(self) -> None:
        """Periodically downloads the latest intraday candlesticks for the tracking universe.

        Batches requests by ``_BATCH_SIZE`` with ``_INTER_REQUEST_SLEEP`` seconds between
        each fetch to avoid triggering yfinance throttling. Iterates over a snapshot of
        ``self.tickers`` so that concurrent ``register_tickers`` calls cannot corrupt iteration.
        """
        while self.running:
            if self._is_market_hours():
                tickers_snapshot = list(self.tickers)
                logger.debug("_fetch_loop: starting universe sweep (%d tickers)", len(tickers_snapshot))
                for i in range(0, len(tickers_snapshot), _BATCH_SIZE):
                    batch = tickers_snapshot[i : i + _BATCH_SIZE]
                    for j, ticker in enumerate(batch):
                        await self._fetch_one_ticker(ticker)
                        # Sleep between tickers but not after the last one in the final batch
                        if not (i + j + 1 >= len(tickers_snapshot)):
                            await asyncio.sleep(_INTER_REQUEST_SLEEP)
            else:
                logger.debug("_fetch_loop: outside market hours — idle")

            await asyncio.sleep(_FETCH_INTERVAL)

    async def _session_loop(self, callback: Callable) -> None:
        """Periodically compiles buffered candlesticks and triggers downstream agent actions.

        Args:
            callback: Async callable receiving the ``session_states`` dict.
        """
        while self.running:
            await asyncio.sleep(_SESSION_INTERVAL)
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

    async def _fetch_one_ticker(self, ticker: str) -> None:
        """Downloads and bulk-inserts all available 5-minute candles for a specific symbol.

        Fetches up to 2 trading days of 5-minute candles and inserts every row into the
        buffer. This warms the buffer to a full indicator-ready depth on the very first
        fetch cycle, eliminating the 70-minute cold start that would result from
        inserting only the latest candle. Duplicate timestamps are safely overwritten
        via the buffer's INSERT OR REPLACE contract.

        Args:
            ticker: Equity ticker symbol to fetch.
        """
        try:
            df: pd.DataFrame = await asyncio.to_thread(fetch_ohlcv_intraday, ticker, "5m", "2d")
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
            df: OHLCV DataFrame with float columns and a datetime index.

        Returns:
            Dict with keys: rsi_14, macd_histogram, bb_percent_b, atr_pct, adx_14,
            vwap_distance, volume_ratio, momentum_30m, momentum_1d, close, timestamp.
        """
        import pandas_ta as ta  # Lazy import to avoid heavy C extension load on startup

        rsi_series = ta.rsi(df["close"], length=14)
        rsi_14 = (
            float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else 50.0
        )

        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        macd_hist = 0.0
        if macd_df is not None and not macd_df.empty:
            # pandas_ta names the histogram column with an 'h' suffix, e.g. MACDh_12_26_9
            hist_col = [c for c in macd_df.columns if c.upper().startswith("MACDH_")]
            if hist_col:
                macd_hist = float(macd_df[hist_col[0]].iloc[-1])
            else:
                logger.warning("_compress_candles: MACD histogram column not found in %s", list(macd_df.columns))

        bb_df = ta.bbands(df["close"], length=20, std=2)
        bb_pct_b = 0.5
        if bb_df is not None and not bb_df.empty:
            # pandas_ta names the %B column with a 'BBP_' prefix, e.g. BBP_20_2.0
            pct_col = [c for c in bb_df.columns if c.upper().startswith("BBP_")]
            if pct_col:
                bb_pct_b = float(bb_df[pct_col[0]].iloc[-1])
            else:
                logger.warning("_compress_candles: BB %%B column not found in %s", list(bb_df.columns))

        atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
        close_last = float(df["close"].iloc[-1])
        atr_pct = (
            (float(atr_series.iloc[-1]) / close_last)
            if (atr_series is not None and not atr_series.empty and close_last)
            else 0.0
        )

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        adx_14 = 20.0
        if adx_df is not None and not adx_df.empty:
            adx_col = [c for c in adx_df.columns if c.upper().startswith("ADX_")]
            if adx_col:
                adx_14 = float(adx_df[adx_col[0]].iloc[-1])

        vwap_distance = 0.0
        try:
            vwap_series = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            if vwap_series is not None and not vwap_series.empty:
                vwap_val = float(vwap_series.iloc[-1])
                if vwap_val and close_last:
                    vwap_distance = (close_last - vwap_val) / vwap_val
        except Exception:
            # pandas_ta raises on zero-volume days; default to 0.0 (no VWAP deviation)
            pass

        vol_series = df["volume"]
        vol_mean = (
            float(vol_series.rolling(20).mean().iloc[-1])
            if len(vol_series) >= 20
            else float(vol_series.mean())
        )
        volume_ratio = (float(vol_series.iloc[-1]) / vol_mean) if vol_mean else 1.0

        close_s = df["close"]
        n = len(close_s)
        # 7-bar lookback ≈ 35 minutes at 5m resolution; 79-bar lookback ≈ 1 trading day
        momentum_30m = (float(close_s.iloc[-1]) / float(close_s.iloc[max(-7, -n)])) - 1.0
        momentum_1d = (float(close_s.iloc[-1]) / float(close_s.iloc[max(-79, -n)])) - 1.0

        return {
            "rsi_14": round(rsi_14, 4),
            "macd_histogram": round(macd_hist, 6),
            "bb_percent_b": round(bb_pct_b, 4),
            "atr_pct": round(atr_pct, 6),
            "adx_14": round(adx_14, 4),
            "vwap_distance": round(vwap_distance, 6),
            "volume_ratio": round(volume_ratio, 4),
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
