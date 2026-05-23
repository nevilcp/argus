"""
argus/data/pipeline.py
======================
Mid-Frequency Trading (MFT) data pipeline for ARGUS v2.

``MFTDataPipeline`` drives the real-time data flow:

1. Every 300 seconds (5 minutes) the ``_fetch_loop`` pulls the latest 5-minute
   candle for each ticker and stores it in the in-memory ``OHLCVBuffer``.

2. Every 1800 seconds (30 minutes) the ``_session_loop`` compresses the
   buffered candles into a feature dict (RSI, MACD, BB, ATR, ADX, VWAP, …)
   and fires the ``on_session_ready`` callback — which triggers the LangGraph
   agent graph for a new decision cycle.

Rate limiting
-------------
Polygon free tier allows 5 requests/minute.  The fetch loop issues one request
per ticker with a 13-second inter-request sleep, batched in groups of 4.

Market hours guard
------------------
``_is_market_hours()`` checks US/Eastern time (09:30 – 16:00, weekdays only)
so the pipeline idles silently outside trading hours without error.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from argus.data.cache import OHLCVBuffer
from argus.data.fetchers import DataFetchError, fetch_ohlcv_intraday

logger = logging.getLogger("argus.pipeline")

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN  = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)

# Batch size chosen so that 4 tickers × 13 s ≈ 52 s < 60 s → < 5 req/min
_BATCH_SIZE        = 4
_INTER_REQUEST_SLEEP = 13   # seconds between individual ticker fetches
_FETCH_INTERVAL      = 300  # seconds between full-universe fetch sweeps
_SESSION_INTERVAL    = 1800 # seconds between decision-cycle callbacks


class MFTDataPipeline:
    """
    Asynchronous mid-frequency data pipeline.

    Parameters
    ----------
    tickers:
        List of equity symbols to track.
    polygon_key:
        Optional Polygon.io API key (reserved for future use when the
        Polygon websocket replaces the yfinance polling approach).
    """

    def __init__(
        self,
        tickers: list[str],
        polygon_key: Optional[str] = None,
    ) -> None:
        self.tickers     = tickers
        self.polygon_key = polygon_key
        self.buffer      = OHLCVBuffer(db_path=":memory:", buffer_size=78)
        self.running     = False
        logger.info(
            "MFTDataPipeline initialised: %d tickers", len(tickers)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def start(self, on_session_ready: Callable) -> None:
        """
        Start the pipeline by launching both the fetch and session loops.

        Both coroutines run concurrently via ``asyncio.gather``.  This method
        blocks until :meth:`stop` is called.

        Parameters
        ----------
        on_session_ready:
            Async callable invoked every ``SESSION_INTERVAL`` seconds when
            the market is open.  Signature: ``async def cb(states: dict) -> None``
            where *states* is the output of :meth:`compress_all`.
        """
        self.running = True
        logger.info("MFTDataPipeline.start: launching fetch and session loops")
        await asyncio.gather(
            self._fetch_loop(),
            self._session_loop(on_session_ready),
        )

    async def stop(self) -> None:
        """Signal both loops to exit on their next iteration."""
        self.running = False
        logger.info("MFTDataPipeline.stop: shutdown requested")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal loops
    # ──────────────────────────────────────────────────────────────────────────

    async def _fetch_loop(self) -> None:
        """
        Periodically fetch the latest 5-minute candle for every tracked ticker.

        Runs in batches of ``_BATCH_SIZE`` with ``_INTER_REQUEST_SLEEP`` seconds
        between individual fetches to stay under the Polygon free-tier limit of
        5 requests/minute.  The full sweep repeats every ``_FETCH_INTERVAL``
        seconds (300 s = 5 min).

        Only operates during US market hours; sleeps through off-hours without
        error.
        """
        while self.running:
            if self._is_market_hours():
                logger.debug("_fetch_loop: starting universe sweep (%d tickers)", len(self.tickers))
                for i in range(0, len(self.tickers), _BATCH_SIZE):
                    batch = self.tickers[i : i + _BATCH_SIZE]
                    for ticker in batch:
                        await self._fetch_one_ticker(ticker)
                        await asyncio.sleep(_INTER_REQUEST_SLEEP)
            else:
                logger.debug("_fetch_loop: outside market hours — idle")

            await asyncio.sleep(_FETCH_INTERVAL)

    async def _session_loop(self, callback: Callable) -> None:
        """
        Every ``SESSION_INTERVAL`` seconds, compress buffered candles and fire
        *callback* if the market is open.

        Parameters
        ----------
        callback:
            Async callable receiving the compressed session states dict.
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
                        type(exc).__name__, exc,
                    )

    # ──────────────────────────────────────────────────────────────────────────
    # Fetch helper
    # ──────────────────────────────────────────────────────────────────────────

    async def _fetch_one_ticker(self, ticker: str) -> None:
        """
        Fetch the latest 5-minute candle for *ticker* and store it in the buffer.

        Uses ``asyncio.to_thread`` to run the blocking yfinance call off the
        event loop.  If the fetch fails for any reason, a warning is logged
        and the loop continues — individual ticker failures must never crash the
        pipeline.

        Parameters
        ----------
        ticker:
            Equity symbol.
        """
        try:
            df: pd.DataFrame = await asyncio.to_thread(
                fetch_ohlcv_intraday, ticker, "5m", "2d"
            )
            if df is None or df.empty:
                logger.warning("_fetch_one_ticker: empty result for %s", ticker)
                return

            # Take the most recent complete candle (iloc[-1])
            latest = df.iloc[-1]
            candle = {
                "timestamp": df.index[-1].isoformat(),
                "open":      float(latest["open"])   if "open"   in latest.index else None,
                "high":      float(latest["high"])   if "high"   in latest.index else None,
                "low":       float(latest["low"])    if "low"    in latest.index else None,
                "close":     float(latest["close"])  if "close"  in latest.index else None,
                "volume":    float(latest["volume"]) if "volume" in latest.index else None,
            }
            self.buffer.insert_candle(ticker, candle)
            logger.debug("_fetch_one_ticker: %s → close=%.2f", ticker, candle["close"] or 0.0)

        except (DataFetchError, Exception) as exc:
            logger.warning(
                "_fetch_one_ticker: %s failed — %s: %s",
                ticker, type(exc).__name__, exc,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Compression
    # ──────────────────────────────────────────────────────────────────────────

    def compress_all(self) -> dict[str, dict]:
        """
        Compress buffered candles for every ticker into a feature dict.

        Returns
        -------
        dict[str, dict]
            ``{ticker: feature_dict}`` for every ticker that has enough
            candles (≥ 14).  Tickers with insufficient data are skipped.
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
                        ticker, type(exc).__name__, exc,
                    )
        logger.info("compress_all: %d tickers compressed", len(states))
        return states

    def _compress_candles(self, df: pd.DataFrame) -> dict:
        """
        Run pandas-ta technical indicators on *df* and return the latest values.

        Indicators computed
        -------------------
        ``rsi_14``       14-period RSI
        ``macd_histogram`` MACD signal line histogram (12/26/9)
        ``bb_percent_b``  Bollinger %B (20-period, 2 σ)
        ``atr_pct``       14-period ATR as % of close price
        ``adx_14``        14-period ADX trend strength
        ``vwap_distance`` (close − VWAP) / VWAP
        ``volume_ratio``  current volume / 20-period rolling mean
        ``momentum_30m``  6-bar log return  (≈ 30 minutes)
        ``momentum_1d``   78-bar log return (≈ 1 trading day)
        ``close``         last close price
        ``timestamp``     ISO-8601 string of the latest bar

        Parameters
        ----------
        df:
            OHLCV DataFrame with DatetimeIndex, at least 14 rows.

        Returns
        -------
        dict
            Feature dict ready for injection into ``TechnicalSignal``.
        """
        import pandas_ta as ta  # lazy import — heavy dep, avoid at module load

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi_series  = ta.rsi(df["close"], length=14)
        rsi_14      = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else 50.0

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_df     = ta.macd(df["close"], fast=12, slow=26, signal=9)
        macd_hist   = 0.0
        if macd_df is not None and not macd_df.empty:
            hist_col = [c for c in macd_df.columns if "h" in c.lower()]
            if hist_col:
                macd_hist = float(macd_df[hist_col[0]].iloc[-1])

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb_df       = ta.bbands(df["close"], length=20, std=2)
        bb_pct_b    = 0.5
        if bb_df is not None and not bb_df.empty:
            pct_col = [c for c in bb_df.columns if "p" in c.lower()]
            if pct_col:
                bb_pct_b = float(bb_df[pct_col[0]].iloc[-1])

        # ── ATR ──────────────────────────────────────────────────────────────
        atr_series  = ta.atr(df["high"], df["low"], df["close"], length=14)
        close_last  = float(df["close"].iloc[-1])
        atr_pct     = (float(atr_series.iloc[-1]) / close_last) if (
            atr_series is not None and not atr_series.empty and close_last
        ) else 0.0

        # ── ADX ──────────────────────────────────────────────────────────────
        adx_df      = ta.adx(df["high"], df["low"], df["close"], length=14)
        adx_14      = 20.0
        if adx_df is not None and not adx_df.empty:
            adx_col = [c for c in adx_df.columns if c.upper().startswith("ADX_")]
            if adx_col:
                adx_14 = float(adx_df[adx_col[0]].iloc[-1])

        # ── VWAP ─────────────────────────────────────────────────────────────
        vwap_distance = 0.0
        try:
            vwap_series = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            if vwap_series is not None and not vwap_series.empty:
                vwap_val = float(vwap_series.iloc[-1])
                if vwap_val and close_last:
                    vwap_distance = (close_last - vwap_val) / vwap_val
        except Exception:
            pass  # VWAP may fail if volume is zero — keep default 0.0

        # ── Volume ratio ─────────────────────────────────────────────────────
        vol_series   = df["volume"]
        vol_mean     = float(vol_series.rolling(20).mean().iloc[-1]) if len(vol_series) >= 20 else float(vol_series.mean())
        volume_ratio = (float(vol_series.iloc[-1]) / vol_mean) if vol_mean else 1.0

        # ── Momentum ─────────────────────────────────────────────────────────
        close_s      = df["close"]
        n            = len(close_s)
        momentum_30m = (float(close_s.iloc[-1]) / float(close_s.iloc[max(-7,  -n)])) - 1.0
        momentum_1d  = (float(close_s.iloc[-1]) / float(close_s.iloc[max(-79, -n)])) - 1.0

        return {
            "rsi_14":          round(rsi_14, 4),
            "macd_histogram":  round(macd_hist, 6),
            "bb_percent_b":    round(bb_pct_b, 4),
            "atr_pct":         round(atr_pct, 6),
            "adx_14":          round(adx_14, 4),
            "vwap_distance":   round(vwap_distance, 6),
            "volume_ratio":    round(volume_ratio, 4),
            "momentum_30m":    round(momentum_30m, 6),
            "momentum_1d":     round(momentum_1d, 6),
            "close":           round(close_last, 4),
            "timestamp":       df.index[-1].isoformat(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Market hours guard
    # ──────────────────────────────────────────────────────────────────────────

    def _is_market_hours(self) -> bool:
        """
        Return ``True`` if the current US/Eastern time is within regular
        trading hours (09:30 – 16:00, Monday – Friday).

        Uses ``zoneinfo.ZoneInfo("America/New_York")`` for DST-aware
        conversion.  Pre-market and after-hours sessions are excluded.
        """
        now_et = datetime.now(_ET)
        if now_et.weekday() >= 5:   # Saturday=5, Sunday=6
            return False
        current = now_et.time().replace(second=0, microsecond=0)
        return _MARKET_OPEN <= current < _MARKET_CLOSE
