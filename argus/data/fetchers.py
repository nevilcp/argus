"""
argus/data/fetchers.py
======================
Centralised data access layer for ARGUS v2.

ALL agents import exclusively from this module — no agent may call yfinance,
fredapi, newsapi, or any other external library directly.

Features
--------
- Exponential back-off retry (3 attempts) on every network call
- Module-level 6-hour in-memory cache for FRED series
- ThreadPoolExecutor parallelism for multi-ticker daily fetches
- Graceful degradation for optional credentials (news)
- Structured logging via ``logger = logging.getLogger("argus.fetchers")``
"""

from __future__ import annotations

import logging
import time
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable, TypeVar

import pandas as pd
import yfinance as yf

from argus.config import settings

# ──────────────────────────────────────────────────────────────────────────────
# Module logger
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("argus.fetchers")

# ──────────────────────────────────────────────────────────────────────────────
# Custom exception
# ──────────────────────────────────────────────────────────────────────────────


class DataFetchError(Exception):
    """Raised when a data fetch fails after all retry attempts."""


# ──────────────────────────────────────────────────────────────────────────────
# Retry decorator
# ──────────────────────────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.5  # seconds; doubles each attempt


def _with_retry(fn: F) -> F:
    """
    Decorator: retry *fn* up to ``_RETRY_ATTEMPTS`` times with exponential
    back-off (1.5 s, 3 s, 6 s).  Re-raises the last exception as
    ``DataFetchError`` if all attempts fail.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return fn(*args, **kwargs)
            except DataFetchError:
                raise  # already wrapped — don't double-wrap
            except Exception as exc:
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "%s attempt %d/%d failed (%s: %s); retrying in %.1fs",
                    fn.__name__, attempt, _RETRY_ATTEMPTS,
                    type(exc).__name__, exc, delay,
                )
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(delay)
        raise DataFetchError(
            f"{fn.__name__} failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    return wrapper  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# Column normaliser
# ──────────────────────────────────────────────────────────────────────────────

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lower-case column names and retain only OHLCV columns.

    yfinance ≥ 0.2.x returns a ``(Price, Ticker)`` MultiIndex where level-0
    is the field name (Close, High, …) and level-1 is the ticker symbol.
    This function flattens that structure and returns a clean DataFrame with
    columns ``[open, high, low, close, volume]`` and a ``DatetimeIndex``.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Level 0 = price field (Close, High, …), level 1 = ticker symbol.
        # Drop the ticker level so we're left with flat field names.
        df = df.droplevel(1, axis=1)

    # Lower-case all column names for consistent access
    df.columns = [str(c).lower() for c in df.columns]

    available = [c for c in _OHLCV_COLUMNS if c in df.columns]
    if not available:
        raise DataFetchError(
            f"DataFrame has no OHLCV columns; found: {list(df.columns)}"
        )
    df = df[available].copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(subset=["close"], inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Price Data
# ──────────────────────────────────────────────────────────────────────────────


@_with_retry
def fetch_ohlcv_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for *ticker* via yfinance.

    Parameters
    ----------
    ticker:
        Equity symbol (e.g. ``"AAPL"``).
    period:
        yfinance period string, e.g. ``"2y"``, ``"1y"``, ``"6mo"``, ``"1mo"``.

    Returns
    -------
    pd.DataFrame
        Columns ``[open, high, low, close, volume]`` with a ``DatetimeIndex``
        sorted ascending.

    Raises
    ------
    DataFetchError
        If yfinance returns an empty DataFrame after all retries.
    """
    logger.debug("fetch_ohlcv_daily: ticker=%s period=%s", ticker, period)
    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise DataFetchError(
            f"yfinance returned empty DataFrame for ticker={ticker!r} period={period!r}"
        )
    df = _normalise_ohlcv(raw)
    df.sort_index(inplace=True)
    logger.info("fetch_ohlcv_daily: %s → %d rows", ticker, len(df))
    return df


@_with_retry
def fetch_ohlcv_intraday(
    ticker: str, interval: str = "5m", period: str = "5d"
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars for *ticker* via yfinance.

    Used by the MFT pipeline (default 5-minute candles, 5-day window).

    Parameters
    ----------
    ticker:
        Equity symbol.
    interval:
        yfinance interval string: ``"1m"``, ``"2m"``, ``"5m"``, ``"15m"``,
        ``"30m"``, ``"60m"``, ``"90m"``, ``"1h"``.
    period:
        yfinance period string: ``"1d"`` – ``"60d"`` (yfinance limit for
        sub-hourly intervals is 60 days).

    Returns
    -------
    pd.DataFrame
        Same structure as :func:`fetch_ohlcv_daily`.

    Raises
    ------
    DataFetchError
        If the result is empty.
    """
    logger.debug(
        "fetch_ohlcv_intraday: ticker=%s interval=%s period=%s",
        ticker, interval, period,
    )
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise DataFetchError(
            f"yfinance returned empty intraday DataFrame for "
            f"ticker={ticker!r} interval={interval!r} period={period!r}"
        )
    df = _normalise_ohlcv(raw)
    df.sort_index(inplace=True)
    logger.info(
        "fetch_ohlcv_intraday: %s [%s] → %d rows", ticker, interval, len(df)
    )
    return df


def fetch_multiple_daily(
    tickers: list[str], period: str = "1y"
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV for multiple *tickers* in parallel.

    Uses a ``ThreadPoolExecutor`` with ``max_workers=5`` to parallelise the
    yfinance calls.  Individual ticker failures are logged as warnings and
    excluded from the result — the function never raises.

    Parameters
    ----------
    tickers:
        List of equity symbols.
    period:
        Passed to :func:`fetch_ohlcv_daily`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps each successfully fetched ticker to its OHLCV DataFrame.
        Missing tickers (failed fetches) are absent from the dict.
    """
    results: dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        future_to_ticker = {
            pool.submit(fetch_ohlcv_daily, ticker, period): ticker
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                logger.warning(
                    "fetch_multiple_daily: failed for %s — %s: %s",
                    ticker, type(exc).__name__, exc,
                )

    logger.info(
        "fetch_multiple_daily: %d/%d tickers fetched successfully",
        len(results), len(tickers),
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Fundamentals
# ──────────────────────────────────────────────────────────────────────────────


@_with_retry
def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch key fundamental metrics for *ticker* via ``yfinance.Ticker.info``.

    Returns a flat dict with standardised keys.  All values default to
    ``None`` when the field is absent from the yfinance payload (common for
    ETFs, pre-revenue companies, or tickers without analyst coverage).

    Keys returned
    -------------
    ``pe_ttm``              trailing P/E ratio
    ``revenue_growth_yoy``  YoY revenue growth rate (decimal, e.g. 0.12 = 12 %)
    ``operating_margin``    operating income / revenue (decimal)
    ``net_margin``          net income / revenue (decimal)
    ``fcf_yield``           free cash flow / market cap (computed; decimal)
    ``debt_to_equity``      total debt / total equity
    ``current_ratio``       current assets / current liabilities
    ``roe``                 return on equity (decimal)
    ``roic``                return on assets used as ROIC proxy (decimal)
    ``sector``              GICS sector string
    ``industry``            GICS industry string
    ``marketCap``           market capitalisation (USD)
    ``p_fcf``               price / free-cash-flow
    ``as_of_date``          ISO-8601 date string (today's date)

    Raises
    ------
    DataFetchError
        If the yfinance call itself fails after all retries.
    """
    logger.debug("fetch_fundamentals: ticker=%s", ticker)
    info: dict = yf.Ticker(ticker).info

    # ── Compute derived fields ───────────────────────────────────────────
    fcf        = info.get("freeCashflow")
    mktcap     = info.get("marketCap")
    fcf_yield  = (fcf / mktcap) if fcf and mktcap else None
    p_fcf      = (mktcap / fcf) if fcf and mktcap and fcf > 0 else None

    result = {
        "pe_ttm":             info.get("trailingPE"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "operating_margin":   info.get("operatingMargins"),
        "net_margin":         info.get("profitMargins"),
        "fcf_yield":          fcf_yield,
        "debt_to_equity":     info.get("debtToEquity"),
        "current_ratio":      info.get("currentRatio"),
        "roe":                info.get("returnOnEquity"),
        "roic":               info.get("returnOnAssets"),   # proxy
        "sector":             info.get("sector"),
        "industry":           info.get("industry"),
        "marketCap":          mktcap,
        "p_fcf":              p_fcf,
        "as_of_date":         date.today().isoformat(),
    }
    logger.info("fetch_fundamentals: %s → %d keys populated",
                ticker, sum(1 for v in result.values() if v is not None))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Macro Data — FRED with 6-hour in-memory cache
# ──────────────────────────────────────────────────────────────────────────────

_FRED_CACHE: dict[str, tuple[datetime, pd.Series]] = {}
_FRED_CACHE_LOCK = Lock()
_FRED_CACHE_TTL = timedelta(hours=6)


@_with_retry
def fetch_fred_series(series_id: str, start: str = "2018-01-01") -> pd.Series:
    """
    Fetch a FRED time series by *series_id* via fredapi.

    Results are cached in memory for 6 hours to avoid redundant API calls
    within a single trading day.

    Parameters
    ----------
    series_id:
        FRED series identifier, e.g. ``"FEDFUNDS"``, ``"T10Y2Y"``.
    start:
        ISO-8601 start date string (default ``"2018-01-01"``).

    Returns
    -------
    pd.Series
        Values with a ``DatetimeIndex``, sorted ascending.

    Raises
    ------
    DataFetchError
        If ``FRED_API_KEY`` is not configured, or the API call fails.
    """
    if not settings.fred_api_key:
        raise DataFetchError(
            "FRED_API_KEY is not set — configure it in .env to use macro data."
        )

    cache_key = f"{series_id}::{start}"
    with _FRED_CACHE_LOCK:
        if cache_key in _FRED_CACHE:
            cached_at, series = _FRED_CACHE[cache_key]
            if datetime.now(timezone.utc).replace(tzinfo=None) - cached_at < _FRED_CACHE_TTL:
                logger.debug("fetch_fred_series: cache hit for %s", series_id)
                return series

    logger.debug("fetch_fred_series: fetching %s from FRED", series_id)
    from fredapi import Fred  # lazy import — avoids hard dep if FRED unused

    fred = Fred(api_key=settings.fred_api_key)
    raw: pd.Series = fred.get_series(series_id, observation_start=start)
    if raw.empty:
        raise DataFetchError(f"FRED returned empty series for {series_id!r}")

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().dropna()

    with _FRED_CACHE_LOCK:
        _FRED_CACHE[cache_key] = (datetime.now(timezone.utc).replace(tzinfo=None), raw)

    logger.info("fetch_fred_series: %s → %d observations", series_id, len(raw))
    return raw


def fetch_macro_bundle() -> dict:
    """
    Fetch the core macro data bundle used by the Macro agent.

    Fetches the following FRED series and derives the latest scalar value
    for each.  Also fetches the current VIX level via yfinance.

    Series fetched
    --------------
    ``fed_funds``          FEDFUNDS — effective federal funds rate
    ``cpi_yoy``            CPIAUCSL — CPI YoY % change (12-month pct_change × 100)
    ``unemployment``       UNRATE — unemployment rate
    ``t10y2y``             T10Y2Y — 10Y minus 2Y Treasury spread (bps)
    ``t10yie``             T10YIE — 10-Year breakeven inflation rate
    ``consumer_sentiment`` UMCSENT — University of Michigan consumer sentiment
    ``vix``                ^VIX via yfinance

    Returns
    -------
    dict
        Keys as listed above; values are the most recent non-NaN float.
        On individual series failure, the value is ``None`` and a warning
        is logged.
    """
    bundle: dict[str, float | None] = {}

    _FRED_MAP = {
        "fed_funds":          "FEDFUNDS",
        "unemployment":       "UNRATE",
        "t10y2y":             "T10Y2Y",
        "t10yie":             "T10YIE",
        "consumer_sentiment": "UMCSENT",
    }

    for key, series_id in _FRED_MAP.items():
        try:
            s = fetch_fred_series(series_id)
            bundle[key] = float(s.dropna().iloc[-1])
        except Exception as exc:
            logger.warning("fetch_macro_bundle: %s (%s) failed — %s", key, series_id, exc)
            bundle[key] = None

    # CPI YoY requires 12-month pct_change
    try:
        cpi_raw = fetch_fred_series("CPIAUCSL")
        cpi_yoy = cpi_raw.pct_change(12) * 100
        bundle["cpi_yoy"] = float(cpi_yoy.dropna().iloc[-1])
    except Exception as exc:
        logger.warning("fetch_macro_bundle: cpi_yoy (CPIAUCSL) failed — %s", exc)
        bundle["cpi_yoy"] = None

    # VIX via yfinance
    try:
        bundle["vix"] = fetch_vix()
    except Exception as exc:
        logger.warning("fetch_macro_bundle: vix fetch failed — %s", exc)
        bundle["vix"] = None

    logger.info(
        "fetch_macro_bundle: populated %d/%d fields",
        sum(1 for v in bundle.values() if v is not None),
        len(bundle),
    )
    return bundle


# ──────────────────────────────────────────────────────────────────────────────
# News & Sentiment
# ──────────────────────────────────────────────────────────────────────────────


def fetch_news(
    ticker: str,
    company_name: str,
    days_back: int = 7,
) -> list[dict]:
    """
    Fetch recent news articles for *ticker* via the NewsAPI.

    Queries ``"{ticker} OR {company_name}"`` across all English sources for
    the past *days_back* days.

    Parameters
    ----------
    ticker:
        Equity symbol used in the query (e.g. ``"AAPL"``).
    company_name:
        Full company name used to enrich the query (e.g. ``"Apple"``).
    days_back:
        Number of calendar days to look back (max 30 for free NewsAPI tier).

    Returns
    -------
    list[dict]
        Each item has keys: ``title``, ``description``, ``published_at``,
        ``source``.  Returns an empty list (never raises) if the API key is
        missing or the rate limit is reached.
    """
    if not settings.newsapi_key:
        logger.debug("fetch_news: NEWSAPI_KEY not set — returning empty list")
        return []

    try:
        from newsapi import NewsApiClient  # lazy import

        client = NewsApiClient(api_key=settings.newsapi_key)
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = f"{ticker} OR {company_name}"

        response = client.get_everything(
            q=query,
            language="en",
            from_param=from_date,
            sort_by="relevancy",
            page_size=50,
        )

        articles = response.get("articles", [])
        result = [
            {
                "title":        a.get("title", ""),
                "description":  a.get("description", ""),
                "published_at": a.get("publishedAt", ""),
                "source":       a.get("source", {}).get("name", ""),
            }
            for a in articles
            if a.get("title")
        ]
        logger.info("fetch_news: %s → %d articles", ticker, len(result))
        return result

    except Exception as exc:
        logger.warning("fetch_news: %s failed — %s: %s", ticker, type(exc).__name__, exc)
        return []


def fetch_social_sentiment(ticker: str) -> dict:
    """
    Fetches social sentiment from StockTwits (primary) and
    Google Trends via pytrends (secondary). No API key required
    for either source. Returns the same dict shape as the
    former social sentiment fetcher so no other code needs updating.
    """
    stocktwits = _fetch_stocktwits(ticker)
    trends     = _fetch_google_trends(ticker)

    mention_count = stocktwits.get("mention_count", 0)
    bull_pct      = stocktwits.get("bull_pct", 0.5)
    bear_pct      = stocktwits.get("bear_pct", 0.5)
    trend_score   = trends.get("trend_score", 50)
    trend_surge   = trends.get("surge", False)

    # Combine StockTwits volume surge with Google Trends surge
    mention_surge = stocktwits.get("mention_surge", False) or trend_surge

    # Volume change pct: use trend score deviation from 50 as proxy
    volume_change_pct = (trend_score - 50) / 50.0

    return {
        "mention_count_7d":   mention_count,
        "mention_surge":      mention_surge,
        "volume_change_pct": round(volume_change_pct, 3),
        "avg_score":          round(bull_pct - bear_pct, 3),  # net sentiment [-1, +1]
        "top_posts":          stocktwits.get("top_messages", []),
        "earnings_within_14d": False,  # Populated separately by _check_earnings_calendar
    }


def _fetch_stocktwits(ticker: str) -> dict:
    """
    Calls the StockTwits public API. No key required.
    Returns message count, bull/bear split, and top message snippets.
    Falls back to empty dict on any error.
    """
    # Temporarily paused pending authentication from Stocktwits team
    logger.info(f"[StockTwits] Fetch disabled for {ticker} pending authentication.")
    return {}

    import requests
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data     = response.json()
        messages = data.get("messages", [])

        bulls = sum(
            1 for m in messages
            if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish"
        )
        bears = sum(
            1 for m in messages
            if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish"
        )
        total = bulls + bears or 1

        top_messages = [
            m.get("body", "")[:120]
            for m in messages[:5]
            if m.get("body")
        ]

        return {
            "mention_count": len(messages),
            "bull_pct":      bulls / total,
            "bear_pct":      bears / total,
            "mention_surge": len(messages) > 20,
            "top_messages":  top_messages,
        }

    except Exception as e:
        logger.warning(f"[StockTwits] Fetch failed for {ticker}: {e}")
        return {}


def _fetch_google_trends(ticker: str) -> dict:
    """
    Fetches 7-day Google search interest for the ticker via pytrends.
    A score > 1.5x the 7-day average is flagged as a surge.
    Falls back gracefully on any error.
    """
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=360, timeout=(8, 20))
        pt.build_payload([ticker], timeframe="now 7-d")
        df = pt.interest_over_time()

        if df.empty or ticker not in df.columns:
            return {"trend_score": 50, "surge": False}

        latest = int(df[ticker].iloc[-1])
        avg    = df[ticker].mean()
        surge  = avg > 0 and latest > avg * 1.5

        return {"trend_score": latest, "surge": bool(surge)}

    except Exception as e:
        logger.warning(f"[GoogleTrends] Fetch failed for {ticker}: {e}")
        return {"trend_score": 50, "surge": False}


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


@_with_retry
def fetch_vix() -> float:
    """
    Fetch the current CBOE VIX level via yfinance.

    Returns
    -------
    float
        Most recent VIX closing price.

    Raises
    ------
    DataFetchError
        If the yfinance download is empty after all retries.
    """
    logger.debug("fetch_vix: downloading ^VIX")
    raw = yf.download("^VIX", period="5d", interval="1d", progress=False, threads=False)
    if raw.empty:
        raise DataFetchError("yfinance returned empty DataFrame for ^VIX")

    # Handle MultiIndex columns produced by newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    col = "Close" if "Close" in raw.columns else raw.columns[0]
    vix = float(raw[col].dropna().iloc[-1])
    logger.info("fetch_vix: current VIX = %.2f", vix)
    return vix
