"""
argus/data/fetchers.py

Centralized data access layer for fetching pricing, fundamentals, and macro series.

Responsibilities:
  - Provide retried access to Yahoo Finance OHLCV and fundamentals
  - Fetch and cache macroeconomic FRED series with a 6-hour TTL
  - Aggregate news headlines and social sentiment from free-tier APIs

Not responsible for:
  - Data compression or indicator calculation (see data/pipeline.py)
  - SQLite buffering (see data/cache.py)
  - Rate-limit governance (see orchestration/governor.py)

Dependencies:
  - yfinance
  - fredapi (optional; requires FRED_API_KEY)
  - newsapi-python (optional; requires NEWSAPI_KEY)
  - pytrends (optional)
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

logger = logging.getLogger("argus.fetchers")


class DataFetchError(Exception):
    """Raised when a data fetch fails after all retry attempts."""


F = TypeVar("F", bound=Callable[..., Any])

_RETRY_ATTEMPTS = 3
# Doubles on each consecutive attempt: 1.5s, 3.0s, 6.0s
_RETRY_BASE_DELAY = 1.5


def _with_retry(fn: F) -> F:
    """Decorator implementing exponential back-off retries on network failures.

    Wraps transient exceptions in DataFetchError after exhausting all attempts.
    Re-raises DataFetchError immediately without re-wrapping to avoid stacking.

    Args:
        fn: The function to wrap.

    Returns:
        Wrapped function with retry semantics.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return fn(*args, **kwargs)
            except DataFetchError:
                raise
            except Exception as exc:
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "%s attempt %d/%d failed (%s: %s); retrying in %.1fs",
                    fn.__name__,
                    attempt,
                    _RETRY_ATTEMPTS,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(delay)
        raise DataFetchError(
            f"{fn.__name__} failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    return wrapper  # type: ignore[return-value]


_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes DataFrame schemas to lower-case OHLCV columns.

    yfinance ≥ 0.2.40 returns MultiIndex columns; this function flattens them
    before column name normalization to ensure downstream consumers always
    receive a simple single-level column schema.

    Args:
        df: Raw DataFrame from yfinance.download().

    Returns:
        Filtered, sorted DataFrame with lowercase OHLCV columns and datetime index.

    Raises:
        DataFetchError: If no recognized OHLCV columns are present after normalization.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    df.columns = [str(c).lower() for c in df.columns]

    available = [c for c in _OHLCV_COLUMNS if c in df.columns]
    if not available:
        raise DataFetchError(f"DataFrame has no OHLCV columns; found: {list(df.columns)}")
    df = df[available].copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(subset=["close"], inplace=True)
    return df


@_with_retry
def fetch_ohlcv_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetches historical daily OHLCV candlestick data from Yahoo Finance.

    Args:
        ticker: Equity ticker symbol (e.g. 'AAPL').
        period: yfinance period string (e.g. '2y', '1y').

    Returns:
        Sorted DataFrame with lowercase OHLCV columns and a datetime index.

    Raises:
        DataFetchError: If the returned DataFrame is empty after all retries.
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
def fetch_ohlcv_intraday(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Fetches intraday OHLCV candlestick intervals from Yahoo Finance.

    Args:
        ticker: Equity ticker symbol.
        interval: yfinance interval string (e.g. '5m', '15m', '1h').
        period: yfinance period string (e.g. '5d', '2d').

    Returns:
        Sorted DataFrame with lowercase OHLCV columns.

    Raises:
        DataFetchError: If the returned DataFrame is empty.
    """
    logger.debug(
        "fetch_ohlcv_intraday: ticker=%s interval=%s period=%s",
        ticker,
        interval,
        period,
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
    logger.info("fetch_ohlcv_intraday: %s [%s] → %d rows", ticker, interval, len(df))
    return df


def fetch_multiple_daily(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetches daily candlestick histories in parallel across a list of tickers.

    Args:
        tickers: List of equity ticker symbols.
        period: yfinance period string applied to all tickers.

    Returns:
        Mapping of ticker → DataFrame. Failed tickers are omitted with a warning.
    """
    results: dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        future_to_ticker = {
            pool.submit(fetch_ohlcv_daily, ticker, period): ticker for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                logger.warning(
                    "fetch_multiple_daily: failed for %s — %s: %s",
                    ticker,
                    type(exc).__name__,
                    exc,
                )

    logger.info(
        "fetch_multiple_daily: %d/%d tickers fetched successfully",
        len(results),
        len(tickers),
    )
    return results


@_with_retry
def fetch_fundamentals(ticker: str) -> dict:
    """Retrieves fundamental balance sheet and income statements from Yahoo Finance.

    Note: ``returnOnAssets`` is used as a ROIC proxy because yfinance does not
    expose an invested-capital figure. This understates ROIC for capital-light
    companies but provides a consistent cross-ticker signal.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Dict of fundamental metrics with keys: pe_ttm, revenue_growth_yoy,
        operating_margin, net_margin, fcf_yield, debt_to_equity, current_ratio,
        roe, roic, sector, industry, marketCap, p_fcf, as_of_date.
    """
    logger.debug("fetch_fundamentals: ticker=%s", ticker)
    info: dict = yf.Ticker(ticker).info

    fcf = info.get("freeCashflow")
    mktcap = info.get("marketCap")
    fcf_yield = (fcf / mktcap) if fcf and mktcap else None
    p_fcf = (mktcap / fcf) if fcf and mktcap and fcf > 0 else None

    # yfinance returns debtToEquity as a percentage (e.g. 79.5 for 79.5%); normalize to decimal
    raw_de = info.get("debtToEquity")
    debt_to_equity = round(raw_de / 100.0, 4) if raw_de is not None else None

    result = {
        "pe_ttm": info.get("trailingPE"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin": info.get("profitMargins"),
        "fcf_yield": fcf_yield,
        "debt_to_equity": debt_to_equity,
        "current_ratio": info.get("currentRatio"),
        "roe": info.get("returnOnEquity"),
        "roic": info.get("returnOnAssets"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": mktcap,
        "p_fcf": p_fcf,
        "as_of_date": date.today().isoformat(),
    }
    logger.info(
        "fetch_fundamentals: %s → %d keys populated",
        ticker,
        sum(1 for v in result.values() if v is not None),
    )
    return result


_FRED_CACHE: dict[str, tuple[datetime, pd.Series]] = {}
_FRED_CACHE_LOCK = Lock()
_FRED_CACHE_TTL = timedelta(hours=6)


@_with_retry
def fetch_fred_series(series_id: str, start: str = "2018-01-01") -> pd.Series:
    """Fetches a macroeconomic timeline from the Federal Reserve Economic Database (FRED).

    Responses are cached in-process for 6 hours to avoid redundant FRED API calls
    across multiple macro analysis cycles within a single trading session.

    Args:
        series_id: FRED series identifier (e.g. 'FEDFUNDS', 'T10Y2Y').
        start: ISO date string for the beginning of the requested history.

    Returns:
        Sorted, NaN-dropped pd.Series with a DatetimeIndex.

    Raises:
        DataFetchError: If FRED_API_KEY is not configured or the series is empty.
    """
    if not settings.fred_api_key:
        raise DataFetchError("FRED_API_KEY is not set — configure it in .env to use macro data.")

    cache_key = f"{series_id}::{start}"
    with _FRED_CACHE_LOCK:
        if cache_key in _FRED_CACHE:
            cached_at, series = _FRED_CACHE[cache_key]
            if datetime.now(timezone.utc).replace(tzinfo=None) - cached_at < _FRED_CACHE_TTL:
                logger.debug("fetch_fred_series: cache hit for %s", series_id)
                return series

    logger.debug("fetch_fred_series: fetching %s from FRED", series_id)
    from fredapi import Fred  # Lazy import; only loaded when FRED is actually used

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
    """Gathers a consolidated set of macroeconomic indicators and interest rates.

    Returns:
        Dict with keys: fed_funds, unemployment, t10y2y, t10yie, consumer_sentiment,
        cpi_yoy, vix. Any key that fails to fetch is set to None.
    """
    bundle: dict[str, float | None] = {}

    _FRED_MAP = {
        "fed_funds": "FEDFUNDS",
        "unemployment": "UNRATE",
        "t10y2y": "T10Y2Y",
        "t10yie": "T10YIE",
        "consumer_sentiment": "UMCSENT",
    }

    for key, series_id in _FRED_MAP.items():
        try:
            s = fetch_fred_series(series_id)
            bundle[key] = float(s.dropna().iloc[-1])
        except Exception as exc:
            logger.warning("fetch_macro_bundle: %s (%s) failed — %s", key, series_id, exc)
            bundle[key] = None

    # CPI YoY requires a 12-month pct_change transform on the raw CPI index
    try:
        cpi_raw = fetch_fred_series("CPIAUCSL")
        cpi_yoy = cpi_raw.pct_change(12) * 100
        bundle["cpi_yoy"] = float(cpi_yoy.dropna().iloc[-1])
    except Exception as exc:
        logger.warning("fetch_macro_bundle: cpi_yoy (CPIAUCSL) failed — %s", exc)
        bundle["cpi_yoy"] = None

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


def fetch_news(
    ticker: str,
    company_name: str,
    days_back: int = 7,
) -> list[dict]:
    """Retrieves recent news articles matching a ticker from NewsAPI.

    Returns an empty list without error when NEWSAPI_KEY is not configured,
    so callers operate in a degraded but non-failing state.

    Args:
        ticker: Equity ticker symbol used in the search query.
        company_name: Human-readable company name appended to the query.
        days_back: Number of days of news history to retrieve (default 7).

    Returns:
        List of dicts with keys: title, description, published_at, source.
    """
    if not settings.newsapi_key:
        logger.debug("fetch_news: NEWSAPI_KEY not set — returning empty list")
        return []

    try:
        from newsapi import NewsApiClient  # Lazy import; only loaded when NewsAPI is used

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
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "published_at": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", ""),
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
    """Aggregates Google Trends signals into a normalized sentiment dict.

    Returns the same dict shape regardless of whether trends data is available
    to prevent downstream callers from needing to guard against missing keys.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Dict with keys: mention_count_7d, mention_surge, volume_change_pct,
        avg_score, top_posts, earnings_within_14d.
    """
    trends = _fetch_google_trends(ticker)
    trend_score = trends.get("trend_score", 50)
    trend_surge = trends.get("surge", False)

    # Maps the 0–100 Google Trends score to a ±1 deviation from baseline (50 = neutral)
    volume_change_pct = (trend_score - 50) / 50.0

    return {
        "mention_count_7d": 0,
        "mention_surge": trend_surge,
        "volume_change_pct": round(volume_change_pct, 3),
        "avg_score": 0.0,
        "top_posts": [],
        "earnings_within_14d": False,
    }



def _fetch_google_trends(ticker: str) -> dict:
    """Fetches 7-day Google search interest for the ticker via pytrends.

    A score > 1.5× the 7-day average is flagged as a surge. Falls back gracefully
    on any error, returning a neutral (50) score with no surge.

    Args:
        ticker: Equity ticker symbol used as the Google Trends keyword.

    Returns:
        Dict with keys: trend_score (int 0–100), surge (bool).
    """
    try:
        from pytrends.request import TrendReq

        pt = TrendReq(hl="en-US", tz=360, timeout=(8, 20))
        pt.build_payload([ticker], timeframe="now 7-d")
        df = pt.interest_over_time()

        if df.empty or ticker not in df.columns:
            return {"trend_score": 50, "surge": False}

        latest = int(df[ticker].iloc[-1])
        avg = df[ticker].mean()
        surge = avg > 0 and latest > avg * 1.5

        return {"trend_score": latest, "surge": bool(surge)}

    except Exception as e:
        logger.warning(f"[GoogleTrends] Fetch failed for {ticker}: {e}")
        return {"trend_score": 50, "surge": False}


@_with_retry
def fetch_vix() -> float:
    """Fetches the current CBOE Volatility Index (VIX) level.

    Returns:
        Most recent VIX close price as a float.

    Raises:
        DataFetchError: If yfinance returns an empty DataFrame for ^VIX.
    """
    logger.debug("fetch_vix: downloading ^VIX")
    raw = yf.download("^VIX", period="5d", interval="1d", progress=False, threads=False)
    if raw.empty:
        raise DataFetchError("yfinance returned empty DataFrame for ^VIX")

    # yfinance MultiIndex schema varies across minor versions; flatten for safety
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    col = "Close" if "Close" in raw.columns else raw.columns[0]
    vix = float(raw[col].dropna().iloc[-1])
    logger.info("fetch_vix: current VIX = %.2f", vix)
    return vix
