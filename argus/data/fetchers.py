"""
argus/data/fetchers.py

Centralized data access layer for fetching pricing, fundamentals, and macro series.

Responsibilities:
  - Provide retried access to Yahoo Finance OHLCV and fundamentals
  - Fetch and cache macroeconomic FRED series with a 6-hour TTL
  - Cache daily OHLCV bars per (ticker, trading day) so a sweep only pays for
    the first fetch of each trading day
  - Aggregate news headlines from free-tier APIs, returning None — never a
    fabricated empty result — when a source can't be reached

Not responsible for:
  - Retry classification and back-off (see data/retry.py)
  - NewsAPI's 100 request/day budget (see data/newsapi_budget.py)
  - Data compression or indicator calculation (see data/indicators.py)
  - Rolling intraday SQLite buffering (see data/cache.py's OHLCVBuffer)
  - Rate-limit governance for Groq LLM calls (see argus/governor.py)

Dependencies:
  - yfinance
  - fredapi (optional; requires FRED_API_KEY)
  - newsapi-python (optional; requires NEWSAPI_KEY)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from argus.config import settings
from argus.data.cache import DailyBarCache, TTLCache
from argus.data.macro_features import (
    CONSUMER_SENTIMENT_SERIES,
    CPI_SERIES,
    FED_FUNDS_SERIES,
    T10Y2Y_SERIES,
    T10YIE_SERIES,
    UNEMPLOYMENT_SERIES,
    cpi_yoy,
)
from argus.data.newsapi_budget import NEWSAPI_DAILY_LIMIT, NewsApiBudget
from argus.data.retry import DataFetchError, with_retry
from argus.params import MACRO, SYSTEM

logger = logging.getLogger(__name__)


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
        DataFetchError: If no close column is present after normalization. Every
            consumer reads ``close``, and guessing another column in its place
            would hand back a number that is not a price.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    df.columns = [str(c).lower() for c in df.columns]

    if "close" not in df.columns:
        raise DataFetchError(f"DataFrame has no close column; found: {list(df.columns)}")
    available = [c for c in _OHLCV_COLUMNS if c in df.columns]
    df = df[available].copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(subset=["close"], inplace=True)
    return df


def _download_ohlcv(ticker: str, *, period: str, interval: str) -> pd.DataFrame:
    """Downloads one ticker's candles from Yahoo Finance as a normalized, sorted frame.

    The one place the daily and intraday fetchers call ``yf.download``. Not
    retried itself; the public fetchers wrap it.

    Args:
        ticker: Equity ticker symbol.
        period: yfinance period string (e.g. '2y', '5d').
        interval: yfinance interval string (e.g. '1d', '5m').

    Returns:
        Sorted DataFrame with lowercase OHLCV columns and a datetime index.

    Raises:
        DataFetchError: If yfinance returns nothing, or no usable OHLCV columns.
    """
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
            f"yfinance returned empty DataFrame for ticker={ticker!r} "
            f"interval={interval!r} period={period!r}"
        )
    df = _normalise_ohlcv(raw)
    df.sort_index(inplace=True)
    logger.info("%s [%s] → %d rows", ticker, interval, len(df))
    return df


@with_retry
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
    return _download_ohlcv(ticker, period=period, interval="1d")


@with_retry
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
    return _download_ohlcv(ticker, period=period, interval=interval)


_MULTI_DAILY_MAX_WORKERS = 5

# Daily bars change once per trading day; caching them turns every run after
# the first of the day into 0 requests for this node instead of one per ticker.
# Constructed lazily (not at import time) so it lands under whatever
# settings.ARGUS_DATA_DIR is current when first used, not whatever it was at
# import time — docker-compose mounts only data/ and chroma_db/, so the old
# cwd-relative default silently landed on the ephemeral container layer and
# was destroyed on every restart.
_DAILY_BAR_CACHE: DailyBarCache | None = None


def _daily_bar_cache() -> DailyBarCache:
    """Returns the module-level DailyBarCache, constructing it under ARGUS_DATA_DIR on first use.

    Returns:
        The shared DailyBarCache instance.
    """
    global _DAILY_BAR_CACHE
    if _DAILY_BAR_CACHE is None:
        db_file = settings.data_path("daily_bars_cache.db")
        db_file.parent.mkdir(parents=True, exist_ok=True)
        _DAILY_BAR_CACHE = DailyBarCache(db_path=str(db_file))
    return _DAILY_BAR_CACHE


def fetch_multiple_daily(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetches daily candlestick histories in parallel across a list of tickers.

    Tickers already refreshed today (UTC) are served from the on-disk
    DailyBarCache instead of hitting the network — daily bars only change once
    per trading day, so a same-day re-run costs 0 requests per cached ticker.

    Args:
        tickers: List of equity ticker symbols.
        period: yfinance period string applied to all tickers.

    Returns:
        Mapping of ticker → DataFrame. Failed tickers are omitted with a warning.
    """
    cache = _daily_bar_cache()
    cache.prune_untracked(set(tickers))
    results: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        cached = cache.get(ticker)
        if cached is not None:
            results[ticker] = cached
        else:
            to_fetch.append(ticker)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=_MULTI_DAILY_MAX_WORKERS) as pool:
            future_to_ticker = {
                pool.submit(fetch_ohlcv_daily, ticker, period): ticker for ticker in to_fetch
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    results[ticker] = df
                    cache.put(ticker, df)
                except Exception as exc:
                    logger.warning(
                        "failed for %s — %s: %s",
                        ticker,
                        type(exc).__name__,
                        exc,
                    )

    logger.info(
        "%d/%d tickers fetched successfully (%d from cache)",
        len(results),
        len(tickers),
        len(tickers) - len(to_fetch),
    )
    return results


def fetch_fundamentals(ticker: str) -> dict:
    """Retrieves fundamental balance sheet and income statements from Yahoo Finance.

    Not decorated with ``with_retry`` itself: ``fetch_ticker_info`` already
    retries the one network call, and a second layer would multiply the
    attempts.

    Note: ``returnOnAssets`` is used as a ROIC proxy because yfinance does not
    expose an invested-capital figure. This understates ROIC for capital-light
    companies but provides a consistent cross-ticker signal.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Dict of fundamental metrics with keys: pe_ttm, revenue_growth_yoy,
        operating_margin, net_margin, fcf_yield, debt_to_equity, current_ratio,
        roe, roic, sector, industry, marketCap, p_fcf, as_of_date.

    Raises:
        DataFetchError: If the info lookup fails after all retries.
    """
    info = fetch_ticker_info(ticker)

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
        "as_of_date": date.today().isoformat(),  # noqa: DTZ011
    }
    logger.info(
        "%s → %d keys populated",
        ticker,
        sum(1 for v in result.values() if v is not None),
    )
    return result


@with_retry
def fetch_ticker_info(ticker: str) -> dict:
    """Fetches yfinance's raw ``.info`` dict for a ticker.

    Shared retried entry point for callers that only need a slice of `.info`
    (e.g. `sector`) — previously agents/risk.py called `yf.Ticker(ticker).info`
    directly, bypassing retry/back-off entirely.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Raw info dict as returned by yfinance.Ticker(ticker).info.
    """
    logger.debug("ticker=%s", ticker)
    return yf.Ticker(ticker).info


@with_retry
def fetch_ticker_calendar(ticker: str) -> Any:
    """Fetches yfinance's raw ``.calendar`` object for a ticker.

    Shared retried entry point for earnings-date lookups — previously
    agents/sentiment.py called `yf.Ticker(ticker).calendar` directly,
    bypassing retry/back-off entirely.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        Raw calendar object from yfinance.Ticker(ticker).calendar. Shape
        (DataFrame or dict) varies across yfinance versions; callers must
        handle both.
    """
    logger.debug("ticker=%s", ticker)
    return yf.Ticker(ticker).calendar


_FRED_CACHE: TTLCache[str, pd.Series] = TTLCache(
    ttl=timedelta(hours=6), max_entries=SYSTEM.fred_cache_max_entries
)


@with_retry
def fetch_fred_series(series_id: str, start: str = MACRO.fred_default_start) -> pd.Series:
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
    cached = _FRED_CACHE.get(cache_key)
    if cached is not None:
        logger.debug("cache hit for %s", series_id)
        return cached

    logger.debug("fetching %s from FRED", series_id)
    from fredapi import Fred  # Lazy import; only loaded when FRED is actually used

    fred = Fred(api_key=settings.fred_api_key)
    raw: pd.Series = fred.get_series(series_id, observation_start=start)
    if raw.empty:
        raise DataFetchError(f"FRED returned empty series for {series_id!r}")

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().dropna()

    _FRED_CACHE.set(cache_key, raw)

    logger.info("%s → %d observations", series_id, len(raw))
    return raw


_MACRO_BUNDLE_SERIES = {
    "fed_funds": FED_FUNDS_SERIES,
    "unemployment": UNEMPLOYMENT_SERIES,
    "t10y2y": T10Y2Y_SERIES,
    "t10yie": T10YIE_SERIES,
    "consumer_sentiment": CONSUMER_SENTIMENT_SERIES,
}


def _latest_fred_value(
    key: str,
    series_id: str,
    transform: Callable[[pd.Series], pd.Series] | None = None,
) -> float | None:
    """Reads the most recent observation of a FRED series, degrading to None on failure.

    Args:
        key: Bundle field name the value lands under, used only for logging.
        series_id: FRED series identifier.
        transform: Applied to the raw series before the last observation is taken,
            for fields that are a derivative of the published index rather than it.

    Returns:
        The last non-NaN observation as a float, or None if the fetch failed.
    """
    try:
        series = fetch_fred_series(series_id)
        if transform is not None:
            series = transform(series)
        return float(series.dropna().iloc[-1])
    except Exception as exc:
        logger.warning("%s (%s) failed — %s", key, series_id, exc)
        return None


def fetch_macro_bundle() -> dict:
    """Gathers a consolidated set of macroeconomic indicators and interest rates.

    Returns:
        Dict with keys: fed_funds, unemployment, t10y2y, t10yie, consumer_sentiment,
        cpi_yoy, vix. Any key that fails to fetch is set to None.
    """
    bundle: dict[str, float | None] = {
        key: _latest_fred_value(key, series_id) for key, series_id in _MACRO_BUNDLE_SERIES.items()
    }
    bundle["cpi_yoy"] = _latest_fred_value("cpi_yoy", CPI_SERIES, cpi_yoy)

    try:
        bundle["vix"] = fetch_vix()
    except Exception as exc:
        logger.warning("vix fetch failed — %s", exc)
        bundle["vix"] = None

    logger.info(
        "populated %d/%d fields",
        sum(1 for v in bundle.values() if v is not None),
        len(bundle),
    )
    return bundle


# NewsAPI's maximum page size; larger than the default 20, so each of the
# scarce daily requests returns as many articles as the tier allows.
_NEWSAPI_PAGE_SIZE = 100

_NEWSAPI_BUDGET = NewsApiBudget()


@with_retry
def _fetch_news_page(client: Any, query: str, from_date: str) -> list[dict]:
    """Fetches and normalizes one page of NewsAPI articles.

    Args:
        client: An initialized NewsApiClient.
        query: NewsAPI query string (e.g. "AAPL OR Apple Inc").
        from_date: ISO date string for the start of the search window.

    Returns:
        List of dicts with keys: title, description, published_at, source.

    Raises:
        DataFetchError: If no budget slot is available. The reservation is made
            here rather than in the caller because every retry of this function is
            another real NewsAPI request.
    """
    if not _NEWSAPI_BUDGET.try_reserve():
        # Exhaustion and a disk error both come back as False; try_reserve already
        # logged the specific cause for a disk error, so this stays cause-agnostic.
        raise DataFetchError(
            f"no NewsAPI budget slot available (daily cap of {NEWSAPI_DAILY_LIMIT} reached, "
            "or the reservation failed)"
        )

    response = client.get_everything(
        q=query,
        language="en",
        from_param=from_date,
        sort_by="relevancy",
        page_size=_NEWSAPI_PAGE_SIZE,
    )
    articles = response.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "published_at": a.get("publishedAt", ""),
            "source": a.get("source", {}).get("name", ""),
        }
        for a in articles
        if a.get("title")
    ]


def fetch_news(
    ticker: str,
    company_name: str,
    days_back: int = 7,
) -> list[dict] | None:
    """Retrieves recent news articles matching a ticker from NewsAPI.

    NewsAPI's free Developer tier delays article availability by roughly 24
    hours and caps history at one month, so results are never same-day and a
    backtest window further back than that returns nothing.

    Args:
        ticker: Equity ticker symbol used in the search query.
        company_name: Human-readable company name appended to the query.
        days_back: Number of days of news history to retrieve (default 7).

    Returns:
        List of dicts with keys: title, description, published_at, source.
        None if NEWSAPI_KEY is not configured, the 100 request/day budget is
        exhausted, or the request failed after retries — distinguishable from
        an empty list, which means the request succeeded but found nothing.
    """
    if not settings.newsapi_key:
        logger.debug("NEWSAPI_KEY not set")
        return None

    try:
        from newsapi import NewsApiClient  # Lazy import; only loaded when NewsAPI is used

        client = NewsApiClient(api_key=settings.newsapi_key)
        from_date = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = f"{ticker} OR {company_name}"

        result = _fetch_news_page(client, query, from_date)
        logger.info("%s → %d articles", ticker, len(result))
        return result

    except DataFetchError as exc:
        logger.warning("%s failed — %s", ticker, exc)
        return None


def fetch_vix() -> float:
    """Fetches the current CBOE Volatility Index (VIX) level.

    Reads the same ``^VIX`` daily history the macro agent does, through the
    same retried download, so a missing ``Close`` column is an error here
    rather than a silent fallback to whichever column comes first.

    Returns:
        Most recent VIX close as a float.

    Raises:
        DataFetchError: If yfinance returns no ``^VIX`` history or no close column.
    """
    vix = float(fetch_ohlcv_daily("^VIX", period="5d")["close"].iloc[-1])
    logger.info("current VIX = %.2f", vix)
    return vix
