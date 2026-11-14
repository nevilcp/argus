"""Retried market, fundamentals, news, and economic data access."""

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


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame schemas to lower-case OHLCV columns."""
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
    """Download one ticker's candles from Yahoo Finance as a normalized, sorted frame."""
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
            f"yfinance returned empty DataFrame for ticker={ticker!r} interval={interval!r} period={period!r}"
        )
    df = _normalize_ohlcv(raw)
    df.sort_index(inplace=True)
    logger.info("%s [%s] -> %d rows", ticker, interval, len(df))
    return df


@with_retry
def fetch_ohlcv_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical daily OHLCV candlestick data from Yahoo Finance."""
    return _download_ohlcv(ticker, period=period, interval="1d")


@with_retry
def fetch_ohlcv_intraday(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Fetch intraday OHLCV candlestick intervals from Yahoo Finance."""
    return _download_ohlcv(ticker, period=period, interval=interval)


_MULTI_DAILY_MAX_WORKERS = 5

# Create the daily cache lazily under the current ARGUS_DATA_DIR so deployment overrides survive restarts.
_DAILY_BAR_CACHE: DailyBarCache | None = None


def _daily_bar_cache() -> DailyBarCache:
    """Return the module-level DailyBarCache, constructing it under ARGUS_DATA_DIR on first use."""
    global _DAILY_BAR_CACHE
    if _DAILY_BAR_CACHE is None:
        db_file = settings.data_path("daily_bars_cache.db")
        db_file.parent.mkdir(parents=True, exist_ok=True)
        _DAILY_BAR_CACHE = DailyBarCache(db_path=str(db_file))
    return _DAILY_BAR_CACHE


def fetch_multiple_daily(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch daily candlestick histories in parallel across a list of tickers."""
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
            future_to_ticker = {pool.submit(fetch_ohlcv_daily, ticker, period): ticker for ticker in to_fetch}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    results[ticker] = df
                    cache.put(ticker, df)
                except Exception as exc:
                    logger.warning(
                        "failed for %s - %s: %s",
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
    """Retrieve fundamental balance sheet and income statements from Yahoo Finance."""
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
        "%s -> %d keys populated",
        ticker,
        sum(1 for v in result.values() if v is not None),
    )
    return result


@with_retry
def fetch_ticker_info(ticker: str) -> dict:
    """Fetch yfinance's raw `.info` dict for a ticker."""
    logger.debug("ticker=%s", ticker)
    return yf.Ticker(ticker).info


@with_retry
def fetch_ticker_calendar(ticker: str) -> Any:
    """Fetch yfinance's raw `.calendar` object for a ticker."""
    logger.debug("ticker=%s", ticker)
    return yf.Ticker(ticker).calendar


_FRED_CACHE: TTLCache[str, pd.Series] = TTLCache(ttl=timedelta(hours=6), max_entries=SYSTEM.fred_cache_max_entries)


@with_retry
def fetch_fred_series(series_id: str, start: str = MACRO.fred_default_start) -> pd.Series:
    """Fetch a macroeconomic timeline from the Federal Reserve Economic Database (FRED)."""
    if not settings.FRED_API_KEY:
        raise DataFetchError("FRED_API_KEY is not set — configure it in .env to use macro data.")

    cache_key = f"{series_id}::{start}"
    cached = _FRED_CACHE.get(cache_key)
    if cached is not None:
        logger.debug("cache hit for %s", series_id)
        return cached

    logger.debug("fetching %s from FRED", series_id)
    from fredapi import Fred

    fred = Fred(api_key=settings.FRED_API_KEY)
    raw: pd.Series = fred.get_series(series_id, observation_start=start)
    if raw.empty:
        raise DataFetchError(f"FRED returned empty series for {series_id!r}")

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().dropna()

    _FRED_CACHE.set(cache_key, raw)

    logger.info("%s -> %d observations", series_id, len(raw))
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
    """Read the most recent observation of a FRED series, degrading to None on failure."""
    try:
        series = fetch_fred_series(series_id)
        if transform is not None:
            series = transform(series)
        return float(series.dropna().iloc[-1])
    except Exception as exc:
        logger.warning("%s (%s) failed - %s", key, series_id, exc)
        return None


def fetch_macro_bundle() -> dict:
    """Gather a consolidated set of macroeconomic indicators and interest rates."""
    bundle: dict[str, float | None] = {
        key: _latest_fred_value(key, series_id) for key, series_id in _MACRO_BUNDLE_SERIES.items()
    }
    bundle["cpi_yoy"] = _latest_fred_value("cpi_yoy", CPI_SERIES, cpi_yoy)

    try:
        bundle["vix"] = fetch_vix()
    except Exception as exc:
        logger.warning("vix fetch failed - %s", exc)
        bundle["vix"] = None

    logger.info(
        "populated %d/%d fields",
        sum(1 for v in bundle.values() if v is not None),
        len(bundle),
    )
    return bundle


# Request the maximum article page so each scarce daily reservation returns as much evidence as possible.
_NEWSAPI_PAGE_SIZE = 100

_NEWSAPI_BUDGET = NewsApiBudget()


@with_retry
def _fetch_news_page(client: Any, query: str, from_date: str) -> list[dict]:
    """Fetch and normalizes one page of NewsAPI articles."""
    if not _NEWSAPI_BUDGET.try_reserve():
        # A rejected reservation may mean exhaustion or disk failure; try_reserve logs persistence failures itself.
        raise DataFetchError(
            f"no NewsAPI budget slot available (daily cap of {NEWSAPI_DAILY_LIMIT} reached, or the reservation failed)"
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
    """Retrieve recent news articles matching a ticker from NewsAPI."""
    if not settings.NEWSAPI_KEY:
        logger.debug("NEWSAPI_KEY not set")
        return None

    try:
        from newsapi import NewsApiClient

        client = NewsApiClient(api_key=settings.NEWSAPI_KEY)
        from_date = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = f"{ticker} OR {company_name}"

        result = _fetch_news_page(client, query, from_date)
        logger.info("%s -> %d articles", ticker, len(result))
        return result

    except DataFetchError as exc:
        logger.warning("%s failed - %s", ticker, exc)
        return None


def fetch_vix() -> float:
    """Fetch the current CBOE Volatility Index (VIX) level."""
    vix = float(fetch_ohlcv_daily("^VIX", period="5d")["close"].iloc[-1])
    logger.info("current VIX = %.2f", vix)
    return vix
