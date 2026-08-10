"""
argus/backtesting/pit_enforcer.py

Point-in-time (PiT) data constraint enforcer for backtest validity.

Responsibilities:
  - Prevent lookahead bias by restricting all data access to dates ≤ the simulation date
  - Provide pre-truncated price Series for use within the Backtrader strategy loop
  - Validate that indicator computations reference only historically available data

Not responsible for:
  - Fetching data from yfinance (see data/fetchers.py)
  - Running the simulation (see backtesting/engine.py)
  - Detecting post-hoc bias (see backtesting/bias_auditor.py)
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger("argus.pit_enforcer")

_PRICE_CACHE: dict[str, pd.Series] = {}


def _get_cached_series(ticker: str, lookback_years: int = 3) -> pd.Series:
    """Fetches and caches a full daily close price Series for a given ticker.

    Results are cached in a module-level dict to avoid redundant yfinance calls
    across bars within the same simulation run. Cache persists across Backtrader
    strategy instances within the same Python process.

    Args:
        ticker: Equity ticker symbol.
        lookback_years: Number of years of history to load (default 3).

    Returns:
        Daily close price Series with a DatetimeIndex.

    Raises:
        ValueError: If yfinance returns an empty DataFrame.
    """
    if ticker not in _PRICE_CACHE:
        df = yf.download(ticker, period=f"{lookback_years}y", progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        close_col = "Close" if "Close" in df.columns else "Adj Close"
        if df.empty or close_col not in df.columns:
            raise ValueError(f"No price data for {ticker!r}")

        series = df[close_col].dropna()
        series.index = pd.to_datetime(series.index)
        _PRICE_CACHE[ticker] = series
        logger.debug("_get_cached_series: cached %d rows for %s", len(series), ticker)

    return _PRICE_CACHE[ticker]


class PointInTimeEnforcer:
    """Ensures data access within Backtrader strategies is always bounded to the simulation date.

    Each enforcer instance is scoped to a single simulation bar date. All data
    lookups return only data available up to and including that date, preventing
    any forward-looking information from contaminating signal computation.
    """

    def __init__(self, simulation_date: date) -> None:
        self.simulation_date = simulation_date
        self._cutoff = pd.Timestamp(simulation_date)
        logger.debug("PointInTimeEnforcer: cutoff = %s", self.simulation_date)

    def get_close_series(self, ticker: str, lookback_days: int = 252) -> pd.Series:
        """Returns a PiT-safe daily close price Series truncated to the simulation date.

        Args:
            ticker: Equity ticker symbol.
            lookback_days: Maximum history to return (default 252 = 1 trading year).

        Returns:
            Daily close price Series with values only up to and including simulation_date.
            Returns an empty Series if the ticker has no history up to the cutoff.
        """
        series = _get_cached_series(ticker)
        available = series[series.index <= self._cutoff]
        result = available.tail(lookback_days)
        logger.debug(
            "get_close_series: %s → %d rows (cutoff=%s)", ticker, len(result), self.simulation_date
        )
        return result

    def get_returns(self, ticker: str, lookback_days: int = 252) -> pd.Series:
        """Returns PiT-safe daily percentage returns from the close price Series.

        Args:
            ticker: Equity ticker symbol.
            lookback_days: Number of returns to return after pct_change.

        Returns:
            Daily return Series, NaN-dropped, bounded to simulation_date.
        """
        closes = self.get_close_series(ticker, lookback_days + 1)
        returns = closes.pct_change().dropna()
        logger.debug(
            "get_returns: %s → %d daily returns (cutoff=%s)",
            ticker,
            len(returns),
            self.simulation_date,
        )
        return returns

    def validate_no_lookahead(self, data_df: pd.DataFrame) -> bool:
        """Verifies that no row in the DataFrame references a future date.

        Args:
            data_df: DataFrame with a DatetimeIndex to validate.

        Returns:
            True if all index values fall on or before the simulation date.
            False and a warning log if any future dates are found.
        """
        future_dates = data_df.index[data_df.index > self._cutoff]
        if not future_dates.empty:
            logger.warning(
                "validate_no_lookahead: %d future dates detected (simulation_date=%s): %s",
                len(future_dates),
                self.simulation_date,
                future_dates.tolist()[:5],
            )
            return False
        return True

    def get_fundamental_snapshot(self, ticker: str, fundamentals_df: pd.DataFrame) -> pd.Series:
        """Returns the most recent fundamental row available before the simulation date.

        Uses the last available row rather than the row matching the exact date, because
        fundamental data is typically reported with a 30–90 day lag from the period end.

        Args:
            ticker: Equity ticker symbol used to filter fundamentals_df.
            fundamentals_df: DataFrame with a DatetimeIndex and a 'ticker' column.

        Returns:
            Pandas Series of the most recent valid fundamental snapshot, or an empty Series.
        """
        ticker_data = fundamentals_df[
            (fundamentals_df["ticker"] == ticker)
            & (fundamentals_df.index <= self._cutoff)
        ]
        if ticker_data.empty:
            logger.warning(
                "get_fundamental_snapshot: no data for %s up to %s",
                ticker,
                self.simulation_date,
            )
            return pd.Series()
        return ticker_data.iloc[-1]

    def get_sentiment_snapshot(self, ticker: str, sentiment_df: pd.DataFrame) -> pd.Series:
        """Returns the most recent sentiment row available before the simulation date.

        Args:
            ticker: Equity ticker symbol used to filter sentiment_df.
            sentiment_df: DataFrame with a DatetimeIndex and a 'ticker' column.

        Returns:
            Pandas Series of the most recent valid sentiment snapshot, or an empty Series.
        """
        ticker_data = sentiment_df[
            (sentiment_df["ticker"] == ticker)
            & (sentiment_df.index <= self._cutoff)
        ]
        if ticker_data.empty:
            logger.warning(
                "get_sentiment_snapshot: no data for %s up to %s",
                ticker,
                self.simulation_date,
            )
            return pd.Series()
        return ticker_data.iloc[-1]
