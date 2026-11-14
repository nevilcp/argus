"""Measured forward returns shared by reconciliation, paper equity, and evaluation."""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from argus.infra.clock import as_naive_timestamp
from argus.ports import MarketDataProvider
from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger(__name__)


def needs_reconciliation(decision: ARGUSDecision) -> bool:
    """Return whether a decision has a position eligible for a measured outcome."""
    return decision.technical is not None and decision.allocation is not None and decision.allocation.allocation_pct > 0


def close_prices(
    ticker: str,
    market_data: MarketDataProvider,
    cache: dict[str, pd.Series | None],
) -> pd.Series | None:
    """Return a ticker's daily close series, fetching each ticker at most once per cache."""
    if ticker not in cache:
        try:
            cache[ticker] = market_data.ohlcv_daily(ticker)["close"]
        except Exception as exc:
            logger.warning(
                "failed to fetch price history for %s (%s): %s",
                ticker,
                type(exc).__name__,
                exc,
            )
            cache[ticker] = None
    return cache[ticker]


def _realized_return_from_prices(
    decision: ARGUSDecision, prices: pd.Series, horizon_days: int
) -> tuple[float, int, str] | None:
    """Pair a decision's entry price with a later close drawn from an already-fetched series."""
    if decision.technical is None:
        raise ValueError("caller must confirm needs_reconciliation(decision)")
    entry_price = decision.technical.current_price
    entry_date = as_naive_timestamp(decision.session_timestamp)
    target_exit_date = entry_date + timedelta(days=horizon_days)

    prices = prices.copy()
    prices.index = pd.DatetimeIndex([as_naive_timestamp(ts) for ts in prices.index])

    on_or_after = prices[prices.index >= target_exit_date].sort_index()
    if on_or_after.empty:
        return None

    exit_date = on_or_after.index[0]
    exit_price = float(on_or_after.iloc[0])

    actual_return_pct = (exit_price - entry_price) / entry_price
    holding_days = (exit_date - entry_date).days
    exit_reason = f"horizon reached ({horizon_days}d target, {holding_days}d actual)"

    return actual_return_pct, holding_days, exit_reason


def compute_realized_return(
    decision: ARGUSDecision,
    market_data: MarketDataProvider,
    horizon_days: int,
    *,
    prices: pd.Series | None = None,
) -> tuple[float, int, str] | None:
    """Compute realized return by pairing the decision's entry price with a later close."""
    if not needs_reconciliation(decision):
        return None
    if prices is None:
        prices = market_data.ohlcv_daily(decision.ticker)["close"]
    return _realized_return_from_prices(decision, prices, horizon_days)
