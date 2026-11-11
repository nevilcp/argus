"""
argus/outcomes.py

What a decision's position actually returned, shared by everything that scores
decisions against later prices.

Responsibilities:
  - needs_reconciliation: whether a decision took a position that can have an outcome
  - compute_realized_return: entry close -> first close on or after the horizon
  - close_prices: a per-call memo so each ticker's price history is fetched once

Not responsible for:
  - Storing outcomes or crediting an agent (see orchestration/reconcile.py)
  - The paper equity curve (see risk/paper_book.py)
  - Evaluation statistics (see backtesting/evaluation.py)

Dependencies:
  - argus.infra.clock (as_naive_timestamp)
  - argus.ports (MarketDataProvider)
  - argus.schemas.signals (ARGUSDecision)
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from argus.infra.clock import as_naive_timestamp
from argus.ports import MarketDataProvider
from argus.schemas.signals import ARGUSDecision

logger = logging.getLogger(__name__)


def needs_reconciliation(decision: ARGUSDecision) -> bool:
    """Whether a decision took a position that could have a realized outcome."""
    return (
        decision.technical is not None
        and decision.allocation is not None
        and decision.allocation.allocation_pct > 0
    )


def close_prices(
    ticker: str,
    market_data: MarketDataProvider,
    cache: dict[str, pd.Series | None],
) -> pd.Series | None:
    """Returns a ticker's daily close series, fetching each ticker at most once per cache.

    Args:
        ticker: Ticker to price.
        market_data: Source of the daily close price series.
        cache: Per-call memo, mutated in place; a failed fetch is cached as
            None so a broken ticker isn't retried for every decision on it.

    Returns:
        The close price series, or None if the fetch failed.
    """
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
    """Pairs a decision's entry price with a later close drawn from an already-fetched series.

    Args:
        decision: Completed ARGUSDecision; caller has already confirmed
            needs_reconciliation(decision).
        prices: Close-price Series for decision.ticker.
        horizon_days: Calendar days after session_timestamp defining the
            target exit date.

    Returns:
        (actual_return_pct, holding_days, exit_reason), or None when the
        price series doesn't yet extend past the target exit date — horizon
        not reached, deferred rather than fabricated.
    """
    assert decision.technical is not None, "caller must confirm needs_reconciliation(decision)"
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
    """Computes realized return by pairing the decision's entry price with a later close.

    Entry price is decision.technical.current_price — the close the
    technical agent scored against at decision time. Exit price is the first
    close on or after session_timestamp + horizon_days, read from
    market_data.ohlcv_daily().

    Args:
        decision: Completed ARGUSDecision.
        market_data: Source of the ticker's daily close price series.
        horizon_days: Calendar days after session_timestamp defining the
            target exit date.
        prices: Pre-fetched close-price Series for decision.ticker. Passed by
            callers batching one fetch per ticker across several decisions
            (see reconcile_decisions, risk/paper_book.py,
            backtesting/evaluation.py); fetched here when omitted.

    Returns:
        (actual_return_pct, holding_days, exit_reason), or None when there is
        nothing to reconcile: no technical signal (no entry price), no
        allocation or a zero/negative-weight allocation (no position was
        taken), or the price series doesn't yet extend past the target exit
        date — horizon not reached, deferred rather than fabricated.
    """
    if not needs_reconciliation(decision):
        return None
    if prices is None:
        prices = market_data.ohlcv_daily(decision.ticker)["close"]
    return _realized_return_from_prices(decision, prices, horizon_days)
