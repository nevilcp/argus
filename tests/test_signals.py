"""
tests/test_signals.py

Tests for schema-level validators in argus/schemas/signals.py that aren't
already covered end to end by the modules exercising them.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from argus.schemas.signals import PortfolioAllocation, PositionAllocation


def _position(ticker: str) -> PositionAllocation:
    return PositionAllocation(
        ticker=ticker,
        allocation_pct=0.05,
        allocation_usd=5_000.0,
        stop_loss=90.0,
        thesis="test thesis",
        composite_conviction=0.6,
        time_horizon="30 days",
    )


def test_portfolio_allocation_rejects_a_duplicate_ticker():
    """A repeated ticker in the portfolio list is rejected.

    Regression test for LD-6: node_log_decisions and risk cap enforcement key
    positions by ticker, so a duplicate would silently overwrite one of the
    two allocations rather than raising.
    """
    with pytest.raises(ValidationError, match="Duplicate ticker"):
        PortfolioAllocation(
            session_id="s1",
            user_investable_capital=100_000.0,
            portfolio=[_position("AAPL"), _position("AAPL")],
            cash_reserve_pct=0.90,
            rebalance_trigger="VIX > 35",
            timestamp=datetime.now(),
        )


def test_portfolio_allocation_accepts_unique_tickers():
    """Distinct tickers pass the uniqueness validator."""
    alloc = PortfolioAllocation(
        session_id="s1",
        user_investable_capital=100_000.0,
        portfolio=[_position("AAPL"), _position("MSFT")],
        cash_reserve_pct=0.90,
        rebalance_trigger="VIX > 35",
        timestamp=datetime.now(),
    )
    assert len(alloc.portfolio) == 2
