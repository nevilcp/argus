"""
tests/test_signals.py

Tests schema-level validators in signals.py not already covered elsewhere.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from argus.schemas.signals import PortfolioAllocation
from tests.helpers.builders import position_allocation


def test_portfolio_allocation_rejects_a_duplicate_ticker():
    """A repeated ticker in the portfolio list is rejected.

    Regression test: node_log_decisions and risk cap enforcement key
    positions by ticker, so a duplicate would silently overwrite one of the
    two allocations rather than raising.
    """
    with pytest.raises(ValidationError, match="Duplicate ticker"):
        PortfolioAllocation(
            session_id="s1",
            user_investable_capital=100_000.0,
            portfolio=[
                position_allocation("AAPL", 0.05, composite_conviction=0.6),
                position_allocation("AAPL", 0.05, composite_conviction=0.6),
            ],
            cash_reserve_pct=0.90,
            rebalance_trigger="VIX > 35",
            timestamp=datetime.now(),
        )


def test_portfolio_allocation_accepts_unique_tickers():
    """Distinct tickers pass the uniqueness validator."""
    alloc = PortfolioAllocation(
        session_id="s1",
        user_investable_capital=100_000.0,
        portfolio=[
            position_allocation("AAPL", 0.05, composite_conviction=0.6),
            position_allocation("MSFT", 0.05, composite_conviction=0.6),
        ],
        cash_reserve_pct=0.90,
        rebalance_trigger="VIX > 35",
        timestamp=datetime.now(),
    )
    assert len(alloc.portfolio) == 2
