"""Tests schema-level validators in signals.py not already covered elsewhere."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from argus.params import FUNDAMENTAL, PORTFOLIO, SENTIMENT
from argus.schemas.prompting import field_list
from argus.schemas.signals import (
    FundamentalSignal,
    FundamentalVerdict,
    PortfolioAllocation,
    PositionAllocation,
    SentimentSignal,
    Trend,
)
from tests.helpers.builders import macro_context, position_allocation


def test_portfolio_allocation_rejects_a_duplicate_ticker():
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


def test_agent_multipliers_validator_fills_missing_keys_without_touching_the_input():
    raw = {"technical": 1.2}

    context = macro_context(agent_multipliers=raw)

    assert raw == {"technical": 1.2}
    assert context.agent_multipliers == {"technical": 1.2, "fundamental": 1.0, "sentiment": 1.0}


@pytest.mark.parametrize("bad", [0.0, -0.5, 2.5])
def test_agent_multipliers_outside_zero_to_two_are_rejected(bad):
    with pytest.raises(ValidationError, match="outside"):
        macro_context(agent_multipliers={"technical": bad})


def test_macro_trend_fields_are_a_shared_enum():
    context = macro_context(interest_rate_trend="RISING", inflation_trajectory="FALLING")

    assert context.interest_rate_trend is Trend.RISING
    assert context.inflation_trajectory is Trend.FALLING
    with pytest.raises(ValidationError):
        macro_context(interest_rate_trend="SIDEWAYS")


def test_text_length_limits_in_the_schemas_come_from_params():
    limits = {
        (FundamentalSignal, "reasoning"): FUNDAMENTAL.reasoning_char_limit,
        (SentimentSignal, "reasoning"): SENTIMENT.reasoning_char_limit,
        (PositionAllocation, "thesis"): PORTFOLIO.thesis_char_limit,
        (PositionAllocation, "advisor_note"): PORTFOLIO.advisor_note_char_limit,
    }
    for (model, name), limit in limits.items():
        constraints = [m for m in model.model_fields[name].metadata if hasattr(m, "max_length")]
        assert [c.max_length for c in constraints] == [limit], f"{model.__name__}.{name}"


def test_the_moat_score_prompt_text_matches_the_field_type():
    assert "moat_score (float 0.0-10.0)" in field_list(FundamentalVerdict)
