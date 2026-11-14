"""Shared decision graph state and per-ticker projections."""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, TypedDict

from argus.schemas.signals import (
    AggregatedSignal,
    ARGUSDecision,
    FundamentalSignal,
    MacroContext,
    PortfolioAllocation,
    RiskAssessment,
    SentimentSignal,
    TechnicalSignal,
    TickerSnapshot,
)


class ARGUSState(TypedDict):
    """Shared graph state initialized with every key before node execution."""

    ticker: str

    universe: list[str]

    total_wealth: float

    invest_pct: float

    risk_tolerance: str

    backtest_mode: bool

    session_seed: int | None

    # Exclude memory rows after this capture time during closed-loop replay; None leaves live reads unfiltered.
    as_of: datetime | None

    price_history: dict[str, dict[str, list]]

    session_states: dict[str, dict]

    macro_context: MacroContext | None

    technical_signals: dict[str, TechnicalSignal]

    fundamental_signals: dict[str, FundamentalSignal]

    sentiment_signals: dict[str, SentimentSignal]

    cultural_memory: dict[str, list[str]]

    risk_assessments: dict[str, RiskAssessment]

    aggregated_signals: dict[str, AggregatedSignal]

    portfolio_allocation: PortfolioAllocation | None

    decisions: list[ARGUSDecision]

    # Parallel specialist and macro nodes append diagnostics through a reducer.
    errors: Annotated[list[str], operator.add]


def initial_state(
    universe: list[str],
    *,
    session_states: dict[str, dict],
    total_wealth: float,
    invest_pct: float,
    risk_tolerance: str,
    # Exclude memory rows after this capture time during closed-loop replay; None leaves live reads unfiltered.
    as_of: datetime | None = None,
    backtest_mode: bool = False,
    session_seed: int | None = None,
) -> ARGUSState:
    """Build the starting ARGUSState for one graph run."""
    return ARGUSState(
        ticker=universe[0],
        universe=universe,
        total_wealth=total_wealth,
        invest_pct=invest_pct,
        risk_tolerance=risk_tolerance,
        backtest_mode=backtest_mode,
        session_seed=session_seed,
        as_of=as_of,
        price_history={},
        session_states=session_states,
        macro_context=None,
        technical_signals={},
        fundamental_signals={},
        sentiment_signals={},
        cultural_memory={"wisdom": [], "warnings": []},
        risk_assessments={},
        aggregated_signals={},
        portfolio_allocation=None,
        decisions=[],
        errors=[],
    )


def build_ticker_snapshots(state: ARGUSState) -> dict[str, TickerSnapshot]:
    """Join every specialist agent's per-ticker output into one TickerSnapshot per ticker."""
    technical_signals = state.get("technical_signals", {})
    fundamental_signals = state.get("fundamental_signals", {})
    sentiment_signals = state.get("sentiment_signals", {})
    risk_assessments = state.get("risk_assessments", {})
    aggregated_signals = state.get("aggregated_signals", {})
    return {
        ticker: TickerSnapshot(
            technical=technical_signals.get(ticker),
            fundamental=fundamental_signals.get(ticker),
            sentiment=sentiment_signals.get(ticker),
            risk=risk_assessments.get(ticker),
            aggregated=aggregated_signals.get(ticker),
        )
        for ticker in state.get("universe", [])
    }
