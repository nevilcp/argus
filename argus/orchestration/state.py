"""
argus/orchestration/state.py
============================
Defines the LangGraph State.
"""

import operator
from typing import Annotated, Any, Optional, TypedDict

from argus.schemas.signals import (
    AggregatedSignal, ARGUSDecision, FundamentalSignal, MacroContext, PortfolioAllocation,
    RiskAssessment, SentimentSignal, TechnicalSignal
)


class ARGUSState(TypedDict):
    # Inputs
    ticker: str
    total_wealth: float
    invest_pct: float
    risk_tolerance: str
    universe: list[str]
    backtest_mode: bool
    session_seed: Optional[int]
    
    # MFT session data
    session_states: dict[str, dict]
    price_history: dict[str, Any]
    
    # Agent outputs
    technical_signals: dict[str, TechnicalSignal]
    macro_context: Optional[MacroContext]
    fundamental_signals: dict[str, FundamentalSignal]
    sentiment_signals: dict[str, SentimentSignal]
    risk_assessments: dict[str, RiskAssessment]
    aggregated_signals: dict[str, AggregatedSignal]
    
    # Cultural memory
    cultural_wisdom: list[str]
    cultural_warnings: list[str]
    
    # Final output
    portfolio_allocation: Optional[PortfolioAllocation]
    
    # Audit trail
    decisions: Annotated[list[ARGUSDecision], operator.add]
    errors: Annotated[list[str], operator.add]
