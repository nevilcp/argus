"""
argus/orchestration/state.py

Shared state dictionary schema for the ARGUS LangGraph execution graph.

Responsibilities:
  - Define the TypedDict shared across all LangGraph nodes
  - Document input/output field contracts per execution stage

Not responsible for:
  - State mutation (handled by individual node functions in graph.py)
  - Schema validation (handled by argus.schemas.signals Pydantic models)
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

import pandas as pd

from argus.schemas.signals import (
    AggregatedSignal,
    FundamentalSignal,
    MacroContext,
    PortfolioAllocation,
    RiskAssessment,
    SentimentSignal,
    TechnicalSignal,
)


class ARGUSState(TypedDict):
    """Complete shared state passed through every node in the ARGUS LangGraph pipeline.

    Fields are organized by the node that populates them. All fields are optional
    by default (None or empty container) until their producing node executes.
    """

    # ── Session Parameters (provided by caller) ──────────────────────────────
    ticker: str
    """Primary ticker symbol for single-asset analysis sessions."""

    universe: list[str]
    """Full set of tickers eligible for portfolio analysis in this session."""

    total_wealth: float
    """Total investable capital available to the user (USD)."""

    invest_pct: float
    """Fraction of total_wealth to deploy into equities (e.g. 0.80 = 80%)."""

    risk_tolerance: str
    """User-defined risk profile: 'CONSERVATIVE', 'MODERATE', or 'AGGRESSIVE'."""

    backtest_mode: bool
    """When True, enables ticker anonymization in fundamental analysis prompts."""

    session_seed: Optional[int]
    """Integer date stamp used to deterministically seed anonymization in backtests."""

    # ── Node 1: fetch_price_history ───────────────────────────────────────────
    price_history: dict[str, pd.Series]
    """Mapping of ticker → daily close price Series, populated by fetch_price_history."""

    session_states: dict[str, dict]
    """Mapping of ticker → compressed technical feature dict (rsi, macd, etc.)."""

    # ── Node 2: macro_analysis ────────────────────────────────────────────────
    macro_context: Optional[MacroContext]
    """Live MacroContext from MacroStatisticalAgent; used as multiplier source by specialist agents."""

    # ── Node 3: Parallel specialist agents ───────────────────────────────────
    technical_signals: dict[str, TechnicalSignal]
    """Mapping of ticker → TechnicalSignal from TechnicalStatisticalAgent."""

    fundamental_signals: dict[str, FundamentalSignal]
    """Mapping of ticker → FundamentalSignal from FundamentalAgent."""

    sentiment_signals: dict[str, SentimentSignal]
    """Mapping of ticker → SentimentSignal from SentimentAgent."""

    cultural_memory: dict[str, list[str]]
    """Dict with keys 'wisdom' and 'warnings' returned by CulturalMemoryManager."""

    # ── Node 4: risk_evaluation ───────────────────────────────────────────────
    risk_assessments: dict[str, RiskAssessment]
    """Mapping of ticker → RiskAssessment from RiskStatisticalEngine."""

    # ── Node 5: signal_aggregation ────────────────────────────────────────────
    aggregated_signals: dict[str, AggregatedSignal]
    """Mapping of ticker → AggregatedSignal from HybridSignalAggregator."""

    # ── Node 6: portfolio_allocation ──────────────────────────────────────────
    portfolio_allocation: Optional[PortfolioAllocation]
    """Final PortfolioAllocation from PortfolioManagerAgent."""

    # ── Node 7: log_decisions ────────────────────────────────────────────────
    decisions: list[Any]
    """List of completed ARGUSDecision snapshots persisted to DecisionLogger."""

    errors: list[str]
    """Accumulated error messages from all nodes, used for diagnostics and alerting."""
