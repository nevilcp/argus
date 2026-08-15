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

import operator
from datetime import datetime
from typing import Annotated, Any, Optional, TypedDict

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

    as_of: Optional[datetime]
    """Point-in-time cutoff for cultural-memory reads: excludes trade outcomes and
    decision documents stored after this timestamp (see memory.cultural's `as_of`
    parameters). None (default) applies no filtering — the live production path.
    Set by backtesting.replay.replay_session's closed_loop arm to the replayed
    session's own capture date, so it can't read outcomes that hadn't happened yet."""

    # ── Node 1: fetch_price_history ───────────────────────────────────────────
    price_history: dict[str, dict[str, list]]
    """Mapping of ticker → {"dates": [...], "prices": [...]}, populated by
    fetch_price_history. SPY rides along with the universe (see
    graph.py:node_fetch_price_history) so it's present even when not requested."""

    session_states: dict[str, dict]
    """Mapping of ticker → compressed technical feature dict (rsi, macd, etc.)."""

    # ── Node 2: macro_analysis ────────────────────────────────────────────────
    macro_context: Optional[MacroContext]
    """Live MacroContext from MacroStatisticalAgent. Runs in parallel with, not ahead
    of, the three specialist agents below — none of them consumes it. Its
    agent_multipliers are read once, by orchestration/aggregator.py, as a per-agent
    vote-weight scalar at signal-aggregation time."""

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
    """List of completed ARGUSDecision snapshots for this session, written once
    by log_decisions and persisted into the LangGraph checkpoint
    (argus_graph.db). Single writer, no reducer: readable later by
    orchestration/reconciliation.py's load_decisions_from_checkpoints(), which
    keeps each thread's own checkpointed list rather than expecting one to
    accumulate across nodes."""

    errors: Annotated[list[str], operator.add]
    """Accumulated error messages from all nodes, used for diagnostics and
    alerting. Needs the operator.add reducer: macro_analysis,
    technical_analysis, fundamental_analysis, and sentiment_analysis all run
    in parallel off fetch_price_history and write this key concurrently, and
    LangGraph raises InvalidUpdateError on concurrent writes to a key
    without one."""
