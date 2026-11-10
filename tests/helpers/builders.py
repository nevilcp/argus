"""
tests/helpers/builders.py

One builder per signal/contract type, each with keyword overrides.

Every builder returns a valid object built from fixed defaults. A test names
only the fields it cares about; any field of the underlying model can be
overridden by keyword, so no test needs its own copy of the full field list.

Not responsible for:
  - Market-data doubles (see tests/helpers/market_data.py)
  - Aggregated signals, which tests derive from the real aggregator
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from argus.schemas.signals import (
    ARGUSDecision,
    FundamentalSignal,
    MacroContext,
    PositionAllocation,
    Regime,
    RiskAssessment,
    RiskVerdict,
    SectorSignal,
    SentimentSignal,
    Signal,
    TechnicalSignal,
    VixRegime,
    YieldCurve,
)


def technical_signal(
    signal: Signal = Signal.BULLISH, conviction: float = 0.7, **fields: Any
) -> TechnicalSignal:
    """Builds a TechnicalSignal with neutral indicators; override any field by keyword."""
    values: dict[str, Any] = {
        "ticker": "TEST",
        "current_price": 100.0,
        "rsi_14": 50.0,
        "macd_histogram": 0.0,
        "bb_percent_b": 0.5,
        "atr_pct": 0.02,
        "adx_14": 20.0,
        "vwap_distance": 0.0,
        "volume_ratio": 1.0,
        "momentum_30m": 0.0,
        "momentum_1d": 0.0,
        "signal": signal,
        "conviction": conviction,
        "net_score": 0.0,
        "timestamp": datetime.now(),
    }
    return TechnicalSignal(**{**values, **fields})


def fundamental_signal(
    signal: Signal = Signal.BULLISH, conviction: float = 0.7, **fields: Any
) -> FundamentalSignal:
    """Builds a FundamentalSignal; override any field by keyword."""
    values: dict[str, Any] = {
        "ticker": "TEST",
        "sector": "Technology",
        "industry": "Software",
        "signal": signal,
        "conviction": conviction,
        "moat_score": 5.0,
        "reasoning": "test",
        "data_as_of_date": datetime.now().date(),
        "timestamp": datetime.now(),
    }
    return FundamentalSignal(**{**values, **fields})


def sentiment_signal(
    signal: Signal = Signal.BULLISH, conviction: float = 0.7, **fields: Any
) -> SentimentSignal:
    """Builds a SentimentSignal; override any field by keyword."""
    values: dict[str, Any] = {
        "ticker": "TEST",
        "finbert_net_score": 0.0,
        "pct_positive": 0.3,
        "pct_negative": 0.3,
        "news_volume_7d": 5,
        "upcoming_catalyst": False,
        "signal": signal,
        "conviction": conviction,
        "sentiment_decay_risk": "LOW",
        "reasoning": "test",
        "timestamp": datetime.now(),
    }
    return SentimentSignal(**{**values, **fields})


def macro_context(regime: Regime = Regime.EXPANSION, **fields: Any) -> MacroContext:
    """Builds a MacroContext in `regime` with unit multipliers; override any field by keyword."""
    values: dict[str, Any] = {
        "fed_funds": 4.0,
        "cpi_yoy": 3.0,
        "unemployment": 4.0,
        "t10y2y": 0.1,
        "consumer_sentiment": 60.0,
        "vix_level": 18.0,
        "macro_regime": regime,
        "regime_confidence": 0.7,
        "model_healthy": True,
        "interest_rate_trend": "STABLE",
        "yield_curve_shape": YieldCurve.NORMAL,
        "vix_regime": VixRegime.MEDIUM,
        "vix_percentile": 40.0,
        "inflation_trajectory": "STABLE",
        "sector_rotation_signal": SectorSignal.GROWTH_FAVORED,
        "agent_multipliers": {"fundamental": 1.0, "technical": 1.0, "sentiment": 1.0},
        "timestamp": datetime.now(),
    }
    return MacroContext(**{**values, **fields})


def risk_assessment(
    verdict: RiskVerdict = RiskVerdict.APPROVE,
    approved_weight: float = 0.15,
    *,
    proposed_weight: float | None = None,
    **fields: Any,
) -> RiskAssessment:
    """Builds a RiskAssessment; `proposed_weight` defaults to `approved_weight`."""
    values: dict[str, Any] = {
        "verdict": verdict,
        "approved_weight": approved_weight,
        "proposed_weight": approved_weight if proposed_weight is None else proposed_weight,
        "var_99": 0.05,
        "portfolio_beta": 1.0,
        "timestamp": datetime.now(),
    }
    return RiskAssessment(**{**values, **fields})


def position_allocation(
    ticker: str = "TEST", allocation_pct: float = 0.10, **fields: Any
) -> PositionAllocation:
    """Builds a PositionAllocation sized against 100 000 USD; override any field by keyword."""
    values: dict[str, Any] = {
        "ticker": ticker,
        "allocation_pct": allocation_pct,
        "allocation_usd": allocation_pct * 100_000.0,
        "stop_loss": 90.0,
        "thesis": "test thesis",
        "composite_conviction": 0.7,
        "time_horizon": "30 days",
    }
    return PositionAllocation(**{**values, **fields})


def argus_decision(
    ticker: str = "TEST", session_timestamp: datetime | None = None, **fields: Any
) -> ARGUSDecision:
    """Builds an ARGUSDecision carrying only what `fields` names (signals, allocation, ...)."""
    values: dict[str, Any] = {
        "ticker": ticker,
        "session_timestamp": datetime.now() if session_timestamp is None else session_timestamp,
    }
    return ARGUSDecision(**{**values, **fields})


def user_profile(invest_pct: float = 0.5, **fields: Any) -> dict[str, Any]:
    """Builds the user-profile dict `PortfolioManagerAgent.allocate` takes."""
    values: dict[str, Any] = {
        "total_wealth": 100_000.0,
        "invest_pct": invest_pct,
        "risk_tolerance": "MODERATE",
    }
    return {**values, **fields}
