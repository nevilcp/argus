"""One builder per signal/contract type, each with keyword overrides."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest import mock

from argus.orchestration.aggregator import HybridSignalAggregator
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


def technical_signal(signal: Signal = Signal.BULLISH, conviction: float = 0.7, **fields: Any) -> TechnicalSignal:
    """Build a TechnicalSignal with neutral indicators; override any field by keyword."""
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


def fundamental_signal(signal: Signal = Signal.BULLISH, conviction: float = 0.7, **fields: Any) -> FundamentalSignal:
    """Build a FundamentalSignal; override any field by keyword."""
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


def sentiment_signal(signal: Signal = Signal.BULLISH, conviction: float = 0.7, **fields: Any) -> SentimentSignal:
    """Build a SentimentSignal; override any field by keyword."""
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
    """Build a MacroContext in `regime` with unit multipliers; override any field by keyword."""
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
    """Build a RiskAssessment; `proposed_weight` defaults to `approved_weight`."""
    values: dict[str, Any] = {
        "verdict": verdict,
        "approved_weight": approved_weight,
        "proposed_weight": approved_weight if proposed_weight is None else proposed_weight,
        "var_99": 0.05,
        "portfolio_beta": 1.0,
        "timestamp": datetime.now(),
    }
    return RiskAssessment(**{**values, **fields})


def position_allocation(ticker: str = "TEST", allocation_pct: float = 0.10, **fields: Any) -> PositionAllocation:
    """Build a PositionAllocation sized against 100 000 USD; override any field by keyword."""
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


def argus_decision(ticker: str = "TEST", session_timestamp: datetime | None = None, **fields: Any) -> ARGUSDecision:
    """Build an ARGUSDecision carrying only what `fields` names (signals, allocation, ...)."""
    values: dict[str, Any] = {
        "ticker": ticker,
        "session_timestamp": datetime.now() if session_timestamp is None else session_timestamp,
    }
    return ARGUSDecision(**{**values, **fields})


def user_profile(invest_pct: float = 0.5, **fields: Any) -> dict[str, Any]:
    """Build the user-profile dict `PortfolioManagerAgent.allocate` takes."""
    values: dict[str, Any] = {
        "total_wealth": 100_000.0,
        "invest_pct": invest_pct,
        "risk_tolerance": "MODERATE",
    }
    return {**values, **fields}


def neutral_cultural_memory() -> mock.Mock:
    """Build a cultural-memory double that retrieves nothing and report the 0.5 prior."""
    return mock.Mock(
        retrieve_wisdom=mock.Mock(return_value=[]),
        retrieve_warnings=mock.Mock(return_value=[]),
        get_agent_accuracy=mock.Mock(return_value=(0.5, 0)),
        store_decision_snapshot=mock.Mock(),
    )


def aggregated_decision(
    technical: TechnicalSignal,
    macro: MacroContext,
    fundamental: FundamentalSignal | None = None,
    sentiment: SentimentSignal | None = None,
    *,
    reliability: dict[str, float] | None = None,
    session_timestamp: datetime | None = None,
    allocation: PositionAllocation | None = None,
) -> ARGUSDecision:
    """Build a TEST decision whose aggregated signal is the real aggregator's output."""
    aggregated = HybridSignalAggregator().aggregate(technical, macro, fundamental, sentiment, reliability=reliability)
    return ARGUSDecision(
        ticker="TEST",
        session_timestamp=datetime.now() if session_timestamp is None else session_timestamp,
        technical=technical,
        fundamental=fundamental,
        sentiment=sentiment,
        macro=macro,
        aggregated=aggregated,
        allocation=allocation,
    )


def matured_decision(ticker: str, start: datetime, price: float = 100.0) -> ARGUSDecision:
    """Build a positioned decision that matures five days after start."""
    return ARGUSDecision(
        ticker=ticker,
        session_timestamp=start,
        technical=technical_signal(Signal.BULLISH, 0.8, ticker=ticker, current_price=price),
        allocation=position_allocation(ticker=ticker),
    )
