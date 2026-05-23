"""
argus/schemas/signals.py
========================
Canonical Pydantic v2 data contracts for ARGUS v2.

Every agent MUST return one of the schemas defined here.  No unstructured text
is passed between agents — all inter-agent communication is typed, validated,
and serialisable to JSON.

Schema hierarchy
----------------
Enumerations
  Signal · Regime · RiskVerdict · VixRegime · YieldCurve · SectorSignal

Agent output schemas (one per specialist agent)
  TechnicalSignal     — Technical Analysis Agent
  MacroContext        — Macro-Economic Agent
  FundamentalSignal   — Fundamental Analysis Agent
  SentimentSignal     — Sentiment Analysis Agent
  RiskAssessment      — Risk Assessment Agent

Portfolio schemas
  PositionAllocation  — single ticker position recommendation
  PortfolioAllocation — full portfolio rebalance plan

Orchestration schemas
  AggregatedSignal    — Aggregator node output (per ticker)
  ARGUSDecision       — Complete decision log entry (all signals combined)

Conviction capping rule (applied via field validator on every signal schema)
  conviction is silently clamped to 0.95 if a value > 0.95 is supplied.
  A warning is emitted via the standard `warnings` module; no exception is
  raised so upstream callers are never disrupted.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from datetime import date, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    computed_field,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────


class Signal(str):
    """Directional signal label returned by every specialist agent."""
    BULLISH  = "BULLISH"
    BEARISH  = "BEARISH"
    NEUTRAL  = "NEUTRAL"

    def __new__(cls, value: str) -> "Signal":  # noqa: D102
        allowed = {cls.BULLISH, cls.BEARISH, cls.NEUTRAL}
        if value not in allowed:
            raise ValueError(f"Signal must be one of {allowed}, got {value!r}")
        return str.__new__(cls, value)


# Re-implement as proper str enum for Pydantic serialisation
from enum import Enum  # noqa: E402  (after str subclass above for clarity)


class Signal(str, Enum):
    """Directional signal label returned by every specialist agent."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Regime(str, Enum):
    """Macroeconomic regime classification produced by the Macro agent."""
    EXPANSION    = "EXPANSION"
    CONTRACTION  = "CONTRACTION"
    TRANSITIONAL = "TRANSITIONAL"


class RiskVerdict(str, Enum):
    """Risk agent disposition on a proposed portfolio action."""
    APPROVE = "APPROVE"  # proceed with proposed_weight unchanged
    VETO    = "VETO"     # block the position entirely
    REDUCE  = "REDUCE"   # approved_weight < proposed_weight


class VixRegime(str, Enum):
    """Categorical bucket for the current VIX level."""
    LOW     = "LOW"      # VIX < 15
    MEDIUM  = "MEDIUM"   # 15 ≤ VIX < 25
    HIGH    = "HIGH"     # 25 ≤ VIX < 35
    EXTREME = "EXTREME"  # VIX ≥ 35  → Governor kill-switch zone


class YieldCurve(str, Enum):
    """Shape of the US Treasury yield curve (10Y-2Y spread)."""
    NORMAL   = "NORMAL"    # spread > +25 bps
    FLAT     = "FLAT"      # -25 bps ≤ spread ≤ +25 bps
    INVERTED = "INVERTED"  # spread < -25 bps (recession signal)


class SectorSignal(str, Enum):
    """Macro-derived sector rotation preference."""
    GROWTH_FAVORED    = "GROWTH_FAVORED"    # risk-on, lower rates
    VALUE_FAVORED     = "VALUE_FAVORED"     # rising rates, reflation
    DEFENSIVE         = "DEFENSIVE"         # high VIX / contraction regime


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_CONVICTION_MAX = 0.95


def _clamp_conviction(v: float) -> float:
    """Silently cap conviction at 0.95; emit a warning if clamping occurs."""
    if v > _CONVICTION_MAX:
        warnings.warn(
            f"conviction={v:.4f} exceeds maximum {_CONVICTION_MAX}; "
            "clamping to 0.95 — check your scoring function.",
            stacklevel=4,
        )
        return _CONVICTION_MAX
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Agent Output Schemas
# ──────────────────────────────────────────────────────────────────────────────


class TechnicalSignal(BaseModel):
    """
    Output of the Technical Analysis Agent.

    Produced once per ticker per decision cycle.  Contains both raw indicator
    values (for auditability) and the weighted composite score that drives the
    directional `signal` label.
    """

    ticker: str = Field(..., description="Equity ticker symbol, e.g. 'AAPL'")
    signal: Signal = Field(..., description="Directional label derived from net_score")
    conviction: float = Field(
        ..., ge=0.0, le=_CONVICTION_MAX,
        description="Normalised confidence in the signal [0, 0.95]",
    )
    net_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Raw weighted composite score before discretisation",
    )

    # ── Indicator values (stored for auditability / memory retrieval) ──────
    rsi_14:          float = Field(..., description="14-period RSI value [0, 100]")
    macd_histogram:  float = Field(..., description="MACD histogram value")
    bb_percent_b:    float = Field(..., description="Bollinger %B [0, 1] (>1 or <0 = breach)")
    atr_pct:         float = Field(..., ge=0.0, description="ATR as % of price (volatility proxy)")
    adx_14:          float = Field(..., ge=0.0, le=100.0, description="14-period ADX trend strength")
    vwap_distance:   float = Field(..., description="(price - VWAP) / VWAP; signed")
    volume_ratio:    float = Field(..., ge=0.0, description="Current volume / 20-day avg volume")
    momentum_30m:    float = Field(..., description="30-minute log-return")
    momentum_1d:     float = Field(..., description="1-day log-return")

    temporal_horizon: Literal["SHORT_TERM"] = "SHORT_TERM"
    api_calls_used:   int = Field(0, ge=0, description="External API calls consumed")
    timestamp:        datetime = Field(..., description="UTC timestamp of signal production")

    @field_validator("conviction", mode="before")
    @classmethod
    def cap_conviction(cls, v: float) -> float:
        """Silently clamp conviction to 0.95 maximum."""
        return _clamp_conviction(v)


class MacroContext(BaseModel):
    """
    Output of the Macro-Economic Agent.

    A single instance is produced per decision cycle (not per ticker) and is
    shared with all other agents via `AgentState`.  The `agent_multipliers`
    dict scales each downstream agent's conviction based on the current regime.
    """

    macro_regime:           Regime      = Field(..., description="HMM-detected economic regime")
    interest_rate_trend:    Literal["RISING", "FALLING", "STABLE"] = Field(
        ..., description="Direction of short-term interest rates"
    )
    yield_curve_shape:      YieldCurve  = Field(..., description="10Y-2Y Treasury spread shape")
    vix_level:              float       = Field(..., ge=0.0, description="CBOE VIX spot level")
    vix_regime:             VixRegime   = Field(..., description="Categorical VIX bucket")
    vix_percentile:         float       = Field(..., ge=0.0, le=100.0,
                                                description="VIX vs. trailing 252-day distribution")
    inflation_trajectory:   Literal["RISING", "FALLING", "STABLE"] = Field(
        ..., description="Direction of CPI year-over-year rate"
    )
    sector_rotation_signal: SectorSignal = Field(
        ..., description="Macro-implied sector preference"
    )
    agent_multipliers: dict[str, float] = Field(
        ...,
        description=(
            "Conviction multipliers keyed by agent name "
            "(fundamental, technical, sentiment). Range [0.5, 1.5]."
        ),
    )
    regime_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="HMM posterior probability for the detected regime",
    )
    api_calls_used: int      = Field(0, ge=0)
    timestamp:      datetime = Field(..., description="UTC timestamp of signal production")

    @field_validator("agent_multipliers", mode="before")
    @classmethod
    def validate_multiplier_keys(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure required agent keys are present; default missing ones to 1.0."""
        required = {"fundamental", "technical", "sentiment"}
        for key in required:
            v.setdefault(key, 1.0)
        for key, mult in v.items():
            if not (0.0 < mult <= 2.0):
                raise ValueError(
                    f"agent_multipliers['{key}']={mult} is outside (0, 2]; "
                    "must be a positive scaling factor"
                )
        return v


class FundamentalSignal(BaseModel):
    """
    Output of the Fundamental Analysis Agent.

    Produced once per ticker.  Financial ratios may be None when data is
    unavailable for a given ticker (e.g., pre-revenue companies).  The LLM
    reasoning field is capped at 400 characters to keep state payloads small.
    """

    ticker:    str           = Field(..., description="Equity ticker symbol")
    anon_id:   str | None    = Field(None, description="Optional anonymised ID for A/B testing")
    signal:    Signal        = Field(..., description="Directional label")
    conviction: float        = Field(..., ge=0.0, le=_CONVICTION_MAX)

    # ── Financial metrics ──────────────────────────────────────────────────
    pe_ttm:              float | None = Field(None, description="Trailing P/E ratio")
    revenue_growth_yoy:  float | None = Field(None, description="YoY revenue growth rate")
    operating_margin:    float | None = Field(None, description="Operating income / revenue")
    fcf_yield:           float | None = Field(None, description="Free cash flow / market cap")
    debt_to_equity:      float | None = Field(None, ge=0.0, description="Total debt / equity")
    roic:                float | None = Field(None, description="Return on invested capital")
    moat_score:          float        = Field(..., ge=0.0, le=10.0,
                                             description="Qualitative competitive moat [0, 10]")

    reasoning:        str       = Field(..., max_length=400,
                                        description="LLM-generated investment thesis (≤400 chars)")
    temporal_horizon: Literal["LONG_TERM"] = "LONG_TERM"
    data_as_of_date:  date      = Field(..., description="Point-in-time date for financial data")
    api_calls_used:   int       = Field(1, ge=0)
    timestamp:        datetime  = Field(..., description="UTC timestamp of signal production")

    @field_validator("conviction", mode="before")
    @classmethod
    def cap_conviction(cls, v: float) -> float:
        return _clamp_conviction(v)


class SentimentSignal(BaseModel):
    """
    Output of the Sentiment Analysis Agent.

    Aggregates FinBERT scores from news headlines and social posts into a
    single signed sentiment score per ticker.  Boolean flags surface
    actionable conditions (upcoming catalyst, social surge) for the Aggregator.
    """

    ticker:     str    = Field(..., description="Equity ticker symbol")
    signal:     Signal = Field(..., description="Directional label")
    conviction: float  = Field(..., ge=0.0, le=_CONVICTION_MAX)

    # ── Sentiment metrics ──────────────────────────────────────────────────
    finbert_net_score:    float = Field(..., ge=-1.0, le=1.0,
                                        description="FinBERT weighted net score [-1, +1]")
    pct_positive:         float = Field(..., ge=0.0, le=1.0,
                                        description="Fraction of articles scored positive")
    pct_negative:         float = Field(..., ge=0.0, le=1.0,
                                        description="Fraction of articles scored negative")
    news_volume_7d:       int   = Field(..., ge=0,
                                        description="Total news articles ingested (7-day window)")
    social_mention_surge: bool  = Field(...,
                                        description="True if social mentions > 2× 30-day average")
    upcoming_catalyst:    bool  = Field(...,
                                        description="True if earnings / FDA / FOMC within 14 days")
    sentiment_decay_risk: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Estimated speed at which the sentiment signal will decay"
    )

    reasoning:        str      = Field(..., max_length=400,
                                       description="LLM rationale for the signal (≤400 chars)")
    temporal_horizon: Literal["EVENT_DRIVEN"] = "EVENT_DRIVEN"
    api_calls_used:   int      = Field(1, ge=0)
    timestamp:        datetime = Field(..., description="UTC timestamp of signal production")

    @field_validator("conviction", mode="before")
    @classmethod
    def cap_conviction(cls, v: float) -> float:
        return _clamp_conviction(v)

    @model_validator(mode="after")
    def pct_sums_to_one_approx(self) -> "SentimentSignal":
        """Warn (not error) if pct_positive + pct_negative exceeds 1.0."""
        total = self.pct_positive + self.pct_negative
        if total > 1.001:
            warnings.warn(
                f"pct_positive ({self.pct_positive:.3f}) + pct_negative "
                f"({self.pct_negative:.3f}) = {total:.3f} > 1.0 for ticker "
                f"{self.ticker!r}; check normalisation.",
                stacklevel=3,
            )
        return self


class RiskAssessment(BaseModel):
    """
    Output of the Risk Assessment Agent.

    Produced once per decision cycle across the entire proposed portfolio.
    `verdict` drives the Governor's routing logic: VETO routes to HALT,
    REDUCE lowers approved_weight, APPROVE passes the portfolio to the
    Portfolio agent unchanged.
    """

    verdict:        RiskVerdict = Field(..., description="Disposition on the proposed portfolio")
    proposed_weight: float      = Field(..., ge=0.0, le=1.0,
                                        description="Aggregate equity exposure before risk review")
    approved_weight: float      = Field(..., ge=0.0, le=1.0,
                                        description="Equity exposure approved by Risk agent")
    veto_reasons:   list[str]   = Field(default_factory=list,
                                        description="Human-readable reasons for VETO / REDUCE")

    # ── Portfolio-level risk metrics ───────────────────────────────────────
    var_99:           float | None = Field(None, description="99% 1-day Value-at-Risk (as + %)")
    cvar:             float | None = Field(None, description="Conditional VaR / Expected Shortfall")
    portfolio_beta:   float | None = Field(None, description="Dollar-weighted beta vs SPY")
    avg_correlation:  float | None = Field(None, ge=-1.0, le=1.0,
                                           description="Average pairwise return correlation")

    # ── Per-ticker maps ────────────────────────────────────────────────────
    stop_losses:  dict[str, float] = Field(
        default_factory=dict,
        description="Ticker → stop-loss price level",
    )
    marginal_var: dict[str, float] = Field(
        default_factory=dict,
        description="Ticker → marginal contribution to portfolio VaR",
    )

    api_calls_used: int      = Field(0, ge=0)
    timestamp:      datetime = Field(..., description="UTC timestamp of signal production")

    @model_validator(mode="after")
    def approved_le_proposed(self) -> "RiskAssessment":
        """approved_weight must never exceed proposed_weight."""
        if self.approved_weight > self.proposed_weight + 1e-6:
            raise ValueError(
                f"approved_weight ({self.approved_weight}) cannot exceed "
                f"proposed_weight ({self.proposed_weight})"
            )
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio Schemas
# ──────────────────────────────────────────────────────────────────────────────


class PositionAllocation(BaseModel):
    """
    A single ticker's position recommendation within a `PortfolioAllocation`.

    `allocation_pct` is hard-capped at 0.15 (MAX_SINGLE_POSITION_PCT) by the
    field constraint — the Portfolio agent must not exceed this even if the
    Kelly criterion suggests a larger position.
    """

    ticker:               str        = Field(..., description="Equity ticker symbol")
    allocation_pct:       float      = Field(..., ge=0.0, le=0.15,
                                             description="Target portfolio weight [0, 0.15]")
    allocation_usd:       float      = Field(..., ge=0.0,
                                             description="Dollar value of the position")
    stop_loss:            float      = Field(..., description="Stop-loss price level")
    target_price:         float | None = Field(None, description="12-month price target (optional)")
    thesis:               str        = Field(..., max_length=120,
                                             description="One-sentence position thesis (≤120 chars)")
    composite_conviction: float      = Field(..., ge=0.0, le=1.0,
                                             description="Aggregated conviction across all agents")
    time_horizon:         str        = Field(...,
                                             description="Expected holding period, e.g. '30 days'")


class PortfolioAllocation(BaseModel):
    """
    Full portfolio rebalance plan produced by the Portfolio Construction Agent.

    The model-level validator ensures the sum of all position weights plus the
    cash reserve does not exceed 1.01 (a 1 % rounding tolerance is permitted
    to avoid false validation failures from floating-point arithmetic).
    """

    session_id:              str                    = Field(
        ..., description="Unique identifier for this advisory session"
    )
    user_investable_capital: float                  = Field(
        ..., gt=0.0, description="Total capital available for investment (USD)"
    )
    portfolio:               list[PositionAllocation] = Field(
        default_factory=list, description="List of recommended positions"
    )
    cash_reserve_pct:        float                  = Field(
        ..., ge=0.05, le=1.00,
        description="Fraction of capital held as cash [0.05, 1.00]",
    )
    expected_sharpe:         float | None           = Field(
        None, description="Forward-looking Sharpe estimate for the proposed portfolio"
    )
    rebalance_trigger:       str                    = Field(
        ..., description="Condition that will next trigger rebalancing, e.g. 'VIX > 35'"
    )
    api_calls_used: int      = Field(1, ge=0)
    timestamp:      datetime = Field(..., description="UTC timestamp of signal production")

    @model_validator(mode="after")
    def total_allocation_le_one(self) -> "PortfolioAllocation":
        """Sum of position weights + cash_reserve_pct must not exceed 1.01."""
        total_equity = sum(p.allocation_pct for p in self.portfolio)
        grand_total  = total_equity + self.cash_reserve_pct
        if grand_total > 1.01:
            raise ValueError(
                f"Total allocations ({total_equity:.4f}) + cash_reserve_pct "
                f"({self.cash_reserve_pct:.4f}) = {grand_total:.4f} > 1.01. "
                "Reduce position sizes or increase cash reserve."
            )
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Schemas
# ──────────────────────────────────────────────────────────────────────────────


class AggregatedSignal(BaseModel):
    """
    Per-ticker signal produced by the Aggregator node.

    Combines TechnicalSignal, FundamentalSignal, and SentimentSignal
    conviction-weighted votes into a single directional view.  If the votes
    are closely split (no agent achieves > 55 % weighted majority), the
    Aggregator may set `debate_triggered = True` to invoke the LLM arbitration
    step before passing control to the Portfolio agent.
    """

    ticker:          str              = Field(..., description="Equity ticker symbol")
    signal:          Signal           = Field(..., description="Aggregated directional label")
    conviction:      float            = Field(..., ge=0.0, le=1.0,
                                             description="Aggregated conviction score")
    weighted_votes:  dict[str, float] = Field(
        ...,
        description=(
            "Agent name → weighted vote value used in aggregation. "
            "Keys include 'technical', 'fundamental', 'sentiment'."
        ),
    )
    debate_triggered: bool       = Field(
        False,
        description="True when agent votes are too split for majority rule",
    )
    skip_reason:      str | None = Field(
        None,
        description="If set, this ticker was excluded from portfolio consideration",
    )


class ARGUSDecision(BaseModel):
    """
    Complete decision log entry combining all agent signals for a single ticker.

    Written to ChromaDB cultural memory at the end of each decision cycle.
    `total_api_calls` is automatically computed from the `api_calls_used`
    fields of all child schemas — no need to set it manually.
    """

    decision_id:       str       = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID v4 unique identifier for this decision record",
    )
    ticker:            str       = Field(..., description="Equity ticker symbol")
    session_timestamp: datetime  = Field(..., description="UTC timestamp of the decision cycle")

    # ── Agent signal snapshots (all optional — populated as agents complete) ──
    technical:   TechnicalSignal   | None = Field(None)
    macro:       MacroContext       | None = Field(None)
    fundamental: FundamentalSignal | None = Field(None)
    sentiment:   SentimentSignal   | None = Field(None)
    risk:        RiskAssessment    | None = Field(None)
    aggregated:  AggregatedSignal  | None = Field(None)
    allocation:  PositionAllocation | None = Field(None)

    @computed_field  # type: ignore[misc]
    @property
    def total_api_calls(self) -> int:
        """Sum api_calls_used from every non-None child schema."""
        sources: list[Any] = [
            self.technical,
            self.macro,
            self.fundamental,
            self.sentiment,
            self.risk,
        ]
        return sum(
            getattr(s, "api_calls_used", 0)
            for s in sources
            if s is not None
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public re-exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enumerations
    "Signal",
    "Regime",
    "RiskVerdict",
    "VixRegime",
    "YieldCurve",
    "SectorSignal",
    # Agent schemas
    "TechnicalSignal",
    "MacroContext",
    "FundamentalSignal",
    "SentimentSignal",
    "RiskAssessment",
    # Portfolio schemas
    "PositionAllocation",
    "PortfolioAllocation",
    # Orchestration schemas
    "AggregatedSignal",
    "ARGUSDecision",
]
