"""
argus/orchestration/aggregator.py

Multi-agent signal aggregation module implementing conflict arbitration.

Responsibilities:
  - Weight individual specialist signals by macro multipliers and, optionally,
    per-regime historical reliability (see cultural.get_agent_accuracy)
  - Apply majority voting and override rules to produce a single AggregatedSignal

Not responsible for:
  - LLM inference or data fetching
  - Portfolio allocation (see agents/portfolio.py)
  - Risk enforcement (see agents/risk.py)

Dependencies:
  - argus.schemas.signals (Signal, AggregatedSignal, MacroContext, etc.)
"""

from __future__ import annotations

import logging
from typing import Optional

from argus.params import AGGREGATOR
from argus.schemas.signals import (
    AggregatedSignal,
    FundamentalSignal,
    MacroContext,
    Regime,
    SentimentSignal,
    Signal,
    TechnicalSignal,
)

logger = logging.getLogger("argus.aggregator")


class HybridSignalAggregator:
    """Arbitrates and fuses independent specialist signals into a consensus AggregatedSignal.

    Arbitration uses a weighted Borda-count style voting scheme. Macro multipliers
    scale each agent's base weight at runtime; the highest-weighted signal pool wins.
    """

    DEFAULT_WEIGHTS = {
        "fundamental": AGGREGATOR.weight_fundamental,
        "technical": AGGREGATOR.weight_technical,
        "sentiment": AGGREGATOR.weight_sentiment,
    }

    def aggregate(
        self,
        technical: Optional[TechnicalSignal],
        macro: Optional[MacroContext],
        fundamental: Optional[FundamentalSignal],
        sentiment: Optional[SentimentSignal],
        reliability: Optional[dict[str, float]] = None,
    ) -> AggregatedSignal:
        """Aggregates specialist signals via conviction-weighted voting with macro multipliers.

        At least one of fundamental, technical, or sentiment must be non-None for
        a meaningful signal to be generated.

        Args:
            technical: Optional TechnicalSignal from TechnicalStatisticalAgent.
            macro: Optional MacroContext from MacroStatisticalAgent; used for multipliers.
            fundamental: Optional FundamentalSignal from FundamentalAgent.
            sentiment: Optional SentimentSignal from SentimentAgent.
            reliability: Optional agent name -> historical win rate in [0, 1]
                (see cultural.get_agent_accuracy), scaling that agent's vote
                weight by win_rate / 0.5. An agent at the 0.5 neutral prior
                (or missing/omitted entirely) votes at its unscaled base
                weight — agents that have been right in this regime count
                for more, agents that haven't count for less.

        Returns:
            AggregatedSignal with a consensus direction, conviction, and weighted vote breakdown.
        """
        ticker = (
            (technical.ticker if technical else None)
            or (fundamental.ticker if fundamental else None)
            or (sentiment.ticker if sentiment else None)
            or "UNKNOWN"
        )

        macro_mults = {}
        if macro:
            macro_mults = macro.agent_multipliers

        sources = {
            "fundamental": fundamental,
            "technical": technical,
            "sentiment": sentiment,
        }

        # Mass that could have been cast by all three agents, reliability excluded — the
        # denominator that makes conviction monotone in participation instead of saturating
        # whenever the agents who did vote happen to agree (see Design Decision A, issue #24)
        w_eff = {name: self.DEFAULT_WEIGHTS[name] * macro_mults.get(name, 1.0) for name in self.DEFAULT_WEIGHTS}
        max_pool = sum(w_eff.values())

        bull_pool: float = 0.0
        bear_pool: float = 0.0
        neutral_pool: float = 0.0
        weighted_votes: dict[str, float] = {}
        agents_present: list[str] = [name for name, signal in sources.items() if signal is not None]
        reliability_used: dict[str, float] = dict(reliability) if reliability else {}

        for name, signal in sources.items():
            if signal is None:
                weighted_votes[name] = 0.0
                continue

            reliability_mult = (reliability or {}).get(name, 0.5) / 0.5
            vote = signal.conviction * w_eff[name] * reliability_mult

            if signal.signal == Signal.BULLISH:
                bull_pool += vote
            elif signal.signal == Signal.BEARISH:
                bear_pool += vote
            else:
                neutral_pool += vote

            weighted_votes[name] = vote

        if bull_pool + bear_pool + neutral_pool == 0.0:
            logger.warning("[Aggregator] No signal contributions for %s", ticker)
            return AggregatedSignal(
                ticker=ticker,
                signal=Signal.NEUTRAL,
                conviction=0.0,
                weighted_votes=weighted_votes,
                agents_present=agents_present,
                reliability=reliability_used,
            )

        bull_pct = bull_pool / max_pool
        bear_pct = bear_pool / max_pool
        neutral_pct = neutral_pool / max_pool

        # A directional call needs a strict majority; an exact tie resolves NEUTRAL
        if bull_pct > bear_pct and bull_pct > neutral_pct:
            consensus = Signal.BULLISH
            conviction = bull_pct
        elif bear_pct > bull_pct and bear_pct > neutral_pct:
            consensus = Signal.BEARISH
            conviction = bear_pct
        else:
            consensus = Signal.NEUTRAL
            conviction = neutral_pct

        # CONTRACTION suppresses low-conviction BULLISH to avoid overallocating in stressed regimes
        if macro and macro.macro_regime == Regime.CONTRACTION:
            if consensus == Signal.BULLISH and conviction < AGGREGATOR.contraction_conviction_threshold:
                logger.info(
                    "[Aggregator] %s: Contraction regime override — suppressing BULLISH (conv=%.2f)",
                    ticker,
                    conviction,
                )
                consensus = Signal.NEUTRAL
                conviction = neutral_pct * AGGREGATOR.contraction_conviction_reduction

        conviction = min(conviction, AGGREGATOR.max_conviction)

        logger.info(
            "[Aggregator] %s: %s (conv=%.2f) bull=%.2f bear=%.2f neutral=%.2f",
            ticker,
            consensus.value,
            conviction,
            bull_pct,
            bear_pct,
            neutral_pct,
        )

        return AggregatedSignal(
            ticker=ticker,
            signal=consensus,
            conviction=conviction,
            weighted_votes=weighted_votes,
            agents_present=agents_present,
            reliability=reliability_used,
        )
