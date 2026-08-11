"""
argus/orchestration/aggregator.py

Multi-agent signal aggregation module implementing conflict arbitration.

Responsibilities:
  - Weight individual specialist signals by macro multipliers
  - Apply majority voting and override rules to produce a single AggregatedSignal
  - Flag split or contested votes for downstream portfolio handling

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
    ) -> AggregatedSignal:
        """Aggregates specialist signals via conviction-weighted voting with macro multipliers.

        When agents disagree (no majority wins), the `debate_triggered` flag is set.
        At least one of fundamental, technical, or sentiment must be non-None for
        a meaningful signal to be generated.

        Args:
            technical: Optional TechnicalSignal from TechnicalStatisticalAgent.
            macro: Optional MacroContext from MacroStatisticalAgent; used for multipliers.
            fundamental: Optional FundamentalSignal from FundamentalAgent.
            sentiment: Optional SentimentSignal from SentimentAgent.

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

        bull_pool: float = 0.0
        bear_pool: float = 0.0
        neutral_pool: float = 0.0
        weighted_votes: dict[str, float] = {}

        for name, signal in sources.items():
            if signal is None:
                weighted_votes[name] = 0.0
                continue

            base_w = self.DEFAULT_WEIGHTS[name]
            mult = macro_mults.get(name, 1.0)
            effective_w = base_w * mult
            vote = signal.conviction * effective_w

            if signal.signal == Signal.BULLISH:
                bull_pool += vote
            elif signal.signal == Signal.BEARISH:
                bear_pool += vote
            else:
                neutral_pool += vote

            weighted_votes[name] = vote

        total = bull_pool + bear_pool + neutral_pool

        if total == 0.0:
            logger.warning("[Aggregator] No signal contributions for %s", ticker)
            return AggregatedSignal(
                ticker=ticker,
                signal=Signal.NEUTRAL,
                conviction=0.0,
                weighted_votes=weighted_votes,
                debate_triggered=False,
                skip_reason="No agent signals available",
            )

        bull_pct = bull_pool / total
        bear_pct = bear_pool / total
        neutral_pct = neutral_pool / total

        debate_triggered = False
        max_pct = max(bull_pct, bear_pct, neutral_pct)

        # Debate threshold: no single direction dominates by > 10 pp over the runner-up
        runner_up = sorted([bull_pct, bear_pct, neutral_pct])[-2]
        if max_pct - runner_up < AGGREGATOR.debate_trigger_margin:
            debate_triggered = True

        if bull_pct >= bear_pct and bull_pct >= neutral_pct:
            consensus = Signal.BULLISH
            conviction = bull_pct
        elif bear_pct > bull_pct and bear_pct >= neutral_pct:
            consensus = Signal.BEARISH
            conviction = bear_pct
        else:
            consensus = Signal.NEUTRAL
            conviction = neutral_pct

        # Contract-based regime override: CONTRACTION suppresses BULLISH signals below a conviction
        # threshold of 0.70 to prevent the system from misallocating during elevated-stress regimes
        if macro and macro.macro_regime == Regime.CONTRACTION:
            if consensus == Signal.BULLISH and conviction < AGGREGATOR.contraction_conviction_threshold:
                logger.info(
                    "[Aggregator] %s: Contraction regime override — suppressing BULLISH (conv=%.2f)",
                    ticker,
                    conviction,
                )
                consensus = Signal.NEUTRAL
                conviction = conviction * AGGREGATOR.contraction_conviction_reduction

        conviction = min(conviction, AGGREGATOR.max_conviction)

        logger.info(
            "[Aggregator] %s: %s (conv=%.2f) bull=%.2f bear=%.2f neutral=%.2f debate=%s",
            ticker,
            consensus.value,
            conviction,
            bull_pct,
            bear_pct,
            neutral_pct,
            debate_triggered,
        )

        return AggregatedSignal(
            ticker=ticker,
            signal=consensus,
            conviction=conviction,
            weighted_votes=weighted_votes,
            debate_triggered=debate_triggered,
        )
