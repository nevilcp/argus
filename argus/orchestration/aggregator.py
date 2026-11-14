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
from typing import ClassVar

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

logger = logging.getLogger(__name__)

# The agents that cast a vote, in voting order. Reliability and ablation
# iterate this; the weights and sources below are keyed by the same names.
SPECIALIST_AGENTS = ("fundamental", "technical", "sentiment")


class HybridSignalAggregator:
    """Arbitrates and fuses independent specialist signals into a consensus AggregatedSignal.

    Each agent casts one conviction-weighted directional vote. Macro multipliers
    scale its base weight at runtime; the highest-weighted signal pool wins.

    Attributes:
        DEFAULT_WEIGHTS: Base per-agent vote weight for fundamental, technical,
            and sentiment, before macro multipliers and reliability scaling
            are applied.
    """

    DEFAULT_WEIGHTS: ClassVar[dict[str, float]] = {
        "fundamental": AGGREGATOR.weight_fundamental,
        "technical": AGGREGATOR.weight_technical,
        "sentiment": AGGREGATOR.weight_sentiment,
    }

    def aggregate(
        self,
        technical: TechnicalSignal | None,
        macro: MacroContext | None,
        fundamental: FundamentalSignal | None,
        sentiment: SentimentSignal | None,
        reliability: dict[str, float] | None = None,
        reliability_n: dict[str, int] | None = None,
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
            reliability_n: Optional agent name -> raw sample count each reliability
                rate was computed from (see cultural.get_agent_accuracy's n).
                Carried onto the result untouched, for agents/portfolio.py's
                Half-Kelly anchor to distinguish "no data" from "measured 0.5".

        Returns:
            AggregatedSignal with a consensus direction, conviction, and weighted vote breakdown.
        """
        ticker = (
            (technical.ticker if technical else None)
            or (fundamental.ticker if fundamental else None)
            or (sentiment.ticker if sentiment else None)
            or "UNKNOWN"
        )
        reliability_used: dict[str, float] = dict(reliability) if reliability else {}
        reliability_n_used: dict[str, int] = dict(reliability_n) if reliability_n else {}

        sources = {
            "fundamental": fundamental,
            "technical": technical,
            "sentiment": sentiment,
        }
        macro_mults = macro.agent_multipliers if macro else {}
        # Mass that could have been cast by all three agents, reliability excluded — the
        # denominator that makes conviction monotone in participation instead of saturating
        # whenever the agents who did vote happen to agree
        w_eff = {
            name: weight * macro_mults.get(name, 1.0)
            for name, weight in self.DEFAULT_WEIGHTS.items()
        }
        max_pool = sum(w_eff.values())

        pools, weighted_votes = _cast_votes(sources, w_eff, reliability_used)

        def build(signal: Signal, conviction: float) -> AggregatedSignal:
            return AggregatedSignal(
                ticker=ticker,
                signal=signal,
                conviction=conviction,
                weighted_votes=weighted_votes,
                agents_present=[name for name, sig in sources.items() if sig is not None],
                reliability=reliability_used,
                reliability_n=reliability_n_used,
            )

        if sum(pools.values()) == 0.0:
            logger.warning("No signal contributions for %s", ticker)
            return build(Signal.NEUTRAL, 0.0)

        bull_pct = pools[Signal.BULLISH] / max_pool
        bear_pct = pools[Signal.BEARISH] / max_pool
        neutral_pct = pools[Signal.NEUTRAL] / max_pool

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
        if (
            macro
            and macro.macro_regime == Regime.CONTRACTION
            and consensus == Signal.BULLISH
            and conviction < AGGREGATOR.contraction_conviction_threshold
        ):
            logger.info(
                "%s: Contraction regime override — suppressing BULLISH (conv=%.2f)",
                ticker,
                conviction,
            )
            consensus = Signal.NEUTRAL
            conviction = neutral_pct * AGGREGATOR.contraction_conviction_reduction

        conviction = min(conviction, AGGREGATOR.max_conviction)

        logger.info(
            "%s: %s (conv=%.2f) bull=%.2f bear=%.2f neutral=%.2f",
            ticker,
            consensus.value,
            conviction,
            bull_pct,
            bear_pct,
            neutral_pct,
        )
        return build(consensus, conviction)


def _cast_votes(
    sources: dict[str, TechnicalSignal | FundamentalSignal | SentimentSignal | None],
    w_eff: dict[str, float],
    reliability: dict[str, float],
) -> tuple[dict[Signal, float], dict[str, float]]:
    """Tallies each present agent's weighted vote into a per-direction pool.

    Args:
        sources: Agent name -> its signal, or None when the agent produced none.
        w_eff: Agent name -> base weight after its macro multiplier.
        reliability: Agent name -> historical win rate; a missing agent votes at
            the neutral prior, i.e. unscaled.

    Returns:
        Tuple of (summed vote per direction, each agent's own weighted vote with
        0.0 for an absent agent).
    """
    pools = {Signal.BULLISH: 0.0, Signal.BEARISH: 0.0, Signal.NEUTRAL: 0.0}
    weighted_votes: dict[str, float] = {}
    for name, signal in sources.items():
        if signal is None:
            weighted_votes[name] = 0.0
            continue
        reliability_mult = (
            reliability.get(name, AGGREGATOR.neutral_reliability) / AGGREGATOR.neutral_reliability
        )
        vote = signal.conviction * w_eff[name] * reliability_mult
        pools[signal.signal] += vote
        weighted_votes[name] = vote
    return pools, weighted_votes
