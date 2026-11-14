"""Specialist consensus with macro and observed reliability weights."""

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

# Keep specialist voting order shared by aggregation, reliability, and attribution.
SPECIALIST_AGENTS = ("fundamental", "technical", "sentiment")


class HybridSignalAggregator:
    """Consensus from conviction-weighted specialist votes, macro multipliers, and outcome reliability."""

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
        """Aggregate specialist signals via conviction-weighted voting with macro multipliers."""
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
        # Normalize against all available vote capacity so conviction increases with participation.
        w_eff = {name: weight * macro_mults.get(name, 1.0) for name, weight in self.DEFAULT_WEIGHTS.items()}
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
                "%s: Contraction regime override - suppressing BULLISH (conv=%.2f)",
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
    """Tally each present agent's weighted vote into a per-direction pool."""
    pools = {Signal.BULLISH: 0.0, Signal.BEARISH: 0.0, Signal.NEUTRAL: 0.0}
    weighted_votes: dict[str, float] = {}
    for name, signal in sources.items():
        if signal is None:
            weighted_votes[name] = 0.0
            continue
        reliability_mult = reliability.get(name, AGGREGATOR.neutral_reliability) / AGGREGATOR.neutral_reliability
        vote = signal.conviction * w_eff[name] * reliability_mult
        pools[signal.signal] += vote
        weighted_votes[name] = vote
    return pools, weighted_votes
