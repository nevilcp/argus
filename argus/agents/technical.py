"""
argus/agents/technical.py

Statistical technical analysis agent executing offline metric calculations.

Responsibilities:
  - Consume compressed multi-frequency data frames (rsi, macd, bollinger, adx, vwap, momentum)
  - Compute a consolidated score per ticker using configurable indicator weights
  - Map directional trade postures without LLM cost or remote API dependency

Not responsible for:
  - Fetching or buffering raw price data (see data/pipeline.py)
  - Risk constraint enforcement (see agents/risk.py)
  - Signal aggregation across agents (see orchestration/aggregator.py)

Dependencies:
  - numpy
  - argus.config.settings.TECHNICAL_INDICATOR_WEIGHTS
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from argus.config import settings
from argus.params import TECHNICAL
from argus.schemas.signals import Signal, TechnicalSignal

logger = logging.getLogger("argus.technical")

# Minimum indicator keys required for a trustworthy signal. analyze() returns
# None (instead of computing from fabricated defaults) when any key is absent.
_REQUIRED_INDICATOR_KEYS: frozenset[str] = frozenset({
    "rsi_14",
    "macd_histogram",
    "bb_percent_b",
    "adx_14",
    "vwap_distance",
    "momentum_30m",
    "momentum_1d",
    "close",
})


def _score_rsi(s: dict) -> float:
    """Evaluates the 14-period RSI and returns a sentiment score in [-1, 1].

    Camps at extreme thresholds (< 25 oversold, > 75 overbought) then
    linearly interpolates through transition bands. A neutral dead-zone
    of ±0.2 score is preserved around RSI 50 to reduce noise.

    Args:
        s: Session state dict containing the key ``rsi_14``.

    Returns:
        Signed score where +1.0 = maximum bullish, -1.0 = maximum bearish.
    """
    rsi = float(s.get("rsi_14", 50.0))

    oversold = TECHNICAL.rsi_oversold
    bullish_transition = TECHNICAL.rsi_bullish_transition
    bullish_transition_score = TECHNICAL.rsi_bullish_transition_score
    overbought = TECHNICAL.rsi_overbought
    bearish_transition = TECHNICAL.rsi_bearish_transition
    neutral_low = TECHNICAL.rsi_neutral_low
    neutral_high = TECHNICAL.rsi_neutral_high
    neutral_span = TECHNICAL.rsi_neutral_span

    if rsi < oversold:
        return 1.0
    if rsi < bullish_transition:
        return 1.0 - (rsi - oversold) / (bullish_transition - oversold) * (1.0 - bullish_transition_score)

    if rsi > overbought:
        return -1.0
    if rsi > bearish_transition:
        return -bullish_transition_score - (rsi - bearish_transition) / (overbought - bearish_transition) * (1.0 - bullish_transition_score)

    if neutral_low <= rsi <= neutral_high:
        return (rsi - 50.0) / neutral_span

    if bullish_transition <= rsi < neutral_low:
        lo, hi = bullish_transition_score, (neutral_low - 50.0) / neutral_span
        return lo + (rsi - bullish_transition) / (neutral_low - bullish_transition) * (hi - lo)

    # Bearish transition band: RSI in [55, 70] (implicit else-branch after all earlier guards)
    lo, hi = (neutral_high - 50.0) / neutral_span, -bullish_transition_score
    return lo + (rsi - neutral_high) / (bearish_transition - neutral_high) * (hi - lo)


def _score_macd(s: dict) -> float:
    """Normalizes the MACD histogram against a 0.5 scale factor, capping to [-1.0, 1.0].

    Args:
        s: Session state dict containing the key ``macd_histogram``.

    Returns:
        Clipped score in [-1.0, 1.0].
    """
    hist = float(s.get("macd_histogram", 0.0))
    return float(np.clip(hist / TECHNICAL.macd_normalization_scale, -1.0, 1.0))


def _score_bollinger(s: dict) -> float:
    """Scores Bollinger %B, treating extreme band breaches as high-conviction mean-reversion signals.

    Breaches below 0.0 (below lower band) are scored bullish; breaches above
    1.0 (above upper band) are scored bearish, with a 0.5 floor/ceiling boost
    to reflect the non-linearity of extreme extensions.

    Args:
        s: Session state dict containing the key ``bb_percent_b``.

    Returns:
        Signed score in [-1.0, 1.0].
    """
    bb = float(s.get("bb_percent_b", 0.5))

    if bb < 0.0:
        return min(1.0, abs(bb) * TECHNICAL.bollinger_breach_multiplier + TECHNICAL.bollinger_breach_boost)

    if bb > 1.0:
        return max(-1.0, -(bb - 1.0) * TECHNICAL.bollinger_breach_multiplier - TECHNICAL.bollinger_breach_boost)

    return 0.5 - bb


def _score_adx_amplified(s: dict, base_direction: float) -> float:
    """Amplifies or dampens indicator directional consensus based on ADX-14 trend strength.

    Dampens signal weight below 20 (multiplier 0.6) and amplifies it above 40
    (multiplier 1.2), linearly interpolating the mid-range. This prevents the
    system from over-weighting directional signals during choppy, trendless markets.

    Args:
        s: Session state dict containing the key ``adx_14``.
        base_direction: Pre-computed weighted direction from primary indicators in [-1, 1].

    Returns:
        Trend-strength-adjusted score clipped to [-1.0, 1.0].
    """
    adx = float(s.get("adx_14", 25.0))
    dampen_threshold = TECHNICAL.adx_dampen_threshold
    dampen_multiplier = TECHNICAL.adx_dampen_multiplier
    amplify_threshold = TECHNICAL.adx_amplify_threshold
    amplify_multiplier = TECHNICAL.adx_amplify_multiplier

    if adx < dampen_threshold:
        multiplier = dampen_multiplier
    elif adx > amplify_threshold:
        multiplier = amplify_multiplier
    else:
        multiplier = dampen_multiplier + (adx - dampen_threshold) / (amplify_threshold - dampen_threshold) * (amplify_multiplier - dampen_multiplier)

    return float(np.clip(base_direction * multiplier, -1.0, 1.0))


def _score_vwap(s: dict) -> float:
    """Calculates price deviation from VWAP, scaling a ±1.5% spread to [-1.0, 1.0].

    Args:
        s: Session state dict containing the key ``vwap_distance``.

    Returns:
        Clipped score in [-1.0, 1.0].
    """
    d = float(s.get("vwap_distance", 0.0))
    return float(np.clip(d / TECHNICAL.vwap_spread_normalization, -1.0, 1.0))


def _score_momentum(s: dict) -> float:
    """Evaluates 30-minute and 1-day momentum, validating direction concordance.

    Requires directional confluence (same sign) before combining signals; conflicting
    timeframes cancel to zero to avoid noise-driven allocations. Applies a 60/40 weight
    skew favouring the 1-day interval, saturated at ±2% bounds.

    Args:
        s: Session state dict containing keys ``momentum_30m`` and ``momentum_1d``.

    Returns:
        Signed score in [-1.0, 1.0], or 0.0 if timeframes conflict.
    """
    m30 = float(s.get("momentum_30m", 0.0))
    m1d = float(s.get("momentum_1d", 0.0))

    # Positive product means both values share the same sign (directional confluence)
    if m30 * m1d > 0.0:
        combined = m30 * TECHNICAL.momentum_30m_weight + m1d * TECHNICAL.momentum_1d_weight
        return float(np.clip(combined / TECHNICAL.momentum_saturation_bound, -1.0, 1.0))

    return 0.0


class TechnicalStatisticalAgent:
    """Agent translating technical indicator matrices into validated trading signals.

    Calculates all indicators locally without remote API dependency or LLM inference,
    making it the cheapest and fastest specialist in the ARGUS pipeline.
    """

    WEIGHTS: dict[str, float] = dict(settings.TECHNICAL_INDICATOR_WEIGHTS)

    def analyze(self, ticker: str, session_state: dict) -> Optional[TechnicalSignal]:
        """Analyzes a single ticker's session state parameters to generate a TechnicalSignal.

        Returns None if any required indicator key is absent or None in session_state,
        preventing signal computation from fabricated defaults.

        Args:
            ticker: Equity ticker symbol (e.g. 'AAPL').
            session_state: Dict of pre-computed technical indicator values. Required keys:
                rsi_14, macd_histogram, bb_percent_b, adx_14, vwap_distance,
                volume_ratio, momentum_30m, momentum_1d, close.

        Returns:
            A validated TechnicalSignal, or None if required fields are missing.
        """
        missing = [
            k for k in _REQUIRED_INDICATOR_KEYS
            if session_state.get(k) is None
        ]
        if missing:
            logger.warning(
                "analyze[%s]: missing required indicator fields %s — skipping to avoid fake signal",
                ticker,
                sorted(missing),
            )
            return None

        s = session_state

        rsi_score = _score_rsi(s)
        macd_score = _score_macd(s)
        bb_score = _score_bollinger(s)
        vwap_score = _score_vwap(s)
        mom_score = _score_momentum(s)

        # Base direction derived from primary indicators drives ADX amplification
        w = self.WEIGHTS
        base_weight_sum = w["rsi"] + w["macd"] + w["bb"]
        base_direction = (
            rsi_score * w["rsi"] + macd_score * w["macd"] + bb_score * w["bb"]
        ) / base_weight_sum
        adx_score = _score_adx_amplified(s, base_direction)

        scores: dict[str, float] = {
            "rsi": rsi_score,
            "macd": macd_score,
            "bb": bb_score,
            "adx": adx_score,
            "vwap": vwap_score,
            "momentum": mom_score,
        }

        logger.debug(
            "analyze[%s] scores: %s",
            ticker,
            {k: round(v, 3) for k, v in scores.items()},
        )

        total_weight = sum(w.values())
        net_score = sum(scores[k] * w[k] for k in scores) / total_weight

        vol_ratio = float(s.get("volume_ratio", 1.0))
        volume_modifier = min(
            max(vol_ratio / TECHNICAL.volume_ratio_divisor, TECHNICAL.volume_modifier_floor),
            TECHNICAL.volume_modifier_ceiling,
        )
        net_score = float(np.clip(net_score * volume_modifier, -1.0, 1.0))

        logger.debug(
            "analyze[%s] net_score=%.4f (vol_mod=%.2f)", ticker, net_score, volume_modifier
        )

        # RSI extremes override net_score direction to prevent contradictory signals
        # (e.g., a bullish net score while RSI is deeply overbought)
        rsi_raw = float(s.get("rsi_14", 50.0))
        if rsi_raw > TECHNICAL.rsi_overbought and net_score > 0.0:
            logger.debug(
                "analyze[%s] RSI=%.2f >75 overbought — clamping net_score %.4f → 0.0",
                ticker,
                rsi_raw,
                net_score,
            )
            net_score = 0.0
        elif rsi_raw < TECHNICAL.rsi_oversold and net_score < 0.0:
            logger.debug(
                "analyze[%s] RSI=%.2f <25 oversold — clamping net_score %.4f → 0.0",
                ticker,
                rsi_raw,
                net_score,
            )
            net_score = 0.0

        abs_score = abs(net_score)

        if abs_score < TECHNICAL.net_score_neutral_threshold:
            signal = Signal.NEUTRAL
            conviction = TECHNICAL.neutral_conviction_floor + abs_score
        elif net_score > 0:
            signal = Signal.BULLISH
            conviction = min(
                TECHNICAL.conviction_base + abs_score * TECHNICAL.conviction_score_multiplier,
                TECHNICAL.conviction_ceiling,
            )
        else:
            signal = Signal.BEARISH
            conviction = min(
                TECHNICAL.conviction_base + abs_score * TECHNICAL.conviction_score_multiplier,
                TECHNICAL.conviction_ceiling,
            )

        logger.debug(
            "analyze[%s]: signal=%s conviction=%.3f net_score=%.4f",
            ticker,
            signal.value,
            conviction,
            net_score,
        )

        return TechnicalSignal(
            ticker=ticker,
            current_price=float(s.get("close", 0.0)),
            rsi_14=float(s.get("rsi_14", 50.0)),
            macd_histogram=float(s.get("macd_histogram", 0.0)),
            bb_percent_b=float(s.get("bb_percent_b", 0.5)),
            atr_pct=float(s.get("atr_pct", 0.0)),
            adx_14=float(s.get("adx_14", 25.0)),
            vwap_distance=float(s.get("vwap_distance", 0.0)),
            volume_ratio=float(s.get("volume_ratio", 1.0)),
            momentum_30m=float(s.get("momentum_30m", 0.0)),
            momentum_1d=float(s.get("momentum_1d", 0.0)),
            signal=signal,
            conviction=conviction,
            net_score=net_score,
            api_calls_used=0,
            timestamp=datetime.now(),
        )

    def batch_analyze(self, session_states: dict[str, dict]) -> dict[str, TechnicalSignal]:
        """Evaluates technical indicators sequentially across a set of tickers.

        Tickers that return None from analyze() (missing required fields) or raise
        an exception are excluded from the output rather than contributing fake signals.

        Args:
            session_states: Mapping of ticker → session state dict.

        Returns:
            Mapping of ticker → TechnicalSignal for each successfully analyzed ticker.
            Tickers with missing data or errors are omitted and logged as warnings.
        """
        results: dict[str, TechnicalSignal] = {}
        for ticker, state in session_states.items():
            try:
                signal = self.analyze(ticker, state)
                if signal is not None:
                    results[ticker] = signal
            except Exception as exc:
                logger.warning(
                    "batch_analyze: %s failed — %s: %s",
                    ticker,
                    type(exc).__name__,
                    exc,
                )
        logger.debug(
            "batch_analyze: %d/%d tickers produced valid signals",
            len(results),
            len(session_states),
        )
        return results
