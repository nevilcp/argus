"""Statistical technical analysis agent executing offline metric calculations."""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from argus.config import settings
from argus.infra.clock import now_local
from argus.params import TECHNICAL
from argus.schemas.signals import (
    Signal,
    TechnicalSignal,
    missing_session_state_keys,
)

logger = logging.getLogger(__name__)


def _score_rsi(s: dict) -> float:
    """Evaluate the 14-period RSI and return a sentiment score in [-1, 1]."""
    rsi = float(s["rsi_14"])

    oversold = TECHNICAL.rsi_oversold
    bullish_transition = TECHNICAL.rsi_bullish_transition
    bullish_transition_score = TECHNICAL.rsi_bullish_transition_score
    overbought = TECHNICAL.rsi_overbought
    bearish_transition = TECHNICAL.rsi_bearish_transition
    neutral_low = TECHNICAL.rsi_neutral_low
    neutral_high = TECHNICAL.rsi_neutral_high

    if rsi < oversold:
        return 1.0
    if rsi < bullish_transition:
        return 1.0 - (rsi - oversold) / (bullish_transition - oversold) * (1.0 - bullish_transition_score)

    if rsi > overbought:
        return -1.0
    if rsi > bearish_transition:
        return -bullish_transition_score - (rsi - bearish_transition) / (overbought - bearish_transition) * (
            1.0 - bullish_transition_score
        )

    if neutral_low <= rsi <= neutral_high:
        return 0.0

    if bullish_transition <= rsi < neutral_low:
        return bullish_transition_score * (1.0 - (rsi - bullish_transition) / (neutral_low - bullish_transition))

    return -bullish_transition_score * (rsi - neutral_high) / (bearish_transition - neutral_high)


def _score_macd(s: dict) -> float:
    """Normalize the MACD histogram to a capped directional score."""
    hist = float(s["macd_histogram"])
    return float(np.clip(hist / TECHNICAL.macd_normalization_scale, -1.0, 1.0))


def _score_bollinger(s: dict) -> float:
    """Score Bollinger band breaches as stronger mean-reversion evidence."""
    bb = float(s["bb_percent_b"])

    if bb < 0.0:
        return min(1.0, abs(bb) * TECHNICAL.bollinger_breach_multiplier + TECHNICAL.bollinger_breach_boost)

    if bb > 1.0:
        return max(
            -1.0,
            -(bb - 1.0) * TECHNICAL.bollinger_breach_multiplier - TECHNICAL.bollinger_breach_boost,
        )

    return 0.5 - bb


def _score_adx_amplified(s: dict, base_direction: float) -> float:
    """Amplify or dampens indicator directional consensus based on ADX-14 trend strength."""
    adx = float(s["adx_14"])
    dampen_threshold = TECHNICAL.adx_dampen_threshold
    dampen_multiplier = TECHNICAL.adx_dampen_multiplier
    amplify_threshold = TECHNICAL.adx_amplify_threshold
    amplify_multiplier = TECHNICAL.adx_amplify_multiplier

    if adx < dampen_threshold:
        multiplier = dampen_multiplier
    elif adx > amplify_threshold:
        multiplier = amplify_multiplier
    else:
        multiplier = dampen_multiplier + (adx - dampen_threshold) / (amplify_threshold - dampen_threshold) * (
            amplify_multiplier - dampen_multiplier
        )

    return float(np.clip(base_direction * multiplier, -1.0, 1.0))


def _score_vwap(s: dict) -> float:
    """Scale price deviation from VWAP into a capped directional score."""
    d = float(s["vwap_distance"])
    return float(np.clip(d / TECHNICAL.vwap_spread_normalization, -1.0, 1.0))


def _score_momentum(s: dict) -> float:
    """Evaluate 30-minute and 1-day momentum, validating direction concordance."""
    m30 = float(s["momentum_30m"])
    m1d = float(s["momentum_1d"])

    if m30 * m1d > 0.0:
        combined = m30 * TECHNICAL.momentum_30m_weight + m1d * TECHNICAL.momentum_1d_weight
        return float(np.clip(combined / TECHNICAL.momentum_saturation_bound, -1.0, 1.0))

    return 0.0


def _clamp_rsi_extreme(ticker: str, rsi: float, net_score: float) -> float:
    """Zero a net score that contradicts an RSI extreme, e.g. bullish while overbought."""
    overbought = rsi > TECHNICAL.rsi_overbought and net_score > 0.0
    oversold = rsi < TECHNICAL.rsi_oversold and net_score < 0.0
    if not (overbought or oversold):
        return net_score
    logger.debug(
        "analyze[%s] RSI=%.2f %s - clamping net_score %.4f -> 0.0",
        ticker,
        rsi,
        "overbought" if overbought else "oversold",
        net_score,
    )
    return 0.0


def _classify(net_score: float) -> tuple[Signal, float]:
    """Map a net score to a signal direction and its conviction."""
    abs_score = abs(net_score)
    if abs_score < TECHNICAL.net_score_neutral_threshold:
        return Signal.NEUTRAL, TECHNICAL.neutral_conviction_floor + abs_score
    conviction = min(
        TECHNICAL.conviction_base + abs_score * TECHNICAL.conviction_score_multiplier,
        TECHNICAL.conviction_ceiling,
    )
    return (Signal.BULLISH if net_score > 0 else Signal.BEARISH), conviction


class TechnicalStatisticalAgent:
    """Agent translating technical indicator matrices into validated trading signals."""

    WEIGHTS: ClassVar[dict[str, float]] = dict(settings.TECHNICAL_INDICATOR_WEIGHTS)

    def analyze(self, ticker: str, session_state: dict) -> TechnicalSignal | None:
        """Analyze a single ticker's session state parameters to generate a TechnicalSignal."""
        missing = missing_session_state_keys(session_state)
        if missing:
            logger.warning(
                "analyze[%s]: missing or non-finite required indicator fields %s - skipping to avoid fake signal",
                ticker,
                missing,
            )
            return None

        s = session_state
        net_score = _clamp_rsi_extreme(ticker, float(s["rsi_14"]), self._net_score(ticker, s))
        signal, conviction = _classify(net_score)

        logger.debug(
            "analyze[%s]: signal=%s conviction=%.3f net_score=%.4f",
            ticker,
            signal.value,
            conviction,
            net_score,
        )

        return TechnicalSignal(
            ticker=ticker,
            current_price=float(s["close"]),
            rsi_14=float(s["rsi_14"]),
            macd_histogram=float(s["macd_histogram"]),
            bb_percent_b=float(s["bb_percent_b"]),
            atr_pct=float(s["atr_pct"]),
            adx_14=float(s["adx_14"]),
            vwap_distance=float(s["vwap_distance"]),
            volume_ratio=float(s["volume_ratio"]),
            momentum_30m=float(s["momentum_30m"]),
            momentum_1d=float(s["momentum_1d"]),
            signal=signal,
            conviction=conviction,
            net_score=net_score,
            api_calls_used=0,
            timestamp=now_local(),
        )

    def _net_score(self, ticker: str, s: dict) -> float:
        """Combine the six indicator scores and the volume modifier into one net score."""
        rsi_score = _score_rsi(s)
        macd_score = _score_macd(s)
        bb_score = _score_bollinger(s)

        w = self.WEIGHTS
        base_weight_sum = w["rsi"] + w["macd"] + w["bb"]
        base_direction = (rsi_score * w["rsi"] + macd_score * w["macd"] + bb_score * w["bb"]) / base_weight_sum

        scores: dict[str, float] = {
            "rsi": rsi_score,
            "macd": macd_score,
            "bb": bb_score,
            "adx": _score_adx_amplified(s, base_direction),
            "vwap": _score_vwap(s),
            "momentum": _score_momentum(s),
        }
        logger.debug(
            "analyze[%s] scores: %s",
            ticker,
            {k: round(v, 3) for k, v in scores.items()},
        )

        net_score = sum(scores[k] * w[k] for k in scores) / sum(w.values())
        volume_modifier = min(
            max(
                float(s["volume_ratio"]) / TECHNICAL.volume_ratio_divisor,
                TECHNICAL.volume_modifier_floor,
            ),
            TECHNICAL.volume_modifier_ceiling,
        )
        logger.debug("analyze[%s] net_score=%.4f (vol_mod=%.2f)", ticker, net_score, volume_modifier)
        return float(np.clip(net_score * volume_modifier, -1.0, 1.0))

    def batch_analyze(self, session_states: dict[str, dict]) -> tuple[dict[str, TechnicalSignal], list[str]]:
        """Evaluate technical indicators sequentially across a set of tickers."""
        results: dict[str, TechnicalSignal] = {}
        errors: list[str] = []
        for ticker, state in session_states.items():
            try:
                signal = self.analyze(ticker, state)
                if signal is not None:
                    results[ticker] = signal
                else:
                    errors.append(f"technical_analysis[{ticker}]: missing required indicator fields")
            except Exception as exc:
                logger.warning(
                    "%s failed - %s: %s",
                    ticker,
                    type(exc).__name__,
                    exc,
                )
                errors.append(f"technical_analysis[{ticker}]: {type(exc).__name__}: {exc}")
        logger.debug(
            "%d/%d tickers produced valid signals",
            len(results),
            len(session_states),
        )
        return results, errors
