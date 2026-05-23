"""
argus/agents/technical.py
=========================
Technical Analysis Agent — purely statistical, zero LLM calls, zero API cost.

Consumes a ``session_state`` dict produced by ``MFTDataPipeline.compress_all()``
and scores six indicator families (RSI, MACD, Bollinger Bands, ADX, VWAP,
Momentum).  Each score is a float in [-1, +1]; positive = bullish.

The scores are combined via ``config.TECHNICAL_INDICATOR_WEIGHTS``, modulated
by a volume-confirmation factor, and discretised into a
:class:`~argus.schemas.signals.TechnicalSignal` Pydantic model.

No external network calls are made.  ``api_calls_used`` is always 0.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

from argus.config import settings
from argus.schemas.signals import Signal, TechnicalSignal

logger = logging.getLogger("argus.technical")


# ──────────────────────────────────────────────────────────────────────────────
# Indicator scoring functions
# Each receives the full session_state dict and returns float ∈ [-1, +1].
# Positive → bullish, negative → bearish.
# ──────────────────────────────────────────────────────────────────────────────


def _score_rsi(s: dict) -> float:
    """
    Score the 14-period RSI.

    Thresholds
    ----------
    RSI < 25 → extreme oversold → +1.0
    RSI < 30 → oversold        → +0.85
    RSI > 75 → extreme overbought → -1.0
    RSI > 70 → overbought      → -0.85
    45 ≤ RSI ≤ 55 → mild directional signal (linear around 50)
    All other regions interpolated linearly between the nearest thresholds.
    """
    rsi = float(s.get("rsi_14", 50.0))

    # Extreme oversold
    if rsi < 25:
        return 1.0
    if rsi < 30:
        # Linear: 25→1.0, 30→0.85
        return 1.0 - (rsi - 25) / (30 - 25) * (1.0 - 0.85)

    # Extreme overbought
    if rsi > 75:
        return -1.0
    if rsi > 70:
        # Linear: 70→-0.85, 75→-1.0
        return -0.85 - (rsi - 70) / (75 - 70) * (1.0 - 0.85)

    # Mild directional: 45–55 range
    if 45.0 <= rsi <= 55.0:
        return (rsi - 50.0) / 25.0

    # Between 30 and 45: oversold side fading to neutral
    if 30.0 <= rsi < 45.0:
        # 30→0.85, 45→(45-50)/25=-0.2
        lo, hi = 0.85, (45.0 - 50.0) / 25.0
        return lo + (rsi - 30.0) / (45.0 - 30.0) * (hi - lo)

    # Between 55 and 70: neutral fading to overbought
    # 55→(55-50)/25=0.2, 70→-0.85
    lo, hi = (55.0 - 50.0) / 25.0, -0.85
    return lo + (rsi - 55.0) / (70.0 - 55.0) * (hi - lo)


def _score_macd(s: dict) -> float:
    """
    Score the MACD histogram.

    Normalises the raw histogram value to [-1, +1] using a divisor of 0.5.
    (Positive histogram → bullish momentum above signal line.)
    """
    hist = float(s.get("macd_histogram", 0.0))
    return float(np.clip(hist / 0.5, -1.0, 1.0))


def _score_bollinger(s: dict) -> float:
    """
    Score the Bollinger %B.

    %B < 0 → price is below the lower band → bullish mean-reversion signal.
    %B > 1 → price is above the upper band → bearish mean-reversion signal.
    0–1    → proportional distance from the midline (0.5 = midband).
    """
    bb = float(s.get("bb_percent_b", 0.5))

    if bb < 0.0:
        # Below lower band — stronger the breach, stronger the bull signal
        return min(1.0, abs(bb) * 2.0 + 0.5)

    if bb > 1.0:
        # Above upper band — stronger the breach, stronger the bear signal
        return max(-1.0, -(bb - 1.0) * 2.0 - 0.5)

    # Within bands: distance from midline
    return 0.5 - bb


def _score_adx_amplified(s: dict, base_direction: float) -> float:
    """
    Amplify or dampen *base_direction* based on ADX trend strength.

    ADX does not produce a directional signal on its own; it scales the
    pre-computed direction from the other indicators.

    Multiplier schedule
    -------------------
    ADX < 20  → weak trend  → multiply by 0.6
    ADX > 40  → strong trend → multiply by 1.2  (capped at ±1.0 after scaling)
    20–40     → linear interpolation between 0.6 and 1.2
    """
    adx = float(s.get("adx_14", 25.0))

    if adx < 20.0:
        multiplier = 0.6
    elif adx > 40.0:
        multiplier = 1.2
    else:
        # Linear: adx=20 → 0.6, adx=40 → 1.2
        multiplier = 0.6 + (adx - 20.0) / 20.0 * 0.6

    return float(np.clip(base_direction * multiplier, -1.0, 1.0))


def _score_vwap(s: dict) -> float:
    """
    Score the distance of the current price from VWAP.

    A positive distance (price above VWAP) is bullish intraday; negative is
    bearish.  ±1.5% is normalised to ±1.0.
    """
    d = float(s.get("vwap_distance", 0.0))
    return float(np.clip(d / 0.015, -1.0, 1.0))


def _score_momentum(s: dict) -> float:
    """
    Score 30-minute and 1-day momentum with confluence check.

    If both momentum fields point in the same direction (product > 0), a
    combined score is computed weighting 1-day momentum more heavily (60 %).
    If they conflict (product ≤ 0), the signal is cancelled out (returns 0.0).

    Normalised to [-1, +1] with ±2 % as the saturation level.
    """
    m30 = float(s.get("momentum_30m", 0.0))
    m1d = float(s.get("momentum_1d", 0.0))

    if m30 * m1d > 0.0:  # same sign → confluence
        combined = m30 * 0.4 + m1d * 0.6
        return float(np.clip(combined / 0.02, -1.0, 1.0))

    return 0.0  # conflicting signals cancel


# ──────────────────────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────────────────────


class TechnicalStatisticalAgent:
    """
    Purely statistical technical analysis agent.

    Consumes the ``session_state`` dict emitted by
    :meth:`~argus.data.pipeline.MFTDataPipeline.compress_all` and returns a
    fully-validated :class:`~argus.schemas.signals.TechnicalSignal`.

    No LLM is invoked; ``api_calls_used`` is always 0.

    Attributes
    ----------
    WEIGHTS:
        Indicator weight mapping loaded from
        ``config.TECHNICAL_INDICATOR_WEIGHTS``.
    """

    WEIGHTS: dict[str, float] = dict(settings.TECHNICAL_INDICATOR_WEIGHTS)

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyze(self, ticker: str, session_state: dict) -> TechnicalSignal:
        """
        Analyse a single ticker's session state and return a TechnicalSignal.

        Parameters
        ----------
        ticker:
            Equity symbol being analysed.
        session_state:
            Feature dict from ``MFTDataPipeline._compress_candles()``.

        Returns
        -------
        TechnicalSignal
            Fully validated Pydantic model ready for injection into AgentState.
        """
        s = session_state  # shorthand

        # ── Step 1: Compute indicator scores ──────────────────────────────────
        rsi_score  = _score_rsi(s)
        macd_score = _score_macd(s)
        bb_score   = _score_bollinger(s)
        vwap_score = _score_vwap(s)
        mom_score  = _score_momentum(s)

        # ADX needs a base direction from the primary indicators
        w = self.WEIGHTS
        base_weight_sum = w["rsi"] + w["macd"] + w["bb"]
        base_direction  = (
            rsi_score  * w["rsi"]  +
            macd_score * w["macd"] +
            bb_score   * w["bb"]
        ) / base_weight_sum
        adx_score = _score_adx_amplified(s, base_direction)

        scores: dict[str, float] = {
            "rsi":      rsi_score,
            "macd":     macd_score,
            "bb":       bb_score,
            "adx":      adx_score,
            "vwap":     vwap_score,
            "momentum": mom_score,
        }

        logger.debug(
            "analyze[%s] scores: %s",
            ticker,
            {k: round(v, 3) for k, v in scores.items()},
        )

        # ── Step 2: Weighted aggregate ────────────────────────────────────────
        total_weight = sum(w.values())
        net_score    = sum(scores[k] * w[k] for k in scores) / total_weight

        # ── Step 3: Volume confirmation modifier ──────────────────────────────
        vol_ratio       = float(s.get("volume_ratio", 1.0))
        volume_modifier = min(max(vol_ratio / 1.5, 0.7), 1.3)
        net_score       = float(np.clip(net_score * volume_modifier, -1.0, 1.0))

        logger.debug(
            "analyze[%s] net_score=%.4f (vol_mod=%.2f)", ticker, net_score, volume_modifier
        )

        # ── Step 4: Discretise to Signal + conviction ─────────────────────────
        abs_score = abs(net_score)

        if abs_score < 0.12:
            signal     = Signal.NEUTRAL
            conviction = 0.30 + abs_score
        elif net_score > 0:
            signal     = Signal.BULLISH
            conviction = min(0.40 + abs_score * 0.55, 0.92)
        else:
            signal     = Signal.BEARISH
            conviction = min(0.40 + abs_score * 0.55, 0.92)

        logger.info(
            "analyze[%s]: signal=%s conviction=%.3f net_score=%.4f",
            ticker, signal.value, conviction, net_score,
        )

        # ── Step 5: Build and return TechnicalSignal ──────────────────────────
        return TechnicalSignal(
            ticker          = ticker,
            signal          = signal,
            conviction      = conviction,
            net_score       = net_score,
            rsi_14          = float(s.get("rsi_14",         50.0)),
            macd_histogram  = float(s.get("macd_histogram",  0.0)),
            bb_percent_b    = float(s.get("bb_percent_b",    0.5)),
            atr_pct         = float(s.get("atr_pct",         0.0)),
            adx_14          = float(s.get("adx_14",         25.0)),
            vwap_distance   = float(s.get("vwap_distance",   0.0)),
            volume_ratio    = float(s.get("volume_ratio",    1.0)),
            momentum_30m    = float(s.get("momentum_30m",    0.0)),
            momentum_1d     = float(s.get("momentum_1d",     0.0)),
            api_calls_used  = 0,
            timestamp       = datetime.now(),
        )

    def batch_analyze(
        self, session_states: dict[str, dict]
    ) -> dict[str, TechnicalSignal]:
        """
        Analyse multiple tickers sequentially (pure CPU — no concurrency needed).

        Parameters
        ----------
        session_states:
            ``{ticker: session_state_dict}`` as returned by
            :meth:`~argus.data.pipeline.MFTDataPipeline.compress_all`.

        Returns
        -------
        dict[str, TechnicalSignal]
            Maps each ticker to its computed signal.  Tickers that fail
            analysis are logged as warnings and excluded from the result.
        """
        results: dict[str, TechnicalSignal] = {}
        for ticker, state in session_states.items():
            try:
                results[ticker] = self.analyze(ticker, state)
            except Exception as exc:
                logger.warning(
                    "batch_analyze: %s failed — %s: %s",
                    ticker, type(exc).__name__, exc,
                )
        logger.info(
            "batch_analyze: %d/%d tickers analysed",
            len(results), len(session_states),
        )
        return results
