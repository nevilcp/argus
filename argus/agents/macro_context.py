"""
argus/agents/macro_context.py

Pure derivations from macro readings to the labels and scalars on a MacroContext.

Responsibilities:
  - Bucket the VIX level and the 10y-2y spread into their regime labels
  - Derive the sector-rotation signal and the per-agent vote multipliers
  - Classify the regime from raw levels when the HMM cannot be used
  - Bucket a value's move against a lagged observation into a Trend

Not responsible for:
  - Fetching anything or caching the result (see agents/macro.py)
  - The HMM regime classifier (see agents/macro_hmm.py)
  - The thresholds themselves, where they live in params.py's MacroParams
"""

from __future__ import annotations

import pandas as pd

from argus.params import MACRO
from argus.schemas.signals import Regime, SectorSignal, Trend, VixRegime, YieldCurve


def rule_based(v: dict) -> tuple[str, float]:
    """Applies static thresholds to classify the regime when the HMM cannot be used.

    Used during cold-start, or when the feature window cannot be assembled. Confidence is fixed
    at 0.6 (CONTRACTION/EXPANSION) or 0.5 (TRANSITIONAL) to signal lower certainty.

    Args:
        v: Dict with optional keys ``vix`` and ``t10y2y``.

    Returns:
        Tuple of (regime_string, confidence).
    """
    vix = v.get("vix")
    t10y2y = v.get("t10y2y")
    vix = vix if vix is not None else MACRO.default_vix
    t10y2y = t10y2y if t10y2y is not None else MACRO.rule_default_t10y2y

    if vix > MACRO.rule_contraction_vix_above or t10y2y < MACRO.rule_contraction_t10y2y_below:
        return Regime.CONTRACTION.value, 0.6
    elif vix < MACRO.rule_expansion_vix_below and t10y2y > MACRO.rule_expansion_t10y2y_above:
        return Regime.EXPANSION.value, 0.6
    else:
        return Regime.TRANSITIONAL.value, 0.5


def vix_regime(vix: float) -> VixRegime:
    """Buckets the VIX on its absolute level, not its percentile.

    EXTREME is the governor's kill-switch zone by level, so the bucket must not
    depend on how the level ranks against recent history.

    Args:
        vix: The VIX level.

    Returns:
        LOW, MEDIUM, HIGH or EXTREME.
    """
    if vix < MACRO.vix_low_max:
        return VixRegime.LOW
    if vix < MACRO.vix_medium_max:
        return VixRegime.MEDIUM
    if vix < MACRO.vix_high_max:
        return VixRegime.HIGH
    return VixRegime.EXTREME


def yield_curve_shape(t10y2y: float) -> YieldCurve:
    """Buckets the 10y-2y Treasury spread.

    Args:
        t10y2y: The spread, in percentage points.

    Returns:
        INVERTED, FLAT or NORMAL.
    """
    if t10y2y < MACRO.yield_curve_inverted_below:
        return YieldCurve.INVERTED
    if t10y2y < MACRO.yield_curve_flat_below:
        return YieldCurve.FLAT
    return YieldCurve.NORMAL


def sector_rotation(regime: Regime, fed_funds: float, curve: YieldCurve) -> SectorSignal:
    """Chooses the sector-rotation stance from the regime, policy rate and curve shape.

    Args:
        regime: The classified macro regime.
        fed_funds: The federal funds rate, in percent.
        curve: The yield-curve bucket.

    Returns:
        GROWTH_FAVORED in a low-rate expansion, DEFENSIVE in a contraction or
        with an inverted curve, VALUE_FAVORED otherwise.
    """
    if regime == Regime.EXPANSION and fed_funds < 4.5:
        return SectorSignal.GROWTH_FAVORED
    if regime == Regime.CONTRACTION or curve == YieldCurve.INVERTED:
        return SectorSignal.DEFENSIVE
    return SectorSignal.VALUE_FAVORED


def agent_multipliers(regime: Regime, vix_percentile: float) -> dict[str, float]:
    """Scales each specialist's vote weight by the volatility regime.

    High VIX amplifies the technical and sentiment signals.

    Args:
        regime: The classified macro regime.
        vix_percentile: Where today's VIX sits in its trailing history, 0-100.

    Returns:
        Agent name → vote-weight multiplier.
    """
    return {
        "fundamental": 1.3 if (regime == Regime.EXPANSION and vix_percentile < 40) else 0.9,
        "technical": 1.2 if vix_percentile > 60 else 1.0,
        "sentiment": 1.15 if vix_percentile > 50 else 0.9,
    }


def trend_from_lag(series: pd.Series, current_value: float, lag: int, threshold: float) -> Trend:
    """Buckets a value's move against the same series `lag` periods back.

    Args:
        series: The series to compare against, oldest observation first.
        current_value: The present-day value, on the series' scale.
        lag: How many periods back to compare against.
        threshold: Move size, in the series' units, that separates a direction
            from STABLE.

    Returns:
        RISING or FALLING when the move exceeds `threshold`; STABLE when it does
        not, and also when the series is no longer than `lag`.
    """
    if len(series) > lag:
        lagged = series.iloc[-(lag + 1)]
        if current_value > lagged + threshold:
            return Trend.RISING
        if current_value < lagged - threshold:
            return Trend.FALLING
    return Trend.STABLE
