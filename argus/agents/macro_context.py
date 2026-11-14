"""Pure macro label, trend, sector, and vote-weight derivations."""

from __future__ import annotations

import pandas as pd

from argus.params import MACRO
from argus.schemas.signals import Regime, SectorSignal, Trend, VixRegime, YieldCurve


def rule_based(v: dict) -> tuple[str, float]:
    """Apply static thresholds to classify the regime when the HMM cannot be used."""
    vix = v.get("vix")
    t10y2y = v.get("t10y2y")

    if (vix is not None and vix > MACRO.rule_contraction_vix_above) or (
        t10y2y is not None and t10y2y < MACRO.rule_contraction_t10y2y_below
    ):
        return Regime.CONTRACTION.value, 0.6
    elif (
        vix is not None
        and t10y2y is not None
        and vix < MACRO.rule_expansion_vix_below
        and t10y2y > MACRO.rule_expansion_t10y2y_above
    ):
        return Regime.EXPANSION.value, 0.6
    else:
        return Regime.TRANSITIONAL.value, 0.5


def vix_regime(vix: float | None) -> VixRegime | None:
    """Bucket the VIX on its absolute level, not its percentile."""
    if vix is None:
        return None
    if vix < MACRO.vix_low_max:
        return VixRegime.LOW
    if vix < MACRO.vix_medium_max:
        return VixRegime.MEDIUM
    if vix < MACRO.vix_high_max:
        return VixRegime.HIGH
    return VixRegime.EXTREME


def yield_curve_shape(t10y2y: float | None) -> YieldCurve | None:
    """Bucket the 10y-2y Treasury spread."""
    if t10y2y is None:
        return None
    if t10y2y < MACRO.yield_curve_inverted_below:
        return YieldCurve.INVERTED
    if t10y2y < MACRO.yield_curve_flat_below:
        return YieldCurve.FLAT
    return YieldCurve.NORMAL


def sector_rotation(regime: Regime, fed_funds: float, curve: YieldCurve | None) -> SectorSignal:
    """Choose the sector-rotation stance from the regime, policy rate and curve shape."""
    if regime == Regime.EXPANSION and fed_funds < 4.5:
        return SectorSignal.GROWTH_FAVORED
    if regime == Regime.CONTRACTION or curve == YieldCurve.INVERTED:
        return SectorSignal.DEFENSIVE
    return SectorSignal.VALUE_FAVORED


def agent_multipliers(regime: Regime, vix_percentile: float) -> dict[str, float]:
    """Scale each specialist's vote weight by the volatility regime."""
    return {
        "fundamental": 1.3 if (regime == Regime.EXPANSION and vix_percentile < 40) else 0.9,
        "technical": 1.2 if vix_percentile > 60 else 1.0,
        "sentiment": 1.15 if vix_percentile > 50 else 0.9,
    }


def trend_from_lag(series: pd.Series, current_value: float, lag: int, threshold: float) -> Trend:
    """Bucket a value's move against the same series `lag` periods back."""
    if len(series) > lag:
        lagged = series.iloc[-(lag + 1)]
        if current_value > lagged + threshold:
            return Trend.RISING
        if current_value < lagged - threshold:
            return Trend.FALLING
    return Trend.STABLE
