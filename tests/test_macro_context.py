"""Tests the pure derivations in agents/macro_context.py and data/macro_features.cpi_yoy."""

import pandas as pd
import pytest

from argus.agents.macro_context import (
    agent_multipliers,
    sector_rotation,
    trend_from_lag,
    vix_regime,
    yield_curve_shape,
)
from argus.data.macro_features import cpi_yoy
from argus.params import MACRO
from argus.schemas.signals import Regime, SectorSignal, Trend, VixRegime, YieldCurve


@pytest.mark.parametrize(
    ("vix", "expected"),
    [
        (MACRO.vix_low_max - 0.01, VixRegime.LOW),
        (MACRO.vix_low_max, VixRegime.MEDIUM),
        (MACRO.vix_medium_max, VixRegime.HIGH),
        (MACRO.vix_high_max, VixRegime.EXTREME),
    ],
)
def test_vix_regime_buckets_change_exactly_at_the_thresholds(vix, expected):
    assert vix_regime(vix) == expected


@pytest.mark.parametrize(
    ("spread", "expected"),
    [
        (MACRO.yield_curve_inverted_below - 0.01, YieldCurve.INVERTED),
        (MACRO.yield_curve_inverted_below, YieldCurve.FLAT),
        (MACRO.yield_curve_flat_below, YieldCurve.NORMAL),
    ],
)
def test_yield_curve_shape_changes_exactly_at_the_thresholds(spread, expected):
    assert yield_curve_shape(spread) == expected


def test_sector_rotation_prefers_growth_only_in_a_low_rate_expansion():
    assert sector_rotation(Regime.EXPANSION, 2.0, YieldCurve.NORMAL) == SectorSignal.GROWTH_FAVORED
    assert sector_rotation(Regime.EXPANSION, 5.0, YieldCurve.NORMAL) == SectorSignal.VALUE_FAVORED


def test_sector_rotation_goes_defensive_on_contraction_or_an_inverted_curve():
    assert sector_rotation(Regime.CONTRACTION, 2.0, YieldCurve.NORMAL) == SectorSignal.DEFENSIVE
    assert sector_rotation(Regime.EXPANSION, 5.0, YieldCurve.INVERTED) == SectorSignal.DEFENSIVE


def test_agent_multipliers_amplify_technical_and_sentiment_at_high_vix_percentile():
    calm = agent_multipliers(Regime.EXPANSION, 30.0)
    stressed = agent_multipliers(Regime.CONTRACTION, 80.0)

    assert calm == {"fundamental": 1.3, "technical": 1.0, "sentiment": 0.9}
    assert stressed == {"fundamental": 0.9, "technical": 1.2, "sentiment": 1.15}


def test_trend_from_lag_compares_against_the_lagged_observation():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    assert trend_from_lag(series, 4.0, lag=2, threshold=0.5) == Trend.RISING
    assert trend_from_lag(series, 1.0, lag=2, threshold=0.5) == Trend.FALLING
    assert trend_from_lag(series, 2.2, lag=2, threshold=0.5) == Trend.STABLE


def test_trend_from_lag_is_stable_when_the_series_is_too_short():
    assert trend_from_lag(pd.Series([1.0, 2.0]), 9.0, lag=2, threshold=0.1) == Trend.STABLE


def test_cpi_yoy_is_the_twelve_month_percent_change():
    cpi = pd.Series([100.0 * 1.02**i for i in range(14)])

    result = cpi_yoy(cpi)

    assert result.iloc[:12].isna().all()
    assert result.iloc[12] == pytest.approx((1.02**12 - 1) * 100.0)


def test_missing_macro_readings_remain_unknown():
    from argus.agents.macro_context import rule_based

    assert vix_regime(None) is None
    assert yield_curve_shape(None) is None
    assert rule_based({"t10y2y": 1.5}) == (Regime.TRANSITIONAL.value, 0.5)
    assert rule_based({"t10y2y": -1.0}) == (Regime.CONTRACTION.value, 0.6)
    assert rule_based({}) == (Regime.TRANSITIONAL.value, 0.5)
