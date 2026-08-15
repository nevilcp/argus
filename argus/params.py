"""
argus/params.py

Single source of truth for every free (hand-set, non-derived) numeric
parameter in ARGUS, each tagged with where its value came from.

A system that mixes textbook constants (RSI period 14), industry convention
(RSI 30/70 overbought/oversold), values tuned against ARGUS's own data, and
outright guesses reads identically in the source unless the provenance is
written down next to the value. This module makes that distinction
mechanical instead of relying on comments that rot.

Provenance categories:
  LITERATURE  — a standard value with a citable source (e.g. RSI/ATR period
                14, VaR confidence 99%).
  CONVENTION  — a widely-used default with no single citation, but common
                enough that deviating from it would need justification
                (e.g. RSI 30/70 bands, half-Kelly sizing).
  CALIBRATED  — tuned against ARGUS's own data or backtests. As of this
                writing there is no honest instance of this category: the
                Phase 1/2 calibration harness measured a constant and PR 7
                deletes it. Any future value in this category must point at
                real evidence.
  ARBITRARY   — a guess with no basis beyond "seemed reasonable". Writing
                ARBITRARY where it is arbitrary is the point of this
                module: it is a to-do list for what needs real evaluation.

PARAMS_VERSION exists so a changed value can be pinned in a fixture or a
backtest run's metadata; bump it whenever any value below changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

PARAMS_VERSION = 5


class Provenance(str, Enum):
    """Where a parameter's value came from: see the module docstring for definitions."""

    LITERATURE = "literature"
    CONVENTION = "convention"
    CALIBRATED = "calibrated"
    ARBITRARY = "arbitrary"


def p(value: Any, provenance: Provenance, note: str = "") -> Any:
    """Declare a dataclass field carrying a provenance tag as metadata.

    The field still holds `value` directly — call sites read it like any
    other attribute (`TECHNICAL.rsi_oversold`), no unwrapping required. The
    provenance/note live in `__dataclass_fields__[name].metadata` for
    introspection (see `all_params` below).

    Args:
        value: The default value the field holds.
        provenance: Category explaining where the value came from.
        note: Free-text detail supporting the provenance category.

    Returns:
        A dataclasses.field() configured with `value` as its default.
    """
    return field(default=value, metadata={"provenance": provenance, "note": note})


@dataclass(frozen=True)
class SystemParams:
    """Timing, universe, and hard-limit parameters formerly inline in config.py."""

    mft_decision_interval_seconds: int = p(
        1800, Provenance.ARBITRARY, "30-minute decision cadence; not evaluated against alternatives"
    )
    candle_buffer_days_retained: int = p(
        2,
        Provenance.CONVENTION,
        "buffer retains this many trading days of intraday bars; matches "
        "data/pipeline.py's yf.download() fetch period so a sweep's candles "
        "always fit without evicting same-day history",
    )
    max_single_position_pct: float = p(
        0.15, Provenance.CONVENTION, "common retail/advisory single-name concentration ceiling"
    )
    max_sector_concentration: float = p(
        0.40, Provenance.ARBITRARY, "no external basis; picked as a loose backstop"
    )
    max_portfolio_beta: float = p(
        1.50, Provenance.ARBITRARY, "no external basis; picked as a loose backstop"
    )
    vix_blackout_threshold: float = p(
        35.0, Provenance.CONVENTION, "VIX > 35 is commonly treated as a high-fear regime"
    )
    max_drawdown_halt: float = p(
        0.15, Provenance.ARBITRARY, "no external basis; picked as a loose backstop"
    )
    lookback_days: int = p(
        252, Provenance.LITERATURE, "252 = standard US trading days per year"
    )


@dataclass(frozen=True)
class TechnicalIndicatorWeights:
    """Per-indicator weights in the technical consensus score (config.py)."""

    rsi: float = p(2.0, Provenance.ARBITRARY, "no basis for relative weighting vs other indicators")
    macd: float = p(2.0, Provenance.ARBITRARY, "no basis for relative weighting vs other indicators")
    bb: float = p(1.5, Provenance.ARBITRARY, "no basis for relative weighting vs other indicators")
    adx: float = p(1.0, Provenance.ARBITRARY, "no basis for relative weighting vs other indicators")
    vwap: float = p(1.0, Provenance.ARBITRARY, "no basis for relative weighting vs other indicators")
    momentum: float = p(1.5, Provenance.ARBITRARY, "no basis for relative weighting vs other indicators")


@dataclass(frozen=True)
class TechnicalParams:
    """Thresholds and curve shapes used by TechnicalStatisticalAgent (agents/technical.py)."""

    rsi_oversold: float = p(25, Provenance.CONVENTION, "RSI < 30 is the standard oversold convention; 25 used as the max-bullish anchor")
    rsi_bullish_transition: float = p(30, Provenance.CONVENTION, "standard RSI oversold boundary")
    rsi_bullish_transition_score: float = p(0.85, Provenance.ARBITRARY, "no basis for this specific score level")
    rsi_overbought: float = p(75, Provenance.CONVENTION, "RSI > 70 is the standard overbought convention; 75 used as the max-bearish anchor")
    rsi_bearish_transition: float = p(70, Provenance.CONVENTION, "standard RSI overbought boundary")
    rsi_neutral_low: float = p(45.0, Provenance.ARBITRARY, "dead-zone boundary; no basis beyond symmetry around 50")
    rsi_neutral_high: float = p(55.0, Provenance.ARBITRARY, "dead-zone boundary; no basis beyond symmetry around 50")

    macd_normalization_scale: float = p(0.5, Provenance.ARBITRARY, "no basis for this specific scale")

    bollinger_breach_multiplier: float = p(2.0, Provenance.ARBITRARY, "no basis for this specific multiplier")
    bollinger_breach_boost: float = p(0.5, Provenance.ARBITRARY, "no basis for this specific boost")

    adx_dampen_threshold: float = p(20.0, Provenance.CONVENTION, "ADX < 20 commonly read as 'no trend'")
    adx_dampen_multiplier: float = p(0.6, Provenance.ARBITRARY, "no basis for this specific multiplier")
    adx_amplify_threshold: float = p(40.0, Provenance.CONVENTION, "ADX > 40 commonly read as 'strong trend'")
    adx_amplify_multiplier: float = p(1.2, Provenance.ARBITRARY, "no basis for this specific multiplier")

    vwap_spread_normalization: float = p(0.015, Provenance.ARBITRARY, "no basis for +/-1.5% as the normalizing spread")

    momentum_30m_weight: float = p(0.4, Provenance.ARBITRARY, "no basis for 40/60 split vs 1d momentum")
    momentum_1d_weight: float = p(0.6, Provenance.ARBITRARY, "no basis for 40/60 split vs 30m momentum")
    momentum_saturation_bound: float = p(0.02, Provenance.ARBITRARY, "no basis for +/-2% as the clipping bound")

    volume_ratio_divisor: float = p(1.5, Provenance.ARBITRARY, "no basis for this specific divisor")
    volume_modifier_floor: float = p(0.7, Provenance.ARBITRARY, "no basis for this specific floor")
    volume_modifier_ceiling: float = p(1.3, Provenance.ARBITRARY, "no basis for this specific ceiling")

    net_score_neutral_threshold: float = p(0.12, Provenance.ARBITRARY, "no basis for this specific cutoff")
    neutral_conviction_floor: float = p(0.30, Provenance.ARBITRARY, "no basis for this specific floor")
    conviction_base: float = p(0.40, Provenance.ARBITRARY, "no basis for this specific base")
    conviction_score_multiplier: float = p(0.55, Provenance.ARBITRARY, "no basis for this specific multiplier")
    conviction_ceiling: float = p(0.92, Provenance.ARBITRARY, "kept below 1.0 so no signal claims certainty")


@dataclass(frozen=True)
class AggregatorParams:
    """Multi-agent voting parameters (orchestration/aggregator.py)."""

    weight_fundamental: float = p(0.35, Provenance.ARBITRARY, "no basis for relative agent weighting")
    weight_technical: float = p(0.35, Provenance.ARBITRARY, "no basis for relative agent weighting")
    weight_sentiment: float = p(0.30, Provenance.ARBITRARY, "no basis for relative agent weighting")
    contraction_conviction_threshold: float = p(0.70, Provenance.ARBITRARY, "no basis for this specific threshold")
    contraction_conviction_reduction: float = p(0.6, Provenance.ARBITRARY, "no basis for this specific multiplier")
    max_conviction: float = p(0.95, Provenance.ARBITRARY, "kept below 1.0 so no aggregate claims certainty")


@dataclass(frozen=True)
class PortfolioParams:
    """Position sizing and LLM-call controls (agents/portfolio.py)."""

    kelly_avg_win_pct: float = p(0.08, Provenance.ARBITRARY, "assumed average winning-trade return; not fit to ARGUS's own trades")
    kelly_avg_loss_pct: float = p(0.04, Provenance.ARBITRARY, "assumed average losing-trade return; not fit to ARGUS's own trades")
    kelly_max_position: float = p(0.15, Provenance.CONVENTION, "matches SystemParams.max_single_position_pct")
    kelly_divisor: float = p(2.0, Provenance.CONVENTION, "half-Kelly is a standard variance-reduction convention")
    llm_temperature: float = p(0.05, Provenance.CONVENTION, "near-zero temperature for reproducible structured output")
    llm_max_tokens: int = p(2400, Provenance.ARBITRARY, "sized to fit observed response length with headroom")
    equity_floor_adjustment: float = p(0.05, Provenance.ARBITRARY, "no basis for this specific adjustment")
    thesis_char_limit: int = p(120, Provenance.ARBITRARY, "no basis beyond keeping per-position text short")
    advisor_note_char_limit: int = p(600, Provenance.ARBITRARY, "no basis beyond keeping the rationale readable")


@dataclass(frozen=True)
class RiskParams:
    """VaR/CVaR, optimization, and structural-gate parameters (agents/risk.py)."""

    sector_cache_ttl_seconds: int = p(86400, Provenance.ARBITRARY, "24h cache lifetime; GICS classifications rarely change")
    returns_lookback_days: int = p(252, Provenance.LITERATURE, "252 = standard US trading days per year")
    var_confidence: float = p(0.99, Provenance.CONVENTION, "99% is a standard VaR/CVaR confidence level")
    cvar_confidence: float = p(0.99, Provenance.CONVENTION, "99% is a standard VaR/CVaR confidence level")
    min_beta_overlap_points: int = p(10, Provenance.ARBITRARY, "no basis for this specific minimum")
    atr_multiplier: float = p(2.5, Provenance.CONVENTION, "2-3x ATR is a common stop-loss distance convention")
    atr_period: int = p(14, Provenance.LITERATURE, "14 is the standard Wilder ATR period")
    min_positions_diversification: int = p(5, Provenance.ARBITRARY, "no basis for this specific minimum")
    max_positions: int = p(20, Provenance.ARBITRARY, "no basis for this specific maximum")
    slsqp_risk_aversion: float = p(1.0, Provenance.ARBITRARY, "equal weighting of conviction return vs variance; not tuned")
    slsqp_min_equity_deployment: float = p(0.50, Provenance.ARBITRARY, "no basis for this specific floor")
    slsqp_ftol: float = p(1e-9, Provenance.CONVENTION, "typical SciPy SLSQP convergence tolerance")
    slsqp_maxiter: int = p(300, Provenance.CONVENTION, "typical SciPy SLSQP iteration cap")
    var_limit: float = p(0.03, Provenance.ARBITRARY, "no basis for this specific daily VaR ceiling")
    cvar_limit: float = p(0.05, Provenance.ARBITRARY, "no basis for this specific CVaR ceiling")
    correlation_limit: float = p(0.75, Provenance.ARBITRARY, "no basis for this specific correlation ceiling")
    reduce_weight_multiplier: float = p(0.5, Provenance.ARBITRARY, "no basis for this specific haircut on statistical violation")


@dataclass(frozen=True)
class MacroParams:
    """RegimeClassifier windowed-inference parameters (agents/macro.py, data/macro_features.py)."""

    inference_window_months: int = p(
        36,
        Provenance.ARBITRARY,
        "3 years of monthly observations; long enough to give the HMM's transition "
        "matrix several observations so one noisy print can't flip the regime; "
        "not backtested against alternatives",
    )
    inference_history_years: int = p(
        10,
        Provenance.ARBITRARY,
        "FRED/VIX history fetched for windowed inference; must exceed the 5-year "
        "VIX percentile warmup plus the inference window itself, with margin",
    )
    fed_funds_publication_lag_days: int = p(
        32, Provenance.LITERATURE, "Federal Reserve H.15 release lag for FEDFUNDS"
    )
    unemployment_publication_lag_days: int = p(
        35, Provenance.LITERATURE, "BLS Employment Situation release lag for UNRATE"
    )
    cpi_publication_lag_days: int = p(
        45, Provenance.LITERATURE, "BLS CPI release lag for CPIAUCSL"
    )
    hmm_n_init: int = p(
        20,
        Provenance.ARBITRARY,
        "random restarts of the fit, keeping the highest-likelihood non-degenerate "
        "one; 3 of 5 single-seed fits collapsed on the shipped data, so one seed "
        "is not enough — 20 is a guess at 'enough', not backtested",
    )
    hmm_min_state_separation: float = p(
        1.0,
        Provenance.ARBITRARY,
        "minimum pairwise Mahalanobis distance between state means (pooled "
        "covariance) a restart must clear to be accepted; the shipped degenerate "
        "model scored ~0.001, the validated target design scored 2.71 — 1.0 is a "
        "floor with margin below the validated fit, not itself validated",
    )
    hmm_min_state_occupancy: float = p(
        0.05,
        Provenance.ARBITRARY,
        "minimum fraction of training observations a state must claim to be "
        "accepted, guarding against a restart that fits one state to a handful "
        "of outlier months",
    )
    contraction_enrichment_min: float = p(
        2.0,
        Provenance.ARBITRARY,
        "minimum ratio of a candidate CONTRACTION state's NBER (USREC) recession "
        "frequency to the sample's overall recession base rate; the validated "
        "target design achieved 3.2x — 2.0 is a floor below that, not itself "
        "validated",
    )
    dwell_time_min_months: float = p(
        3.0,
        Provenance.ARBITRARY,
        "minimum acceptable mean state dwell time in the training-gate check; "
        "below this a state is switching faster than any real macro regime does",
    )
    dwell_time_max_months: float = p(
        120.0,
        Provenance.ARBITRARY,
        "maximum acceptable mean state dwell time in the training-gate check; "
        "above this a state is effectively never leaving, i.e. tracking a rate "
        "era rather than a regime",
    )


@dataclass(frozen=True)
class ReconciliationParams:
    """Outcome-reconciliation horizon (orchestration/reconciliation.py)."""

    horizon_days: int = p(
        5,
        Provenance.ARBITRARY,
        "pre-registered before any evaluation result was observed; still no "
        "empirical basis for this specific value",
    )
    min_abs_return_for_storage: float = p(
        0.01,
        Provenance.ARBITRARY,
        "matches cultural.store_trade_outcome's SUCCESSFUL/FAILED/FLAT |return| "
        "classification threshold; kept here so both live in one place",
    )


@dataclass(frozen=True)
class MemoryParams:
    """Reliability-weighting parameters (memory/cultural.py, orchestration/aggregator.py)."""

    accuracy_shrinkage_k: float = p(
        10.0,
        Provenance.ARBITRARY,
        "pseudo-observation count pulling small-sample agent accuracy toward "
        "the 0.5 prior; no basis for this specific strength",
    )


SYSTEM = SystemParams()
TECHNICAL_INDICATOR_WEIGHTS = TechnicalIndicatorWeights()
TECHNICAL = TechnicalParams()
AGGREGATOR = AggregatorParams()
PORTFOLIO = PortfolioParams()
RISK = RiskParams()
MACRO = MacroParams()
RECONCILIATION = ReconciliationParams()
MEMORY = MemoryParams()

_ALL_GROUPS: dict[str, Any] = {
    "SYSTEM": SYSTEM,
    "TECHNICAL_INDICATOR_WEIGHTS": TECHNICAL_INDICATOR_WEIGHTS,
    "TECHNICAL": TECHNICAL,
    "AGGREGATOR": AGGREGATOR,
    "PORTFOLIO": PORTFOLIO,
    "RISK": RISK,
    "MACRO": MACRO,
    "RECONCILIATION": RECONCILIATION,
    "MEMORY": MEMORY,
}


def all_params() -> list[dict[str, Any]]:
    """Flat listing of every declared parameter, for tests and audits.

    Each entry: group name, field name, current value, Provenance, note.
    Used by tests/test_params.py to assert every field carries provenance
    metadata (i.e. nobody added a bare `float = 0.5` field and skipped `p(...)`).
    """
    rows: list[dict[str, Any]] = []
    for group_name, instance in _ALL_GROUPS.items():
        for f in fields(instance):
            rows.append(
                {
                    "group": group_name,
                    "field": f.name,
                    "value": getattr(instance, f.name),
                    "provenance": f.metadata.get("provenance"),
                    "note": f.metadata.get("note", ""),
                }
            )
    return rows
