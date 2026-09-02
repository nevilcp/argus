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
                Phase 1/2 calibration harness measured a constant that has
                since been removed. Any future value in this category must
                point at real evidence.
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

PARAMS_VERSION = 17


class Provenance(str, Enum):
    """Where a parameter's value came from.

    See the module docstring for the category definitions.
    """

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
    """Timing, universe, and hard-limit parameters moved out of config.py."""

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
    lookback_days: int = p(
        252, Provenance.LITERATURE, "252 = standard US trading days per year"
    )
    freshness_margin_seconds: int = p(
        60,
        Provenance.ARBITRARY,
        "buffer against fetch/session timing jitter added to the MFT cache-write "
        "and bar-age budgets in data/pipeline.py; not evaluated against alternatives",
    )
    min_inter_request_seconds: float = p(
        0.5,
        Provenance.ARBITRARY,
        "minimum pause between consecutive per-ticker yfinance requests within a "
        "sweep, to avoid rate limiting; a fixed floor rather than a fill-the-interval "
        "spread, since pacing exists for rate limiting, not to occupy the cadence "
        "budget — not evaluated against alternatives",
    )
    max_tracked_tickers: int = p(
        100,
        Provenance.ARBITRARY,
        "ceiling on MFTDataPipeline.tickers enforced in register_tickers; stops "
        "unbounded universe growth from repeated /analyze requests (API-1) — not "
        "evaluated against alternatives",
    )
    tracked_ticker_ttl_seconds: int = p(
        86400,
        Provenance.ARBITRARY,
        "how long a non-seed ticker (one registered by an /analyze request rather "
        "than the pipeline's initial universe) stays tracked without being "
        "re-requested before register_tickers evicts it (API-12) — not evaluated "
        "against alternatives",
    )
    analyze_retry_after_seconds: int = p(
        5,
        Provenance.ARBITRARY,
        "Retry-After header value on /analyze's 429 when a run is already in "
        "progress; a guess at a graph run's typical duration, not measured",
    )


@dataclass(frozen=True)
class KillSwitchParams:
    """Drawdown circuit-breaker tiers and halt-dump retention.

    Used by risk/kill_switch.py. The three tiers previously lived as a
    hardcoded dict there, alongside an unrelated, unused
    SystemParams.max_drawdown_halt = 0.15 that silently contradicted the
    AGGRESSIVE tier's 0.18 — this group is now the single source of truth
    for all three.
    """

    conservative_drawdown_halt: float = p(
        0.08, Provenance.ARBITRARY, "no external basis; picked as a loose backstop"
    )
    moderate_drawdown_halt: float = p(
        0.12, Provenance.ARBITRARY, "no external basis; picked as a loose backstop"
    )
    aggressive_drawdown_halt: float = p(
        0.18, Provenance.ARBITRARY, "no external basis; picked as a loose backstop"
    )
    max_halt_dumps_retained: int = p(
        20, Provenance.ARBITRARY, "no basis for this specific retention count"
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
    """Thresholds and curve shapes used by TechnicalStatisticalAgent.

    See agents/technical.py.
    """

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
    contraction_conviction_threshold: float = p(
        0.45,
        Provenance.ARBITRARY,
        "restated post-issue-24-Decision-A against achievable conviction (~0.665 for a "
        "two-agent unanimous call), not the pre-A 1.0 scale; no basis for this specific value",
    )
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
    llm_max_tokens: int = p(1000, Provenance.ARBITRARY, "gpt-oss-120b at reasoning_effort='low' on a reconstructed dense-portfolio prompt peaked at 785 completion tokens (up to 458 reasoning) across 4 runs; 1000 leaves ~25% headroom")
    equity_floor_adjustment: float = p(0.05, Provenance.ARBITRARY, "no basis for this specific adjustment")
    cash_reserve_floor_pct: float = p(0.05, Provenance.ARBITRARY, "matches PortfolioAllocation.cash_reserve_pct's schema floor")
    thesis_char_limit: int = p(120, Provenance.ARBITRARY, "no basis beyond keeping per-position text short")
    advisor_note_char_limit: int = p(600, Provenance.ARBITRARY, "no basis beyond keeping the rationale readable")


@dataclass(frozen=True)
class RiskParams:
    """VaR/CVaR, optimization, and structural-gate parameters.

    See agents/risk.py.
    """

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
    slsqp_max_total_deployment: float = p(1.0, Provenance.CONVENTION, "a long-only book cannot exceed its capital")
    slsqp_zero_cap_epsilon: float = p(1e-3, Provenance.ARBITRARY, "cap below this is treated as no room to allocate, not a real position")
    slsqp_ftol: float = p(1e-9, Provenance.CONVENTION, "typical SciPy SLSQP convergence tolerance")
    slsqp_maxiter: int = p(300, Provenance.CONVENTION, "typical SciPy SLSQP iteration cap")
    var_limit: float = p(0.03, Provenance.ARBITRARY, "no basis for this specific daily VaR ceiling")
    cvar_limit: float = p(0.05, Provenance.ARBITRARY, "no basis for this specific CVaR ceiling")
    correlation_limit: float = p(0.75, Provenance.ARBITRARY, "no basis for this specific correlation ceiling")
    reduce_weight_multiplier: float = p(0.5, Provenance.ARBITRARY, "no basis for this specific haircut on statistical violation")


@dataclass(frozen=True)
class MacroParams:
    """RegimeClassifier windowed-inference parameters.

    See agents/macro.py and data/macro_features.py.
    """

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
    retention_margin_days: int = p(
        2,
        Provenance.ARBITRARY,
        "buffer beyond horizon_days before a decision, checkpoint, or PENDING "
        "snapshot is pruned as growing-store cleanup (PR 6) — wide enough that "
        "a delayed reconcile pass still finds it first, no basis for this "
        "specific width",
    )
    unresolved_retirement_days: int = p(
        30,
        Provenance.ARBITRARY,
        "age beyond horizon_days at which a decision that has never received "
        "a reconciled outcome is retired anyway, so the store stays bounded "
        "even though it was never scored — scheduled reconcile ticks can be "
        "dropped by the platform for days at a time, so this must comfortably "
        "outlast that, but no basis for this specific width",
    )


@dataclass(frozen=True)
class MemoryParams:
    """Reliability-weighting parameters.

    See memory/cultural.py and orchestration/aggregator.py.
    """

    accuracy_shrinkage_k: float = p(
        10.0,
        Provenance.ARBITRARY,
        "pseudo-observation count pulling small-sample agent accuracy toward "
        "the 0.5 prior; no basis for this specific strength",
    )


@dataclass(frozen=True)
class StructuredOutputParams:
    """Retry/back-off controls for the structured-output decoder.

    See structured_output.py. One attempt budget replacing the three
    per-agent copies of ``range(3)`` + ``time.sleep(2**attempt)`` this
    decoder consolidates (fundamental.py, sentiment.py, portfolio.py) —
    none of those three had any basis beyond "seemed reasonable" either,
    so the value carries forward unchanged rather than being re-guessed.
    """

    max_attempts: int = p(
        3,
        Provenance.ARBITRARY,
        "matches the per-agent retry loops this decoder replaces; not evaluated "
        "against alternatives",
    )
    backoff_base_seconds: float = p(
        1.0,
        Provenance.ARBITRARY,
        "matches the per-agent time.sleep(2**attempt) back-off this decoder "
        "replaces; not evaluated against alternatives",
    )
    backoff_max_seconds: float = p(
        30.0,
        Provenance.ARBITRARY,
        "ceiling on the exponential back-off; the per-agent loops this decoder "
        "replaces had no cap since 3 attempts alone tops out at 4s, kept here as "
        "a defensive bound if max_attempts is ever raised",
    )


@dataclass(frozen=True)
class CollectorParams:
    """Thresholds the unattended collector uses to judge a completed cycle's health.

    See orchestration/collector.py's CycleOutcome: a cycle that ran but produced
    too few usable allocations is DEGRADED, not SUCCESS, even though nothing
    raised — the LLM agents' "degrade, never fabricate" behavior means a dead
    upstream (issue #90's missing-secret case, or any other) shows up as null
    fields on every decision rather than an exception.
    """

    min_decisions_with_allocation: int = p(
        1,
        Provenance.ARBITRARY,
        "a completed cycle needs at least one decision with a real allocation "
        "to have produced anything worth reconciling; below this the cycle is "
        "reported DEGRADED and the job fails. 1 is a floor, not a fraction — "
        "not tuned against real degraded-run data",
    )


SYSTEM = SystemParams()
KILL_SWITCH = KillSwitchParams()
TECHNICAL_INDICATOR_WEIGHTS = TechnicalIndicatorWeights()
TECHNICAL = TechnicalParams()
AGGREGATOR = AggregatorParams()
PORTFOLIO = PortfolioParams()
RISK = RiskParams()
MACRO = MacroParams()
RECONCILIATION = ReconciliationParams()
MEMORY = MemoryParams()
STRUCTURED_OUTPUT = StructuredOutputParams()
COLLECTOR = CollectorParams()

_ALL_GROUPS: dict[str, Any] = {
    "SYSTEM": SYSTEM,
    "KILL_SWITCH": KILL_SWITCH,
    "TECHNICAL_INDICATOR_WEIGHTS": TECHNICAL_INDICATOR_WEIGHTS,
    "TECHNICAL": TECHNICAL,
    "AGGREGATOR": AGGREGATOR,
    "PORTFOLIO": PORTFOLIO,
    "RISK": RISK,
    "MACRO": MACRO,
    "RECONCILIATION": RECONCILIATION,
    "MEMORY": MEMORY,
    "STRUCTURED_OUTPUT": STRUCTURED_OUTPUT,
    "COLLECTOR": COLLECTOR,
}


def all_params() -> list[dict[str, Any]]:
    """Flat listing of every declared parameter, for tests and audits.

    Each entry: group name, field name, current value, Provenance, note.
    Used by tests/test_params.py to assert every field carries provenance
    metadata (i.e. nobody added a bare `float = 0.5` field and skipped
    `p(...)`).
    """
    return [
        {
            "group": group_name,
            "field": f.name,
            "value": getattr(instance, f.name),
            "provenance": f.metadata.get("provenance"),
            "note": f.metadata.get("note", ""),
        }
        for group_name, instance in _ALL_GROUPS.items()
        for f in fields(instance)
    ]
