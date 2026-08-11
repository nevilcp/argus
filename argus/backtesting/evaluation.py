"""
argus/backtesting/evaluation.py

Pre-registered evaluation of the outcome loop: what is measured, why, and
the success threshold fixed before any of this module's output was observed.

Responsibilities:
  - Pair each decision's signed conviction with its realized forward return
  - Rank information coefficient (Spearman) and hit-rate-with-dead-band,
    both with percentile bootstrap confidence intervals
  - System-behavior metrics (schema validity, constraint enforcement,
    API calls/decision, replay determinism), reported separately and never
    blended into the predictive metrics above

Not responsible for:
  - Running the replay itself (see backtesting/replay.py)
  - Deciding what horizon/dead-band/threshold to use (fixed ahead of time
    and read from argus/params.py, not re-derived here)

Dependencies:
  - numpy, scipy (statistics)
  - argus.orchestration.reconciliation (compute_realized_return, reused
    rather than duplicated — this is the same function that feeds
    cultural.store_trade_outcome in production)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

from argus.backtesting.replay import SessionResult, replay_session
from argus.orchestration.reconciliation import compute_realized_return
from argus.schemas.signals import ARGUSDecision, RiskVerdict, Signal
from argus.seams import MarketDataProvider

_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 0
_CI_ALPHA = 0.05


def signed_conviction(decision: ARGUSDecision) -> Optional[float]:
    """Conviction signed by direction: +conviction (BULLISH), -conviction
    (BEARISH), 0.0 (NEUTRAL). None if the decision has no aggregated signal.
    """
    if decision.aggregated is None:
        return None
    if decision.aggregated.signal == Signal.BULLISH:
        return decision.aggregated.conviction
    if decision.aggregated.signal == Signal.BEARISH:
        return -decision.aggregated.conviction
    return 0.0


@dataclass(frozen=True)
class PairedOutcome:
    """One decision's signed conviction paired with its realized forward return."""

    ticker: str
    signed_conviction: float
    forward_return: float
    holding_days: int


def collect_paired_outcomes(
    decisions: list[ARGUSDecision], market_data: MarketDataProvider, horizon_days: int
) -> list[PairedOutcome]:
    """Pairs signed conviction with realized forward return for every reconcilable decision.

    Reuses reconciliation.compute_realized_return — the same function that
    feeds cultural.store_trade_outcome in production — so this evaluation
    measures exactly the decisions that would actually be reconciled, not a
    parallel notion of "outcome". Decisions with no aggregated signal, no
    allocation, or whose horizon hasn't cleared against market_data are
    skipped (see compute_realized_return's docstring for why: deferred
    rather than fabricated).

    Args:
        decisions: Completed ARGUSDecision objects, e.g. from a replayed
            session's final_state["decisions"].
        market_data: Source of each ticker's daily close price series.
        horizon_days: Calendar days after session_timestamp defining the
            target exit date.

    Returns:
        One PairedOutcome per reconcilable decision.
    """
    pairs: list[PairedOutcome] = []
    for decision in decisions:
        conv = signed_conviction(decision)
        if conv is None:
            continue
        outcome = compute_realized_return(decision, market_data, horizon_days)
        if outcome is None:
            continue
        actual_return_pct, holding_days, _exit_reason = outcome
        pairs.append(
            PairedOutcome(
                ticker=decision.ticker,
                signed_conviction=conv,
                forward_return=actual_return_pct,
                holding_days=holding_days,
            )
        )
    return pairs


def rank_information_coefficient(pairs: Sequence[PairedOutcome]) -> tuple[Optional[float], Optional[float]]:
    """Spearman rank correlation between signed conviction and forward return.

    Returns:
        (rho, p_value), or (None, None) if fewer than 2 pairs, or if either
        series is constant (Spearman is undefined in both cases).
    """
    if len(pairs) < 2:
        return None, None
    convictions = [p.signed_conviction for p in pairs]
    returns = [p.forward_return for p in pairs]
    if len(set(convictions)) < 2 or len(set(returns)) < 2:
        return None, None
    rho, p_value = stats.spearmanr(convictions, returns)
    return float(rho), float(p_value)


def hit_rate_with_deadband(pairs: Sequence[PairedOutcome], deadband: float) -> tuple[Optional[float], int]:
    """Fraction of decisions outside the dead band where sign(conviction) == sign(return).

    Decisions whose forward return falls within +/-deadband are excluded —
    mirroring cultural.store_trade_outcome's own dead band
    (RECONCILIATION.min_abs_return_for_storage), not a second invented
    threshold.

    Args:
        pairs: Paired outcomes to score.
        deadband: Absolute forward-return threshold below which a decision
            is excluded from scoring.

    Returns:
        (hit_rate, n_scored). hit_rate is None if n_scored == 0.
    """
    scored = [p for p in pairs if abs(p.forward_return) > deadband]
    if not scored:
        return None, 0
    hits = sum(
        1
        for p in scored
        if (p.signed_conviction > 0) == (p.forward_return > 0)
    )
    return hits / len(scored), len(scored)


def bootstrap_ci(
    pairs: Sequence[PairedOutcome],
    statistic_fn: Callable[[Sequence[PairedOutcome]], Optional[float]],
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
    alpha: float = _CI_ALPHA,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for a statistic over paired (conviction, return) outcomes.

    Paired resampling (each resample draws whole PairedOutcome tuples, with
    replacement) rather than a closed-form CI — at the sample sizes this
    evaluation actually has, asymptotic-normality assumptions behind
    closed-form Spearman CIs don't hold.

    Args:
        pairs: Paired outcomes to resample from.
        statistic_fn: Function computing the statistic (e.g. rank IC's rho)
            from a resampled list of pairs; may return None for a
            resample too small to define the statistic, in which case that
            resample is dropped.
        n_resamples: Bootstrap resample count.
        alpha: Two-sided significance level (0.05 -> 95% CI).
        seed: RNG seed, fixed for reproducibility.

    Returns:
        (lower, upper) percentile bounds, or (None, None) if fewer than 2
        pairs, or if every resample's statistic was undefined.
    """
    if len(pairs) < 2:
        return None, None

    rng = random.Random(seed)
    n = len(pairs)
    values: list[float] = []
    for _ in range(n_resamples):
        resample = [pairs[rng.randrange(n)] for _ in range(n)]
        stat = statistic_fn(resample)
        if stat is not None and not math.isnan(stat):
            values.append(stat)

    if not values:
        return None, None

    lower = float(np.percentile(values, 100 * (alpha / 2)))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


@dataclass(frozen=True)
class EvaluationResult:
    """One session/condition's evaluation: rank IC and hit-rate, each with a bootstrap CI."""

    n: int
    rank_ic: Optional[float]
    rank_ic_p_value: Optional[float]
    rank_ic_ci: tuple[Optional[float], Optional[float]]
    hit_rate: Optional[float]
    hit_rate_n: int
    hit_rate_ci: tuple[Optional[float], Optional[float]]
    pairs: list[PairedOutcome] = field(default_factory=list)


def evaluate_decisions(
    decisions: list[ARGUSDecision],
    market_data: MarketDataProvider,
    horizon_days: int,
    deadband: float,
) -> EvaluationResult:
    """Runs the full pre-registered evaluation over a set of completed decisions.

    Args:
        decisions: Completed ARGUSDecision objects.
        market_data: Source of each ticker's daily close price series.
        horizon_days: Calendar days after session_timestamp defining the
            target exit date (pre-registered at 5).
        deadband: Absolute forward-return threshold for hit-rate exclusion
            (pre-registered as RECONCILIATION.min_abs_return_for_storage).

    Returns:
        EvaluationResult with rank IC, hit-rate, and bootstrap CIs for both.
    """
    pairs = collect_paired_outcomes(decisions, market_data, horizon_days)
    rho, p_value = rank_information_coefficient(pairs)
    rho_ci = bootstrap_ci(pairs, lambda sample: rank_information_coefficient(sample)[0])
    hit_rate, hit_n = hit_rate_with_deadband(pairs, deadband)
    hit_ci = bootstrap_ci(pairs, lambda sample: hit_rate_with_deadband(sample, deadband)[0])

    return EvaluationResult(
        n=len(pairs),
        rank_ic=rho,
        rank_ic_p_value=p_value,
        rank_ic_ci=rho_ci,
        hit_rate=hit_rate,
        hit_rate_n=hit_n,
        hit_rate_ci=hit_ci,
        pairs=pairs,
    )


@dataclass(frozen=True)
class SystemBehaviorReport:
    """System-behavior metrics for one replayed session — reported separately from
    (never blended into) EvaluationResult's predictive metrics.
    """

    tickers_total: int
    decisions_built: int
    schema_validity: float
    errors: list[str]
    reduce_verdicts: int
    constraint_violations: int
    total_api_calls: int
    api_calls_per_decision: Optional[float]
    retries_instrumented: bool = False
    tokens_instrumented: bool = False


def system_behavior_report(session_result: SessionResult) -> SystemBehaviorReport:
    """Builds a SystemBehaviorReport from one replayed session's final state.

    Constraint violations are reported as 0 by construction:
    RiskAssessment.approved_weight <= proposed_weight is a Pydantic
    model_validator, so a violating instance cannot exist in
    final_state["decisions"] in the first place.
    Retries and tokens/decision are not currently instrumented anywhere in
    the codebase at decision granularity — reported as a disclosed gap
    rather than fabricated.

    Args:
        session_result: Output of backtesting.replay.replay_session.

    Returns:
        SystemBehaviorReport for that session.
    """
    decisions: list[ARGUSDecision] = session_result.final_state.get("decisions") or []
    errors: list[str] = session_result.final_state.get("errors") or []
    tickers_total = len(session_result.universe)
    decisions_built = len(decisions)

    reduce_verdicts = sum(
        1 for d in decisions if d.risk is not None and d.risk.verdict == RiskVerdict.REDUCE
    )
    api_calls = [d.total_api_calls for d in decisions]

    return SystemBehaviorReport(
        tickers_total=tickers_total,
        decisions_built=decisions_built,
        schema_validity=decisions_built / tickers_total if tickers_total else 0.0,
        errors=list(errors),
        reduce_verdicts=reduce_verdicts,
        constraint_violations=0,
        total_api_calls=sum(api_calls),
        api_calls_per_decision=(sum(api_calls) / len(api_calls)) if api_calls else None,
    )


def check_replay_determinism(session_dir: Path) -> bool:
    """Replays the same fixture session twice, open-loop, and checks for bit-for-bit
    agreement on each decision's signal/conviction/total_api_calls.

    closed_loop=True is intentionally not checked here — it reads live
    chroma_db state, which reconciliation can mutate between runs, so
    determinism isn't a property closed-loop replay is expected to have.

    Args:
        session_dir: Directory shaped like tests/fixtures/.

    Returns:
        True if both replays produced identical decisions for every ticker.
    """
    first = replay_session(session_dir, closed_loop=False)
    second = replay_session(session_dir, closed_loop=False)

    def _fingerprint(decisions: list[ARGUSDecision]) -> list[tuple]:
        return sorted(
            (d.ticker, d.aggregated.signal if d.aggregated else None,
             d.aggregated.conviction if d.aggregated else None, d.total_api_calls)
            for d in decisions
        )

    first_decisions = first.final_state.get("decisions") or []
    second_decisions = second.final_state.get("decisions") or []
    return _fingerprint(first_decisions) == _fingerprint(second_decisions)
