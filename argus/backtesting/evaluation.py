"""Pre-registered outcome evaluation and separate system behavior metrics."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from argus.backtesting.metrics import trade_level_metrics
from argus.backtesting.replay import SessionResult, replay_session
from argus.outcomes import compute_realized_return, needs_reconciliation
from argus.params import EVALUATION
from argus.ports import MarketDataProvider
from argus.schemas.signals import ARGUSDecision, RiskVerdict, Signal


def signed_conviction(decision: ARGUSDecision) -> float | None:
    """Return positive bullish, negative bearish, or zero neutral conviction; return None if absent."""
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
    """Pair signed conviction with realized forward return for every reconcilable decision."""
    pairs: list[PairedOutcome] = []
    prices_by_ticker: dict[str, pd.Series] = {}
    for decision in decisions:
        conv = signed_conviction(decision)
        if conv is None or not needs_reconciliation(decision):
            continue
        # Fetch each ticker once and propagate failures so partial evaluation cannot pass silently.
        if decision.ticker not in prices_by_ticker:
            prices_by_ticker[decision.ticker] = market_data.ohlcv_daily(decision.ticker)["close"]
        outcome = compute_realized_return(decision, market_data, horizon_days, prices=prices_by_ticker[decision.ticker])
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


def rank_information_coefficient(
    pairs: Sequence[PairedOutcome],
) -> tuple[float | None, float | None]:
    """Compute Spearman correlation between signed conviction and measured forward return."""
    if len(pairs) < 2:
        return None, None
    convictions = [p.signed_conviction for p in pairs]
    returns = [p.forward_return for p in pairs]
    if len(set(convictions)) < 2 or len(set(returns)) < 2:
        return None, None
    rho, p_value = stats.spearmanr(convictions, returns)
    return float(rho), float(p_value)


def hit_rate_with_deadband(pairs: Sequence[PairedOutcome], deadband: float) -> tuple[float | None, int]:
    """Measure directional hit rate excluding neutral calls and return inside the dead band."""
    scored = [p for p in pairs if p.signed_conviction != 0 and abs(p.forward_return) > deadband]
    if not scored:
        return None, 0
    hits = sum(1 for p in scored if (p.signed_conviction > 0) == (p.forward_return > 0))
    return hits / len(scored), len(scored)


def bootstrap_ci(
    pairs: Sequence[PairedOutcome],
    statistic_fn: Callable[[Sequence[PairedOutcome]], float | None],
    n_resamples: int = EVALUATION.bootstrap_resamples,
    alpha: float = EVALUATION.ci_alpha,
    seed: int = EVALUATION.bootstrap_seed,
) -> tuple[float | None, float | None]:
    """Estimate a percentile confidence interval by resampling whole paired outcomes."""
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
    rank_ic: float | None
    rank_ic_p_value: float | None
    rank_ic_ci: tuple[float | None, float | None]
    hit_rate: float | None
    hit_rate_n: int
    hit_rate_ci: tuple[float | None, float | None]
    pairs: list[PairedOutcome] = field(default_factory=list)


def evaluate_decisions(
    decisions: list[ARGUSDecision],
    market_data: MarketDataProvider,
    horizon_days: int,
    deadband: float,
) -> EvaluationResult:
    """Run the full pre-registered evaluation over a set of completed decisions."""
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


def pre_registered_verdict(open_loop: EvaluationResult, closed_loop: EvaluationResult) -> bool:
    """Apply the pre-registered success rule to the two replay conditions."""
    if open_loop.rank_ic is None or closed_loop.rank_ic is None:
        return False
    ci_lower = closed_loop.rank_ic_ci[0]
    return ci_lower is not None and ci_lower > 0 and closed_loop.rank_ic > open_loop.rank_ic


def trade_level_win_loss_stats(pairs: Sequence[PairedOutcome]) -> dict:
    """Compute trade-level win, loss, and profit-factor statistics without a daily equity curve."""
    if not pairs:
        return {}
    return trade_level_metrics([{"return_pct": p.forward_return, "holding_days": p.holding_days} for p in pairs])


@dataclass(frozen=True)
class SystemBehaviorReport:
    """System-behavior metrics for one replayed session."""

    tickers_total: int
    decisions_built: int
    schema_validity: float
    errors: list[str]
    reduce_verdicts: int
    constraint_violations: int
    total_api_calls: int
    api_calls_per_decision: float | None
    retries_instrumented: bool = False
    tokens_instrumented: bool = False


def system_behavior_report(session_result: SessionResult) -> SystemBehaviorReport:
    """Build a SystemBehaviorReport from one replayed session's final state."""
    decisions: list[ARGUSDecision] = session_result.final_state.get("decisions") or []
    errors: list[str] = session_result.final_state.get("errors") or []
    tickers_total = len(session_result.universe)
    decisions_built = len(decisions)
    total_api_calls = sum(d.total_api_calls for d in decisions)

    return SystemBehaviorReport(
        tickers_total=tickers_total,
        decisions_built=decisions_built,
        schema_validity=decisions_built / tickers_total if tickers_total else 0.0,
        errors=list(errors),
        reduce_verdicts=sum(1 for d in decisions if d.risk is not None and d.risk.verdict == RiskVerdict.REDUCE),
        constraint_violations=0,
        total_api_calls=total_api_calls,
        api_calls_per_decision=total_api_calls / decisions_built if decisions_built else None,
    )


def check_replay_determinism(session_dir: Path) -> bool:
    """Replay the same fixture session twice, open-loop, and diffs the results."""

    def fingerprint(result: SessionResult) -> list[tuple]:
        """Return sorted ticker, signal, conviction, and API-call tuples for comparison."""
        decisions: list[ARGUSDecision] = result.final_state.get("decisions") or []
        return sorted(
            (
                d.ticker,
                d.aggregated.signal if d.aggregated else None,
                d.aggregated.conviction if d.aggregated else None,
                d.total_api_calls,
            )
            for d in decisions
        )

    first = replay_session(session_dir, closed_loop=False)
    second = replay_session(session_dir, closed_loop=False)
    return fingerprint(first) == fingerprint(second)
