"""Deterministic capital protection and portfolio risk optimization agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from argus.config import settings
from argus.data.cache import TTLCache
from argus.infra.clock import now_local
from argus.params import RISK, SYSTEM
from argus.ports import MarketDataProvider
from argus.schemas.signals import UNKNOWN_SECTOR, RiskAssessment, RiskVerdict
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger(__name__)

# Share note prefixes with the graph risk node so degraded measurements remain visible in graph errors.
DIVERSIFICATION_FLOOR_PREFIX = "Below diversification floor:"
COVARIANCE_NOTE_PREFIX = "Covariance:"

# Bound sector lookups independently of TTL while allowing daily corporate reclassification refreshes.
_SECTOR_CACHE: TTLCache[str, str] = TTLCache(
    ttl=timedelta(seconds=RISK.sector_cache_ttl_seconds),
    max_entries=RISK.sector_cache_max_entries,
)


def get_sector(ticker: str, market_data: MarketDataProvider) -> str:
    """Fetch the GICS sector, caching live successes while isolating fixture sessions."""
    is_live = market_data.is_live

    if is_live:
        cached = _SECTOR_CACHE.get(ticker)
        if cached is not None:
            return cached

    try:
        info = market_data.ticker_info(ticker)
        sector = info.get("sector", UNKNOWN_SECTOR)
        if is_live:
            _SECTOR_CACHE.set(ticker, sector)
        return sector
    except Exception as exc:
        # Leave transient failures uncached so they cannot pin the unknown sector for a full TTL.
        logger.warning("Failed to fetch sector for %s, defaulting to %r: %s", ticker, UNKNOWN_SECTOR, exc)
        return UNKNOWN_SECTOR


def compute_asset_returns(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    lookback: int = RISK.returns_lookback_days,
    dropped: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate unweighted daily asset returns over the lookback window."""
    returns_dict = {}
    for pos in positions:
        ticker = pos["ticker"]
        if ticker in price_history:
            series = price_history[ticker].tail(lookback + 1)
            ret = series.pct_change().dropna()
            if len(ret) < RISK.min_beta_overlap_points:
                if dropped is not None:
                    dropped.append(ticker)
                continue
            returns_dict[ticker] = ret

    df = pd.DataFrame(returns_dict).dropna()
    return df


def compute_portfolio_returns(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    lookback: int = RISK.returns_lookback_days,
) -> pd.DataFrame:
    """Calculate weighted daily portfolio returns over the lookback window."""
    return weight_asset_returns(compute_asset_returns(positions, price_history, lookback), positions)


def weight_asset_returns(asset_returns: pd.DataFrame, positions: list[dict]) -> pd.DataFrame:
    """Scale each asset's daily returns by its proposed weight."""
    # Align weights to surviving asset columns so dropped tickers cannot reappear as NaN columns.
    weights = pd.Series({pos["ticker"]: pos["weight"] for pos in positions}).reindex(asset_returns.columns)
    return asset_returns.mul(weights)


def historical_var(portfolio_returns: pd.Series, confidence: float = RISK.var_confidence) -> float | None:
    """Evaluate historical Value-at-Risk (VaR) at a target confidence percentile."""
    if portfolio_returns.empty:
        return None
    percentile = (1.0 - confidence) * 100.0
    val = float(np.percentile(portfolio_returns.dropna(), percentile))
    return abs(val)


def conditional_var(portfolio_returns: pd.Series, confidence: float = RISK.cvar_confidence) -> float | None:
    """Calculate expected loss beyond the historical VaR threshold."""
    if portfolio_returns.empty:
        return None
    var = historical_var(portfolio_returns, confidence)
    if var is None:
        raise RuntimeError("Non-empty return history did not yield a measurable VaR")
    tail = portfolio_returns[portfolio_returns <= -var]
    return abs(float(tail.mean())) if len(tail) > 0 else var


def ols_portfolio_beta(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    benchmark_ticker: str = "SPY",
    market_data: MarketDataProvider | None = None,
    lookback: int = RISK.returns_lookback_days,
) -> float | None:
    """Compute weighted portfolio beta against a benchmark over aligned dates."""
    if not positions:
        return 0.0

    if benchmark_ticker not in price_history:
        try:
            spy_df = (market_data or LiveMarketDataProvider()).ohlcv_daily(benchmark_ticker, period="1y")
            spy_returns = spy_df["close"].tail(lookback + 1).pct_change().dropna()
        except Exception:
            logger.warning("benchmark fetch failed, beta unavailable")
            return None
    else:
        spy_returns = price_history[benchmark_ticker].tail(lookback + 1).pct_change().dropna()

    spy_var = spy_returns.var()
    if spy_var == 0 or pd.isna(spy_var):
        return None

    betas = []
    weights = []

    for pos in positions:
        ticker = pos["ticker"]
        w = pos["weight"]
        if ticker in price_history:
            ret = price_history[ticker].tail(lookback + 1).pct_change().dropna()
            df = pd.DataFrame({"asset": ret, "spy": spy_returns}).dropna()
            if len(df) > RISK.min_beta_overlap_points:
                cov = df.cov().iloc[0, 1]
                beta = cov / spy_var
                betas.append(beta)
                weights.append(w)

    if len(betas) != len(positions):
        return None

    return float(np.average(betas, weights=weights))


def avg_pairwise_correlation(returns_matrix: pd.DataFrame) -> float | None:
    """Compute mean pairwise correlation across asset return series."""
    if returns_matrix.shape[1] < 2:
        return 0.0
    corr = returns_matrix.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    mean_corr = float(upper.stack().mean())
    return mean_corr if not np.isnan(mean_corr) else None


def close_to_close_stop_losses(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    stop_multiplier: float = RISK.stop_multiplier,
) -> dict[str, float]:
    """Derive dynamic stop-loss bounds from the mean absolute close-to-close change."""
    stops = {}
    for pos in positions:
        ticker = pos["ticker"]
        if ticker in price_history:
            series = price_history[ticker]
            if len(series) > RISK.stop_lookback_period:
                mean_abs_change = series.diff().abs().dropna().tail(RISK.stop_lookback_period).mean()
                latest_close = float(series.iloc[-1])
                stops[ticker] = max(0.0, latest_close - (mean_abs_change * stop_multiplier))
    return stops


def component_var(returns_matrix: pd.DataFrame, portfolio_returns: pd.Series) -> dict[str, float]:
    """Calculate marginal Value-at-Risk (mVaR) contributions for each portfolio asset."""
    mvar = {}
    port_std = portfolio_returns.std()
    if port_std == 0 or pd.isna(port_std):
        return dict.fromkeys(returns_matrix.columns, 0.0)

    for col in returns_matrix.columns:
        cov = np.cov(returns_matrix[col], portfolio_returns)[0, 1]
        mvar[col] = float(cov / port_std)
    return mvar


@dataclass(frozen=True)
class _RiskStats:
    """The measured statistics shared by the REDUCE and APPROVE assessments."""

    var_99: float | None
    cvar: float | None
    beta: float | None
    avg_correlation: float | None
    optimal_weights: dict[str, float]
    optimizer_converged: bool | None


def _veto_assessment(total_weight: float, reasons: list[str]) -> RiskAssessment:
    """Build the VETO assessment: nothing approved and no statistics were computed."""
    return RiskAssessment(
        verdict=RiskVerdict.VETO,
        approved_weight=0.0,
        proposed_weight=total_weight,
        veto_reasons=reasons,
        optimal_weights={},
        var_99=None,
        cvar=None,
        marginal_var=None,
        portfolio_beta=None,
        avg_correlation=None,
        stop_loss=None,
        api_calls_used=0,
        timestamp=now_local(),
    )


def _reduce_assessment(total_weight: float, reasons: list[str], stats: _RiskStats) -> RiskAssessment:
    """Build the REDUCE assessment: the book is scaled down and no stop or marginal VaR is set."""
    return RiskAssessment(
        verdict=RiskVerdict.REDUCE,
        approved_weight=min(total_weight * RISK.reduce_weight_multiplier, total_weight),
        proposed_weight=total_weight,
        veto_reasons=reasons,
        optimal_weights=stats.optimal_weights,
        optimizer_converged=stats.optimizer_converged,
        var_99=stats.var_99,
        cvar=stats.cvar,
        marginal_var=None,
        portfolio_beta=stats.beta,
        avg_correlation=stats.avg_correlation,
        stop_loss=None,
        api_calls_used=0,
        timestamp=now_local(),
    )


def _approve_assessment(
    total_weight: float,
    notes: list[str],
    stats: _RiskStats,
    marginal_var: float | None,
    stop_loss: float | None,
) -> RiskAssessment:
    """Build the APPROVE assessment: the full proposed weight with the primary ticker's risk."""
    return RiskAssessment(
        verdict=RiskVerdict.APPROVE,
        approved_weight=total_weight,
        proposed_weight=total_weight,
        veto_reasons=notes,
        optimal_weights=stats.optimal_weights,
        optimizer_converged=stats.optimizer_converged,
        var_99=stats.var_99,
        cvar=stats.cvar,
        marginal_var=marginal_var,
        portfolio_beta=stats.beta,
        avg_correlation=stats.avg_correlation,
        stop_loss=stop_loss,
        api_calls_used=0,
        timestamp=now_local(),
    )


class RiskStatisticalAgent:
    """Core risk management gateway enforcing statistical allocation limits."""

    def __init__(self, market_data: MarketDataProvider | None = None) -> None:
        self.max_position_pct = settings.MAX_SINGLE_POSITION_PCT
        self.max_sector_pct = settings.MAX_SECTOR_CONCENTRATION
        self.vix_blackout = settings.VIX_BLACKOUT_THRESHOLD
        self.max_port_beta = settings.MAX_PORTFOLIO_BETA
        self.market_data = market_data or LiveMarketDataProvider()

    def _structural_violations(
        self,
        proposed_positions: list[dict],
        current_vix: float | None,
        total_weight: float,
    ) -> tuple[list[str], str | None]:
        """Run gate 1: per-position caps, book size, and the VIX blackout."""
        violations = [
            f"{pos['ticker']} weight {pos['weight']:.1%} > limit {self.max_position_pct:.1%}"
            for pos in proposed_positions
            if pos["weight"] > self.max_position_pct
        ]

        # Report below-floor diversification as information; a small book alone does not require a veto.
        diversification_note: str | None = None
        if (
            len(proposed_positions) > 1
            and len(proposed_positions) < RISK.min_positions_diversification
            and total_weight > 0
        ):
            diversification_note = (
                f"{DIVERSIFICATION_FLOOR_PREFIX} {len(proposed_positions)} positions "
                f"(min {RISK.min_positions_diversification})"
            )

        if len(proposed_positions) > RISK.max_positions:
            # Keep the universe cap for direct callers as well as HTTP requests.
            violations.append(f"Over-diversification: {len(proposed_positions)} positions (max {RISK.max_positions})")

        if current_vix is None:
            violations.append("VIX unavailable — blackout check cannot be performed")
        elif current_vix >= self.vix_blackout:
            violations.append(f"VIX {current_vix:.1f} >= blackout threshold {self.vix_blackout:.1f}")

        return violations, diversification_note

    def _solve_optimal_weights(
        self,
        proposed_positions: list[dict],
        returns_df: pd.DataFrame,
        dropped_tickers: list[str],
        sector_to_tickers: dict[str, list],
        convictions: dict[str, float] | None,
    ) -> tuple[dict[str, float], bool | None, list[str]]:
        """Run gate 2: the SLSQP solve for allocation ceilings under per-sector caps."""
        optimizer_notes: list[str] = []
        if len(proposed_positions) <= 1:
            return {}, None, optimizer_notes

        if dropped_tickers:
            optimizer_notes.append(
                f"{COVARIANCE_NOTE_PREFIX} excluded {', '.join(dropped_tickers)} — fewer than "
                f"{RISK.min_beta_overlap_points} overlapping return observations"
            )
        if returns_df.empty:
            return {}, None, optimizer_notes

        cov = returns_df.cov() * SYSTEM.trading_days_per_year
        if not np.isfinite(cov.to_numpy()).all():
            # Reject non-finite covariance after alignment because SLSQP cannot recover from it.
            optimizer_notes.append(
                f"{COVARIANCE_NOTE_PREFIX} non-finite values in the covariance matrix — SLSQP optimization skipped"
            )
            return {}, None, optimizer_notes

        tickers = list(returns_df.columns)
        n = len(tickers)

        def _obj(w: np.ndarray, _conv: dict[str, float] | None = convictions) -> float:
            """Minimize variance penalized by signed conviction, bound at definition time."""
            variance = float(np.dot(w, np.dot(cov, w)))
            if _conv:
                alpha = sum(w[i] * _conv.get(tickers[i], 0.0) for i in range(n))
                return -alpha + RISK.slsqp_risk_aversion * variance
            return variance

        w0 = np.full(n, min(self.max_position_pct, 1.0 / n))
        bounds = tuple((0.0, self.max_position_pct) for _ in range(n))

        cons = []
        for sec_tickers in sector_to_tickers.values():
            idxs = [i for i, t in enumerate(tickers) if t in sec_tickers]
            cap = self.max_sector_pct
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, idx=idxs, c=cap: c - sum(w[i] for i in idx),
                }
            )
        # A long-only book cannot deploy more capital than it has.
        cons.append({"type": "ineq", "fun": lambda w: RISK.slsqp_max_total_deployment - np.sum(w)})

        res = minimize(
            _obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": RISK.slsqp_ftol, "maxiter": RISK.slsqp_maxiter},
        )
        optimizer_converged = bool(res.success)
        if not res.success:
            logger.warning("SLSQP solve did not converge: %s", res.message)
            return {}, optimizer_converged, optimizer_notes

        optimal_weights = {tickers[i]: float(res.x[i]) for i in range(n)}
        logger.debug(
            "SLSQP optimal weights: %s",
            {t: f"{w:.3f}" for t, w in optimal_weights.items()},
        )
        return optimal_weights, optimizer_converged, optimizer_notes

    def _statistical_violations(
        self,
        var99: float | None,
        cvar: float | None,
        beta: float | None,
        corr: float | None,
        sector_weights: dict[str, float],
        has_sector_violation: bool,
    ) -> list[str]:
        """Run gate 3: the VaR, CVaR, beta, correlation, and sector-cap thresholds."""
        stat_violations: list[str] = []
        if var99 is not None and var99 > RISK.var_limit:
            stat_violations.append(f"VaR 99%: {var99:.2%} > {RISK.var_limit:.0%} limit")
        if cvar is not None and cvar > RISK.cvar_limit:
            stat_violations.append(f"CVaR: {cvar:.2%} > {RISK.cvar_limit:.0%} limit")
        if beta is not None and beta > self.max_port_beta:
            stat_violations.append(f"Beta {beta:.2f} > {self.max_port_beta:.2f}")
        if corr is not None and corr > RISK.correlation_limit:
            stat_violations.append(f"Avg correlation > {RISK.correlation_limit} — reduce overlap")
        if has_sector_violation:
            for sec, w in sector_weights.items():
                if w > self.max_sector_pct:
                    stat_violations.append(
                        f"Sector '{sec}' weight {w:.1%} > limit {self.max_sector_pct:.1%} (optimizer applied caps)"
                    )

        return stat_violations

    def evaluate(
        self,
        proposed_positions: list[dict],
        price_history: dict[str, pd.Series],
        current_vix: float | None,
        convictions: dict[str, float] | None = None,
    ) -> RiskAssessment:
        """Evaluate a proposed portfolio posture against structural, statistical, and sector caps."""
        total_weight = sum(p["weight"] for p in proposed_positions)

        violations, diversification_note = self._structural_violations(proposed_positions, current_vix, total_weight)
        diversification_notes = [diversification_note] if diversification_note else []

        if violations:
            logger.debug("Risk evaluate: VETO due to %s", violations)
            return _veto_assessment(total_weight, violations + diversification_notes)

        sector_weights: dict[str, float] = {}
        sector_to_tickers: dict[str, list] = {}
        for pos in proposed_positions:
            sector = get_sector(pos["ticker"], self.market_data)
            sector_weights[sector] = sector_weights.get(sector, 0.0) + pos["weight"]
            sector_to_tickers.setdefault(sector, []).append(pos["ticker"])

        has_sector_violation = any(w > self.max_sector_pct for w in sector_weights.values())

        dropped_tickers: list[str] = []
        asset_returns = compute_asset_returns(proposed_positions, price_history, dropped=dropped_tickers)
        optimal_weights, optimizer_converged, optimizer_notes = self._solve_optimal_weights(
            proposed_positions, asset_returns, dropped_tickers, sector_to_tickers, convictions
        )

        returns = weight_asset_returns(asset_returns, proposed_positions)
        port_returns = returns.sum(axis=1) if not returns.empty else pd.Series(dtype=float)

        normalized_returns = port_returns / total_weight if total_weight > 0 else port_returns
        var99 = historical_var(normalized_returns)
        cvar = conditional_var(normalized_returns)
        beta = ols_portfolio_beta(proposed_positions, price_history, market_data=self.market_data)
        corr = avg_pairwise_correlation(returns)

        # Report unmeasurable statistics as None instead of fabricating zero and silently passing a gate.
        degraded_notes: list[str] = []
        if var99 is None or cvar is None:
            degraded_notes.append("VaR/CVaR not measurable — insufficient return history for this book")
        if beta is None:
            degraded_notes.append("Portfolio beta not measurable — insufficient benchmark overlap")
        if corr is None:
            degraded_notes.append("Average correlation not measurable — positions share no overlapping return history")
        notes = diversification_notes + optimizer_notes + degraded_notes

        stat_violations = self._statistical_violations(var99, cvar, beta, corr, sector_weights, has_sector_violation)
        stats = _RiskStats(var99, cvar, beta, corr, optimal_weights, optimizer_converged)

        if stat_violations:
            logger.debug("Risk evaluate: REDUCE due to %s", stat_violations)
            return _reduce_assessment(total_weight, stat_violations + notes, stats)

        logger.debug(
            "Risk evaluate: APPROVE (VaR99: %s, Beta: %s)",
            f"{var99:.2%}" if var99 is not None else "not measurable",
            beta,
        )
        primary_ticker = proposed_positions[0]["ticker"] if proposed_positions else ""
        stop_val = close_to_close_stop_losses(proposed_positions, price_history).get(primary_ticker)
        mvar_val = component_var(returns, port_returns).get(primary_ticker)
        return _approve_assessment(total_weight, notes, stats, mvar_val, stop_val)
