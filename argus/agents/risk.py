"""
argus/agents/risk.py

Deterministic capital protection and portfolio risk optimization agent.

Responsibilities:
  - Compute portfolio-level metrics (VaR, CVaR, Beta, average pairwise correlation)
  - Run a sequential SLSQP solver to cap exposures across tickers and sectors
  - Enforce structural, statistical, and blackout gates before returning a verdict

Not responsible for:
  - Portfolio allocation decisions (see agents/portfolio.py)
  - Kill-switch monitoring (see risk/kill_switch.py)
  - LLM-backed signal synthesis (see agents/fundamental.py, agents/sentiment.py)

Dependencies:
  - scipy (SLSQP optimizer)
  - yfinance (GICS sector lookup)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from argus.config import settings
from argus.params import RISK
from argus.schemas.signals import RiskAssessment, RiskVerdict
from argus.seams import LiveMarketDataProvider, MarketDataProvider

logger = logging.getLogger("argus.risk")

# Stores (sector_string, cached_at) per ticker; 24h TTL reflects corporate reclassifications
# (M&A, spin-offs) without a restart
_SECTOR_CACHE: dict[str, tuple[str, datetime]] = {}
_SECTOR_CACHE_TTL_SECONDS = RISK.sector_cache_ttl_seconds  # 24 hours


def get_sector(ticker: str) -> str:
    """Dynamically retrieves and caches the GICS sector classification for a ticker.

    Uses a module-level dict with a 24-hour TTL to avoid redundant yfinance
    round-trips within a session while still reflecting corporate reclassifications
    (e.g. acquisitions, spin-offs) across multi-day server runs.

    Args:
        ticker: Equity ticker symbol.

    Returns:
        GICS sector string (e.g. 'Technology'), or 'Unknown' on fetch failure.
    """
    if ticker in _SECTOR_CACHE:
        sector, cached_at = _SECTOR_CACHE[ticker]
        if (datetime.now() - cached_at).total_seconds() < _SECTOR_CACHE_TTL_SECONDS:
            return sector

    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown")
        _SECTOR_CACHE[ticker] = (sector, datetime.now())
        return sector
    except Exception:
        logger.warning(f"Failed to fetch sector for {ticker}, defaulting to 'Unknown'")
        _SECTOR_CACHE[ticker] = ("Unknown", datetime.now())
        return "Unknown"


def compute_portfolio_returns(
    positions: list[dict], price_history: dict[str, pd.Series], lookback: int = RISK.returns_lookback_days
) -> pd.DataFrame:
    """Calculates daily historical returns adjusted by proposed weights over a lookback window.

    Args:
        positions: List of dicts with keys ``ticker`` and ``weight``.
        price_history: Mapping of ticker → daily close price Series.
        lookback: Number of trading days to include (default 252 = 1 year).

    Returns:
        DataFrame of weighted daily returns per ticker, with NaN rows dropped.
    """
    returns_dict = {}
    for pos in positions:
        ticker = pos["ticker"]
        weight = pos["weight"]
        if ticker in price_history:
            series = price_history[ticker].tail(lookback + 1)
            returns_dict[ticker] = series.pct_change().dropna() * weight

    df = pd.DataFrame(returns_dict).dropna()
    return df


def historical_var(portfolio_returns: pd.Series, confidence: float = RISK.var_confidence) -> float:
    """Evaluates historical Value-at-Risk (VaR) at a target confidence percentile.

    Args:
        portfolio_returns: Daily portfolio return Series.
        confidence: Confidence level for VaR calculation (default 0.99 = 99%).

    Returns:
        Positive VaR value representing the worst-case daily loss at the given confidence.
    """
    if portfolio_returns.empty:
        return 0.0
    percentile = (1.0 - confidence) * 100.0
    val = float(np.percentile(portfolio_returns.dropna(), percentile))
    return abs(val)


def conditional_var(portfolio_returns: pd.Series, confidence: float = RISK.cvar_confidence) -> float:
    """Calculates Conditional Value-at-Risk (CVaR / Expected Shortfall) in the tail distribution.

    Args:
        portfolio_returns: Daily portfolio return Series.
        confidence: Confidence level (default 0.99).

    Returns:
        Positive CVaR value, or the VaR value if no returns fall below the VaR threshold.
    """
    if portfolio_returns.empty:
        return 0.0
    var = historical_var(portfolio_returns, confidence)
    tail = portfolio_returns[portfolio_returns <= -var]
    return abs(float(tail.mean())) if len(tail) > 0 else var


def ols_portfolio_beta(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    benchmark_ticker: str = "SPY",
    market_data: Optional[MarketDataProvider] = None,
) -> float:
    """Computes weighted Ordinary Least Squares (OLS) portfolio beta against a benchmark index.

    Args:
        positions: List of dicts with keys ``ticker`` and ``weight``.
        price_history: Mapping of ticker → daily close price Series.
        benchmark_ticker: Index to use as market proxy (default 'SPY').
        market_data: Source for the benchmark series when it isn't already in
            ``price_history``. Defaults to a live fetch.

    Returns:
        Dollar-weighted portfolio beta. Returns 1.0 if the benchmark cannot be fetched or
        if fewer than 10 overlapping data points exist for any constituent.
    """
    if not positions:
        return 0.0

    if benchmark_ticker not in price_history:
        try:
            spy_df = (market_data or LiveMarketDataProvider()).ohlcv_daily(
                benchmark_ticker, period="1y"
            )
            spy_returns = spy_df["close"].pct_change().dropna()
        except Exception:
            logger.warning("ols_portfolio_beta: benchmark fetch failed, returning 1.0")
            return 1.0
    else:
        spy_returns = price_history[benchmark_ticker].pct_change().dropna()

    spy_var = spy_returns.var()
    if spy_var == 0 or pd.isna(spy_var):
        return 1.0

    betas = []
    weights = []

    for pos in positions:
        ticker = pos["ticker"]
        w = pos["weight"]
        if ticker in price_history:
            ret = price_history[ticker].pct_change().dropna()
            df = pd.DataFrame({"asset": ret, "spy": spy_returns}).dropna()
            if len(df) > RISK.min_beta_overlap_points:
                cov = df.cov().iloc[0, 1]
                beta = cov / spy_var
                betas.append(beta)
                weights.append(w)

    if not betas:
        return 1.0

    return float(np.average(betas, weights=weights))


def avg_pairwise_correlation(returns_matrix: pd.DataFrame) -> float:
    """Computes the mean pairwise correlation coefficients for a matrix of asset returns.

    Args:
        returns_matrix: DataFrame where each column is a weighted return series for one asset.

    Returns:
        Mean upper-triangle pairwise correlation, or 0.0 for single-asset portfolios.
    """
    if returns_matrix.shape[1] < 2:
        return 0.0
    corr = returns_matrix.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return float(upper.stack().mean())


def atr_stop_losses(
    positions: list[dict], price_history: dict[str, pd.Series], atr_multiplier: float = RISK.atr_multiplier
) -> dict[str, float]:
    """Derives dynamic stop-loss bounds using Average True Range (ATR-14) metrics.

    Args:
        positions: List of dicts with key ``ticker``.
        price_history: Mapping of ticker → daily close price Series.
        atr_multiplier: Multiplier applied to ATR-14 to set the stop distance (default 2.5).

    Returns:
        Mapping of ticker → stop-loss price level. Tickers with fewer than 14 data points
        are excluded.
    """
    stops = {}
    for pos in positions:
        ticker = pos["ticker"]
        if ticker in price_history:
            series = price_history[ticker]
            if len(series) > RISK.atr_period:
                # NOTE: true_range is close-to-close diff, not H-L-Cprev — understates ATR for
                # volatile stocks with large intraday gaps; pass OHLCV data here for precision
                true_range = series.diff().abs().dropna()
                atr_14 = true_range.tail(RISK.atr_period).mean()
                latest_close = float(series.iloc[-1])
                stops[ticker] = max(0.0, latest_close - (atr_14 * atr_multiplier))
    return stops


def component_var(returns_matrix: pd.DataFrame, portfolio_returns: pd.Series) -> dict[str, float]:
    """Calculates marginal Value-at-Risk (mVaR) contributions for each portfolio asset.

    Args:
        returns_matrix: DataFrame of weighted returns per asset.
        portfolio_returns: Aggregated portfolio return Series.

    Returns:
        Mapping of asset column name → marginal VaR contribution.
        Returns 0.0 for all assets if portfolio standard deviation is zero or NaN.
    """
    mvar = {}
    port_std = portfolio_returns.std()
    if port_std == 0 or pd.isna(port_std):
        return {col: 0.0 for col in returns_matrix.columns}

    for col in returns_matrix.columns:
        cov = np.cov(returns_matrix[col], portfolio_returns)[0, 1]
        mvar[col] = float(cov / port_std)
    return mvar


class RiskStatisticalEngine:
    """Core risk management gateway enforcing statistical allocation limits.

    Runs three sequential gates:
      1. Structural rules (position caps, diversification, VIX blackout)
      2. SLSQP optimization for multi-asset portfolios
      3. Statistical thresholds (VaR, CVaR, Beta, correlation)

    Returns a RiskAssessment with APPROVE, REDUCE, or VETO verdict.
    """

    def __init__(self, market_data: Optional[MarketDataProvider] = None) -> None:
        """Loads gate thresholds from settings.

        Args:
            market_data: Provider for price/beta lookups; defaults to live fetches.
        """
        self.max_position_pct = settings.MAX_SINGLE_POSITION_PCT
        self.max_sector_pct = settings.MAX_SECTOR_CONCENTRATION
        self.vix_blackout = settings.VIX_BLACKOUT_THRESHOLD
        self.max_port_beta = settings.MAX_PORTFOLIO_BETA
        self.market_data = market_data or LiveMarketDataProvider()

    def evaluate(
        self,
        proposed_positions: list[dict],
        price_history: dict[str, pd.Series],
        current_vix: float,
        convictions: Optional[dict[str, float]] = None,
    ) -> RiskAssessment:
        """Evaluates a proposed portfolio posture against structural, statistical, and sector caps.

        Invokes a sequential quadratic solver (SLSQP) to establish optimal allocation ceilings,
        runs structural rule verifications, and calculates trailing stop-loss values.

        Args:
            proposed_positions: List of dicts with keys ``ticker`` and ``weight``.
            price_history: Mapping of ticker → daily close price Series.
            current_vix: Current CBOE VIX level for blackout checking.
            convictions: Optional mapping of ticker → signed conviction score used by the SLSQP
                objective. Positive = BULLISH, negative = BEARISH, zero = NEUTRAL.

        Returns:
            RiskAssessment with verdict, optimal weights, and risk statistics.
        """
        violations = []

        total_weight = sum(p["weight"] for p in proposed_positions)

        for pos in proposed_positions:
            if pos["weight"] > self.max_position_pct:
                violations.append(
                    f"{pos['ticker']} weight {pos['weight']:.1%} > limit {self.max_position_pct:.1%}"
                )

        if (
            len(proposed_positions) > 1
            and len(proposed_positions) < RISK.min_positions_diversification
            and total_weight > 0
        ):
            violations.append(
                f"Insufficient diversification: {len(proposed_positions)} positions (min {RISK.min_positions_diversification})"
            )

        if len(proposed_positions) > RISK.max_positions:
            violations.append(
                f"Over-diversification: {len(proposed_positions)} positions (max {RISK.max_positions})"
            )

        if current_vix >= self.vix_blackout:
            violations.append(
                f"VIX {current_vix:.1f} >= blackout threshold {self.vix_blackout:.1f}"
            )

        if violations:
            logger.debug("Risk evaluate: VETO due to %s", violations)
            return RiskAssessment(
                verdict=RiskVerdict.VETO,
                approved_weight=0.0,
                proposed_weight=total_weight,
                veto_reasons=violations,
                optimal_weights={},
                var_99=0.0,
                cvar=0.0,
                marginal_var=0.0,
                portfolio_beta=0.0,
                avg_correlation=0.0,
                stop_loss=0.0,
                api_calls_used=0,
                timestamp=datetime.now(),
            )

        sector_weights: dict[str, float] = {}
        sector_to_tickers: dict[str, list] = {}
        for pos in proposed_positions:
            sector = get_sector(pos["ticker"])
            sector_weights[sector] = sector_weights.get(sector, 0.0) + pos["weight"]
            sector_to_tickers.setdefault(sector, []).append(pos["ticker"])

        has_sector_violation = any(w > self.max_sector_pct for w in sector_weights.values())

        optimal_weights: dict[str, float] = {}
        if len(proposed_positions) > 1:
            returns_df = compute_portfolio_returns(proposed_positions, price_history)
            if not returns_df.empty:
                cov = returns_df.cov() * 252
                tickers = list(returns_df.columns)
                n = len(tickers)

                def _obj(w: np.ndarray, _conv: Optional[dict[str, float]] = convictions) -> float:
                    """Minimises variance penalised by signed conviction.

                    Signed conviction (passed from graph.py):
                      +conviction for BULLISH → optimizer drives weight UP
                      -conviction for BEARISH → optimizer drives weight to 0
                       0 for NEUTRAL          → pure variance minimisation

                    ``_conv`` is bound at definition time to prevent closure capture
                    from picking up a mutated reference if convictions changes later.
                    """
                    variance = float(np.dot(w, np.dot(cov, w)))
                    if _conv:
                        alpha = sum(w[i] * _conv.get(tickers[i], 0.0) for i in range(n))
                        # Lambda=1.0 trades off return conviction against variance equally
                        return -alpha + RISK.slsqp_risk_aversion * variance
                    return variance

                w0 = np.full(n, min(self.max_position_pct, 1.0 / n))
                bounds = tuple((0.0, self.max_position_pct) for _ in range(n))

                cons = []
                for sec, sec_tickers in sector_to_tickers.items():
                    idxs = [i for i, t in enumerate(tickers) if t in sec_tickers]
                    cap = self.max_sector_pct
                    cons.append(
                        {
                            "type": "ineq",
                            "fun": lambda w, idx=idxs, c=cap: c - sum(w[i] for i in idx),
                        }
                    )
                # Require aggregate equity deployment of at least 50%
                cons.append(
                    {"type": "ineq", "fun": lambda w: np.sum(w) - RISK.slsqp_min_equity_deployment}
                )

                res = minimize(
                    _obj,
                    w0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons,
                    options={"ftol": RISK.slsqp_ftol, "maxiter": RISK.slsqp_maxiter},
                )
                if res.success:
                    optimal_weights = {tickers[i]: float(res.x[i]) for i in range(n)}
                    logger.debug(
                        "[Risk] SLSQP optimal weights: %s",
                        {t: f"{w:.3f}" for t, w in optimal_weights.items()},
                    )

        returns = compute_portfolio_returns(proposed_positions, price_history)
        port_returns = returns.sum(axis=1) if not returns.empty else pd.Series(dtype=float)

        var99 = historical_var(port_returns)
        cvar = conditional_var(port_returns)
        beta = ols_portfolio_beta(proposed_positions, price_history, market_data=self.market_data)
        corr = avg_pairwise_correlation(returns)

        stat_violations: list[str] = []
        if var99 > RISK.var_limit:
            stat_violations.append(f"VaR 99%: {var99:.2%} > {RISK.var_limit:.0%} limit")
        if cvar > RISK.cvar_limit:
            stat_violations.append(f"CVaR: {cvar:.2%} > {RISK.cvar_limit:.0%} limit")
        if beta > self.max_port_beta:
            stat_violations.append(f"Beta {beta:.2f} > {self.max_port_beta:.2f}")
        if corr > RISK.correlation_limit:
            stat_violations.append(f"Avg correlation > {RISK.correlation_limit} — reduce overlap")
        if has_sector_violation:
            for sec, w in sector_weights.items():
                if w > self.max_sector_pct:
                    stat_violations.append(
                        f"Sector '{sec}' weight {w:.1%} > limit {self.max_sector_pct:.1%} "
                        f"(optimizer applied caps)"
                    )

        if stat_violations:
            logger.debug("Risk evaluate: REDUCE due to %s", stat_violations)
            return RiskAssessment(
                verdict=RiskVerdict.REDUCE,
                approved_weight=min(total_weight * RISK.reduce_weight_multiplier, total_weight),
                proposed_weight=total_weight,
                veto_reasons=stat_violations,
                optimal_weights=optimal_weights,
                var_99=var99,
                cvar=cvar,
                marginal_var=0.0,
                portfolio_beta=beta,
                avg_correlation=corr,
                stop_loss=0.0,
                api_calls_used=0,
                timestamp=datetime.now(),
            )

        stops = atr_stop_losses(proposed_positions, price_history)
        mvar = component_var(returns, port_returns)

        primary_ticker = proposed_positions[0]["ticker"] if proposed_positions else ""
        stop_val = stops.get(primary_ticker, 0.0)
        mvar_val = mvar.get(primary_ticker, 0.0)

        logger.debug("Risk evaluate: APPROVE (VaR99: %.2f%%, Beta: %.2f)", var99 * 100, beta)
        return RiskAssessment(
            verdict=RiskVerdict.APPROVE,
            approved_weight=total_weight,
            proposed_weight=total_weight,
            veto_reasons=[],
            optimal_weights=optimal_weights,
            var_99=var99,
            cvar=cvar,
            marginal_var=mvar_val,
            portfolio_beta=beta,
            avg_correlation=corr,
            stop_loss=stop_val,
            api_calls_used=0,
            timestamp=datetime.now(),
        )
