"""
argus/agents/risk.py
====================
Risk Statistical Engine — deterministic capital protection layer.

Calculates portfolio-level statistics (VaR, CVaR, Beta, Correlation) and enforces
hard diversification and volatility thresholds before approving allocations.
Zero LLM calls. Target runtime < 200ms.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from argus.config import settings
from argus.schemas.signals import RiskAssessment, RiskVerdict

logger = logging.getLogger("argus.risk")

# ──────────────────────────────────────────────────────────────────────────────
# Universe & Sectors
# ──────────────────────────────────────────────────────────────────────────────

GICS_SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "BAC": "Financials",
    "JNJ": "Health Care",
    "UNH": "Health Care",
    "XOM": "Energy",
    "CVX": "Energy",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "NEE": "Utilities",
    "DUK": "Utilities",
    "CAT": "Industrials",
    "HON": "Industrials",
    "NUE": "Materials",
    "PLD": "Real Estate",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions (Pure Statistical Math)
# ──────────────────────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    lookback: int = 252
) -> pd.DataFrame:
    """Compute weight-adjusted daily returns for all positions over lookback."""
    returns_dict = {}
    for pos in positions:
        ticker = pos["ticker"]
        weight = pos["weight"]
        if ticker in price_history:
            series = price_history[ticker].tail(lookback + 1)
            # Daily percentage change, multiplied by weight
            returns_dict[ticker] = series.pct_change().dropna() * weight
            
    df = pd.DataFrame(returns_dict).dropna()
    return df


def historical_var(portfolio_returns: pd.Series, confidence: float = 0.99) -> float:
    """Calculate historical Value-at-Risk using percentile simulation."""
    if portfolio_returns.empty:
        return 0.0
    percentile = (1.0 - confidence) * 100.0
    val = float(np.percentile(portfolio_returns.dropna(), percentile))
    return abs(val)


def conditional_var(portfolio_returns: pd.Series, confidence: float = 0.99) -> float:
    """Calculate Conditional VaR (Expected Shortfall)."""
    if portfolio_returns.empty:
        return 0.0
    var = historical_var(portfolio_returns, confidence)
    tail = portfolio_returns[portfolio_returns <= -var]
    return abs(float(tail.mean())) if len(tail) > 0 else var


def ols_portfolio_beta(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    benchmark_ticker: str = "SPY"
) -> float:
    """Calculate the weighted OLS beta of the portfolio vs a benchmark."""
    if not positions:
        return 0.0

    if benchmark_ticker not in price_history:
        try:
            from argus.data.fetchers import fetch_ohlcv_daily
            # Fetch benchmark history on demand if missing
            spy_df = fetch_ohlcv_daily(benchmark_ticker, period="1y")
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
            if len(df) > 10:
                cov = df.cov().iloc[0, 1]
                beta = cov / spy_var
                betas.append(beta)
                weights.append(w)

    if not betas:
        return 1.0

    return float(np.average(betas, weights=weights))


def avg_pairwise_correlation(returns_matrix: pd.DataFrame) -> float:
    """Calculate the average pairwise correlation across the portfolio."""
    if returns_matrix.shape[1] < 2:
        return 0.0
    corr = returns_matrix.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return float(upper.stack().mean())


def atr_stop_losses(
    positions: list[dict],
    price_history: dict[str, pd.Series],
    atr_multiplier: float = 2.5
) -> dict[str, float]:
    """Compute ATR-based stop-loss levels per position."""
    stops = {}
    for pos in positions:
        ticker = pos["ticker"]
        if ticker in price_history:
            series = price_history[ticker]
            if len(series) > 14:
                true_range = series.diff().abs().dropna()
                atr_14 = true_range.tail(14).mean()
                latest_close = float(series.iloc[-1])
                stops[ticker] = max(0.0, latest_close - (atr_14 * atr_multiplier))
    return stops


def component_var(
    returns_matrix: pd.DataFrame,
    portfolio_returns: pd.Series
) -> dict[str, float]:
    """Calculate the marginal VaR contribution of each component."""
    mvar = {}
    port_std = portfolio_returns.std()
    if port_std == 0 or pd.isna(port_std):
        return {col: 0.0 for col in returns_matrix.columns}

    for col in returns_matrix.columns:
        cov = np.cov(returns_matrix[col], portfolio_returns)[0, 1]
        mvar[col] = float(cov / port_std)
    return mvar


# ──────────────────────────────────────────────────────────────────────────────
# Main Risk Engine
# ──────────────────────────────────────────────────────────────────────────────

class RiskStatisticalEngine:
    """
    Evaluates proposed portfolio allocations against risk limits.
    Returns a RiskAssessment schema enforcing deterministic constraints.
    """

    def __init__(self) -> None:
        # Load from settings, provide fallbacks if missing
        self.max_position_pct = getattr(settings, "MAX_SINGLE_POSITION_PCT", 0.15)
        self.max_sector_pct = getattr(settings, "MAX_SECTOR_CONCENTRATION", 0.35)
        self.vix_blackout = getattr(settings, "VIX_BLACKOUT_THRESHOLD", 35.0)
        self.max_port_beta = getattr(settings, "MAX_PORTFOLIO_BETA", 1.5)

    def evaluate(
        self,
        proposed_positions: list[dict],
        price_history: dict[str, pd.Series],
        current_vix: float
    ) -> RiskAssessment:
        """
        Evaluate the proposed allocation.

        Parameters
        ----------
        proposed_positions:
            List of dicts: ``[{"ticker": "AAPL", "weight": 0.10}, ...]``
        price_history:
            Dict mapping tickers to pandas Series of daily closing prices.
        current_vix:
            Current level of the VIX index.

        Returns
        -------
        RiskAssessment
            Schema with APPROVE, REDUCE, or VETO verdict and details.
        """
        violations = []

        # ── GATE 1 — Hard rules ───────────────────────────────────────────────
        total_weight = sum(p["weight"] for p in proposed_positions)
        
        for pos in proposed_positions:
            if pos["weight"] > self.max_position_pct:
                violations.append(f"{pos['ticker']} weight {pos['weight']:.1%} > limit {self.max_position_pct:.1%}")

        if len(proposed_positions) > 1 and len(proposed_positions) < 5 and total_weight > 0:
            violations.append(f"Insufficient diversification: {len(proposed_positions)} positions (min 5)")
            
        if len(proposed_positions) > 20:
            violations.append(f"Over-diversification: {len(proposed_positions)} positions (max 20)")

        if current_vix >= self.vix_blackout:
            violations.append(f"VIX {current_vix:.1f} >= blackout threshold {self.vix_blackout:.1f}")

        # Sector Concentration Check
        sector_weights: dict[str, float] = {}
        for pos in proposed_positions:
            sector = GICS_SECTORS.get(pos["ticker"], "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + pos["weight"]

        for sector, weight in sector_weights.items():
            if weight > self.max_sector_pct:
                violations.append(f"Sector '{sector}' weight {weight:.1%} > limit {self.max_sector_pct:.1%}")

        if violations:
            logger.info("Risk evaluate: VETO due to %s", violations)
            return RiskAssessment(
                verdict=RiskVerdict.VETO,
                veto_reasons=violations,
                var_99=0.0,
                cvar=0.0,
                portfolio_beta=0.0,
                avg_correlation=0.0,
                stop_losses={},
                marginal_var={},
                approved_weight=0.0,
                proposed_weight=total_weight,
                api_calls_used=0,
                timestamp=datetime.now()
            )

        # ── GATE 2 — Statistical risk measures ────────────────────────────────
        returns = compute_portfolio_returns(proposed_positions, price_history)
        port_returns = returns.sum(axis=1) if not returns.empty else pd.Series(dtype=float)

        var99 = historical_var(port_returns)
        cvar = conditional_var(port_returns)
        beta = ols_portfolio_beta(proposed_positions, price_history)
        corr = avg_pairwise_correlation(returns)

        # ── GATE 3 — Statistical thresholds ───────────────────────────────────
        stat_violations = []
        if var99 > 0.03:
            stat_violations.append(f"VaR 99%: {var99:.2%} > 3% limit")
        if cvar > 0.05:
            stat_violations.append(f"CVaR: {cvar:.2%} > 5% limit")
        if beta > self.max_port_beta:
            stat_violations.append(f"Beta {beta:.2f} > {self.max_port_beta:.2f}")
        if corr > 0.75:
            stat_violations.append("Avg correlation > 0.75 — reduce overlap")

        if stat_violations:
            logger.info("Risk evaluate: REDUCE due to %s", stat_violations)
            return RiskAssessment(
                verdict=RiskVerdict.REDUCE,
                veto_reasons=stat_violations,
                var_99=var99,
                cvar=cvar,
                portfolio_beta=beta,
                avg_correlation=corr,
                stop_losses={},
                marginal_var={},
                approved_weight=min(total_weight * 0.5, total_weight), # Reduce by half hypothetically
                proposed_weight=total_weight,
                api_calls_used=0,
                timestamp=datetime.now()
            )

        # ── PASS — Compute stops and return APPROVE ───────────────────────────
        stops = atr_stop_losses(proposed_positions, price_history)
        mvar = component_var(returns, port_returns)

        logger.info("Risk evaluate: APPROVE (VaR99: %.2f%%, Beta: %.2f)", var99 * 100, beta)
        return RiskAssessment(
            verdict=RiskVerdict.APPROVE,
            veto_reasons=[],
            var_99=var99,
            cvar=cvar,
            portfolio_beta=beta,
            avg_correlation=corr,
            stop_losses=stops,
            marginal_var=mvar,
            approved_weight=total_weight,
            proposed_weight=total_weight,
            api_calls_used=0,
            timestamp=datetime.now()
        )
