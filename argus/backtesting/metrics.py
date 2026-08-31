"""
argus/backtesting/metrics.py

Pure return-series risk and performance metrics. Its trade-level block
(win rate, profit factor, avg win/loss) is reused by
backtesting/evaluation.py's trade_level_win_loss_stats; the return-series
block (Sharpe, Sortino, max drawdown, alpha/beta, VaR/CVaR) has no caller
yet — nothing in this repo produces a daily equity curve for it to score.

Responsibilities:
  - Compute return, drawdown, benchmark-adjusted, tail-risk, and trade-level statistics
  - Produce a flat dict suitable for JSON reporting

Not responsible for:
  - Running a backtest or replaying sessions (see backtesting/replay.py)

Dependencies:
  - numpy
  - scipy
  - pandas
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

_TRADING_DAYS_PER_YEAR = 252
_ANNUALIZATION = math.sqrt(_TRADING_DAYS_PER_YEAR)
_VAR_PERCENTILE = 5  # 5th percentile == the 95% historical VaR level


def compute_all_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
    trade_log: Optional[list[dict]] = None,
) -> dict:
    """Computes return, drawdown, benchmark-adjusted, tail-risk, and trade-level statistics.

    All float values are rounded to 4 decimal places. NaN or Inf results are replaced
    with None to ensure JSON-serializability of the output dict.

    Args:
        strategy_returns: Daily return Series for the backtest strategy.
        benchmark_returns: Daily return Series for the benchmark (e.g. SPY).
        risk_free_rate: Annual risk-free rate used in Sharpe/Sortino computation (default 5%).
        trade_log: Optional list of trade dicts with key ``return_pct``;
            activates trade-level statistics when provided.

    Returns:
        Dict of computed metrics. Returns an empty dict if strategy_returns is None or empty.
    """
    if strategy_returns is None or strategy_returns.empty:
        return {}

    s_ret, b_ret = _align(strategy_returns.dropna(), benchmark_returns.dropna())
    if s_ret.empty:
        return {}

    metrics = {
        **_return_and_drawdown_metrics(s_ret, risk_free_rate),
        **_benchmark_metrics(s_ret, b_ret, risk_free_rate),
        **_tail_risk_metrics(s_ret),
    }
    if trade_log is not None:
        metrics.update(_trade_level_metrics(trade_log))

    return {key: _json_safe(value) for key, value in metrics.items()}


def _align(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Restricts both series to the dates they share; an empty side drops the benchmark."""
    if strategy_returns.empty or benchmark_returns.empty:
        return strategy_returns, pd.Series(dtype=float)
    shared = strategy_returns.index.intersection(benchmark_returns.index)
    return strategy_returns.loc[shared], benchmark_returns.loc[shared]


def _return_and_drawdown_metrics(returns: pd.Series, risk_free_rate: float) -> dict:
    """Return, volatility, risk-adjusted-ratio and drawdown statistics of the strategy alone."""
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    downside_std = float(returns[returns < 0].std())
    daily_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR
    annualized_return = mean_ret * _TRADING_DAYS_PER_YEAR

    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    # Consecutive underwater days: the cumulative count restarts at every day back at a high.
    underwater = drawdowns < 0
    underwater_run = underwater.astype(int).groupby((~underwater).cumsum()).cumsum()
    active_drawdowns = drawdowns[underwater]

    return {
        "annualized_return": annualized_return,
        "cumulative_return": float(cum_returns.iloc[-1] - 1),
        "annualized_volatility": std_ret * _ANNUALIZATION if std_ret else 0.0,
        "sharpe_ratio": (mean_ret - daily_rf) / std_ret * _ANNUALIZATION if std_ret > 0 else 0.0,
        "sortino_ratio": (
            (mean_ret - daily_rf) / downside_std * _ANNUALIZATION if downside_std > 0 else None
        ),
        "calmar_ratio": annualized_return / abs(max_drawdown) if max_drawdown < 0 else None,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_days": int(underwater_run.max()),
        "avg_drawdown": float(active_drawdowns.mean()) if not active_drawdowns.empty else 0.0,
    }


def _benchmark_metrics(returns: pd.Series, benchmark: pd.Series, risk_free_rate: float) -> dict:
    """Alpha, beta, information ratio and R^2 against the benchmark.

    Every value stays None unless the benchmark actually covers the same dates
    and varies — a degraded, flagged result rather than an invented one.
    """
    metrics: dict = {
        "alpha_annualized": None,
        "beta": None,
        "information_ratio": None,
        "r_squared": None,
    }
    if len(returns) <= 1 or len(benchmark) != len(returns):
        return metrics

    mean_ret = float(returns.mean())
    daily_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR
    benchmark_variance = float(np.var(benchmark, ddof=1))

    if float(benchmark.std()) > 0 and benchmark_variance > 0:
        beta = float(np.cov(returns, benchmark)[0, 1] / benchmark_variance)
        alpha_daily = mean_ret - (daily_rf + beta * (float(benchmark.mean()) - daily_rf))
        metrics["beta"] = beta
        metrics["alpha_annualized"] = alpha_daily * _TRADING_DAYS_PER_YEAR
        metrics["r_squared"] = float(np.corrcoef(returns, benchmark)[0, 1] ** 2)

    active_return = returns - benchmark
    tracking_error = float(active_return.std())
    if tracking_error > 0:
        metrics["information_ratio"] = float(active_return.mean()) / tracking_error * _ANNUALIZATION

    return metrics


def _tail_risk_metrics(returns: pd.Series) -> dict:
    """Historical VaR/CVaR and the distribution shape of the return series."""
    var_95 = float(np.percentile(returns, _VAR_PERCENTILE))
    tail = returns[returns <= var_95]

    try:
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))
    except Exception:
        skewness = 0.0
        kurtosis = 0.0

    return {
        "var_95_historical": var_95,
        "cvar_95": float(tail.mean()) if not tail.empty else var_95,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def _trade_level_metrics(trade_log: list[dict]) -> dict:
    """Win/loss, profit-factor and holding-period statistics over discrete trades."""
    total_trades = len(trade_log)
    wins = [t for t in trade_log if t.get("return_pct", 0) > 0]
    losses = [t for t in trade_log if t.get("return_pct", 0) <= 0]

    avg_win_pct = float(np.mean([t["return_pct"] for t in wins])) if wins else 0.0
    avg_loss_pct = float(np.mean([t["return_pct"] for t in losses])) if losses else 0.0

    gross_wins = sum(t["return_pct"] for t in wins)
    gross_losses = abs(sum(t["return_pct"] for t in losses))
    if gross_losses > 0:
        profit_factor = float(gross_wins / gross_losses)
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    holding_days = [t.get("holding_days", 0) for t in trade_log]

    return {
        "win_rate": len(wins) / total_trades if total_trades > 0 else 0.0,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "win_loss_ratio": abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0.0,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "avg_holding_days": float(np.mean(holding_days)) if holding_days else 0.0,
    }


def _json_safe(value: object) -> object:
    """NaN and Inf have no JSON representation; surviving floats round to 4 dp."""
    if not isinstance(value, float):
        return value
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, 4)
