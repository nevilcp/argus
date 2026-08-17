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

    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()

    if not strategy_returns.empty and not benchmark_returns.empty:
        idx = strategy_returns.index.intersection(benchmark_returns.index)
        s_ret = strategy_returns.loc[idx]
        b_ret = benchmark_returns.loc[idx]
    else:
        s_ret = strategy_returns
        b_ret = pd.Series(dtype=float)

    n_days = len(s_ret)
    if n_days == 0:
        return {}

    mean_ret = float(s_ret.mean())
    std_ret = float(s_ret.std())

    annualized_return = mean_ret * 252
    cumulative_return = float((1 + s_ret).cumprod().iloc[-1] - 1)
    annualized_volatility = std_ret * math.sqrt(252) if std_ret else 0.0

    if std_ret > 0:
        sharpe_ratio = (mean_ret - risk_free_rate / 252) / std_ret * math.sqrt(252)
    else:
        sharpe_ratio = 0.0

    neg_rets = s_ret[s_ret < 0]
    downside_std = float(neg_rets.std())
    if downside_std > 0:
        sortino_ratio = (mean_ret - risk_free_rate / 252) / downside_std * math.sqrt(252)
    else:
        sortino_ratio = None

    cum_returns = (1 + s_ret).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max

    max_drawdown = float(drawdowns.min())

    if max_drawdown < 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = None

    underwater = drawdowns < 0
    duration_series = underwater.astype(int).groupby((~underwater).cumsum()).cumsum()
    max_drawdown_duration_days = int(duration_series.max())

    active_dds = drawdowns[drawdowns < 0]
    avg_drawdown = float(active_dds.mean()) if not active_dds.empty else 0.0

    alpha_annualized = None
    beta = None
    information_ratio = None
    r_squared = None

    if len(s_ret) > 1 and len(b_ret) == len(s_ret):
        b_mean = b_ret.mean()
        b_std = b_ret.std()

        if b_std > 0:
            cov = np.cov(s_ret, b_ret)[0, 1]
            var_b = np.var(b_ret, ddof=1)
            if var_b > 0:
                beta = float(cov / var_b)
                alpha_daily = mean_ret - (
                    risk_free_rate / 252 + beta * (b_mean - risk_free_rate / 252)
                )
                alpha_annualized = alpha_daily * 252

                corr = np.corrcoef(s_ret, b_ret)[0, 1]
                r_squared = float(corr**2)

        active_return = s_ret - b_ret
        tracking_error = active_return.std()
        if tracking_error > 0:
            information_ratio = float((active_return.mean() / tracking_error) * math.sqrt(252))

    var_95_historical = float(np.percentile(s_ret, 5)) if len(s_ret) > 0 else 0.0
    cvar_95_rets = s_ret[s_ret <= var_95_historical]
    cvar_95 = float(cvar_95_rets.mean()) if not cvar_95_rets.empty else var_95_historical

    try:
        skewness = float(stats.skew(s_ret))
        kurtosis = float(stats.kurtosis(s_ret))
    except Exception:
        skewness = 0.0
        kurtosis = 0.0

    metrics = {
        "annualized_return": annualized_return,
        "cumulative_return": cumulative_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_days": max_drawdown_duration_days,
        "avg_drawdown": avg_drawdown,
        "alpha_annualized": alpha_annualized,
        "beta": beta,
        "information_ratio": information_ratio,
        "r_squared": r_squared,
        "var_95_historical": var_95_historical,
        "cvar_95": cvar_95,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

    if trade_log is not None:
        total_trades = len(trade_log)
        wins = [t for t in trade_log if t.get("return_pct", 0) > 0]
        losses = [t for t in trade_log if t.get("return_pct", 0) <= 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win_pct = float(np.mean([t["return_pct"] for t in wins])) if wins else 0.0
        avg_loss_pct = float(np.mean([t["return_pct"] for t in losses])) if losses else 0.0
        win_loss_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0.0

        gross_wins = sum(t["return_pct"] for t in wins)
        gross_losses = abs(sum(t["return_pct"] for t in losses))
        profit_factor = (
            float(gross_wins / gross_losses)
            if gross_losses > 0
            else (float("inf") if gross_wins > 0 else 0.0)
        )

        holding_days = [t.get("holding_days", 0) for t in trade_log]
        avg_holding_days = float(np.mean(holding_days)) if holding_days else 0.0

        metrics.update(
            {
                "win_rate": win_rate,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "win_loss_ratio": win_loss_ratio,
                "profit_factor": profit_factor,
                "total_trades": total_trades,
                "avg_holding_days": avg_holding_days,
            }
        )

    # Replace NaN/Inf with None for JSON-safe output; round survivors to 4 dp
    for k, v in metrics.items():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                metrics[k] = None
            else:
                metrics[k] = round(v, 4)

    return metrics
