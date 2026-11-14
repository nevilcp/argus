"""Trade-level performance statistics."""

from __future__ import annotations

import math

import numpy as np


def trade_level_metrics(trade_log: list[dict]) -> dict:
    """Compute win/loss, profit-factor and holding-period statistics over discrete trades."""
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

    metrics = {
        "win_rate": len(wins) / total_trades if total_trades > 0 else 0.0,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "win_loss_ratio": abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0.0,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "avg_holding_days": float(np.mean(holding_days)) if holding_days else 0.0,
    }
    return {key: _json_safe(value) for key, value in metrics.items()}


def _json_safe(value: object) -> object:
    """Replace non-finite floats with None and round finite floats to four decimal places."""
    if not isinstance(value, float):
        return value
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, 4)
