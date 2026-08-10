"""
argus/backtesting/walk_forward.py

Walk-forward validation orchestrator for sequential in-sample/out-of-sample testing.

Responsibilities:
  - Partition the full date range into rolling training and test windows
  - Execute an independent backtest run for each out-of-sample window
  - Aggregate per-window Sharpe ratios into a consistency score

Not responsible for:
  - Individual backtest simulation (see backtesting/engine.py)
  - Post-run bias detection (see backtesting/bias_auditor.py)
  - Phase calibration (see backtesting/phase1_calibration.py)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from argus.backtesting.engine import run_backtest

logger = logging.getLogger("argus.walk_forward")


def run_walk_forward_validation(
    universe: list[str],
    start: str = "2023-01-03",
    end: str = "2024-12-31",
    train_months: int = 6,
    test_months: int = 1,
    risk_tolerance: str = "MODERATE",
) -> dict:
    """Orchestrates sequential in-sample training and out-of-sample testing windows.

    Generates rolling windows that advance by one month each iteration.
    Only the out-of-sample test window is run through the backtest engine;
    the training period is used implicitly by the strategy's rolling indicators.

    The consistency score measures the volatility of Sharpe ratios across windows:
        consistency = 1 - (std_sharpe / max(|avg_sharpe|, 0.01))
    A score ≥ 0.5 indicates robust, low-variance performance.

    Args:
        universe: List of equity ticker symbols to include in each window.
        start: ISO date string for the first window's start (default 2023-01-03).
        end: ISO date string for the final window's end (default 2024-12-31).
        train_months: Number of months per in-sample training window (default 6).
        test_months: Number of months per out-of-sample test window (default 1).
        risk_tolerance: Risk tier passed to each backtest run (default 'MODERATE').

    Returns:
        Dict with keys:
            windows: List of per-window result dicts with window_start, window_end keys.
            n_windows: Number of completed windows.
            avg_sharpe: Average Sharpe ratio across windows, or None if no windows ran.
            std_sharpe: Standard deviation of Sharpe ratios, or None.
            consistency_score: Stability score in (-inf, 1], or None.
            pass_criteria: True if avg_sharpe ≥ 0.80.
    """
    windows = []
    current_start = pd.Timestamp(start)
    end_date = pd.Timestamp(end)

    while True:
        test_start = current_start + pd.DateOffset(months=train_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end_date:
            break

        logger.info(
            "run_walk_forward_validation: window test %s → %s",
            test_start.date(),
            test_end.date(),
        )

        result = run_backtest(
            universe=universe,
            start=test_start.strftime("%Y-%m-%d"),
            end=test_end.strftime("%Y-%m-%d"),
            risk_tolerance=risk_tolerance,
        )

        result["window_start"] = test_start.isoformat()
        result["window_end"] = test_end.isoformat()
        windows.append(result)

        current_start += pd.DateOffset(months=1)

    sharpes = [w.get("sharpe") for w in windows if w.get("sharpe") is not None]

    avg_sharpe = float(np.mean(sharpes)) if sharpes else None
    std_sharpe = float(np.std(sharpes)) if sharpes else None

    consistency = None
    if sharpes:
        denom = max(abs(avg_sharpe), 0.01) if avg_sharpe is not None else 0.01
        if std_sharpe is not None:
            consistency = 1 - (std_sharpe / denom)

    return {
        "windows": windows,
        "n_windows": len(windows),
        "avg_sharpe": round(avg_sharpe, 4) if avg_sharpe is not None else None,
        "std_sharpe": round(std_sharpe, 4) if std_sharpe is not None else None,
        "consistency_score": round(consistency, 4) if consistency is not None else None,
        "pass_criteria": (avg_sharpe >= 0.80) if avg_sharpe is not None else False,
    }
