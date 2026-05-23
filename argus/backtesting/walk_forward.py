"""
argus/backtesting/walk_forward.py
=================================
Walk-forward validation orchestrator for ARGUS.
"""

import logging
import pandas as pd
import numpy as np

from argus.backtesting.engine import run_backtest

logger = logging.getLogger("argus.walk_forward")

def run_walk_forward_validation(
    universe: list[str],
    start: str = "2023-01-03",
    end: str   = "2024-12-31",
    train_months: int = 6,
    test_months: int  = 1,
    risk_tolerance: str = "MODERATE"
) -> dict:
    windows = []
    current_start = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    
    while True:
        test_start = current_start + pd.DateOffset(months=train_months)
        test_end   = test_start + pd.DateOffset(months=test_months)
        if test_end > end_date: 
            break
        
        logger.info(f"Walk-forward window: test {test_start.date()} to {test_end.date()}")
        
        result = run_backtest(
            universe=universe,
            start=test_start.strftime("%Y-%m-%d"),
            end=test_end.strftime("%Y-%m-%d"),
            risk_tolerance=risk_tolerance
        )
        
        result["window_start"] = test_start.isoformat()
        result["window_end"]   = test_end.isoformat()
        windows.append(result)
        
        current_start += pd.DateOffset(months=1)
    
    # Aggregate stats across windows
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
