"""Backtesting sub-package for ARGUS.

metrics.py: pure return-series and trade-level statistics (Sharpe, Sortino,
max drawdown, win rate, profit factor); its trade-level block is wired into
evaluation.py, its return-series block has no caller yet (see metrics.py).
evaluation.py: pre-registered rank IC / hit-rate / trade-level evaluation.
replay.py: replays recorded fixture sessions through the real graph; there is
no multi-year walk-forward engine here.
"""
