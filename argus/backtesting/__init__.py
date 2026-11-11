"""Backtesting sub-package for ARGUS.

metrics.py: trade-level statistics (win rate, profit factor, average win/loss),
used by evaluation.py. There is no return-series (Sharpe, drawdown, VaR) engine:
nothing here produces a daily equity curve.
evaluation.py: pre-registered rank IC / hit-rate / trade-level evaluation.
replay.py: replays recorded fixture sessions through the real graph; there is
no multi-year walk-forward engine here.
"""
