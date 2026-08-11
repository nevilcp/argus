"""Backtesting sub-package for ARGUS.

metrics.py: pure return-series statistics (Sharpe, Sortino, max drawdown, IC).
replay.py: replays recorded fixture sessions through the real graph — see
docs/adr/0009-no-multiyear-backtest.md for why there is no multi-year
walk-forward engine here.
"""
