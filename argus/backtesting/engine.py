"""
argus/backtesting/engine.py

Backtrader strategy and execution engine integrating ARGUS agents with point-in-time constraints.

Responsibilities:
  - Define the ARGUSBacktestStrategy Backtrader strategy class
  - Coordinate agent decision cycles within the Backtrader bar loop
  - Execute portfolio rebalancing against the Backtrader broker

Not responsible for:
  - Walk-forward orchestration (see backtesting/walk_forward.py)
  - Bias detection (see backtesting/bias_auditor.py)
  - Post-trade performance metrics (see backtesting/metrics.py)

Dependencies:
  - backtrader
  - yfinance (for data feed construction)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import backtrader as bt
import pandas as pd
import yfinance as yf

from argus.agents.fundamental import FundamentalAgent
from argus.agents.macro import MacroStatisticalAgent
from argus.agents.portfolio import PortfolioManagerAgent
from argus.agents.risk import RiskStatisticalEngine
from argus.agents.sentiment import SentimentAgent
from argus.agents.technical import TechnicalStatisticalAgent
from argus.backtesting.pit_enforcer import PointInTimeEnforcer
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.schemas.signals import PortfolioAllocation

logger = logging.getLogger("argus.backtest_engine")


class ARGUSBacktestStrategy(bt.Strategy):
    """Backtrader strategy coordinating agent decisions and portfolio rebalancing.

    Instantiates all ARGUS agents independently to avoid contaminating live-session
    state. Rebalances on the first bar and then every ``rebalance_days`` bars.
    """

    params = dict(
        invest_pct=0.80,
        risk_tolerance="MODERATE",
        rebalance_days=21,
        min_conviction=0.50,
        backtest_mode=True,
    )

    def __init__(self):
        self.bar_count = 0
        self.trade_log = []
        self.pit_enforcer: Optional[PointInTimeEnforcer] = None

        self.agents = {
            "technical": TechnicalStatisticalAgent(),
            "macro": MacroStatisticalAgent(),
            "risk": RiskStatisticalEngine(),
            "fundamental": FundamentalAgent(),
            "sentiment": SentimentAgent(),
            "portfolio": PortfolioManagerAgent(),
        }
        self.aggregator = HybridSignalAggregator()
        self.final_metrics = {}

    def next(self):
        """Executes the rebalancing loop on simulation bar updates.

        Skips bars that do not fall on a rebalance boundary to reduce
        the number of LLM calls during simulation.
        """
        self.bar_count += 1
        if self.bar_count != 1 and (self.bar_count - 1) % self.p.rebalance_days != 0:
            return

        sim_date = self.data.datetime.date(0)
        self.pit_enforcer = PointInTimeEnforcer(sim_date)

        user_profile = {
            "total_wealth": self.broker.getvalue(),
            "invest_pct": self.p.invest_pct,
            "risk_tolerance": self.p.risk_tolerance,
        }

        all_signals = {}
        macro = self.agents["macro"].analyze()

        for data_feed in self.datas:
            ticker = data_feed._name
            if ticker == "SPY":
                continue

            try:
                close_series = self.pit_enforcer.get_close_series(ticker)
                session_state = self._simulate_session_state(ticker, close_series)

                tech = self.agents["technical"].analyze(ticker, session_state)
                fund = self.agents["fundamental"].analyze(
                    ticker, backtest_mode=True, session_seed=int(sim_date.strftime("%Y%m%d"))
                )
                sent = self.agents["sentiment"].analyze(ticker)

                try:
                    spy_close = self.pit_enforcer.get_close_series("SPY")
                except Exception:
                    spy_close = close_series

                pit_hist = {ticker: close_series, "SPY": spy_close}
                proposed = [{"ticker": ticker, "weight": 0.10}]
                risk = self.agents["risk"].evaluate(proposed, pit_hist, macro.vix_level)

                agg = self.aggregator.aggregate(tech, macro, fund, sent)
                all_signals[ticker] = {
                    "technical": tech,
                    "fundamental": fund,
                    "sentiment": sent,
                    "risk": risk,
                    "aggregated": agg,
                }

            except Exception as e:
                logger.warning(f"[Backtest] Error processing {ticker} on {sim_date}: {e}")

        if all_signals:
            allocation = self.agents["portfolio"].allocate(user_profile, all_signals, macro)
            self._execute_allocation(allocation)
            self._log_trades(sim_date, allocation)

    def _simulate_session_state(self, ticker: str, close: pd.Series) -> dict:
        """Approximates technical candles using historical daily close prices.

        Args:
            ticker: Equity ticker symbol.
            close: Daily close price Series from the PointInTimeEnforcer.

        Returns:
            Lightweight session state dict sufficient for TechnicalStatisticalAgent.
        """
        if close.empty:
            return {"recent_prices": [], "volume": []}

        recent = close.tail(20).tolist()
        return {"recent_prices": recent, "volume": [1000] * len(recent)}

    def _execute_allocation(self, allocation: PortfolioAllocation):
        """Transmits order targets to the broker based on normalized weights.

        The agent computes allocation_pct relative to investable capital (e.g. 80%),
        but Backtrader's order_target_percent operates on total broker value (100%).
        The invest_pct normalization aligns the two scales.

        Args:
            allocation: Validated PortfolioAllocation with per-ticker weights.
        """
        alloc_dict = {pos.ticker: pos.allocation_pct for pos in allocation.portfolio}

        for data_feed in self.datas:
            ticker = data_feed._name
            if ticker == "SPY":
                continue

            target_pct = alloc_dict.get(ticker, 0.0) * self.p.invest_pct
            self.order_target_percent(data_feed, target=target_pct)

    def _log_trades(self, date: date, allocation: PortfolioAllocation):
        """Logs the portfolio allocation state at a specific simulation date.

        Args:
            date: Current simulation bar date.
            allocation: PortfolioAllocation executed at this bar.
        """
        self.trade_log.append(
            {
                "date": date.isoformat(),
                "allocation": [p.model_dump() for p in allocation.portfolio] if allocation else [],
            }
        )

    def stop(self):
        """Logs summary metrics when the backtest run completes."""
        self.final_metrics = {
            "total_bars": self.bar_count,
            "trades_logged": len(self.trade_log),
        }


def run_backtest(
    universe: list[str],
    start: str,
    end: str,
    initial_cash: float = 100_000.0,
    invest_pct: float = 0.80,
    risk_tolerance: str = "MODERATE",
) -> dict:
    """Configures and runs a Backtrader Cerebro simulation for the selected universe.

    SPY is always added to the universe for Beta calculations even if not in the
    caller-supplied list.

    Args:
        universe: List of equity tickers to include in the backtest.
        start: ISO start date string (e.g. '2021-01-04').
        end: ISO end date string (e.g. '2022-12-30').
        initial_cash: Starting broker cash (default $100,000).
        invest_pct: Fraction of equity to target in each rebalance (default 0.80).
        risk_tolerance: Risk tier passed to ARGUSBacktestStrategy.

    Returns:
        Dict with keys: sharpe, max_drawdown, rtot, total_trades, final_value, trade_log.
        Returns ``{'error': str}`` on failure.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ARGUSBacktestStrategy, invest_pct=invest_pct, risk_tolerance=risk_tolerance)

    universe_with_spy = list(set(universe + ["SPY"]))

    for ticker in universe_with_spy:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # bt.feeds.PandasData requires flat columns; yfinance ≥ 0.2.40 returns MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                data = bt.feeds.PandasData(dataname=df, name=ticker)
                cerebro.adddata(data, name=ticker)
        except Exception as e:
            logger.warning(f"[Backtest] Failed to fetch data for {ticker}: {e}")

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0)

    try:
        results = cerebro.run()
        if not results:
            return {"error": "Cerebro run returned no results."}

        strat = results[0]

        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trades = strat.analyzers.trades.get_analysis()

        return {
            "sharpe": sharpe.get("sharperatio"),
            "max_drawdown": drawdown.get("max", {}).get("drawdown", 0.0) / 100.0,
            "rtot": returns.get("rtot"),
            "total_trades": trades.get("total", {}).get("total"),
            "final_value": cerebro.broker.getvalue(),
            "trade_log": strat.trade_log,
        }
    except Exception as e:
        logger.error(f"[Backtest] Cerebro run failed: {e}")
        return {"error": str(e)}
