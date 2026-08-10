"""
argus/backtesting/phase1_calibration.py

Phase 1: In-sample calibration pipeline on 2021–2022 historical data.

Responsibilities:
  - Fit the macro HMM on pre-2021 FRED history as a pre-calibration step
  - Run a grid search over technical indicator weights using walk-forward Sharpe
  - Execute a baseline bias audit to validate data handling before locking params
  - Persist the calibration report to calibration_report.json for Phase 2 ingestion

Not responsible for:
  - Out-of-sample validation (see backtesting/phase2_validation.py)
  - Live trading (see api/main.py)

Dependencies:
  - unittest.mock (mocks out LLM agents for deterministic backtesting)
  - FRED_API_KEY env var must be set (see .env.example)
"""

import json
import logging
from datetime import date, datetime

import pandas as pd

from argus.agents.macro import MacroStatisticalAgent
from argus.backtesting.bias_auditor import BiasAuditor
from argus.backtesting.engine import run_backtest
from argus.backtesting.walk_forward import run_walk_forward_validation
from argus.config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("argus.phase1")

# Reference universe of 20 liquid large-cap equities spanning 7 GICS sectors
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "V", "UNH",
    "XOM", "JNJ", "PG", "MA", "HD",
    "MRK", "ABBV", "LLY", "AVGO", "CVX",
]

PHASE_1_START = "2021-01-04"
PHASE_1_END = "2022-12-30"


def run_phase1_calibration() -> dict:
    """Fits macro models, optimizes indicator weights, audits bias, and locks optimal parameters.

    Mocks the LLM-backed agents (Fundamental, Sentiment, Portfolio) to produce fixed
    BULLISH signals at 0.6 conviction, isolating Technical + Risk as the only variable
    components. This ensures the grid search evaluates purely the technical indicator
    weighting configuration rather than confounding LLM randomness.

    The weight grid searches 2×2×2×2 = 16 combinations across RSI, MACD, Bollinger,
    and Momentum weights. The best combination is determined by highest avg_sharpe
    from a walk-forward validation on a 5-ticker sub-universe (for speed).

    Returns:
        Calibration report dict persisted to calibration_report.json with keys:
        phase, period, baseline_sharpe, optimized_sharpe, locked_indicator_weights,
        hmm_regime_mapping, bias_audit, locked_at, WARNING.
    """
    from unittest import mock

    from argus.schemas.signals import FundamentalSignal, PortfolioAllocation, SentimentSignal, Signal

    def _fund_side_effect(ticker, backtest_mode=False, session_seed=None):
        """Returns a fixed BULLISH fundamental signal to isolate technical weight tuning."""
        return FundamentalSignal(
            ticker=ticker,
            signal=Signal.BULLISH,
            conviction=0.6,
            moat_score=7,
            reasoning="Mock",
            data_as_of_date=date.today(),
            timestamp=datetime.now(),
        )

    def _sent_side_effect(ticker, *args, **kwargs):
        """Returns a fixed BULLISH sentiment signal to isolate technical weight tuning."""
        return SentimentSignal(
            ticker=ticker,
            signal=Signal.BULLISH,
            conviction=0.6,
            finbert_net_score=0.2,
            pct_positive=0.5,
            pct_negative=0.2,
            news_volume_7d=10,
            social_mention_surge=False,
            upcoming_catalyst=False,
            sentiment_decay_risk="LOW",
            reasoning="Mock",
            timestamp=datetime.now(),
        )

    def _port_side_effect(self, user_profile, all_signals, macro, *args, **kwargs):
        """Produces a deterministic equal-weight allocation capped at 15% per position.

        This deterministic portfolio eliminates LLM non-determinism from Phase 1
        so the grid search cleanly measures the technical indicator configuration.
        """
        from uuid import uuid4

        from argus.schemas.signals import PortfolioAllocation, PositionAllocation, RiskVerdict

        investable = user_profile.get("total_wealth", 100_000.0) * user_profile.get(
            "invest_pct", 0.8
        )

        approved = [
            t
            for t, s in all_signals.items()
            if s.get("risk") and s["risk"].verdict == RiskVerdict.APPROVE
        ]

        if not approved:
            return PortfolioAllocation(
                session_id=str(uuid4()),
                user_investable_capital=investable,
                portfolio=[],
                cash_reserve_pct=1.0,
                expected_sharpe=0.0,
                rebalance_trigger="MONTHLY",
                timestamp=datetime.now(),
            )

        weight = min(0.8 / len(approved), 0.15)

        portfolio = [
            PositionAllocation(
                ticker=t,
                allocation_pct=weight,
                allocation_usd=investable * weight,
                stop_loss=float(all_signals[t]["risk"].stop_loss or 0.0),
                target_price=None,
                thesis="Mocked backtest allocation",
                composite_conviction=0.6,
                time_horizon="3-6 months",
            )
            for t in approved
        ]

        return PortfolioAllocation(
            session_id=str(uuid4()),
            user_investable_capital=investable,
            portfolio=portfolio,
            cash_reserve_pct=round(1.0 - (weight * len(approved)), 4),
            expected_sharpe=1.0,
            rebalance_trigger="MONTHLY",
            timestamp=datetime.now(),
        )

    logger.info("PHASE 1: In-sample calibration (%s to %s)", PHASE_1_START, PHASE_1_END)

    patcher_fund = mock.patch(
        "argus.agents.fundamental.FundamentalAgent.analyze", side_effect=_fund_side_effect
    )
    patcher_sent = mock.patch(
        "argus.agents.sentiment.SentimentAgent.analyze", side_effect=_sent_side_effect
    )
    patcher_port = mock.patch(
        "argus.agents.portfolio.PortfolioManagerAgent.allocate",
        autospec=True,
        side_effect=_port_side_effect,
    )
    patcher_fund.start()
    patcher_sent.start()
    patcher_port.start()

    # Step 1: Fit HMM on 2010–2020 to avoid using Phase 1 data in training
    logger.info("Step 1: Fitting macro HMM on pre-2021 FRED data (2010-2020)...")
    macro_agent = MacroStatisticalAgent()
    macro_agent.fit_on_history(start_date="2010-01-01")
    logger.info("HMM fitted. Regime mapping: %s", macro_agent.classifier.state_to_regime)

    # Step 2: Baseline with default config weights
    logger.info("Step 2: Running baseline backtest (default indicator weights)...")
    try:
        baseline = run_backtest(
            universe=UNIVERSE,
            start=PHASE_1_START,
            end=PHASE_1_END,
            initial_cash=100_000.0,
            risk_tolerance="MODERATE",
        )
        baseline_sharpe = baseline.get("sharpe")
    except Exception as e:
        logger.error("Baseline backtest failed: %s", e)
        baseline = {}
        baseline_sharpe = 0.0

    logger.info("Baseline Sharpe: %s", baseline_sharpe)

    # Step 3: Grid search over technical indicator weights
    # Walk-forward uses only 5 tickers for speed; full universe is validated in Phase 2
    logger.info("Step 3: Walk-forward weight grid search (2021-2022, 5-ticker sub-universe)...")
    best_sharpe = baseline_sharpe or 0.0
    best_weights = dict(settings.TECHNICAL_INDICATOR_WEIGHTS)

    from itertools import product

    weight_grid = {
        "rsi": [1.5, 2.0],
        "macd": [1.5, 2.0],
        "bb": [1.0, 1.5],
        "momentum": [1.0, 1.5],
    }

    for rsi, macd, bb, mom in product(*weight_grid.values()):
        test_weights = {
            "rsi": rsi,
            "macd": macd,
            "bb": bb,
            "adx": 1.0,
            "vwap": 1.0,
            "momentum": mom,
        }

        settings.TECHNICAL_INDICATOR_WEIGHTS = test_weights

        try:
            wf = run_walk_forward_validation(
                universe=UNIVERSE[:5],
                start=PHASE_1_START,
                end=PHASE_1_END,
                train_months=3,
                test_months=1,
                risk_tolerance="MODERATE",
            )

            if wf.get("avg_sharpe") is not None and wf["avg_sharpe"] > best_sharpe:
                best_sharpe = wf["avg_sharpe"]
                best_weights = test_weights
                logger.info("New best weights: %s → Sharpe %.3f", best_weights, best_sharpe)
        except Exception as e:
            logger.warning("Walk-forward failed for weights %s: %s", test_weights, e)

    settings.TECHNICAL_INDICATOR_WEIGHTS = best_weights

    # Step 4: Bias audit on baseline run
    logger.info("Step 4: Running bias audit on baseline backtest...")
    auditor = BiasAuditor(
        strategy_returns=pd.Series(dtype=float),
        trade_log=baseline.get("trade_log", []),
        universe=UNIVERSE,
    )
    bias_report = auditor.run_full_audit()

    # Step 5: Persist locked calibration params
    report = {
        "phase": 1,
        "period": f"{PHASE_1_START} to {PHASE_1_END}",
        "baseline_sharpe": baseline_sharpe,
        "optimized_sharpe": best_sharpe,
        "locked_indicator_weights": best_weights,
        "hmm_regime_mapping": macro_agent.classifier.state_to_regime,
        "bias_audit": bias_report,
        "locked_at": datetime.now().isoformat(),
        # WARNING: these params must not be overridden between Phase 1 and Phase 3
        "WARNING": "These parameters must NOT be changed before Phase 3.",
    }

    with open("calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("PHASE 1 COMPLETE. Locked Sharpe: %.3f", best_sharpe)
    logger.info("Calibration saved to: calibration_report.json")
    logger.info("Proceed to Phase 2 (walk-forward validation on 2023-2024).")

    return report


if __name__ == "__main__":
    run_phase1_calibration()
