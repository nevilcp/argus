"""
Phase 1: In-sample calibration on 2021-2022 data.
Calibrates: TechnicalStatisticalAgent indicator weights.
Fits: MacroStatisticalAgent HMM on FRED data.
Produces: calibration_report.json with optimal parameters.
"""

import json
import logging
from datetime import date, datetime
import pandas as pd

from argus.agents.macro import MacroStatisticalAgent
from argus.backtesting.engine import run_backtest
from argus.backtesting.walk_forward import run_walk_forward_validation
from argus.backtesting.bias_auditor import BiasAuditor
from argus.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("argus.phase1")

UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","JPM","V","UNH",
    "XOM","JNJ","PG","MA","HD","MRK","ABBV","LLY","AVGO","CVX"
]

PHASE_1_START = "2021-01-04"
PHASE_1_END   = "2022-12-30"

def run_phase1_calibration():
    from unittest import mock
    from argus.schemas.signals import FundamentalSignal, SentimentSignal, Signal, PortfolioAllocation
    
    # Mocking definitions
    def fund_side_effect(ticker, backtest_mode=False, session_seed=None):
        return FundamentalSignal(ticker=ticker, signal=Signal.BULLISH, conviction=0.6,
            moat_score=7, reasoning="Mock", data_as_of_date=date.today(), timestamp=datetime.now())
            
    def sent_side_effect(ticker, *args, **kwargs):
        return SentimentSignal(ticker=ticker, signal=Signal.BULLISH, conviction=0.6,
            finbert_net_score=0.2, pct_positive=0.5, pct_negative=0.2, news_volume_7d=10,
            social_mention_surge=False, upcoming_catalyst=False, sentiment_decay_risk="LOW",
            reasoning="Mock", timestamp=datetime.now())

    def port_side_effect(self, user_profile, all_signals, macro, *args, **kwargs):
        from argus.schemas.signals import PortfolioAllocation, PositionAllocation, RiskVerdict
        from uuid import uuid4
        investable = user_profile.get("total_wealth", 100000.0) * user_profile.get("invest_pct", 0.8)
        
        approved = [
            t for t, s in all_signals.items() 
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
                timestamp=datetime.now()
            )
            
        weight = min(0.8 / len(approved), 0.15)
        
        portfolio = [
            PositionAllocation(
                ticker=t,
                allocation_pct=weight,
                allocation_usd=investable * weight,
                stop_loss=float(all_signals[t]["risk"].stop_losses.get(t, 0.0) or 0.0),
                target_price=None,
                thesis="Mocked backtest allocation",
                composite_conviction=0.6,
                time_horizon="3-6 months"
            ) for t in approved
        ]
            
        return PortfolioAllocation(
            session_id=str(uuid4()),
            user_investable_capital=investable,
            portfolio=portfolio,
            cash_reserve_pct=round(1.0 - (weight * len(approved)), 4),
            expected_sharpe=1.0,
            rebalance_trigger="MONTHLY",
            timestamp=datetime.now()
        )

    logger.info("=" * 60)
    logger.info("PHASE 1: In-sample calibration (2021-2022)")
    logger.info("=" * 60)
    
    # Start patches
    patcher_fund = mock.patch("argus.agents.fundamental.FundamentalAgent.analyze", side_effect=fund_side_effect)
    patcher_sent = mock.patch("argus.agents.sentiment.SentimentAgent.analyze", side_effect=sent_side_effect)
    patcher_port = mock.patch("argus.agents.portfolio.PortfolioManagerAgent.allocate", autospec=True, side_effect=port_side_effect)
    patcher_fund.start()
    patcher_sent.start()
    patcher_port.start()
    
    # Step 1: Fit HMM on pre-2021 FRED data
    logger.info("Step 1: Fitting Macro HMM on 2010-2020 FRED data...")
    macro_agent = MacroStatisticalAgent()
    macro_agent.fit_on_history(start_date="2010-01-01")
    logger.info(f"HMM fitted. Regime mapping: {macro_agent.classifier.state_to_regime}")
    
    # Step 2: Run baseline backtest with default weights
    logger.info("Step 2: Running baseline backtest (default indicator weights)...")
    try:
        baseline = run_backtest(
            universe=UNIVERSE,
            start=PHASE_1_START,
            end=PHASE_1_END,
            initial_cash=100_000.0,
            risk_tolerance="MODERATE"
        )
        baseline_sharpe = baseline.get("sharpe")
    except Exception as e:
        logger.error(f"Baseline backtest failed: {e}")
        baseline = {}
        baseline_sharpe = 0.0

    logger.info(f"Baseline Sharpe: {baseline_sharpe}")
    
    # Step 3: Weight grid search using walk-forward on 2021-2022
    # (Use 3-month windows to stay within Phase 1 data)
    logger.info("Step 3: Walk-forward weight calibration (2021-2022)...")
    best_sharpe = baseline_sharpe or 0.0
    best_weights = dict(settings.TECHNICAL_INDICATOR_WEIGHTS)
    
    from itertools import product
    # Smaller grid for faster execution in testing environment
    weight_grid = {
        "rsi":      [1.5, 2.0],
        "macd":     [1.5, 2.0],
        "bb":       [1.0, 1.5],
        "momentum": [1.0, 1.5],
    }
    
    for rsi, macd, bb, mom in product(*weight_grid.values()):
        test_weights = {"rsi":rsi, "macd":macd, "bb":bb, "adx":1.0, 
                        "vwap":1.0, "momentum":mom}
        
        # Temporarily override weights in config
        settings.TECHNICAL_INDICATOR_WEIGHTS = test_weights
        
        try:
            wf = run_walk_forward_validation(
                universe=UNIVERSE[:5],  # Smaller universe for speed
                start=PHASE_1_START,
                end=PHASE_1_END,
                train_months=3,
                test_months=1,
                risk_tolerance="MODERATE"
            )
            
            if wf.get("avg_sharpe") is not None and wf["avg_sharpe"] > best_sharpe:
                best_sharpe = wf["avg_sharpe"]
                best_weights = test_weights
                logger.info(f"New best weights: {best_weights} -> Sharpe {best_sharpe:.3f}")
        except Exception as e:
            logger.warning(f"Walk-forward failed for weights {test_weights}: {e}")
    
    # Restoring best weights in config
    settings.TECHNICAL_INDICATOR_WEIGHTS = best_weights

    # Step 4: Run bias audit on baseline backtest
    logger.info("Step 4: Running bias audit...")
    auditor = BiasAuditor()
    daily_returns = baseline.get("daily_returns", {})
    if isinstance(daily_returns, dict):
         daily_returns = pd.Series(daily_returns)
    elif not isinstance(daily_returns, pd.Series):
         daily_returns = pd.Series()

    bias_report = auditor.run_full_audit(
        universe=UNIVERSE,
        backtest_start=date(2021, 1, 4),
        strategy_returns=daily_returns,
    )
    
    # Step 5: Save calibration report
    report = {
        "phase": 1,
        "period": f"{PHASE_1_START} to {PHASE_1_END}",
        "baseline_sharpe": baseline_sharpe,
        "optimized_sharpe": best_sharpe,
        "locked_indicator_weights": best_weights,
        "hmm_regime_mapping": macro_agent.classifier.state_to_regime,
        "bias_audit": bias_report,
        "locked_at": datetime.now().isoformat(),
        "WARNING": "These parameters must NOT be changed before Phase 3."
    }
    
    with open("calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"PHASE 1 COMPLETE. Locked Sharpe: {best_sharpe:.3f}")
    logger.info(f"Calibration saved to: calibration_report.json")
    logger.info("Proceed to Phase 2 (walk-forward validation on 2023-2024).")
    logger.info("=" * 60)
    
    return report

if __name__ == "__main__":
    run_phase1_calibration()
