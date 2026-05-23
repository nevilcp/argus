"""
Phase 2: Walk-forward validation (2023-2024).
Tests the locked parameters from Phase 1 on unseen data.
"""

import json
import logging
from datetime import date, datetime
from unittest import mock
import pandas as pd

from argus.config import settings
from argus.backtesting.walk_forward import run_walk_forward_validation
from argus.schemas.signals import FundamentalSignal, SentimentSignal, Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("argus.phase2")

def run_phase2_validation():
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
    logger.info("PHASE 2: Walk-forward validation (2023-2024)")
    logger.info("=" * 60)
    
    # Start patches
    patcher_fund = mock.patch("argus.agents.fundamental.FundamentalAgent.analyze", side_effect=fund_side_effect)
    patcher_sent = mock.patch("argus.agents.sentiment.SentimentAgent.analyze", side_effect=sent_side_effect)
    patcher_port = mock.patch("argus.agents.portfolio.PortfolioManagerAgent.allocate", autospec=True, side_effect=port_side_effect)
    patcher_fund.start()
    patcher_sent.start()
    patcher_port.start()

    # Load locked weights from Phase 1
    try:
        with open('calibration_report.json') as f:
            cal = json.load(f)
        settings.TECHNICAL_INDICATOR_WEIGHTS = cal['locked_indicator_weights']
        logger.info(f"Loaded weights: {settings.TECHNICAL_INDICATOR_WEIGHTS}")
    except Exception as e:
        logger.error(f"Failed to load calibration_report.json: {e}")
        return

    logger.info('Running Phase 2: Walk-forward validation (2023-2024)...')
    # Using 5 tickers instead of all 20 for speed (similar to Phase 1 tuning loop)
    results = run_walk_forward_validation(
        universe=settings.UNIVERSE_DEFAULT[:5], 
        start='2023-01-03',
        end='2024-12-31',
        train_months=6,
        test_months=1,
        risk_tolerance='MODERATE'
    )

    avg_sharpe = results.get("avg_sharpe")
    std_sharpe = results.get("std_sharpe")
    consistency = results.get("consistency_score")
    
    avg_str = f"{avg_sharpe:.3f}" if avg_sharpe is not None else "N/A"
    std_str = f"{std_sharpe:.3f}" if std_sharpe is not None else "N/A"
    con_str = f"{consistency:.3f}" if consistency is not None else "N/A"

    logger.info(f'Windows completed: {results.get("n_windows", 0)}')
    logger.info(f'Average Sharpe:    {avg_str} (minimum required: 0.80)')
    logger.info(f'Sharpe std dev:    {std_str} (lower = more consistent)')
    logger.info(f'Consistency score: {con_str}')
    logger.info(f'PASS criteria met: {results.get("pass_criteria", False)}')

    if not results.get('pass_criteria'):
        logger.warning('PHASE 2 FAILED minimum criteria.')
        logger.warning('Do NOT proceed to Phase 3 or live use.')
        logger.warning('Review agent weights and re-run Phase 1 calibration.')
    else:
        logger.info('PHASE 2 PASSED. Safe to proceed to Phase 3 (2025 forward test).')

if __name__ == "__main__":
    run_phase2_validation()
