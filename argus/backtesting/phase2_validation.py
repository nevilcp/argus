"""
argus/backtesting/phase2_validation.py

Phase 2: Out-of-sample walk-forward validation on 2023–2024 data.

Responsibilities:
  - Load the locked technical indicator weights from calibration_report.json
  - Run a 6-month train / 1-month test walk-forward validation across 2023–2024
  - Log pass/fail outcome against the avg_sharpe ≥ 0.80 threshold

Not responsible for:
  - Weight calibration (see backtesting/phase1_calibration.py)
  - Live trading (see api/main.py)

Dependencies:
  - calibration_report.json must exist (produced by phase1_calibration.py)
"""

import json
import logging
from datetime import date, datetime
from unittest import mock

from argus.backtesting.walk_forward import run_walk_forward_validation
from argus.config import settings
from argus.schemas.signals import FundamentalSignal, SentimentSignal, Signal

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("argus.phase2")


def run_phase2_validation() -> None:
    """Validates locked configuration weights on unseen out-of-sample data windows.

    Applies the same LLM mock strategy as Phase 1 to ensure that Technical + Risk
    are the only variable components during out-of-sample evaluation. LLM agents are
    mocked identically to Phase 1 to prevent result contamination from live model calls.

    Reads locked_indicator_weights from calibration_report.json. Aborts if the file
    is missing, since running Phase 2 before a completed Phase 1 produces invalid results.

    Pass criteria: avg_sharpe across all monthly windows ≥ 0.80.
    Failure means the locked Phase 1 weights do not generalize out-of-sample.
    """

    def _fund_side_effect(ticker, backtest_mode=False, session_seed=None):
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
        """Deterministic equal-weight allocation; mirrors the Phase 1 mock for comparability."""
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

    logger.info("PHASE 2: Walk-forward validation (2023-2024 out-of-sample)")

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

    # Load the exact weights locked at the end of Phase 1
    try:
        with open("calibration_report.json") as f:
            cal = json.load(f)
        settings.TECHNICAL_INDICATOR_WEIGHTS = cal["locked_indicator_weights"]
        logger.info("Loaded locked weights: %s", settings.TECHNICAL_INDICATOR_WEIGHTS)
    except Exception as e:
        logger.error("Failed to load calibration_report.json: %s", e)
        logger.error("Run phase1_calibration.py before phase2_validation.py.")
        return

    # 5-ticker sub-universe mirrors Phase 1 grid search scope; full universe reserved for Phase 3
    results = run_walk_forward_validation(
        universe=settings.UNIVERSE_DEFAULT[:5],
        start="2023-01-03",
        end="2024-12-31",
        train_months=6,
        test_months=1,
        risk_tolerance="MODERATE",
    )

    avg_sharpe = results.get("avg_sharpe")
    std_sharpe = results.get("std_sharpe")
    consistency = results.get("consistency_score")

    logger.info("Windows completed: %d", results.get("n_windows", 0))
    logger.info(
        "Average Sharpe:    %s (minimum required: 0.80)",
        f"{avg_sharpe:.3f}" if avg_sharpe is not None else "N/A",
    )
    logger.info(
        "Sharpe std dev:    %s (lower = more consistent)",
        f"{std_sharpe:.3f}" if std_sharpe is not None else "N/A",
    )
    logger.info(
        "Consistency score: %s",
        f"{consistency:.3f}" if consistency is not None else "N/A",
    )
    logger.info("PASS criteria met: %s", results.get("pass_criteria", False))

    if not results.get("pass_criteria"):
        logger.warning("PHASE 2 FAILED minimum criteria (avg_sharpe < 0.80).")
        logger.warning("Do NOT proceed to Phase 3 or live use.")
        logger.warning("Review agent weights and re-run Phase 1 calibration.")
    else:
        logger.info("PHASE 2 PASSED. Safe to proceed to Phase 3 (2025 forward test).")


if __name__ == "__main__":
    run_phase2_validation()
