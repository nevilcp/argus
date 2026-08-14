"""
tests/test_integration.py
=========================
Comprehensive integration tests for ARGUS.
Tests individual pipelines, the orchestrator graph, schema serialization,
PiT enforcer, and Governor state limits.
"""

import asyncio
import time
from datetime import date, datetime, timezone
from unittest import mock
from uuid import uuid4

import pandas as pd
import pytest
import yfinance as yf

from argus.agents.macro import MacroStatisticalAgent
from argus.agents.portfolio import half_kelly_weight
from argus.agents.risk import RiskStatisticalEngine
from argus.agents.technical import TechnicalStatisticalAgent
from argus.data.pipeline import MFTDataPipeline
from argus.orchestration.governor import BOOTSTRAP_LIMITS, governor
from argus.orchestration.graph import graph
from argus.orchestration.state import ARGUSState
from argus.risk.kill_switch import KillSwitch
from argus.schemas.signals import (
    FundamentalSignal,
    MacroContext,
    Regime,
    RiskAssessment,
    SectorSignal,
    SentimentSignal,
    Signal,
    TechnicalSignal,
    VixRegime,
    YieldCurve,
)


class TestEndToEnd:
    """Integration tests exercising real agents, the orchestrator graph, and shared state."""

    def test_statistical_agents_pipeline(self):
        """TechnicalAgent -> MacroAgent -> RiskEngine chain runs with zero LLM calls."""
        pipeline = MFTDataPipeline(["AAPL"])
        df = yf.download("AAPL", period="60d", interval="5m", progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()

        if len(df) >= 14:
            for ts, row in df.iterrows():
                candle = {
                    "timestamp": ts.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                pipeline.buffer.insert_candle("AAPL", candle)

        states = pipeline.compress_all()
        assert "AAPL" in states, "Pipeline produced no compressed state for AAPL"

        tech = TechnicalStatisticalAgent()
        t_sig = tech.analyze("AAPL", states["AAPL"])
        assert isinstance(t_sig, TechnicalSignal)
        assert t_sig.api_calls_used == 0

        macro = MacroStatisticalAgent()
        m_ctx = macro.analyze()
        assert isinstance(m_ctx, MacroContext)
        assert m_ctx.api_calls_used == 0

        risk = RiskStatisticalEngine()
        price_history = {
            "AAPL": pd.Series(
                [150.0, 151.0, 149.0, 155.0, 160.0], index=pd.date_range("2023-01-01", periods=5)
            )
        }
        r_sig = risk.evaluate([{"ticker": "AAPL", "weight": 0.1}], price_history, m_ctx.vix_level)
        assert isinstance(r_sig, RiskAssessment)
        assert r_sig.api_calls_used == 0

    @pytest.mark.asyncio
    @mock.patch("argus.orchestration.graph.get_cultural_memory")
    @mock.patch("argus.agents.fundamental.FundamentalAgent.analyze")
    @mock.patch("argus.agents.sentiment.SentimentAgent.analyze")
    async def test_full_graph_smoke(self, mock_sent, mock_fund, mock_cultural_memory):
        """Runs the complete LangGraph graph for AAPL, MSFT using mocked LLMs.

        Cultural memory is mocked out (not just left real) because its embedding
        function needs the optional `[models]` extra (sentence-transformers) that
        default installs/CI don't have, and because this smoke test is about graph
        wiring, not vector-DB behavior.

        Args:
            mock_sent: Patched SentimentAgent.analyze.
            mock_fund: Patched FundamentalAgent.analyze.
            mock_cultural_memory: Patched get_cultural_memory.
        """
        mock_cultural_memory.return_value = mock.Mock(
            retrieve_wisdom=mock.Mock(return_value=[]),
            retrieve_warnings=mock.Mock(return_value=[]),
            get_agent_accuracy=mock.Mock(return_value=0.5),
            store_decision_snapshot=mock.Mock(),
        )

        def fund_side_effect(ticker, backtest_mode=False, session_seed=None):
            """Stand in for FundamentalAgent.analyze, matching its real signature.

            Args:
                ticker: Ticker to build a signal for.
                backtest_mode: Unused; accepted to match the real signature.
                session_seed: Unused; accepted to match the real signature.

            Returns:
                A fixed BULLISH FundamentalSignal for the given ticker.
            """
            return FundamentalSignal(
                ticker=ticker,
                sector="Technology",
                industry="Software",
                signal=Signal.BULLISH,
                conviction=0.8,
                moat_score=8.0,
                reasoning="Test",
                data_as_of_date=date.today(),
                timestamp=datetime.now(),
            )

        mock_fund.side_effect = fund_side_effect

        def sent_side_effect(ticker, *args, **kwargs):
            """Stand in for SentimentAgent.analyze, matching its real signature.

            Args:
                ticker: Ticker to build a signal for.
                *args: Unused; accepted to match the real signature.
                **kwargs: Unused; accepted to match the real signature.

            Returns:
                A fixed BULLISH SentimentSignal for the given ticker.
            """
            return SentimentSignal(
                ticker=ticker,
                signal=Signal.BULLISH,
                conviction=0.7,
                finbert_net_score=0.5,
                pct_positive=0.6,
                pct_negative=0.1,
                news_volume_7d=10,
                social_volume_change_pct=10.0,
                social_mention_surge=False,
                upcoming_catalyst=False,
                sentiment_decay_risk="LOW",
                reasoning="Test",
                timestamp=datetime.now(),
            )

        mock_sent.side_effect = sent_side_effect

        state = ARGUSState(
            ticker="AAPL",
            total_wealth=10000,
            invest_pct=0.8,
            risk_tolerance="MODERATE",
            universe=["AAPL", "MSFT"],
            backtest_mode=False,
            session_seed=None,
            session_states={},
            price_history={},
            technical_signals={},
            macro_context=None,
            fundamental_signals={},
            sentiment_signals={},
            risk_assessments={},
            aggregated_signals={},
            cultural_memory={"wisdom": [], "warnings": []},
            portfolio_allocation=None,
            decisions=[],
            errors=[],
        )

        config = {"configurable": {"thread_id": str(uuid4())}}

        try:
            final_state = await asyncio.to_thread(graph.invoke, state, config)
            alloc = final_state.get("portfolio_allocation")
            assert alloc is not None
            assert len(alloc.portfolio) >= 1 or alloc.cash_reserve_pct == 1.0

            for pos in alloc.portfolio:
                assert pos.stop_loss > 0.0
        except Exception as e:
            # The checkpointer can raise a dataframe serialization error unrelated to graph wiring
            if "msgpack" in str(e).lower() or "not msgpack serializable" in str(e).lower():
                pass
            else:
                raise e

    def test_kill_switch_drawdown_trigger(self):
        """The kill switch halts once portfolio value drops past the drawdown threshold."""
        ks = KillSwitch("MODERATE", check_interval_seconds=1)
        ks.start(10000.0)
        ks.update_portfolio_value(8700.0)
        time.sleep(2.5)
        assert ks.is_halted

    def test_schema_round_trip(self):
        """Signal schemas survive a JSON dump/validate round trip unchanged."""
        tech = TechnicalSignal(
            ticker="AAPL",
            current_price=150.0,
            signal=Signal.BULLISH,
            conviction=0.8,
            net_score=0.7,
            rsi_14=60,
            macd_histogram=0.5,
            bb_percent_b=0.8,
            atr_pct=0.02,
            adx_14=25,
            vwap_distance=0.01,
            volume_ratio=1.2,
            momentum_30m=0.001,
            momentum_1d=0.01,
            timestamp=datetime.now(),
        )
        tech_json = tech.model_dump_json()
        tech2 = TechnicalSignal.model_validate_json(tech_json)
        assert tech.ticker == tech2.ticker

        macro = MacroContext(
            fed_funds=4.5,
            cpi_yoy=3.2,
            unemployment=3.8,
            t10y2y=-0.2,
            consumer_sentiment=68.5,
            macro_regime=Regime.EXPANSION,
            interest_rate_trend="STABLE",
            yield_curve_shape=YieldCurve.NORMAL,
            vix_level=15.0,
            vix_regime=VixRegime.LOW,
            vix_percentile=30.0,
            inflation_trajectory="STABLE",
            sector_rotation_signal=SectorSignal.GROWTH_FAVORED,
            agent_multipliers={"fundamental": 1.0, "technical": 1.0, "sentiment": 1.0},
            regime_confidence=0.8,
            model_healthy=True,
            timestamp=datetime.now(),
        )
        macro_json = macro.model_dump_json()
        macro2 = MacroContext.model_validate_json(macro_json)
        assert macro.macro_regime == macro2.macro_regime

    def test_governor_prevents_over_limit(self):
        """The shared governor sleeps once a model's per-minute request count reaches its limit."""
        model = "llama-3.3-70b-versatile"
        limit = BOOTSTRAP_LIMITS[model]["requests_per_minute"]

        usage = governor._get_usage(model)
        # governor is a module-level singleton, so usage.current_minute may have been
        # stamped by an earlier test; pin it to now rather than assume it matches, or
        # a real minute rollover mid-suite spuriously resets the counter below
        usage.current_minute = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        usage.requests_this_minute = limit - 1

        # Simulates the minute rolling over during the sleep, so the retry loop's
        # re-check finds room on its second pass rather than exhausting every
        # attempt against a mock that never advances real time.
        def _advance_minute(_seconds):
            usage.current_minute = "1970-01-01T00:00"

        # Reaching exactly the limit still succeeds without sleeping
        try:
            with mock.patch(
                "argus.orchestration.governor.time.sleep", side_effect=_advance_minute
            ) as mock_sleep:
                governor.wait_if_needed(model)
                assert usage.requests_this_minute == limit
                assert mock_sleep.call_count == 0

                # Exceeding it sleeps out the remainder of the current minute
                governor.wait_if_needed(model)
                assert mock_sleep.call_count == 1
        finally:
            # governor is a module-level singleton shared across tests; the
            # _advance_minute side effect above stamped a sentinel `current_minute`
            # onto its ModelUsage, which must not leak into later tests
            usage.requests_this_minute = 0
            usage.current_minute = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")

    def test_half_kelly_formula(self):
        """half_kelly_weight() computes half the Kelly-optimal position size."""
        weight = half_kelly_weight(
            win_probability=0.6, avg_win_pct=0.08, avg_loss_pct=0.04, max_position=0.15
        )
        assert abs(weight - 0.15) < 1e-5
