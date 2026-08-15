"""
argus/orchestration/graph.py

LangGraph execution orchestrator for the ARGUS multi-agent decision network.

Coordinates asynchronous fetch pipelines, parallel specialized analyst agents,
centralized risk assessment, dynamic signal aggregation, and long-term cultural
memory ingestion into a stateful execution graph.

Responsibilities:
  - Define node functions that map ARGUSState → partial state updates
  - Build and compile the LangGraph StateGraph DAG via build_graph()

Not responsible for:
  - Agent implementation logic (see individual agent modules)
  - Schema validation (see schemas/signals.py)
  - API serving (see api/main.py)

Dependencies:
  - langgraph
  - argus.agents.* (all six specialist agents)
  - SQLite checkpointer (argus_graph.db) for resumable runs

Agent wiring is a build_graph() parameter, not a module-level singleton:
every agent that touches the network or an LLM accepts a
MarketDataProvider/LLMClient, defaulting to the real implementation, so a
test can build a fixture-backed graph without patching this module's
internals. `graph` below is the default, real-data instance — the one
api/main.py imports.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from argus.agents.fundamental import FundamentalAgent
from argus.agents.macro import MacroStatisticalAgent
from argus.agents.portfolio import PortfolioManagerAgent
from argus.agents.risk import RiskStatisticalEngine
from argus.agents.sentiment import SentimentAgent
from argus.agents.technical import TechnicalStatisticalAgent
from argus.memory.cultural import get_cultural_memory
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.orchestration.state import ARGUSState
from argus.schemas.signals import AggregatedSignal, ARGUSDecision, RiskVerdict
from argus.seams import LiveMarketDataProvider, LLMClient, MarketDataProvider

logger = logging.getLogger("argus.graph")

# Every schemas.signals model nested in ARGUSState must be listed here or the
# checkpointer's msgpack deserializer degrades it to a plain dict; shared with
# reconciliation.py's checkpoint loader so the two allowlists can't drift apart
_CHECKPOINT_MODEL_CLASSES = (
    "MacroContext",
    "TechnicalSignal",
    "FundamentalSignal",
    "SentimentSignal",
    "RiskAssessment",
    "AggregatedSignal",
    "PositionAllocation",
    "PortfolioAllocation",
    "ARGUSDecision",
)


def build_checkpoint_serde() -> JsonPlusSerializer:
    """Returns the JsonPlusSerializer used to read/write ARGUSState checkpoints."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("argus.schemas.signals", cls) for cls in _CHECKPOINT_MODEL_CLASSES
        ]
    )


def build_graph(
    market_data: Optional[MarketDataProvider] = None,
    fundamental_llm: Optional[LLMClient] = None,
    sentiment_llm: Optional[LLMClient] = None,
    portfolio_llm: Optional[LLMClient] = None,
    checkpoint_db_path: str = "argus_graph.db",
):
    """Constructs and compiles the ARGUS decision graph.

    Args:
        market_data: Shared data source injected into every agent that fetches
            market/macro data (macro, fundamental, sentiment, risk). Defaults
            to a live LiveMarketDataProvider for each agent independently.
        fundamental_llm: LLMClient for FundamentalAgent. Defaults to a real
            Groq client.
        sentiment_llm: LLMClient for SentimentAgent. Defaults to a real Groq
            client.
        portfolio_llm: LLMClient for PortfolioManagerAgent. Defaults to a real
            Groq client.
        checkpoint_db_path: SQLite file the compiled graph checkpoints
            ARGUSState to after each run. Exposed (rather than hardcoded) so
            tests — and argus/orchestration/reconciliation.py's own tests —
            can point it at a temp file instead of the real
            argus_graph.db production callers use.

    Returns:
        A compiled LangGraph graph, ready for .invoke()/.ainvoke().
    """
    market_data = market_data or LiveMarketDataProvider()

    macro_agent = MacroStatisticalAgent(market_data=market_data)
    tech_agent = TechnicalStatisticalAgent()
    fund_agent = FundamentalAgent(llm_client=fundamental_llm, market_data=market_data)
    sent_agent = SentimentAgent(llm_client=sentiment_llm, market_data=market_data)
    risk_engine = RiskStatisticalEngine(market_data=market_data)
    aggregator = HybridSignalAggregator()
    portfolio_agent = PortfolioManagerAgent(llm_client=portfolio_llm)

    def node_fetch_price_history(state: ARGUSState) -> dict:
        """Fetches raw daily pricing data for risk covariance calculation.

        Session states (technical indicators) are sourced exclusively from the MFT
        live cache injected by the API layer. Daily compression is intentionally
        omitted here — computing indicators from daily closes produces a different
        signal resolution than the intraday 5-minute data the TechnicalAgent expects.
        Tickers absent from the MFT cache will carry no session_state and will be
        excluded from downstream technical and portfolio allocation nodes.

        Args:
            state: ARGUSState containing ``universe`` list of tickers and
                ``session_states`` pre-populated from the MFT live cache.

        Returns:
            Partial state update with ``price_history`` only. ``session_states``
            is passed through unchanged from the incoming state.
        """
        history_dfs = market_data.multiple_daily(state["universe"], period="1y")
        history_serializable = {}

        for ticker, df in history_dfs.items():
            history_serializable[ticker] = {
                "dates": df.index.astype(str).tolist(),
                "prices": df["close"].tolist(),
            }

        # session_states already populated from MFT cache by API layer; pass through unchanged
        return {
            "price_history": history_serializable,
            "session_states": state.get("session_states") or {},
        }

    def node_macro_analysis(state: ARGUSState) -> dict:
        """Classifies current macroeconomic regimes and volatility indices.

        Args:
            state: ARGUSState (no required fields; macro agent uses its own data sources).

        Returns:
            Partial state update with ``macro_context``.
        """
        macro_ctx = macro_agent.analyze()
        if macro_ctx is None:
            logger.error(
                "node_macro_analysis: MacroStatisticalAgent returned None (FRED data unavailable). "
                "Routing to END — no downstream agents will run this session."
            )
            return {
                "macro_context": None,
                "errors": ["macro_analysis: MacroStatisticalAgent returned None (FRED data unavailable)"],
            }
        return {"macro_context": macro_ctx}

    def node_technical_analysis(state: ARGUSState) -> dict:
        """Computes statistical technical indicator metrics across active assets.

        Args:
            state: ARGUSState with ``session_states`` populated by ``node_fetch_price_history``.

        Returns:
            Partial state update with ``technical_signals``.
        """
        session_states = state.get("session_states", {})
        signals, errors = tech_agent.batch_analyze(session_states)
        return {"technical_signals": signals, "errors": errors}

    def node_fundamental_analysis(state: ARGUSState) -> dict:
        """Audits corporate health metrics and qualitative economic moat vectors.

        Args:
            state: ARGUSState with ``universe``, ``backtest_mode``, and ``session_seed``.

        Returns:
            Partial state update with ``fundamental_signals``.
        """
        signals, errors = fund_agent.batch_analyze(
            state["universe"],
            backtest_mode=state.get("backtest_mode", False),
            session_seed=state.get("session_seed"),
        )
        return {"fundamental_signals": signals, "errors": errors}

    def node_sentiment_analysis(state: ARGUSState) -> dict:
        """Scores news articles and social media sentiment signals.

        Args:
            state: ARGUSState with ``universe`` list of tickers.

        Returns:
            Partial state update with ``sentiment_signals``.
        """
        signals, errors = sent_agent.batch_analyze(state["universe"])
        return {"sentiment_signals": signals, "errors": errors}

    def node_retrieve_cultural_memory(state: ARGUSState) -> dict:
        """Retrieves semantic trading wisdom and historical trade patterns matching the current regime.

        Args:
            state: ARGUSState with ``macro_context`` to scope the semantic similarity query.

        Returns:
            Partial state update with ``cultural_memory`` dict containing 'wisdom' and 'warnings' lists.
        """
        macro = state.get("macro_context")
        if not macro:
            return {"cultural_memory": {"wisdom": [], "warnings": []}}

        try:
            memory = get_cultural_memory()
        except ImportError as exc:
            # sentence-transformers/torch are an optional [models] extra; their
            # absence degrades to the same empty result as no macro context
            logger.warning("node_retrieve_cultural_memory: cultural memory unavailable: %s", exc)
            return {"cultural_memory": {"wisdom": [], "warnings": []}}

        wisdom = memory.retrieve_wisdom(macro, "mixed technicals")
        warnings = memory.retrieve_warnings(macro)

        return {
            "cultural_memory": {
                "wisdom": wisdom if wisdom else [],
                "warnings": warnings if warnings else [],
            }
        }

    def node_signal_aggregation(state: ARGUSState) -> dict:
        """Consolidates specialized analyst signals into weighted consensus indicators.

        Args:
            state: ARGUSState with ``technical_signals``, ``fundamental_signals``,
                ``sentiment_signals``, and ``macro_context`` populated.

        Returns:
            Partial state update with ``aggregated_signals``.
        """
        aggs: dict[str, AggregatedSignal] = {}
        macro = state["macro_context"]
        if not macro:
            return {"aggregated_signals": aggs}

        # Per-regime reliability doesn't depend on ticker, so compute it once per
        # session rather than once per ticker.
        regime = macro.macro_regime.value
        try:
            memory = get_cultural_memory()
            reliability = {
                name: memory.get_agent_accuracy(name, regime=regime)
                for name in ("technical", "fundamental", "sentiment")
            }
        except ImportError as exc:
            # Same optional-dependency guard as node_retrieve_cultural_memory;
            # degrade every agent to the 0.5 neutral prior rather than crash
            logger.warning("node_signal_aggregation: cultural memory unavailable: %s", exc)
            reliability = {name: 0.5 for name in ("technical", "fundamental", "sentiment")}

        for ticker in state["universe"]:
            tech = state.get("technical_signals", {}).get(ticker)
            fund = state.get("fundamental_signals", {}).get(ticker)
            sent = state.get("sentiment_signals", {}).get(ticker)
            if tech is None and fund is None and sent is None:
                continue
            aggs[ticker] = aggregator.aggregate(tech, macro, fund, sent, reliability=reliability)
        return {"aggregated_signals": aggs}

    def node_risk_evaluation(state: ARGUSState) -> dict:
        """Enforces structural, statistical, and sector concentration risk caps.

        Evaluates the full portfolio covariance first, then maps SLSQP-derived
        allocation caps back to per-ticker RiskAssessments. Portfolio-level VETO
        verdicts propagate down to suppress all individual asset approvals.

        Args:
            state: ARGUSState with ``price_history``, ``macro_context``, ``universe``,
                and ``aggregated_signals`` populated.

        Returns:
            Partial state update with ``risk_assessments``.
        """
        import pandas as pd

        history_raw = state.get("price_history", {})
        history = {
            ticker: pd.Series(data["prices"], index=pd.to_datetime(data["dates"]))
            for ticker, data in history_raw.items()
        }
        macro_ctx = state.get("macro_context")
        vix = macro_ctx.vix_level if macro_ctx else 20.0
        universe = state["universe"]

        # Extract signed convictions: positive = BULLISH, negative = BEARISH, zero = NEUTRAL
        convictions = {}
        aggs = state.get("aggregated_signals", {})
        if aggs:
            for t, sig in aggs.items():
                sign = (
                    1.0
                    if sig.signal.value == "BULLISH"
                    else (-1.0 if sig.signal.value == "BEARISH" else 0.0)
                )
                convictions[t] = sig.conviction * sign

        # Joint evaluation captures inter-asset covariance and global diversification limits
        base_weight = min(0.15, 1.0 / len(universe)) if universe else 0.15
        full_portfolio = [{"ticker": t, "weight": base_weight} for t in universe]
        portfolio_result = risk_engine.evaluate(full_portfolio, history, vix, convictions=convictions)

        portfolio_vetoed = portfolio_result.verdict == RiskVerdict.VETO
        ticker_cap_map: dict[str, float] = portfolio_result.optimal_weights

        assessments = {}

        for ticker in universe:
            single = risk_engine.evaluate([{"ticker": ticker, "weight": 0.15}], history, vix)

            # Propagate portfolio-level correlation stats for uniform telemetry
            single = single.model_copy(update={"avg_correlation": portfolio_result.avg_correlation})

            if ticker in ticker_cap_map and not portfolio_vetoed:
                cap = ticker_cap_map[ticker]
                # Downgrade verdict monotonically: never upgrade a VETO to REDUCE or APPROVE
                new_verdict = (
                    single.verdict
                    if single.verdict == RiskVerdict.VETO
                    else (RiskVerdict.REDUCE if cap < single.approved_weight else single.verdict)
                )
                single = single.model_copy(
                    update={
                        "verdict": new_verdict,
                        "approved_weight": min(cap, single.approved_weight),
                        "veto_reasons": single.veto_reasons + [f"[Portfolio] SLSQP cap: {cap:.1%}"],
                    }
                )

            # Portfolio-level veto overrides all individual verdicts to prevent partial allocation
            if portfolio_vetoed:
                single = single.model_copy(
                    update={
                        "verdict": RiskVerdict.VETO,
                        "approved_weight": 0.0,
                        "veto_reasons": single.veto_reasons
                        + [f"[Portfolio] {r}" for r in portfolio_result.veto_reasons],
                    }
                )

            assessments[ticker] = single

        return {"risk_assessments": assessments}

    def node_portfolio_allocation(state: ARGUSState) -> dict:
        """Determines capital allocations using specialist ratings, risk targets, and memory heuristics.

        Args:
            state: ARGUSState with all signal, risk, and cultural memory fields populated.

        Returns:
            Partial state update with ``portfolio_allocation``.
        """
        signals_dict = {}
        for ticker in state["universe"]:
            signals_dict[ticker] = {
                "technical": state.get("technical_signals", {}).get(ticker),
                "fundamental": state.get("fundamental_signals", {}).get(ticker),
                "sentiment": state.get("sentiment_signals", {}).get(ticker),
                "risk": state.get("risk_assessments", {}).get(ticker),
                "aggregated": state.get("aggregated_signals", {}).get(ticker),
            }

        profile = {
            "total_wealth": state.get("total_wealth"),
            "invest_pct": state.get("invest_pct"),
            "risk_tolerance": state.get("risk_tolerance"),
        }

        macro = state.get("macro_context")
        if not macro:
            return {"errors": ["portfolio_allocation: skipped, no macro_context in state"]}

        mem = state.get("cultural_memory", {})
        wisdom = mem.get("wisdom", [])
        warnings = mem.get("warnings", [])

        alloc = portfolio_agent.allocate(profile, signals_dict, macro, wisdom, warnings)
        if alloc is None:
            logger.error(
                "node_portfolio_allocation: PortfolioManagerAgent returned None (LLM API failure). "
                "Skipping portfolio_allocation to allow graceful exit."
            )
            return {
                "errors": [
                    "portfolio_allocation: PortfolioManagerAgent returned None (LLM API failure)"
                ]
            }
        return {"portfolio_allocation": alloc}

    def route_after_macro(state: ARGUSState):
        """Fans out to the four parallel analyst nodes, or short-circuits to END.

        A None macro_context means the core FRED data bundle was entirely
        unavailable (see MacroStatisticalAgent.analyze's docstring) — every
        downstream node depends on macro_context, so there is nothing useful
        left for this session to do.

        Args:
            state: ARGUSState with ``macro_context`` populated by node_macro_analysis.

        Returns:
            END if macro_context is None, otherwise a list of Send calls fanning
            out to technical_analysis, fundamental_analysis, sentiment_analysis,
            and retrieve_cultural_memory.
        """
        if state.get("macro_context") is None:
            return END
        return [
            Send("technical_analysis", state),
            Send("fundamental_analysis", state),
            Send("sentiment_analysis", state),
            Send("retrieve_cultural_memory", state),
        ]

    def node_log_decisions(state: ARGUSState) -> dict:
        """Persists structural decision profiles to vector database collections.

        Args:
            state: ARGUSState with all fields populated from prior nodes.

        Returns:
            Partial state update with ``decisions`` list of completed ARGUSDecision objects.
        """
        allocation = state.get("portfolio_allocation")
        if not allocation:
            logger.warning("[log_decisions] No portfolio_allocation in state; skipping.")
            return {"decisions": []}

        positions = {pos.ticker: pos for pos in allocation.portfolio}
        now = datetime.now()
        decisions: list[ARGUSDecision] = []

        for ticker in state.get("universe", []):
            try:
                decision = ARGUSDecision(
                    ticker=ticker,
                    session_timestamp=now,
                    technical=state.get("technical_signals", {}).get(ticker),
                    macro=state.get("macro_context"),
                    fundamental=state.get("fundamental_signals", {}).get(ticker),
                    sentiment=state.get("sentiment_signals", {}).get(ticker),
                    risk=state.get("risk_assessments", {}).get(ticker),
                    aggregated=state.get("aggregated_signals", {}).get(ticker),
                    allocation=positions.get(ticker),
                )
                decisions.append(decision)

                # Snapshot stored pre-confirmation to enable pre-settlement similarity retrieval
                get_cultural_memory().store_decision_snapshot(decision)

            except Exception as exc:
                logger.warning("[log_decisions] Failed to build decision for %s: %s", ticker, exc)

        logger.info(
            "[log_decisions] Logged %d/%d decisions to cultural memory.",
            len(decisions),
            len(state.get("universe", [])),
        )
        return {"decisions": decisions}

    builder = StateGraph(ARGUSState)

    builder.add_node("fetch_price_history", node_fetch_price_history)
    builder.add_node("macro_analysis", node_macro_analysis)
    builder.add_node("technical_analysis", node_technical_analysis)
    builder.add_node("fundamental_analysis", node_fundamental_analysis)
    builder.add_node("sentiment_analysis", node_sentiment_analysis)
    builder.add_node("retrieve_cultural_memory", node_retrieve_cultural_memory)
    builder.add_node("signal_aggregation", node_signal_aggregation)
    builder.add_node("risk_evaluation", node_risk_evaluation)
    builder.add_node("portfolio_allocation", node_portfolio_allocation)
    builder.add_node("log_decisions", node_log_decisions)

    builder.set_entry_point("fetch_price_history")
    builder.add_edge("fetch_price_history", "macro_analysis")

    builder.add_conditional_edges("macro_analysis", route_after_macro)

    builder.add_edge(
        [
            "technical_analysis",
            "fundamental_analysis",
            "sentiment_analysis",
            "retrieve_cultural_memory",
        ],
        "signal_aggregation",
    )
    builder.add_edge("signal_aggregation", "risk_evaluation")
    builder.add_edge("risk_evaluation", "portfolio_allocation")
    builder.add_edge("portfolio_allocation", "log_decisions")
    builder.add_edge("log_decisions", END)

    try:
        if checkpoint_db_path != ":memory:":
            Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn, serde=build_checkpoint_serde())
    except Exception:
        checkpointer = None

    return builder.compile(checkpointer=checkpointer)


# Default, real-data graph — what api/main.py imports and invokes in production.
graph = build_graph()
