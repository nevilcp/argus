"""
argus/orchestration/graph.py

LangGraph execution orchestrator for the ARGUS multi-agent decision network.

Coordinates asynchronous fetch pipelines, parallel specialized analyst agents,
centralized risk assessment, dynamic signal aggregation, and long-term cultural
memory ingestion into a stateful execution graph.

Responsibilities:
  - Define and compile the LangGraph StateGraph DAG
  - Instantiate and share all module-level agent singletons
  - Provide node functions that map ARGUSState → partial state updates

Not responsible for:
  - Agent implementation logic (see individual agent modules)
  - Schema validation (see schemas/signals.py)
  - API serving (see api/main.py)

Dependencies:
  - langgraph
  - argus.agents.* (all six specialist agents)
  - SQLite checkpointer (argus_graph.db) for resumable runs
"""

import logging
import sqlite3
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from argus.agents.fundamental import FundamentalAgent
from argus.agents.macro import MacroStatisticalAgent
from argus.agents.portfolio import PortfolioManagerAgent
from argus.agents.risk import RiskStatisticalEngine
from argus.agents.sentiment import SentimentAgent
from argus.agents.technical import TechnicalStatisticalAgent
from argus.data.fetchers import fetch_multiple_daily
from argus.memory.cultural import cultural_memory
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.orchestration.state import ARGUSState
from argus.schemas.signals import ARGUSDecision, RiskVerdict

logger = logging.getLogger("argus.graph")

# Module-level singletons avoid re-initialization overhead across graph invocations
macro_agent = MacroStatisticalAgent()
tech_agent = TechnicalStatisticalAgent()
fund_agent = FundamentalAgent()
sent_agent = SentimentAgent()
risk_engine = RiskStatisticalEngine()
aggregator = HybridSignalAggregator()
portfolio_agent = PortfolioManagerAgent()


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
    history_dfs = fetch_multiple_daily(state["universe"], period="1y")
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
            "Downstream agents will be skipped."
        )
        return {"macro_context": None}
    return {"macro_context": macro_ctx}


def node_technical_analysis(state: ARGUSState) -> dict:
    """Computes statistical technical indicator metrics across active assets.

    Args:
        state: ARGUSState with ``session_states`` populated by ``node_fetch_price_history``.

    Returns:
        Partial state update with ``technical_signals``.
    """
    session_states = state.get("session_states", {})
    signals = tech_agent.batch_analyze(session_states)
    return {"technical_signals": signals}


def node_fundamental_analysis(state: ARGUSState) -> dict:
    """Audits corporate health metrics and qualitative economic moat vectors.

    Args:
        state: ARGUSState with ``universe``, ``backtest_mode``, and ``session_seed``.

    Returns:
        Partial state update with ``fundamental_signals``.
    """
    signals = fund_agent.batch_analyze(
        state["universe"],
        backtest_mode=state.get("backtest_mode", False),
        session_seed=state.get("session_seed"),
    )
    return {"fundamental_signals": signals}


def node_sentiment_analysis(state: ARGUSState) -> dict:
    """Scores news articles and social media sentiment signals.

    Args:
        state: ARGUSState with ``universe`` list of tickers.

    Returns:
        Partial state update with ``sentiment_signals``.
    """
    signals = sent_agent.batch_analyze(state["universe"])
    return {"sentiment_signals": signals}


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

    wisdom = cultural_memory.retrieve_wisdom(macro, "mixed technicals")
    warnings = cultural_memory.retrieve_warnings(macro)

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
    aggs = {}
    macro = state["macro_context"]
    if not macro:
        return {"aggregated_signals": aggs}

    for ticker in state["universe"]:
        tech = state.get("technical_signals", {}).get(ticker)
        if not tech:
            continue
        fund = state.get("fundamental_signals", {}).get(ticker)
        sent = state.get("sentiment_signals", {}).get(ticker)
        aggs[ticker] = aggregator.aggregate(tech, macro, fund, sent)
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
    vix = state["macro_context"].vix_level if state.get("macro_context") else 20.0
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

    # Joint portfolio evaluation captures inter-asset covariance and global diversification limits
    base_weight = min(0.15, 1.0 / len(universe)) if universe else 0.15
    full_portfolio = [{"ticker": t, "weight": base_weight} for t in universe]
    portfolio_result = risk_engine.evaluate(full_portfolio, history, vix, convictions=convictions)

    portfolio_vetoed = portfolio_result.verdict == RiskVerdict.VETO
    ticker_cap_map: dict[str, float] = portfolio_result.optimal_weights

    assessments = {}

    for ticker in universe:
        single = risk_engine.evaluate([{"ticker": ticker, "weight": 0.15}], history, vix)

        # Propagate portfolio-level correlation stats to individual records for uniform telemetry
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
        return {}

    mem = state.get("cultural_memory", {})
    wisdom = mem.get("wisdom", [])

    alloc = portfolio_agent.allocate(profile, signals_dict, macro, wisdom)
    if alloc is None:
        logger.error(
            "node_portfolio_allocation: PortfolioManagerAgent returned None (LLM API failure). "
            "Skipping portfolio_allocation to allow graceful exit."
        )
        return {}
    return {"portfolio_allocation": alloc}


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

    # Index allocations by ticker for O(1) lookup during decision assembly
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

            # Snapshot is stored before outcome confirmation to enable pre-settlement similarity retrieval
            cultural_memory.store_decision_snapshot(decision)

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

builder.add_conditional_edges(
    "macro_analysis",
    lambda s: [
        Send("technical_analysis", s),
        Send("fundamental_analysis", s),
        Send("sentiment_analysis", s),
        Send("retrieve_cultural_memory", s),
    ],
)

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
    conn = sqlite3.connect("argus_graph.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
except Exception:
    checkpointer = None

graph = builder.compile(checkpointer=checkpointer)
