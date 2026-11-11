"""
argus/orchestration/graph.py

LangGraph execution orchestrator for the ARGUS multi-agent decision network.

Coordinates asynchronous fetch pipelines, parallel specialized analyst agents,
centralized risk assessment, dynamic signal aggregation, and long-term cultural
memory ingestion into a stateful execution graph.

Responsibilities:
  - Construct the agents and bundle them into a GraphDeps
  - Build and compile the LangGraph StateGraph DAG via build_graph()

Not responsible for:
  - The node functions themselves (see orchestration/nodes.py)
  - Agent implementation logic (see individual agent modules)
  - Schema validation (see schemas/signals.py)
  - API serving (see api/main.py)

Dependencies:
  - langgraph
  - argus.agents.* (all six specialist agents)
  - argus.orchestration.checkpoint (the SQLite checkpointer, argus_graph.db)

Agent wiring is a build_graph() parameter, not a module-level singleton:
every agent that touches the network or an LLM accepts a
MarketDataProvider/LLMClient, defaulting to the real implementation, so a
test can build a fixture-backed graph without patching this module's
internals. Callers — api/lifespan.py, scripts/collect_session.py — call
build_graph() themselves rather than importing a shared instance.
"""

from collections.abc import Callable
from functools import partial

from langgraph.graph import END, StateGraph

from argus.agents.fundamental import FundamentalAgent
from argus.agents.macro import MacroStatisticalAgent
from argus.agents.portfolio import PortfolioManagerAgent
from argus.agents.risk import RiskStatisticalAgent
from argus.agents.sentiment import SentimentAgent
from argus.agents.technical import TechnicalStatisticalAgent
from argus.config import settings
from argus.memory.cultural import CulturalMemoryManager, get_cultural_memory
from argus.orchestration.aggregator import HybridSignalAggregator
from argus.orchestration.checkpoint import CheckpointedGraph
from argus.orchestration.nodes import (
    GraphDeps,
    node_fetch_price_history,
    node_fundamental_analysis,
    node_log_decisions,
    node_macro_analysis,
    node_portfolio_allocation,
    node_retrieve_cultural_memory,
    node_risk_evaluation,
    node_sentiment_analysis,
    node_signal_aggregation,
    node_technical_analysis,
)
from argus.orchestration.state import ARGUSState
from argus.ports import LLMClient, MarketDataProvider
from argus.seams import LiveMarketDataProvider


def build_graph(
    market_data: MarketDataProvider | None = None,
    fundamental_llm: LLMClient | None = None,
    sentiment_llm: LLMClient | None = None,
    portfolio_llm: LLMClient | None = None,
    checkpoint_db_path: str | None = None,
    cultural_memory_provider: Callable[[], CulturalMemoryManager] | None = None,
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
        checkpoint_db_path: SQLite file each invocation checkpoints ARGUSState
            to. Exposed (rather than hardcoded) so tests — and
            argus/orchestration/reconcile.py's own tests — can point it
            at a temp file instead of the real store production callers use.
            None (default) resolves to ``<ARGUS_DATA_DIR>/argus_graph.db`` at
            call time, so tests/conftest.py's per-test ARGUS_DATA_DIR override
            also isolates a call site that omits this argument.
        cultural_memory_provider: Returns the cultural memory store each time
            a node needs it. Defaults to the process-wide
            ``get_cultural_memory``; a test or replay passes its own, so no
            patching of this module is needed.

    Returns:
        A CheckpointedGraph, ready for .invoke(). Recompiles with a fresh
        checkpointer connection on every call rather than baking one
        connection into a single compiled graph — see checkpoint.CheckpointedGraph.
    """
    if checkpoint_db_path is None:
        checkpoint_db_path = str(settings.data_path("argus_graph.db"))

    market_data = market_data or LiveMarketDataProvider()

    deps = GraphDeps(
        market_data=market_data,
        macro_agent=MacroStatisticalAgent(market_data=market_data),
        tech_agent=TechnicalStatisticalAgent(),
        fund_agent=FundamentalAgent(llm_client=fundamental_llm, market_data=market_data),
        sent_agent=SentimentAgent(llm_client=sentiment_llm, market_data=market_data),
        risk_agent=RiskStatisticalAgent(market_data=market_data),
        aggregator=HybridSignalAggregator(),
        portfolio_agent=PortfolioManagerAgent(llm_client=portfolio_llm),
        cultural_memory_provider=cultural_memory_provider or get_cultural_memory,
    )

    builder = StateGraph(ARGUSState)

    builder.add_node("fetch_price_history", partial(node_fetch_price_history, deps=deps))
    builder.add_node("macro_analysis", partial(node_macro_analysis, deps=deps))
    builder.add_node("technical_analysis", partial(node_technical_analysis, deps=deps))
    builder.add_node("fundamental_analysis", partial(node_fundamental_analysis, deps=deps))
    builder.add_node("sentiment_analysis", partial(node_sentiment_analysis, deps=deps))
    builder.add_node("retrieve_cultural_memory", partial(node_retrieve_cultural_memory, deps=deps))
    builder.add_node("signal_aggregation", partial(node_signal_aggregation, deps=deps))
    builder.add_node("risk_evaluation", partial(node_risk_evaluation, deps=deps))
    builder.add_node("portfolio_allocation", partial(node_portfolio_allocation, deps=deps))
    builder.add_node("log_decisions", partial(node_log_decisions, deps=deps))

    builder.set_entry_point("fetch_price_history")

    # macro_analysis runs in parallel with the three specialists, not ahead of
    # them: none of them consumes macro_context (see state.py's docstring), so
    # gating them behind it needlessly serializes fundamental's ~20 sequential
    # Groq round-trips behind the HMM window and drops a whole session's output
    # whenever only the FRED bundle is unavailable.
    builder.add_edge("fetch_price_history", "macro_analysis")
    builder.add_edge("fetch_price_history", "technical_analysis")
    builder.add_edge("fetch_price_history", "fundamental_analysis")
    builder.add_edge("fetch_price_history", "sentiment_analysis")

    builder.add_edge(
        ["macro_analysis", "technical_analysis", "fundamental_analysis", "sentiment_analysis"],
        "signal_aggregation",
    )

    # retrieve_cultural_memory runs after signal_aggregation (not beside it) so it
    # can summarize this session's own aggregated_signals into its similarity
    # query; wisdom/warnings are consumed only by portfolio_allocation, so this
    # is safe to run in parallel with risk_evaluation.
    builder.add_edge("signal_aggregation", "retrieve_cultural_memory")
    builder.add_edge("signal_aggregation", "risk_evaluation")
    builder.add_edge(["retrieve_cultural_memory", "risk_evaluation"], "portfolio_allocation")
    builder.add_edge("portfolio_allocation", "log_decisions")
    builder.add_edge("log_decisions", END)

    return CheckpointedGraph(builder, checkpoint_db_path)
