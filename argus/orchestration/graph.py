"""Dependency wiring and parallel execution of the decision graph."""

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
    """Construct and compile the ARGUS decision graph."""
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

    # Run macro beside independent specialists so a slow or failed macro fetch cannot gate their analysis.
    builder.add_edge("fetch_price_history", "macro_analysis")
    builder.add_edge("fetch_price_history", "technical_analysis")
    builder.add_edge("fetch_price_history", "fundamental_analysis")
    builder.add_edge("fetch_price_history", "sentiment_analysis")

    builder.add_edge(
        ["macro_analysis", "technical_analysis", "fundamental_analysis", "sentiment_analysis"],
        "signal_aggregation",
    )

    # Query memory after aggregation supplies its technical posture; only allocation needs the result.
    builder.add_edge("signal_aggregation", "retrieve_cultural_memory")
    builder.add_edge("signal_aggregation", "risk_evaluation")
    builder.add_edge(["retrieve_cultural_memory", "risk_evaluation"], "portfolio_allocation")
    builder.add_edge("portfolio_allocation", "log_decisions")
    builder.add_edge("log_decisions", END)

    return CheckpointedGraph(builder, checkpoint_db_path)
