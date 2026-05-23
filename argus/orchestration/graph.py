"""
argus/orchestration/graph.py
============================
LangGraph execution orchestrator.
"""

import sqlite3
from typing import Any

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

# Instantiate singletons/agents
macro_agent = MacroStatisticalAgent()
tech_agent = TechnicalStatisticalAgent()
fund_agent = FundamentalAgent()
sent_agent = SentimentAgent()
risk_engine = RiskStatisticalEngine()
aggregator = HybridSignalAggregator()
portfolio_agent = PortfolioManagerAgent()

def node_fetch_price_history(state: ARGUSState) -> dict:
    history_dfs = fetch_multiple_daily(state["universe"], period="1y")
    history_serializable = {}
    for ticker, df in history_dfs.items():
        history_serializable[ticker] = {
            "dates": df.index.astype(str).tolist(),
            "prices": df["close"].tolist()
        }
    return {"price_history": history_serializable}

def node_macro_analysis(state: ARGUSState) -> dict:
    macro_ctx = macro_agent.analyze()
    return {"macro_context": macro_ctx}

def node_technical_analysis(state: ARGUSState) -> dict:
    session_states = state.get("session_states", {})
    signals = tech_agent.batch_analyze(session_states)
    return {"technical_signals": signals}

def node_fundamental_analysis(state: ARGUSState) -> dict:
    signals = fund_agent.batch_analyze(
        state["universe"], 
        backtest_mode=state.get("backtest_mode", False),
        session_seed=state.get("session_seed")
    )
    return {"fundamental_signals": signals}

def node_sentiment_analysis(state: ARGUSState) -> dict:
    signals = sent_agent.batch_analyze(state["universe"])
    return {"sentiment_signals": signals}

def node_risk_evaluation(state: ARGUSState) -> dict:
    import pandas as pd
    assessments = {}
    history_raw = state.get("price_history", {})
    history = {}
    for ticker, data in history_raw.items():
        history[ticker] = pd.Series(data["prices"], index=pd.to_datetime(data["dates"]))
        
    vix = state["macro_context"].vix_level if state.get("macro_context") else 20.0
    
    for ticker in state["universe"]:
        proposed = [{"ticker": ticker, "weight": 0.15}]
        assessments[ticker] = risk_engine.evaluate(proposed, history, vix)
    return {"risk_assessments": assessments}

def node_retrieve_cultural_memory(state: ARGUSState) -> dict:
    macro = state.get("macro_context")
    if not macro:
        return {"cultural_wisdom": [], "cultural_warnings": []}
    
    wisdom = cultural_memory.retrieve_wisdom(macro, "mixed technicals")
    warnings = cultural_memory.retrieve_warnings(macro)
    return {
        "cultural_wisdom": [wisdom] if wisdom else [], 
        "cultural_warnings": [warnings] if warnings else []
    }

def node_signal_aggregation(state: ARGUSState) -> dict:
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

def node_portfolio_allocation(state: ARGUSState) -> dict:
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
        "total_wealth": state.get("total_wealth", 100000),
        "invest_pct": state.get("invest_pct", 1.0),
        "risk_tolerance": state.get("risk_tolerance", "MODERATE")
    }
    
    macro = state.get("macro_context")
    if not macro:
        return {}
        
    wisdom = state.get("cultural_wisdom", [])
    
    alloc = portfolio_agent.allocate(profile, signals_dict, macro, wisdom)
    return {"portfolio_allocation": alloc}

def node_log_decisions(state: ARGUSState) -> dict:
    return {"decisions": []}


builder = StateGraph(ARGUSState)

builder.add_node("fetch_price_history", node_fetch_price_history)
builder.add_node("macro_analysis", node_macro_analysis)
builder.add_node("technical_analysis", node_technical_analysis)
builder.add_node("fundamental_analysis", node_fundamental_analysis)
builder.add_node("sentiment_analysis", node_sentiment_analysis)
builder.add_node("risk_evaluation", node_risk_evaluation)
builder.add_node("retrieve_cultural_memory", node_retrieve_cultural_memory)
builder.add_node("signal_aggregation", node_signal_aggregation)
builder.add_node("portfolio_allocation", node_portfolio_allocation)
builder.add_node("log_decisions", node_log_decisions)

builder.set_entry_point("fetch_price_history")
builder.add_edge("fetch_price_history", "macro_analysis")

builder.add_conditional_edges("macro_analysis", 
    lambda s: [
        Send("technical_analysis", s), 
        Send("fundamental_analysis", s),
        Send("sentiment_analysis", s), 
        Send("retrieve_cultural_memory", s)
    ]
)

builder.add_edge(["technical_analysis", "fundamental_analysis", 
                  "sentiment_analysis", "retrieve_cultural_memory"], "risk_evaluation")
builder.add_edge("risk_evaluation", "signal_aggregation")
builder.add_edge("signal_aggregation", "portfolio_allocation")
builder.add_edge("portfolio_allocation", "log_decisions")
builder.add_edge("log_decisions", END)

try:
    conn = sqlite3.connect("argus_graph.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
except Exception:
    checkpointer = None

graph = builder.compile(checkpointer=checkpointer)
