"""
argus/orchestration/aggregator.py
=================================
Aggregates Technical, Fundamental, and Sentiment signals.
"""

import logging
from collections import defaultdict
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from argus.config import settings
from argus.orchestration.governor import governor
from argus.schemas.signals import (
    AggregatedSignal, FundamentalSignal, MacroContext, SentimentSignal, Signal, TechnicalSignal
)

logger = logging.getLogger("argus.aggregator")

def _resolve_conflict(fundamental: FundamentalSignal, sentiment: SentimentSignal) -> str:
    """Uses Groq 8B for lightweight conflict resolution between agents."""
    governor.wait_if_needed("llama-3.1-8b-instant", 300)
    
    prompt = f"""
Fundamental says {fundamental.signal.value}, Sentiment says {sentiment.signal.value}. Which is more reliable 
given that moat_score={fundamental.moat_score} and finbert_score={sentiment.finbert_net_score:.2f}?
Respond with one word: BULLISH, BEARISH, or NEUTRAL.
"""
    api_key = settings.groq_api_key or "DUMMY_KEY_FOR_TESTING"
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=10, groq_api_key=api_key)
    
    try:
        response = llm.invoke([HumanMessage(content=prompt.strip())])
        raw = response.content.strip().upper()
        if "BULLISH" in raw: return "BULLISH"
        if "BEARISH" in raw: return "BEARISH"
        return "NEUTRAL"
    except Exception as e:
        logger.warning(f"[Aggregator] Conflict resolution failed: {e}")
        return "NEUTRAL"


class HybridSignalAggregator:
    def aggregate(
        self,
        technical: TechnicalSignal,
        macro: MacroContext,
        fundamental: Optional[FundamentalSignal],
        sentiment: Optional[SentimentSignal],
    ) -> AggregatedSignal:
        
        multipliers = macro.agent_multipliers
        weighted_votes = defaultdict(float)
        
        signal_inputs = [
            ("technical", technical.signal, technical.conviction * multipliers.get("technical", 1.0))
        ]
        
        if fundamental:
            signal_inputs.append(
                ("fundamental", fundamental.signal, fundamental.conviction * multipliers.get("fundamental", 1.0))
            )
        if sentiment:
            signal_inputs.append(
                ("sentiment", sentiment.signal, sentiment.conviction * multipliers.get("sentiment", 1.0))
            )
            
        for name, sig, wt in signal_inputs:
            weighted_votes[sig.value] += wt
            
        total = sum(weighted_votes.values())
        best_sig = max(weighted_votes, key=weighted_votes.get) if total > 0 else Signal.NEUTRAL.value
        best_score = weighted_votes[best_sig] / total if total > 0 else 0.0
        
        debate_triggered = False
        
        if (fundamental and sentiment and 
            fundamental.signal != Signal.NEUTRAL and 
            sentiment.signal != Signal.NEUTRAL and 
            fundamental.signal != sentiment.signal and 
            best_score < 0.52):
            
            debate_triggered = True
            resolution = _resolve_conflict(fundamental, sentiment)
            weighted_votes[resolution] += 0.25
            
            total = sum(weighted_votes.values())
            best_sig = max(weighted_votes, key=weighted_votes.get)
            best_score = weighted_votes[best_sig] / total
            
        return AggregatedSignal(
            ticker=technical.ticker,
            signal=Signal(best_sig),
            conviction=round(min(best_score, 0.92), 3),
            weighted_votes=dict(weighted_votes),
            debate_triggered=debate_triggered,
        )
