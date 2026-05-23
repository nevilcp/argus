"""
argus/memory/cultural.py
========================
Long-Term Cultural Memory for ARGUS v2.
Stores successful trade patterns as vector embeddings in ChromaDB to provide
the Portfolio Manager with historical wisdom.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from argus.schemas.signals import ARGUSDecision, MacroContext

logger = logging.getLogger("argus.cultural_memory")

class CulturalMemoryManager:
    """Manages the vector database for long-term pattern recognition."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        import chromadb
        from chromadb.utils import embedding_functions

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Use local sentence-transformer for embeddings (free, no API)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # 22MB, downloads once
        )

        self.collection = self.client.get_or_create_collection(
            name="argus_wisdom",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("[Memory] Cultural Memory Manager initialized at %s", persist_dir)

    def store_trade_outcome(
        self,
        decision: ARGUSDecision,
        actual_return_pct: float,
        holding_days: int,
        exit_reason: str
    ) -> None:
        """Store meaningful trades (>1% win or >1% loss) into ChromaDB."""
        if actual_return_pct > 0.01:
            prefix = "SUCCESSFUL"
        elif actual_return_pct < -0.01:
            prefix = "FAILED"
        else:
            return  # Don't clutter memory with flat trades

        macro_regime = decision.macro.macro_regime.value if decision.macro else "unknown"
        vix_regime = decision.macro.vix_regime if decision.macro else "unknown"
        tech_sig = decision.technical.signal.value if decision.technical else "N/A"
        tech_rsi = getattr(decision.technical, "rsi_14", 0.0) if decision.technical else 0.0
        tech_macd = getattr(decision.technical, "macd_histogram", 0.0) if decision.technical else 0.0
        fund_sig = decision.fundamental.signal.value if decision.fundamental else "N/A"
        fund_moat = decision.fundamental.moat_score if decision.fundamental else "N/A"
        sent_sig = decision.sentiment.signal.value if decision.sentiment else "N/A"
        sent_finbert = getattr(decision.sentiment, "finbert_net_score", 0.0) if decision.sentiment else "N/A"
        agg_conv = decision.aggregated.conviction if decision.aggregated else "N/A"
        
        # Primary driver estimation
        primary_driver = "unknown"
        if decision.technical and decision.technical.conviction > 0.8:
            primary_driver = "technical"
        elif decision.fundamental and decision.fundamental.conviction > 0.8:
            primary_driver = "fundamental"
        elif decision.sentiment and getattr(decision.sentiment, 'conviction', 0.0) > 0.8:
            primary_driver = "sentiment"

        document = f"""{prefix} PATTERN:
Macro regime: {macro_regime}
VIX regime: {vix_regime}
Technical signal: {tech_sig} 
  (RSI={tech_rsi:.0f}, MACD hist={tech_macd:.3f})
Fundamental signal: {fund_sig}
  (Moat={fund_moat})
Sentiment signal: {sent_sig}
  (FinBERT={sent_finbert if isinstance(sent_finbert, str) else f"{sent_finbert:.2f}"})
Aggregated conviction: {agg_conv}
Outcome: {actual_return_pct*100:+.1f}% in {holding_days} days. Exit: {exit_reason}"""

        doc_id = f"trade_{decision.decision_id}"

        self.collection.upsert(
            documents=[document],
            ids=[doc_id],
            metadatas=[{
                "regime": macro_regime,
                "outcome": prefix,
                "return_pct": float(round(actual_return_pct, 4)),
                "holding_days": int(holding_days),
                "timestamp": decision.session_timestamp.isoformat(),
                "primary_driver": primary_driver,
            }]
        )
        logger.info("[Memory] Stored %s trade pattern. Total: %d", prefix, self.collection.count())

    def retrieve_wisdom(
        self,
        current_macro: MacroContext,
        current_technical_summary: str,
        n_results: int = 5
    ) -> list[str]:
        """Retrieve historical SUCCESSFUL patterns similar to current state."""
        if self.collection.count() == 0:
            return []

        query = f"Macro regime {current_macro.macro_regime.value}, VIX {current_macro.vix_regime}, {current_technical_summary}"

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where={"outcome": "SUCCESSFUL"},
            )
            return results["documents"][0] if results and results["documents"] else []
        except Exception as e:
            logger.warning("[Memory] Failed to retrieve wisdom: %s", e)
            return []

    def retrieve_warnings(
        self,
        current_macro: MacroContext,
        n_results: int = 3
    ) -> list[str]:
        """Retrieve FAILED patterns in the current regime as warnings."""
        if self.collection.count() == 0:
            return []
            
        try:
            query = f"Failed trades in {current_macro.macro_regime.value} regime"
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where={
                    "$and": [
                        {"outcome": "FAILED"},
                        {"regime": current_macro.macro_regime.value}
                    ]
                }
            )
            return results["documents"][0] if results and results["documents"] else []
        except Exception as e:
            logger.warning("[Memory] Failed to retrieve warnings: %s", e)
            return []

    def get_agent_accuracy(
        self,
        agent_name: str,
        regime: Optional[str] = None
    ) -> float:
        """
        Approximate win rate for trades where a specific agent was the primary driver.
        agent_name should be 'technical', 'fundamental', or 'sentiment'.
        """
        if self.collection.count() == 0:
            return 0.5
            
        try:
            where_clause = {"primary_driver": agent_name.lower()}
            if regime:
                where_clause = {
                    "$and": [
                        {"primary_driver": agent_name.lower()},
                        {"regime": regime}
                    ]
                }
                
            results = self.collection.get(where=where_clause)
            metadatas = results.get("metadatas", [])
            if not metadatas:
                return 0.5
                
            wins = sum(1 for m in metadatas if m.get("outcome") == "SUCCESSFUL")
            return float(wins) / len(metadatas)
        except Exception as e:
            logger.warning("[Memory] Failed to compute agent accuracy: %s", e)
            return 0.5

    def summary_stats(self) -> dict:
        """Returns aggregate statistics of the memory vault."""
        count = self.collection.count()
        if count == 0:
            return {
                "total_stored": 0,
                "successful_count": 0,
                "failed_count": 0,
                "avg_return_pct": 0.0,
                "best_regime_for_wins": "N/A"
            }
            
        try:
            results = self.collection.get()
            metadatas = results.get("metadatas", [])
            wins = 0
            fails = 0
            total_ret = 0.0
            regime_wins: dict[str, int] = {}
            
            for m in metadatas:
                outcome = m.get("outcome")
                ret = m.get("return_pct", 0.0)
                regime = m.get("regime", "unknown")
                
                total_ret += ret
                if outcome == "SUCCESSFUL":
                    wins += 1
                    regime_wins[regime] = regime_wins.get(regime, 0) + 1
                elif outcome == "FAILED":
                    fails += 1
                    
            best_regime = "N/A"
            if regime_wins:
                best_regime = max(regime_wins.items(), key=lambda x: x[1])[0]
                
            return {
                "total_stored": count,
                "successful_count": wins,
                "failed_count": fails,
                "avg_return_pct": total_ret / count if count > 0 else 0.0,
                "best_regime_for_wins": best_regime
            }
        except Exception as e:
            logger.warning("[Memory] Failed to compute summary stats: %s", e)
            return {"total_stored": count}


# Module-level singleton
cultural_memory = CulturalMemoryManager()
