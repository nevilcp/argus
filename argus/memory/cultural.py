"""
argus/memory/cultural.py

Long-term vector memory interface for ARGUS.

Integrates ChromaDB and sentence-transformer embeddings to store and query trade outcomes
and decision snapshots, providing qualitative historical wisdom to the Portfolio Manager.

Responsibilities:
  - Persist successful and failed trade outcomes as embedded documents
  - Retrieve semantically similar historical patterns for current macro and technical contexts
  - Expose agent-level accuracy statistics and regime diagnostics

Not responsible for:
  - Real-time signal generation (see agents/)
  - SQLite decision archiving (see data/cache.py)
  - Portfolio allocation (see agents/portfolio.py)

Dependencies:
  - chromadb
  - sentence-transformers (all-MiniLM-L6-v2, downloaded on first import)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from argus.schemas.signals import ARGUSDecision, MacroContext

logger = logging.getLogger("argus.cultural_memory")


class CulturalMemoryManager:
    """Manages persistent vector databases, semantic indexing, and similarity lookups.

    Uses a local SentenceTransformer embedding model (all-MiniLM-L6-v2) to avoid
    per-query API costs and ensure retrieval works offline. The collection uses
    cosine similarity (hnsw:space=cosine) to match multi-dimensional market context.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        import chromadb
        from chromadb.utils import embedding_functions

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Model downloaded once to ~/.cache/sentence-transformers on first invocation
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="argus_wisdom",
            embedding_function=self.ef,  # type: ignore[arg-type]  # chromadb/sentence-transformers stub mismatch
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("[Memory] Cultural Memory Manager initialized at %s", persist_dir)

    def store_trade_outcome(
        self, decision: ARGUSDecision, actual_return_pct: float, holding_days: int, exit_reason: str
    ) -> None:
        """Persists successful or failed trade outcomes to index trading history patterns.

        Excludes flat or minor return trades (|return| ≤ 1%) to prevent signal dilution
        from low-information outcomes in the similarity index.

        Args:
            decision: Completed ARGUSDecision containing all agent signals.
            actual_return_pct: Realized return as a decimal (e.g. 0.05 = +5%).
            holding_days: Number of calendar days the position was held.
            exit_reason: Free-text description of the exit trigger.
        """
        if actual_return_pct > 0.01:
            prefix = "SUCCESSFUL"
        elif actual_return_pct < -0.01:
            prefix = "FAILED"
        else:
            return

        macro_regime = decision.macro.macro_regime.value if decision.macro else "unknown"
        vix_regime = decision.macro.vix_regime if decision.macro else "unknown"
        tech_sig = decision.technical.signal.value if decision.technical else "N/A"
        tech_rsi = getattr(decision.technical, "rsi_14", 0.0) if decision.technical else 0.0
        tech_macd = (
            getattr(decision.technical, "macd_histogram", 0.0) if decision.technical else 0.0
        )
        fund_sig = decision.fundamental.signal.value if decision.fundamental else "N/A"
        fund_moat = decision.fundamental.moat_score if decision.fundamental else "N/A"
        sent_sig = decision.sentiment.signal.value if decision.sentiment else "N/A"
        sent_finbert = (
            getattr(decision.sentiment, "finbert_net_score", 0.0) if decision.sentiment else "N/A"
        )
        agg_conv = decision.aggregated.conviction if decision.aggregated else "N/A"

        primary_driver = "unknown"
        if decision.technical and decision.technical.conviction > 0.8:
            primary_driver = "technical"
        elif decision.fundamental and decision.fundamental.conviction > 0.8:
            primary_driver = "fundamental"
        elif decision.sentiment and getattr(decision.sentiment, "conviction", 0.0) > 0.8:
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
Outcome: {actual_return_pct * 100:+.1f}% in {holding_days} days. Exit: {exit_reason}"""

        doc_id = f"trade_{decision.decision_id}"

        self.collection.upsert(
            documents=[document],
            ids=[doc_id],
            metadatas=[
                {
                    "regime": macro_regime,
                    "outcome": prefix,
                    "return_pct": float(round(actual_return_pct, 4)),
                    "holding_days": int(holding_days),
                    "timestamp": decision.session_timestamp.isoformat(),
                    "primary_driver": primary_driver,
                }
            ],
        )
        logger.info("[Memory] Stored %s trade pattern. Total: %d", prefix, self.collection.count())

    def retrieve_wisdom(
        self, current_macro: MacroContext, current_technical_summary: str, n_results: int = 5
    ) -> list[str]:
        """Queries the vector collection to retrieve successful historic patterns similar to the current posture.

        Args:
            current_macro: MacroContext used to construct the similarity query.
            current_technical_summary: Free-text description of current technical posture.
            n_results: Maximum number of similar patterns to return.

        Returns:
            List of matching document strings from the SUCCESSFUL outcome filter.
        """
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

    def retrieve_warnings(self, current_macro: MacroContext, n_results: int = 3) -> list[str]:
        """Retrieves failed historic patterns within the current macro regime to serve as warnings.

        Args:
            current_macro: MacroContext used to scope the regime filter.
            n_results: Maximum number of warning documents to return.

        Returns:
            List of matching document strings from the FAILED outcome filter for the current regime.
        """
        if self.collection.count() == 0:
            return []

        try:
            query = f"Failed trades in {current_macro.macro_regime.value} regime"
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where={
                    "$and": [{"outcome": "FAILED"}, {"regime": current_macro.macro_regime.value}]
                },
            )
            return results["documents"][0] if results and results["documents"] else []
        except Exception as e:
            logger.warning("[Memory] Failed to retrieve warnings: %s", e)
            return []

    def get_agent_accuracy(self, agent_name: str, regime: Optional[str] = None) -> float:
        """Computes statistical win rates for trades driven primarily by a specific specialist agent.

        Args:
            agent_name: Agent identifier string (e.g. 'technical', 'fundamental', 'sentiment').
            regime: Optional regime filter (e.g. 'EXPANSION'); queries all regimes if None.

        Returns:
            Win rate as a float in [0, 1]. Returns 0.5 (neutral prior) when no data exists.
        """
        if self.collection.count() == 0:
            return 0.5

        try:
            where_clause: dict[str, Any] = {"primary_driver": agent_name.lower()}
            if regime:
                where_clause = {
                    "$and": [{"primary_driver": agent_name.lower()}, {"regime": regime}]
                }

            results = self.collection.get(where=where_clause)  # type: ignore[arg-type]
            metadatas = results.get("metadatas", [])
            if not metadatas:
                return 0.5

            wins = sum(1 for m in metadatas if m.get("outcome") == "SUCCESSFUL")
            return float(wins) / len(metadatas)
        except Exception as e:
            logger.warning("[Memory] Failed to compute agent accuracy: %s", e)
            return 0.5

    def summary_stats(self) -> dict:
        """Compiles aggregate performance statistics and regime diagnostics from the memory database.

        Returns:
            Dict with keys: total_stored, successful_count, failed_count,
            avg_return_pct, best_regime_for_wins.
        """
        count = self.collection.count()
        if count == 0:
            return {
                "total_stored": 0,
                "successful_count": 0,
                "failed_count": 0,
                "avg_return_pct": 0.0,
                "best_regime_for_wins": "N/A",
            }

        try:
            results = self.collection.get()
            metadatas = results.get("metadatas") or []
            wins = 0
            fails = 0
            total_ret = 0.0
            regime_wins: dict[str, int] = {}

            for m in metadatas:
                outcome = m.get("outcome")
                ret = float(m.get("return_pct", 0.0) or 0.0)  # type: ignore[arg-type]  # chromadb metadata values are typed as a broad scalar union
                regime = str(m.get("regime", "unknown"))

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
                "best_regime_for_wins": best_regime,
            }
        except Exception as e:
            logger.warning("[Memory] Failed to compute summary stats: %s", e)
            return {"total_stored": count}

    def store_decision_snapshot(self, decision: "ARGUSDecision") -> None:
        """Persists a real-time decision profile before outcomes are finalized.

        Populates the vector store with decision-making context, enabling retrieval
        of historical similarities before trade settlement data is processed.

        Args:
            decision: In-flight ARGUSDecision captured at signal generation time.
        """
        try:
            macro_regime = decision.macro.macro_regime.value if decision.macro else "unknown"
            vix_regime = decision.macro.vix_regime.value if decision.macro else "unknown"
            agg_signal = decision.aggregated.signal.value if decision.aggregated else "N/A"
            agg_conv = decision.aggregated.conviction if decision.aggregated else 0.0
            fund_sig = decision.fundamental.signal.value if decision.fundamental else "N/A"
            tech_sig = decision.technical.signal.value if decision.technical else "N/A"
            sent_sig = decision.sentiment.signal.value if decision.sentiment else "N/A"
            alloc_pct = decision.allocation.allocation_pct if decision.allocation else 0.0
            thesis = decision.allocation.thesis if decision.allocation else "No position taken"

            document = (
                f"DECISION SNAPSHOT [{decision.ticker}]:\n"
                f"Macro regime: {macro_regime} | VIX: {vix_regime}\n"
                f"Technical: {tech_sig} | Fundamental: {fund_sig} | Sentiment: {sent_sig}\n"
                f"Aggregated: {agg_signal} (conviction={agg_conv:.2f})\n"
                f"Position taken: {decision.allocation is not None} @ {alloc_pct:.1%}\n"
                f"Thesis: {thesis}"
            )

            self.collection.upsert(
                documents=[document],
                ids=[f"snapshot_{decision.decision_id}"],
                metadatas=[
                    {
                        "regime": macro_regime,
                        "outcome": "PENDING",
                        "ticker": decision.ticker,
                        "agg_signal": agg_signal,
                        "allocated": str(decision.allocation is not None),
                        "timestamp": decision.session_timestamp.isoformat(),
                        "decision_id": decision.decision_id,
                    }
                ],
            )
            logger.info(
                "[Memory] Snapshotted decision for %s (ID=%s)",
                decision.ticker,
                decision.decision_id,
            )
        except Exception as e:
            logger.warning("[Memory] Failed to snapshot decision for %s: %s", decision.ticker, e)


_cultural_memory: Optional[CulturalMemoryManager] = None


def get_cultural_memory(persist_dir: str = "./chroma_db") -> CulturalMemoryManager:
    """Returns the process-wide CulturalMemoryManager, constructing it on first call.

    Lazy on purpose: construction pulls in sentence-transformers (and its
    torch dependency) to build the embedding function, and both are an
    optional `[models]` extra (see pyproject.toml, ADR 0007) — importing
    this module must not require them, only actually using cultural memory
    does.
    """
    global _cultural_memory
    if _cultural_memory is None:
        _cultural_memory = CulturalMemoryManager(persist_dir)
    return _cultural_memory
