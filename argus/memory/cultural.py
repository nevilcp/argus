"""
argus/memory/cultural.py

Long-term vector memory interface for ARGUS.

Integrates ChromaDB and sentence-transformer embeddings to store and query trade outcomes
and decision snapshots, providing qualitative historical wisdom to the Portfolio Manager.

Responsibilities:
  - Persist successful and failed trade outcomes as embedded documents
  - Retrieve semantically similar historical patterns for current macro and technical contexts
  - Expose agent-level accuracy statistics and regime diagnostics
  - Expire PENDING decision snapshots that never settled (see
    expire_pending_snapshots), so the collection stays bounded

Not responsible for:
  - Real-time signal generation (see agents/)
  - Credit assignment / primary_driver computation (see
    orchestration/reconciliation.py)
  - Portfolio allocation (see agents/portfolio.py)

Dependencies:
  - chromadb
  - sentence-transformers (all-MiniLM-L6-v2, downloaded on first import)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from threading import Lock
from typing import Any, Optional

from argus.config import settings
from argus.params import MEMORY, RECONCILIATION
from argus.schemas.signals import ARGUSDecision, MacroContext

logger = logging.getLogger("argus.cultural_memory")


def _where(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Combines metadata filter conditions into a single ChromaDB ``where`` clause."""
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _signal_or_na(agent_signal: Any) -> str:
    """Returns an agent signal block's enum value, or "N/A" when the agent didn't run."""
    return agent_signal.signal.value if agent_signal else "N/A"


class CulturalMemoryManager:
    """Manages persistent vector databases, semantic indexing, and similarity lookups.

    Uses a local SentenceTransformer embedding model (all-MiniLM-L6-v2) to avoid
    per-query API costs and ensure retrieval works offline. The collection uses
    cosine similarity (hnsw:space=cosine) to match multi-dimensional market context.

    Attributes:
        client: ChromaDB PersistentClient backing the store.
        ef: SentenceTransformer embedding function used for indexing and queries.
        collection: The ``argus_wisdom`` ChromaDB collection.
        persist_dir: Absolute filesystem directory the collection persists to.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        """Opens (or creates) the persistent ChromaDB collection and embedding function.

        Args:
            persist_dir: Filesystem directory backing the ChromaDB PersistentClient.
        """
        # Deferred to construction time: embedding_functions pulls in
        # sentence-transformers, an optional `models` extra (see pyproject.toml),
        # so importing this module must not require it
        import chromadb
        from chromadb.utils import embedding_functions

        # Resolved to an absolute path so a caller launching from an unexpected
        # working directory gets a loud warning instead of a silent, empty store
        persist_dir = os.path.abspath(persist_dir)
        is_fresh = not os.path.isdir(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        if is_fresh:
            logger.warning(
                "[Memory] %s did not exist; created a fresh, empty cultural memory "
                "store there. If this is unexpected, check the process's working directory.",
                persist_dir,
            )

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )

        # Model downloaded once to ~/.cache/sentence-transformers on first invocation
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="argus_wisdom",
            embedding_function=self.ef,  # type: ignore[arg-type]  # chromadb/sentence-transformers stub mismatch
            metadata={"hnsw:space": "cosine"},
        )
        self.persist_dir = persist_dir
        logger.info("[Memory] Cultural Memory Manager initialized at %s", persist_dir)

    def already_reconciled(self, decision_ids: list[str]) -> set[str]:
        """Returns the subset of decision_ids that already have a stored trade outcome.

        A single batched ``get`` rather than one existence check per decision,
        so a caller reconciling a growing history of decisions (see
        orchestration/reconciliation.py) can filter its work list down to
        unreconciled decisions before touching market data at all.

        Args:
            decision_ids: Candidate ARGUSDecision.decision_id values.

        Returns:
            The subset already stored under a ``trade_{decision_id}`` row.
        """
        if not decision_ids:
            return set()
        ids = [f"trade_{decision_id}" for decision_id in decision_ids]
        results = self.collection.get(ids=ids)
        return {stored_id.removeprefix("trade_") for stored_id in results.get("ids") or []}

    def store_trade_outcome(
        self,
        decision: ARGUSDecision,
        actual_return_pct: float,
        holding_days: int,
        exit_reason: str,
        primary_driver: str,
    ) -> None:
        """Persists trade outcomes to index trading history patterns.

        Flat or minor return trades (|return| ≤ 1%) are stored as "FLAT" rather than
        excluded: get_agent_accuracy's win rate is P(win | stored), so dropping flat
        trades from storage would have conditioned that rate on |return| > 1% while
        still reporting it as a plain win rate. FLAT documents don't match either
        retrieve_wisdom's or retrieve_warnings' outcome filter, so they only affect
        the accuracy denominator.

        Args:
            decision: Completed ARGUSDecision containing all agent signals.
            actual_return_pct: Realized return as a decimal (e.g. 0.05 = +5%).
            holding_days: Number of calendar days the position was held.
            exit_reason: Free-text description of the exit trigger.
            primary_driver: Agent name ('technical'/'fundamental'/'sentiment'/'unknown')
                credited via leave-one-out ablation
                (see argus/orchestration/reconciliation.py:credit_primary_driver).
                Signal arbitration is this method's caller's job, not this
                module's — see its docstring's "Not responsible for" list.
        """
        if actual_return_pct > RECONCILIATION.min_abs_return_for_storage:
            prefix = "SUCCESSFUL"
        elif actual_return_pct < -RECONCILIATION.min_abs_return_for_storage:
            prefix = "FAILED"
        else:
            prefix = "FLAT"

        technical = decision.technical
        macro_regime = decision.macro.macro_regime.value if decision.macro else "unknown"
        vix_regime = decision.macro.vix_regime.value if decision.macro else "unknown"
        tech_sig = _signal_or_na(technical)
        tech_rsi = getattr(technical, "rsi_14", 0.0) if technical else 0.0
        tech_macd = getattr(technical, "macd_histogram", 0.0) if technical else 0.0
        fund_sig = _signal_or_na(decision.fundamental)
        fund_moat = decision.fundamental.moat_score if decision.fundamental else "N/A"
        sent_sig = _signal_or_na(decision.sentiment)
        sent_finbert = decision.sentiment.finbert_net_score if decision.sentiment else None
        agg_conv = decision.aggregated.conviction if decision.aggregated else "N/A"

        document = f"""{prefix} PATTERN:
Macro regime: {macro_regime}
VIX regime: {vix_regime}
Technical signal: {tech_sig}
  (RSI={tech_rsi:.0f}, MACD hist={tech_macd:.3f})
Fundamental signal: {fund_sig}
  (Moat={fund_moat})
Sentiment signal: {sent_sig}
  (FinBERT={"N/A" if sent_finbert is None else f"{sent_finbert:.2f}"})
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

        # The PENDING snapshot has now settled into a trade_{id} row; leaving it
        # in place would double-count the decision (dragging down avg_return_pct
        # via its zero return_pct) in summary_stats
        snapshot_id = f"snapshot_{decision.decision_id}"
        try:
            self.collection.delete(ids=[snapshot_id])
        except Exception as e:
            logger.warning("[Memory] Failed to delete snapshot %s: %s", snapshot_id, e)

        logger.info("[Memory] Stored %s trade pattern. Total: %d", prefix, self.collection.count())

    def _retrieve_by_outcome(
        self,
        query: str,
        outcome: str,
        regime: str,
        n_results: int,
        as_of: Optional[datetime],
        label: str,
    ) -> list[str]:
        """Runs one outcome- and regime-filtered similarity query, returning [] on failure.

        Shared by retrieve_wisdom and retrieve_warnings so the two filter identically.

        Args:
            query: Free-text similarity query.
            outcome: Stored ``outcome`` metadata value to filter on ("SUCCESSFUL"/"FAILED").
            regime: Macro regime to scope the lookup to.
            n_results: Maximum number of documents to return.
            as_of: When given, excludes documents stored after this timestamp.
            label: Noun used in the failure log line.

        Returns:
            Matching document strings, or [] if the store is empty or the query failed.
        """
        if self.collection.count() == 0:
            return []

        conditions: list[dict[str, Any]] = [{"outcome": outcome}, {"regime": regime}]
        if as_of is not None:
            conditions.append({"timestamp": {"$lte": as_of.isoformat()}})

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=_where(conditions),
            )
            return results["documents"][0] if results and results["documents"] else []
        except Exception as e:
            logger.warning("[Memory] Failed to retrieve %s: %s", label, e)
            return []

    def retrieve_wisdom(
        self,
        current_macro: MacroContext,
        current_technical_summary: str,
        n_results: int = 5,
        as_of: Optional[datetime] = None,
    ) -> list[str]:
        """Queries the vector collection to retrieve successful historic patterns similar to the current posture.

        Filters on the current regime, symmetrically with retrieve_warnings — both
        land in the same portfolio prompt, so successes shouldn't be drawn from every
        regime while failures are confined to this one. Rows stored with
        regime="unknown" (from scripts/remediate_macro_regime_tags.py) never match,
        since current_macro.macro_regime is always one of Regime's three live values —
        remediated rows are deliberately not credited to today's regime.

        Args:
            current_macro: MacroContext used to construct the similarity query and
                to scope the regime filter.
            current_technical_summary: Free-text description of current technical posture.
            n_results: Maximum number of similar patterns to return.
            as_of: When given, excludes documents stored after this timestamp. Must
                be naive, matching the naive datetime.now() this module stores
                (see pyproject.toml's DTZ tracking comment). None (default) applies
                no filtering.

        Returns:
            List of matching document strings from the SUCCESSFUL outcome filter.
        """
        regime = current_macro.macro_regime.value
        vix_regime = current_macro.vix_regime.value
        return self._retrieve_by_outcome(
            query=f"Macro regime {regime}, VIX {vix_regime}, {current_technical_summary}",
            outcome="SUCCESSFUL",
            regime=regime,
            n_results=n_results,
            as_of=as_of,
            label="wisdom",
        )

    def retrieve_warnings(
        self,
        current_macro: MacroContext,
        n_results: int = 3,
        as_of: Optional[datetime] = None,
    ) -> list[str]:
        """Retrieves failed historic patterns within the current macro regime to serve as warnings.

        Args:
            current_macro: MacroContext used to scope the regime filter.
            n_results: Maximum number of warning documents to return.
            as_of: When given, excludes documents stored after this timestamp. Must
                be naive, matching the naive datetime.now() this module stores.
                None (default) applies no filtering.

        Returns:
            List of matching document strings from the FAILED outcome filter for the current regime.
        """
        regime = current_macro.macro_regime.value
        return self._retrieve_by_outcome(
            query=f"Failed trades in {regime} regime",
            outcome="FAILED",
            regime=regime,
            n_results=n_results,
            as_of=as_of,
            label="warnings",
        )

    def get_agent_accuracy(
        self,
        agent_name: str,
        regime: Optional[str] = None,
        as_of: Optional[datetime] = None,
    ) -> tuple[float, int]:
        """Computes shrunk statistical win rates for trades driven primarily by a specific specialist agent.

        Win rate is shrunk toward the 0.5 neutral prior by
        MEMORY.accuracy_shrinkage_k pseudo-observations (wins + k*0.5) / (n + k),
        so an agent with 3 observations isn't trusted like one with 300 — it
        converges to the raw win rate as n grows and collapses to 0.5 as
        n -> 0. Feeds orchestration/aggregator.py's reliability weighting. n
        includes FLAT outcomes (store_trade_outcome's |return| ≤ 1% bucket) as
        non-wins, so this is a true win rate over all stored trades, not one
        conditioned on |return| > 1%.

        Args:
            agent_name: Agent identifier string (e.g. 'technical', 'fundamental', 'sentiment').
            regime: Optional regime filter (e.g. 'EXPANSION'); queries all regimes if None.
            as_of: When given, excludes trades stored after this timestamp. Must be
                naive, matching the naive datetime.now() this module stores. None
                (default) applies no filtering; replay's closed-loop evaluation
                passes it to avoid reading future outcomes.

        Returns:
            (rate, n) — the shrunk win rate in [0, 1], and the raw sample count it
            was computed from. n distinguishes "0.5 because no data exists" from
            "0.5 because that's the measured rate"; callers that need to gate on
            data actually existing (e.g. the Kelly anchor) should check n, not rate.
        """
        if self.collection.count() == 0:
            return 0.5, 0

        try:
            conditions: list[dict[str, Any]] = [{"primary_driver": agent_name.lower()}]
            if regime:
                conditions.append({"regime": regime})
            if as_of is not None:
                conditions.append({"timestamp": {"$lte": as_of.isoformat()}})

            results = self.collection.get(where=_where(conditions))  # type: ignore[arg-type]
            metadatas = results.get("metadatas", [])
            if not metadatas:
                return 0.5, 0

            wins = sum(1 for m in metadatas if m.get("outcome") == "SUCCESSFUL")
            n = len(metadatas)
            k = MEMORY.accuracy_shrinkage_k
            return (float(wins) + k * 0.5) / (n + k), n
        except Exception as e:
            logger.warning("[Memory] Failed to compute agent accuracy: %s", e)
            return 0.5, 0

    def summary_stats(self) -> dict[str, Any]:
        """Compiles aggregate performance statistics and regime diagnostics from the memory database.

        avg_return_pct averages over settled rows only (SUCCESSFUL/FAILED/FLAT
        outcomes carry a real return_pct; PENDING snapshots don't). Dividing by
        total_stored instead — which includes every still-open PENDING
        snapshot — structurally biases the average toward 0.0.

        Returns:
            Dict with keys: total_stored, successful_count, failed_count,
            pending_count, avg_return_pct, best_regime_for_wins.
        """
        count = self.collection.count()
        if count == 0:
            return {
                "total_stored": 0,
                "successful_count": 0,
                "failed_count": 0,
                "pending_count": 0,
                "avg_return_pct": 0.0,
                "best_regime_for_wins": "N/A",
            }

        try:
            results = self.collection.get()
            metadatas = results.get("metadatas") or []
            wins = 0
            fails = 0
            pending = 0
            settled = 0
            total_ret = 0.0
            regime_wins: dict[str, int] = {}

            for m in metadatas:
                outcome = m.get("outcome")
                regime = str(m.get("regime", "unknown"))

                if outcome == "PENDING":
                    pending += 1
                    continue

                settled += 1
                total_ret += float(m.get("return_pct", 0.0) or 0.0)  # type: ignore[arg-type]  # chromadb metadata values are typed as a broad scalar union
                if outcome == "SUCCESSFUL":
                    wins += 1
                    regime_wins[regime] = regime_wins.get(regime, 0) + 1
                elif outcome == "FAILED":
                    fails += 1

            best_regime = max(regime_wins, key=lambda r: regime_wins[r], default="N/A")

            return {
                "total_stored": count,
                "successful_count": wins,
                "failed_count": fails,
                "pending_count": pending,
                "avg_return_pct": total_ret / settled if settled > 0 else 0.0,
                "best_regime_for_wins": best_regime,
            }
        except Exception as e:
            logger.warning("[Memory] Failed to compute summary stats: %s", e)
            return {"total_stored": count}

    def store_decision_snapshot(self, decision: ARGUSDecision) -> bool:
        """Persists a real-time decision profile before outcomes are finalized.

        Populates the vector store with decision-making context, enabling retrieval
        of historical similarities before trade settlement data is processed.

        Args:
            decision: In-flight ARGUSDecision captured at signal generation time.

        Returns:
            True if the snapshot was written, False if the write failed.
        """
        try:
            macro_regime = decision.macro.macro_regime.value if decision.macro else "unknown"
            vix_regime = decision.macro.vix_regime.value if decision.macro else "unknown"
            agg_signal = _signal_or_na(decision.aggregated)
            agg_conv = decision.aggregated.conviction if decision.aggregated else 0.0
            fund_sig = _signal_or_na(decision.fundamental)
            tech_sig = _signal_or_na(decision.technical)
            sent_sig = _signal_or_na(decision.sentiment)
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
            return True
        except Exception as e:
            logger.warning("[Memory] Failed to snapshot decision for %s: %s", decision.ticker, e)
            return False

    def expire_pending_snapshots(self, cutoff: datetime) -> int:
        """Deletes PENDING decision snapshots older than cutoff.

        A snapshot only leaves PENDING once store_trade_outcome settles it
        (see its docstring); a decision that never took a position, or whose
        reconciliation permanently fails (e.g. the ticker's price history is
        never fetchable again), would otherwise sit PENDING indefinitely,
        growing the collection and dragging summary_stats.pending_count up
        forever. Called on the same reconciliation cadence as
        orchestration.reconciliation.prune_checkpoints and
        compact_decisions_jsonl, with the same horizon-plus-margin cutoff — a
        snapshot that old has already had its chance to settle.

        Args:
            cutoff: Snapshots whose ``timestamp`` metadata is before this are
                deleted. Should be tz-naive, matching the naive
                session_timestamps snapshots are stored with.

        Returns:
            Count of snapshots deleted.
        """
        try:
            results = self.collection.get(where={"outcome": "PENDING"})
        except Exception as e:
            logger.warning("[Memory] Failed to query PENDING snapshots for expiry: %s", e)
            return 0

        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        stale_ids = []
        for doc_id, meta in zip(ids, metadatas):
            timestamp_raw = meta.get("timestamp")
            if not timestamp_raw:
                continue
            try:
                if datetime.fromisoformat(str(timestamp_raw)) < cutoff:
                    stale_ids.append(doc_id)
            except ValueError:
                continue

        if not stale_ids:
            return 0

        try:
            self.collection.delete(ids=stale_ids)
        except Exception as e:
            logger.warning("[Memory] Failed to delete stale PENDING snapshot(s): %s", e)
            return 0

        logger.info("[Memory] Expired %d stale PENDING snapshot(s)", len(stale_ids))
        return len(stale_ids)


_cultural_memory: Optional[CulturalMemoryManager] = None
_cultural_memory_lock = Lock()


def get_cultural_memory(persist_dir: Optional[str] = None) -> CulturalMemoryManager:
    """Returns the process-wide CulturalMemoryManager, constructing it on first call.

    Lazy on purpose: construction pulls in sentence-transformers (and its
    torch dependency) to build the embedding function, and both are an
    optional `[models]` extra (see pyproject.toml) — importing this module
    must not require them, only actually using cultural memory does.

    Args:
        persist_dir: Filesystem directory backing the ChromaDB PersistentClient,
            used only on the first call that constructs the manager — a later
            call with a different persist_dir cannot reconfigure the singleton
            and is logged rather than silently ignored. None resolves to
            settings.ARGUS_CHROMA_DIR, or <ARGUS_DATA_DIR>/chroma_db if that's
            also unset.

    Returns:
        The process-wide CulturalMemoryManager singleton.
    """
    if persist_dir is None:
        persist_dir = settings.ARGUS_CHROMA_DIR or f"{settings.ARGUS_DATA_DIR}/chroma_db"

    global _cultural_memory
    if _cultural_memory is None:
        with _cultural_memory_lock:
            if _cultural_memory is None:
                _cultural_memory = CulturalMemoryManager(persist_dir)
                return _cultural_memory
    if os.path.abspath(persist_dir) != _cultural_memory.persist_dir:
        logger.warning(
            "[Memory] get_cultural_memory(persist_dir=%r) ignored; the process-wide "
            "singleton is already initialized at %s.",
            persist_dir,
            _cultural_memory.persist_dir,
        )
    return _cultural_memory
