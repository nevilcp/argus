"""ChromaDB decision snapshots, settled outcomes, and regime-filtered retrieval."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from chromadb.api.types import Where

from argus.config import settings
from argus.infra.clock import as_naive_timestamp
from argus.params import MEMORY, RECONCILIATION
from argus.schemas.signals import ARGUSDecision, MacroContext, Outcome

logger = logging.getLogger(__name__)

_SETTLED_IN: dict[Literal["$in", "$nin"], list[str | int | float | bool]] = {"$in": ["SUCCESSFUL", "FAILED", "FLAT"]}
_SETTLED_WHERE: Where = {"outcome": _SETTLED_IN}

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModelMismatchError(RuntimeError):
    """The persisted collection was embedded with a different model than this process uses."""


def _check_embedding_model_identity(
    persisted_metadata: dict[str, Any] | None, expected_model: str
) -> dict[str, Any] | None:
    """Check the stored embedding model identity without overwriting existing provenance."""
    persisted_metadata = persisted_metadata or {}
    recorded_model = persisted_metadata.get("embedding_model")
    if recorded_model is None:
        # Backfill only embedding provenance because Chroma rejects hnsw:space in collection.modify metadata.
        return {"embedding_model": expected_model}
    if recorded_model != expected_model:
        raise EmbeddingModelMismatchError(
            f"Cultural memory collection was embedded with {recorded_model!r}, but this "
            f"process is configured for {expected_model!r}. Vectors from different "
            "embedding models are not comparable — point persist_dir at a fresh "
            "directory or reindex the collection under the new model before continuing."
        )
    return None


def _where(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine metadata filter conditions into a single ChromaDB `where` clause."""
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _outcome_for(actual_return_pct: float) -> Outcome:
    """Classify a realized return as successful, failed, or flat (within the storage band)."""
    if actual_return_pct > RECONCILIATION.min_abs_return_for_storage:
        return Outcome.SUCCESSFUL
    if actual_return_pct < -RECONCILIATION.min_abs_return_for_storage:
        return Outcome.FAILED
    return Outcome.FLAT


def _render_trade_document(
    prefix: Outcome,
    decision: ARGUSDecision,
    actual_return_pct: float,
    holding_days: int,
    exit_reason: str,
) -> str:
    """Render the text document stored (and embedded) for a settled trade."""
    technical = decision.technical
    macro_regime = decision.macro.macro_regime.value if decision.macro else "unknown"
    vix_regime = decision.macro.vix_regime.value if decision.macro and decision.macro.vix_regime else "unknown"
    tech_rsi = f"{technical.rsi_14:.0f}" if technical else "N/A"
    tech_macd = f"{technical.macd_histogram:.3f}" if technical else "N/A"
    fund_moat = decision.fundamental.moat_score if decision.fundamental else "N/A"
    sent_finbert = decision.sentiment.finbert_net_score if decision.sentiment else None
    agg_conv = decision.aggregated.conviction if decision.aggregated else "N/A"

    return f"""{prefix.value} PATTERN:
Macro regime: {macro_regime}
VIX regime: {vix_regime}
Technical signal: {_signal_or_na(technical)}
  (RSI={tech_rsi}, MACD hist={tech_macd})
Fundamental signal: {_signal_or_na(decision.fundamental)}
  (Moat={fund_moat})
Sentiment signal: {_signal_or_na(decision.sentiment)}
  (FinBERT={"N/A" if sent_finbert is None else f"{sent_finbert:.2f}"})
Aggregated conviction: {agg_conv}
Outcome: {actual_return_pct * 100:+.1f}% in {holding_days} days. Exit: {exit_reason}"""


def _signal_or_na(agent_signal: Any) -> str:
    """Return an agent signal block's enum value, or "N/A" when the agent didn't run."""
    return agent_signal.signal.value if agent_signal else "N/A"


class CulturalMemoryManager:
    """Local vector memory for decision snapshots and settled, regime-filtered outcomes."""

    def __init__(self, persist_dir: str | None = None):
        if persist_dir is None:
            persist_dir = default_chroma_dir()
        # Import optional embedding dependencies only when constructing the memory store.
        import chromadb
        from chromadb.utils import embedding_functions

        # Resolve the storage path so unexpected working directories produce a visible warning.
        persist_dir = os.path.abspath(persist_dir)
        is_fresh = not os.path.isdir(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        if is_fresh:
            logger.warning(
                "%s did not exist; created a fresh, empty cultural memory "
                "store there. If this is unexpected, check the process's working directory.",
                persist_dir,
            )

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )

        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL_NAME)

        self.collection = self.client.get_or_create_collection(
            name="argus_wisdom",
            embedding_function=self.ef,  # type: ignore[arg-type]  # chromadb/sentence-transformers stub mismatch
            metadata={"hnsw:space": "cosine", "embedding_model": _EMBEDDING_MODEL_NAME},
        )
        backfill_metadata = _check_embedding_model_identity(self.collection.metadata, _EMBEDDING_MODEL_NAME)
        if backfill_metadata is not None:
            self.collection.modify(metadata=backfill_metadata)
            logger.info(
                "Backfilled embedding_model=%s onto a pre-existing collection with no recorded model",
                _EMBEDDING_MODEL_NAME,
            )
        self.persist_dir = persist_dir
        logger.info("Cultural Memory Manager initialized at %s", persist_dir)

    def already_reconciled(self, decision_ids: list[str]) -> set[str]:
        """Return the subset of decision_ids that already have a stored trade outcome."""
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
        """Store successful, failed, and flat outcomes, replacing any pending snapshot."""
        prefix = _outcome_for(actual_return_pct)
        macro_regime = decision.macro.macro_regime.value if decision.macro else "unknown"
        document = _render_trade_document(prefix, decision, actual_return_pct, holding_days, exit_reason)

        doc_id = f"trade_{decision.decision_id}"

        self.collection.upsert(
            documents=[document],
            ids=[doc_id],
            metadatas=[
                {
                    "regime": macro_regime,
                    "outcome": prefix.value,
                    "return_pct": float(round(actual_return_pct, 4)),
                    "holding_days": int(holding_days),
                    "timestamp": decision.session_timestamp.isoformat(),
                    "primary_driver": primary_driver,
                }
            ],
        )

        # Remove the settled PENDING snapshot so one decision cannot inflate counts or dilute average returns.
        snapshot_id = f"snapshot_{decision.decision_id}"
        try:
            self.collection.delete(ids=[snapshot_id])
        except Exception as e:
            logger.warning("Failed to delete snapshot %s: %s", snapshot_id, e)

        self._prune_trade_outcomes()
        logger.info("Stored %s trade pattern. Total: %d", prefix, self.collection.count())

    def _prune_trade_outcomes(self) -> None:
        """Keep the most recent settled outcomes, with IDs breaking timestamp ties."""
        rows = self.collection.get(
            where=_SETTLED_WHERE,
            include=["metadatas"],
        )
        trades = sorted(
            (str(meta.get("timestamp", "")), doc_id)
            for doc_id, meta in zip(rows.get("ids") or [], rows.get("metadatas") or [], strict=True)
            if doc_id.startswith("trade_")
        )
        excess = len(trades) - MEMORY.max_trade_outcomes
        if excess > 0:
            self.collection.delete(ids=[doc_id for _, doc_id in trades[:excess]])

    def _retrieve_by_outcome(
        self,
        query: str,
        outcome: Outcome,
        regime: str,
        n_results: int,
        as_of: datetime | None,
        label: str,
    ) -> list[str]:
        """Run one outcome- and regime-filtered similarity query, returning [] on failure."""
        if self.collection.count() == 0:
            return []

        conditions: list[dict[str, Any]] = [{"outcome": outcome.value}, {"regime": regime}]
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
            logger.warning("Failed to retrieve %s: %s", label, e)
            return []

    def retrieve_wisdom(
        self,
        current_macro: MacroContext,
        current_technical_summary: str,
        n_results: int = 5,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Query the vector collection to retrieve successful historic patterns."""
        regime = current_macro.macro_regime.value
        vix_regime = current_macro.vix_regime.value if current_macro.vix_regime else "unknown"
        return self._retrieve_by_outcome(
            query=(f"Macro regime {regime}, VIX {vix_regime}, {current_technical_summary}"),
            outcome=Outcome.SUCCESSFUL,
            regime=regime,
            n_results=n_results,
            as_of=as_of,
            label="wisdom",
        )

    def retrieve_warnings(
        self,
        current_macro: MacroContext,
        n_results: int = 3,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Retrieve failed historic patterns within the current macro regime to serve as warnings."""
        regime = current_macro.macro_regime.value
        return self._retrieve_by_outcome(
            query=f"Failed trades in {regime} regime",
            outcome=Outcome.FAILED,
            regime=regime,
            n_results=n_results,
            as_of=as_of,
            label="warnings",
        )

    def get_agent_accuracy(
        self,
        agent_name: str,
        regime: str | None = None,
        as_of: datetime | None = None,
    ) -> tuple[float, int]:
        """Return per-agent win rate shrunk toward 0.5 and its sample count, including flat non-wins."""
        if self.collection.count() == 0:
            return 0.5, 0

        try:
            conditions: list[dict[str, Any]] = [
                {"primary_driver": agent_name.lower()},
                _SETTLED_WHERE,
            ]
            if regime:
                conditions.append({"regime": regime})
            if as_of is not None:
                conditions.append({"timestamp": {"$lte": as_of.isoformat()}})

            results = self.collection.get(where=_where(conditions), include=["metadatas"])  # type: ignore[arg-type]
            metadatas = results.get("metadatas", [])
            if not metadatas:
                return 0.5, 0

            wins = sum(1 for m in metadatas if m.get("outcome") == Outcome.SUCCESSFUL)
            n = len(metadatas)
            k = MEMORY.accuracy_shrinkage_k
            return (float(wins) + k * 0.5) / (n + k), n
        except Exception as e:
            logger.warning("Failed to compute agent accuracy: %s", e)
            return 0.5, 0

    def summary_stats(self) -> dict[str, Any]:
        """Summarize settled returns separately from pending decision counts."""
        count = 0
        try:
            count = self.collection.count()
        except Exception as exc:
            logger.warning("Failed to count memory rows: %s", exc)
        empty = {
            "total_stored": count,
            "successful_count": 0,
            "failed_count": 0,
            "pending_count": 0,
            "avg_return_pct": 0.0,
            "best_regime_for_wins": "N/A",
        }
        if count == 0:
            return empty

        try:
            results = self.collection.get(where=_SETTLED_WHERE, include=["metadatas"])
            metadatas = results.get("metadatas") or []
            wins = 0
            fails = 0
            pending_rows = self.collection.get(where={"outcome": "PENDING"}, include=[])
            pending = len(pending_rows.get("ids") or [])
            settled = 0
            total_ret = 0.0
            regime_wins: dict[str, int] = {}

            for m in metadatas:
                outcome = m.get("outcome")
                regime = str(m.get("regime", "unknown"))

                settled += 1
                total_ret += float(m.get("return_pct", 0.0) or 0.0)  # type: ignore[arg-type]
                if outcome == Outcome.SUCCESSFUL:
                    wins += 1
                    regime_wins[regime] = regime_wins.get(regime, 0) + 1
                elif outcome == Outcome.FAILED:
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
            logger.warning("Failed to compute summary stats: %s", e)
            return empty

    def store_decision_snapshot(self, decision: ARGUSDecision) -> bool:
        """Persist a real-time decision profile before outcomes are finalized."""
        try:
            macro_regime = decision.macro.macro_regime.value if decision.macro else "unknown"
            vix_regime = decision.macro.vix_regime.value if decision.macro and decision.macro.vix_regime else "unknown"
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
                        "outcome": Outcome.PENDING.value,
                        "ticker": decision.ticker,
                        "agg_signal": agg_signal,
                        "allocated": str(decision.allocation is not None),
                        "timestamp": decision.session_timestamp.isoformat(),
                        "decision_id": decision.decision_id,
                    }
                ],
            )
            logger.info(
                "Snapshotted decision for %s (ID=%s)",
                decision.ticker,
                decision.decision_id,
            )
            return True
        except Exception as e:
            logger.warning("Failed to snapshot decision for %s: %s", decision.ticker, e)
            return False

    def expire_pending_snapshots(self, cutoff: datetime) -> int:
        """Delete PENDING decision snapshots older than cutoff."""
        try:
            results = self.collection.get(where={"outcome": Outcome.PENDING.value})
        except Exception as e:
            logger.warning("Failed to query PENDING snapshots for expiry: %s", e)
            return 0

        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        stale_ids = []
        for doc_id, meta in zip(ids, metadatas, strict=True):
            timestamp_raw = meta.get("timestamp")
            if not timestamp_raw:
                continue
            try:
                timestamp = datetime.fromisoformat(str(timestamp_raw))
            except ValueError:
                continue
            # Normalize timestamps before comparing with the naive retention cutoff.
            if as_naive_timestamp(timestamp) < cutoff:
                stale_ids.append(doc_id)

        if not stale_ids:
            return 0

        try:
            self.collection.delete(ids=stale_ids)
        except Exception as e:
            logger.warning("Failed to delete stale PENDING snapshot(s): %s", e)
            return 0

        logger.info("Expired %d stale PENDING snapshot(s)", len(stale_ids))
        return len(stale_ids)


_cultural_memory: CulturalMemoryManager | None = None
_cultural_memory_lock = Lock()


def default_chroma_dir() -> str:
    """Return where the cultural-memory store lives when no directory is given."""
    return settings.ARGUS_CHROMA_DIR or str(settings.data_path("chroma_db"))


def get_cultural_memory(persist_dir: str | None = None) -> CulturalMemoryManager:
    """Return the process-wide CulturalMemoryManager, constructing it on first call."""
    if persist_dir is None:
        persist_dir = default_chroma_dir()

    global _cultural_memory
    if _cultural_memory is None:
        with _cultural_memory_lock:
            if _cultural_memory is None:
                _cultural_memory = CulturalMemoryManager(persist_dir)
                return _cultural_memory
    if os.path.abspath(persist_dir) != _cultural_memory.persist_dir:
        logger.warning(
            "get_cultural_memory(persist_dir=%r) ignored; the process-wide singleton is already initialized at %s.",
            persist_dir,
            _cultural_memory.persist_dir,
        )
    return _cultural_memory
