"""Tests for CulturalMemoryManager."""

from datetime import datetime
from unittest import mock

import pytest

import argus.memory.cultural as cultural_module
from argus.memory.cultural import (
    CulturalMemoryManager,
    EmbeddingModelMismatchError,
    _check_embedding_model_identity,
    default_chroma_dir,
    get_cultural_memory,
)
from argus.params import MEMORY, RECONCILIATION
from argus.schemas.signals import (
    ARGUSDecision,
    Regime,
    VixRegime,
)
from tests.helpers.builders import macro_context


def _manager_with_metadatas(metadatas: list[dict]) -> CulturalMemoryManager:
    manager = object.__new__(CulturalMemoryManager)
    manager.collection = mock.Mock()
    manager.collection.count.return_value = len(metadatas)

    def get_rows(*, where=None, include=None, **kwargs):
        def matches(meta, condition):
            if "$and" in condition:
                return all(matches(meta, part) for part in condition["$and"])
            for key, expected in condition.items():
                value = meta.get(key)
                if isinstance(expected, dict):
                    if "$in" in expected and value not in expected["$in"]:
                        return False
                    if "$lte" in expected and value is not None and value > expected["$lte"]:
                        return False
                elif value != expected:
                    return False
            return True

        rows = [(f"trade_{i}", m) for i, m in enumerate(metadatas) if where is None or matches(m, where)]
        return {"ids": [i for i, _ in rows], "metadatas": [m for _, m in rows]}

    manager.collection.get.side_effect = get_rows
    return manager


def _manager_with_query_collection() -> CulturalMemoryManager:
    manager = object.__new__(CulturalMemoryManager)
    manager.collection = mock.Mock()
    manager.collection.count.return_value = 5
    manager.collection.query.return_value = {"documents": [["doc"]]}
    return manager


def _manager_with_pending(ids: list[str], timestamps: list[str]) -> CulturalMemoryManager:
    manager = object.__new__(CulturalMemoryManager)
    manager.collection = mock.Mock()
    manager.collection.get.return_value = {
        "ids": ids,
        "metadatas": [{"outcome": "PENDING", "timestamp": ts} for ts in timestamps],
    }
    return manager


def test_embedding_identity_check_backfills_a_collection_with_no_recorded_model():
    result = _check_embedding_model_identity({"hnsw:space": "cosine"}, "all-MiniLM-L6-v2")
    assert result == {"embedding_model": "all-MiniLM-L6-v2"}


def test_embedding_identity_check_is_a_noop_when_the_model_matches():
    metadata = {"hnsw:space": "cosine", "embedding_model": "all-MiniLM-L6-v2"}
    assert _check_embedding_model_identity(metadata, "all-MiniLM-L6-v2") is None


def test_embedding_identity_check_raises_on_a_mismatched_model():
    metadata = {"hnsw:space": "cosine", "embedding_model": "all-mpnet-base-v2"}
    with pytest.raises(EmbeddingModelMismatchError):
        _check_embedding_model_identity(metadata, "all-MiniLM-L6-v2")


def test_embedding_identity_check_handles_no_metadata_at_all():
    result = _check_embedding_model_identity(None, "all-MiniLM-L6-v2")
    assert result == {"embedding_model": "all-MiniLM-L6-v2"}


def test_zero_observations_returns_prior():
    manager = _manager_with_metadatas([])
    assert manager.get_agent_accuracy("technical") == (0.5, 0)


def test_small_sample_shrinks_toward_prior_instead_of_reporting_the_raw_rate():
    # 2/2 wins -> raw win rate 1.0, which shrinkage should pull well below
    metadatas = [{"outcome": "SUCCESSFUL", "primary_driver": "technical"}] * 2
    manager = _manager_with_metadatas(metadatas)

    accuracy, n = manager.get_agent_accuracy("technical")
    k = MEMORY.accuracy_shrinkage_k
    expected = (2 + k * 0.5) / (2 + k)

    assert accuracy == expected
    assert 0.5 < accuracy < 1.0
    assert n == 2


def test_large_sample_converges_to_the_raw_win_rate():
    metadatas = [{"outcome": "SUCCESSFUL", "primary_driver": "technical"}] * 900 + [
        {"outcome": "FAILED", "primary_driver": "technical"}
    ] * 100
    manager = _manager_with_metadatas(metadatas)

    accuracy, n = manager.get_agent_accuracy("technical")
    assert abs(accuracy - 0.9) < 0.01
    assert n == 1000


def test_flat_outcomes_count_as_non_wins_in_the_denominator():
    metadatas = [
        {"outcome": "SUCCESSFUL", "primary_driver": "technical"},
        {"outcome": "FLAT", "primary_driver": "technical"},
    ]
    manager = _manager_with_metadatas(metadatas)

    accuracy, n = manager.get_agent_accuracy("technical")
    k = MEMORY.accuracy_shrinkage_k
    expected = (1 + k * 0.5) / (2 + k)

    assert accuracy == expected
    assert n == 2


def test_get_agent_accuracy_as_of_filters_on_timestamp():
    manager = _manager_with_metadatas([{"outcome": "SUCCESSFUL", "primary_driver": "technical"}])
    as_of = datetime(2026, 1, 1)

    manager.get_agent_accuracy("technical", regime="EXPANSION", as_of=as_of)

    where = manager.collection.get.call_args.kwargs["where"]
    assert where == {
        "$and": [
            {"primary_driver": "technical"},
            {"outcome": {"$in": ["SUCCESSFUL", "FAILED", "FLAT"]}},
            {"regime": "EXPANSION"},
            {"timestamp": {"$lte": as_of.isoformat()}},
        ]
    }


def test_store_trade_outcome_persists_flat_returns_instead_of_dropping_them():
    manager = _manager_with_metadatas([])
    decision = ARGUSDecision(ticker="AAPL", session_timestamp=datetime.now())

    manager.store_trade_outcome(
        decision,
        actual_return_pct=RECONCILIATION.min_abs_return_for_storage / 2,
        holding_days=3,
        exit_reason="time_stop",
        primary_driver="technical",
    )

    manager.collection.upsert.assert_called_once()
    metadata = manager.collection.upsert.call_args.kwargs["metadatas"][0]
    assert metadata["outcome"] == "FLAT"


def test_store_trade_outcome_writes_vix_regime_value_not_enum_repr():
    manager = _manager_with_metadatas([])
    decision = ARGUSDecision(
        ticker="AAPL",
        session_timestamp=datetime.now(),
        macro=macro_context(vix_regime=VixRegime.HIGH),
    )

    manager.store_trade_outcome(
        decision,
        actual_return_pct=0.05,
        holding_days=3,
        exit_reason="target_hit",
        primary_driver="technical",
    )

    document = manager.collection.upsert.call_args.kwargs["documents"][0]
    assert "VIX regime: HIGH" in document
    assert "VixRegime" not in document


def test_store_trade_outcome_deletes_the_settled_pending_snapshot():
    manager = _manager_with_metadatas([])
    decision = ARGUSDecision(ticker="AAPL", session_timestamp=datetime.now())

    manager.store_trade_outcome(
        decision,
        actual_return_pct=0.05,
        holding_days=3,
        exit_reason="target_hit",
        primary_driver="technical",
    )

    manager.collection.delete.assert_called_once_with(ids=[f"snapshot_{decision.decision_id}"])


def test_already_reconciled_returns_ids_with_a_stored_trade_row():
    manager = _manager_with_metadatas([])
    manager.collection.get.side_effect = None
    manager.collection.get.return_value = {"ids": ["trade_abc", "trade_def"]}

    result = manager.already_reconciled(["abc", "def", "ghi"])

    manager.collection.get.assert_called_once_with(ids=["trade_abc", "trade_def", "trade_ghi"])
    assert result == {"abc", "def"}


def test_already_reconciled_empty_input_skips_the_query():
    manager = _manager_with_metadatas([])

    assert manager.already_reconciled([]) == set()
    manager.collection.get.assert_not_called()


def test_summary_stats_averages_return_over_settled_rows_only():
    manager = _manager_with_metadatas(
        [
            {"outcome": "SUCCESSFUL", "return_pct": 0.10, "regime": "EXPANSION"},
            {"outcome": "FAILED", "return_pct": -0.04, "regime": "EXPANSION"},
            {"outcome": "PENDING", "regime": "EXPANSION"},
            {"outcome": "PENDING", "regime": "EXPANSION"},
        ]
    )

    stats = manager.summary_stats()

    assert stats["total_stored"] == 4
    assert stats["pending_count"] == 2
    assert stats["successful_count"] == 1
    assert stats["failed_count"] == 1
    assert stats["avg_return_pct"] == pytest.approx((0.10 - 0.04) / 2)


def test_summary_stats_all_pending_reports_zero_average_without_dividing_by_zero():
    manager = _manager_with_metadatas(
        [
            {"outcome": "PENDING", "regime": "EXPANSION"},
        ]
    )

    stats = manager.summary_stats()

    assert stats["pending_count"] == 1
    assert stats["avg_return_pct"] == 0.0


def test_expire_pending_snapshots_deletes_only_entries_before_cutoff():
    manager = _manager_with_pending(
        ids=["snapshot_old", "snapshot_new"],
        timestamps=["2026-01-01T00:00:00", "2026-01-20T00:00:00"],
    )

    deleted = manager.expire_pending_snapshots(cutoff=datetime(2026, 1, 10))

    assert deleted == 1
    manager.collection.get.assert_called_once_with(where={"outcome": "PENDING"})
    manager.collection.delete.assert_called_once_with(ids=["snapshot_old"])


def test_expire_pending_snapshots_handles_tz_aware_timestamps_without_aborting():
    manager = _manager_with_pending(
        ids=["aware_old", "naive_old", "aware_new"],
        timestamps=[
            "2026-01-01T00:00:00+00:00",
            "2026-01-02T00:00:00",
            "2026-01-20T05:00:00-05:00",
        ],
    )

    deleted = manager.expire_pending_snapshots(cutoff=datetime(2026, 1, 10))

    assert deleted == 2
    manager.collection.delete.assert_called_once_with(ids=["aware_old", "naive_old"])


def test_expire_pending_snapshots_skips_an_unparseable_timestamp():
    manager = _manager_with_pending(ids=["garbled", "old"], timestamps=["not-a-date", "2026-01-01T00:00:00"])

    assert manager.expire_pending_snapshots(cutoff=datetime(2026, 1, 10)) == 1
    manager.collection.delete.assert_called_once_with(ids=["old"])


def test_expire_pending_snapshots_nothing_stale_does_not_call_delete():
    manager = _manager_with_pending(ids=["snapshot_new"], timestamps=["2026-01-20T00:00:00"])

    deleted = manager.expire_pending_snapshots(cutoff=datetime(2026, 1, 1))

    assert deleted == 0
    manager.collection.delete.assert_not_called()


def test_expire_pending_snapshots_empty_store_is_a_noop():
    manager = _manager_with_pending(ids=[], timestamps=[])

    assert manager.expire_pending_snapshots(cutoff=datetime(2026, 1, 1)) == 0
    manager.collection.delete.assert_not_called()


def test_store_decision_snapshot_returns_true_on_success():
    manager = _manager_with_metadatas([])
    decision = ARGUSDecision(ticker="AAPL", session_timestamp=datetime.now())

    assert manager.store_decision_snapshot(decision) is True


def test_store_decision_snapshot_returns_false_on_failure():
    manager = _manager_with_metadatas([])
    manager.collection.upsert.side_effect = RuntimeError("chroma write failed")
    decision = ARGUSDecision(ticker="AAPL", session_timestamp=datetime.now())

    assert manager.store_decision_snapshot(decision) is False


def test_retrieve_wisdom_and_retrieve_warnings_apply_a_symmetric_regime_filter():
    macro = macro_context(regime=Regime.CONTRACTION)

    wisdom_manager = _manager_with_query_collection()
    wisdom_manager.retrieve_wisdom(macro, "mixed technicals")
    wisdom_where = wisdom_manager.collection.query.call_args.kwargs["where"]
    assert wisdom_where == {"$and": [{"outcome": "SUCCESSFUL"}, {"regime": "CONTRACTION"}]}

    warnings_manager = _manager_with_query_collection()
    warnings_manager.retrieve_warnings(macro)
    warnings_where = warnings_manager.collection.query.call_args.kwargs["where"]
    assert warnings_where == {"$and": [{"outcome": "FAILED"}, {"regime": "CONTRACTION"}]}


def test_retrieve_wisdom_as_of_filters_on_timestamp():
    manager = _manager_with_query_collection()
    as_of = datetime(2026, 1, 1)

    manager.retrieve_wisdom(macro_context(regime=Regime.EXPANSION), "mixed technicals", as_of=as_of)

    where = manager.collection.query.call_args.kwargs["where"]
    assert where == {
        "$and": [
            {"outcome": "SUCCESSFUL"},
            {"regime": "EXPANSION"},
            {"timestamp": {"$lte": as_of.isoformat()}},
        ]
    }


def test_get_cultural_memory_ignores_a_later_persist_dir_and_warns(monkeypatch, caplog, tmp_path):
    fake_manager = mock.Mock(persist_dir=str(tmp_path / "first"))
    monkeypatch.setattr(cultural_module, "_cultural_memory", fake_manager)

    with caplog.at_level("WARNING", logger="argus.memory.cultural"):
        returned = get_cultural_memory(str(tmp_path / "second"))

    assert returned is fake_manager
    assert any("ignored" in r.message for r in caplog.records)


def test_default_chroma_dir_prefers_the_explicit_setting(monkeypatch):
    monkeypatch.setattr(cultural_module.settings, "ARGUS_CHROMA_DIR", "/srv/chroma")

    assert default_chroma_dir() == "/srv/chroma"


def test_default_chroma_dir_falls_back_to_the_data_dir(monkeypatch):
    monkeypatch.setattr(cultural_module.settings, "ARGUS_CHROMA_DIR", "")
    monkeypatch.setattr(cultural_module.settings, "ARGUS_DATA_DIR", "/srv/data")

    assert default_chroma_dir() == "/srv/data/chroma_db"


def test_trade_document_renders_missing_indicators_as_unavailable():
    manager = _manager_with_metadatas([])
    decision = ARGUSDecision(
        ticker="AAPL",
        session_timestamp=datetime.now(),
        macro=macro_context(vix_level=None, vix_regime=None),
    )
    manager.store_trade_outcome(decision, 0.05, 3, "target_hit", "unknown")
    document = manager.collection.upsert.call_args.kwargs["documents"][0]
    assert "RSI=N/A, MACD hist=N/A" in document
    assert "VIX regime: unknown" in document


def test_trade_retention_evicts_oldest_rows_and_preserves_pending(monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(cultural_module, "MEMORY", replace(MEMORY, max_trade_outcomes=2))
    manager = _manager_with_metadatas(
        [
            {"outcome": "SUCCESSFUL", "timestamp": "2026-01-03"},
            {"outcome": "FLAT", "timestamp": "2026-01-01"},
            {"outcome": "FAILED", "timestamp": "2026-01-02"},
            {"outcome": "PENDING", "timestamp": "2025-01-01"},
        ]
    )
    manager._prune_trade_outcomes()
    manager.collection.delete.assert_called_once_with(ids=["trade_1"])


def test_summary_failure_returns_the_full_response_shape():
    manager = _manager_with_metadatas([{"outcome": "PENDING"}])
    manager.collection.get.side_effect = OSError("unavailable")
    stats = manager.summary_stats()
    assert stats == {
        "total_stored": 1,
        "successful_count": 0,
        "failed_count": 0,
        "pending_count": 0,
        "avg_return_pct": 0.0,
        "best_regime_for_wins": "N/A",
    }
