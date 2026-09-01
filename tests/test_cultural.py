"""Tests for CulturalMemoryManager.

Builds the manager via object.__new__ plus a stub .collection rather than
the real constructor, which pulls in chromadb and sentence-transformers
(the optional [models] extra) just to exercise arithmetic and query-shaping
over already-stored metadata.
"""

from datetime import datetime
from unittest import mock

import pytest

import argus.memory.cultural as cultural_module
from argus.memory.cultural import CulturalMemoryManager, get_cultural_memory
from argus.params import MEMORY, RECONCILIATION
from argus.schemas.signals import (
    ARGUSDecision,
    MacroContext,
    Regime,
    SectorSignal,
    VixRegime,
    YieldCurve,
)


def _manager_with_metadatas(metadatas: list[dict]) -> CulturalMemoryManager:
    manager = object.__new__(CulturalMemoryManager)
    manager.collection = mock.Mock()
    manager.collection.count.return_value = len(metadatas)
    manager.collection.get.return_value = {"metadatas": metadatas}
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


def _macro(
    regime: Regime = Regime.EXPANSION, vix_regime: VixRegime = VixRegime.MEDIUM
) -> MacroContext:
    return MacroContext(
        fed_funds=4.0,
        cpi_yoy=3.0,
        unemployment=4.0,
        t10y2y=0.1,
        consumer_sentiment=60.0,
        vix_level=18.0,
        macro_regime=regime,
        regime_confidence=0.7,
        model_healthy=True,
        interest_rate_trend="STABLE",
        yield_curve_shape=YieldCurve.NORMAL,
        vix_regime=vix_regime,
        vix_percentile=40.0,
        inflation_trajectory="STABLE",
        sector_rotation_signal=SectorSignal.GROWTH_FAVORED,
        agent_multipliers={"fundamental": 1.0, "technical": 1.0, "sentiment": 1.0},
        timestamp=datetime.now(),
    )


def test_zero_observations_returns_prior():
    """With no stored outcomes, accuracy falls back to the prior with n=0."""
    manager = _manager_with_metadatas([])
    assert manager.get_agent_accuracy("technical") == (0.5, 0)


def test_small_sample_shrinks_toward_prior_instead_of_reporting_the_raw_rate():
    """A small sample's accuracy is pulled well below its raw win rate."""
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
    """As the sample grows, accuracy converges to the raw win rate."""
    metadatas = (
        [{"outcome": "SUCCESSFUL", "primary_driver": "technical"}] * 900
        + [{"outcome": "FAILED", "primary_driver": "technical"}] * 100
    )
    manager = _manager_with_metadatas(metadatas)

    accuracy, n = manager.get_agent_accuracy("technical")
    assert abs(accuracy - 0.9) < 0.01
    assert n == 1000


def test_flat_outcomes_count_as_non_wins_in_the_denominator():
    """A FLAT-tagged trade lowers accuracy, not vanishes from the sample.

    get_agent_accuracy's win rate must be P(win | stored), not
    P(win | |return| > 1%) — a FLAT trade must inflate n without inflating
    wins.
    """
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
    """as_of adds a timestamp upper bound to the metadata filter, alongside regime."""
    manager = _manager_with_metadatas([{"outcome": "SUCCESSFUL", "primary_driver": "technical"}])
    as_of = datetime(2026, 1, 1)

    manager.get_agent_accuracy("technical", regime="EXPANSION", as_of=as_of)

    where = manager.collection.get.call_args.kwargs["where"]
    assert where == {
        "$and": [
            {"primary_driver": "technical"},
            {"regime": "EXPANSION"},
            {"timestamp": {"$lte": as_of.isoformat()}},
        ]
    }


def test_store_trade_outcome_persists_flat_returns_instead_of_dropping_them():
    """A |return| <= 1% trade is stored as FLAT, not silently discarded."""
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
    """The stored document text uses VixRegime's .value, not its str(Enum) repr.

    Python >= 3.11 includes the class name in a str-Enum's default
    __format__, so an f-string without .value would write "VixRegime.HIGH"
    into the document instead of "HIGH".
    """
    manager = _manager_with_metadatas([])
    decision = ARGUSDecision(
        ticker="AAPL",
        session_timestamp=datetime.now(),
        macro=_macro(vix_regime=VixRegime.HIGH),
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
    """Storing a trade outcome removes the snapshot_{id} row it settles.

    Leaving the PENDING snapshot in place after the trade settles would
    double-count the decision in summary_stats via the snapshot's zero
    return_pct.
    """
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
    """Only decision_ids with a trade_{id} row come back, stripped of the prefix.

    reconcile_decisions() calls this once per batch, before touching market
    data, to skip decisions already reconciled on a prior run.
    """
    manager = _manager_with_metadatas([])
    manager.collection.get.return_value = {"ids": ["trade_abc", "trade_def"]}

    result = manager.already_reconciled(["abc", "def", "ghi"])

    manager.collection.get.assert_called_once_with(ids=["trade_abc", "trade_def", "trade_ghi"])
    assert result == {"abc", "def"}


def test_already_reconciled_empty_input_skips_the_query():
    """An empty decision_ids list returns immediately without touching the collection."""
    manager = _manager_with_metadatas([])

    assert manager.already_reconciled([]) == set()
    manager.collection.get.assert_not_called()


def test_summary_stats_averages_return_over_settled_rows_only():
    """avg_return_pct excludes PENDING rows from both the numerator and denominator.

    PENDING snapshots carry no return_pct, so dividing by total_stored
    (which includes them) would structurally bias the average toward 0.0.
    """
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
    """An all-PENDING store reports avg_return_pct=0.0 rather than raising ZeroDivisionError."""
    manager = _manager_with_metadatas(
        [
            {"outcome": "PENDING", "regime": "EXPANSION"},
        ]
    )

    stats = manager.summary_stats()

    assert stats["pending_count"] == 1
    assert stats["avg_return_pct"] == 0.0


def test_expire_pending_snapshots_deletes_only_entries_before_cutoff():
    """A snapshot older than cutoff is deleted; one at or after it survives."""
    manager = _manager_with_pending(
        ids=["snapshot_old", "snapshot_new"],
        timestamps=["2026-01-01T00:00:00", "2026-01-20T00:00:00"],
    )

    deleted = manager.expire_pending_snapshots(cutoff=datetime(2026, 1, 10))

    assert deleted == 1
    manager.collection.get.assert_called_once_with(where={"outcome": "PENDING"})
    manager.collection.delete.assert_called_once_with(ids=["snapshot_old"])


def test_expire_pending_snapshots_nothing_stale_does_not_call_delete():
    """When every PENDING snapshot is at or after cutoff, delete() is never called."""
    manager = _manager_with_pending(ids=["snapshot_new"], timestamps=["2026-01-20T00:00:00"])

    deleted = manager.expire_pending_snapshots(cutoff=datetime(2026, 1, 1))

    assert deleted == 0
    manager.collection.delete.assert_not_called()


def test_expire_pending_snapshots_empty_store_is_a_noop():
    """No PENDING rows at all is a no-op, not an error."""
    manager = _manager_with_pending(ids=[], timestamps=[])

    assert manager.expire_pending_snapshots(cutoff=datetime(2026, 1, 1)) == 0
    manager.collection.delete.assert_not_called()


def test_store_decision_snapshot_returns_true_on_success():
    """A successful upsert reports success so callers can count real writes."""
    manager = _manager_with_metadatas([])
    decision = ARGUSDecision(ticker="AAPL", session_timestamp=datetime.now())

    assert manager.store_decision_snapshot(decision) is True


def test_store_decision_snapshot_returns_false_on_failure():
    """A failed upsert reports failure instead of silently swallowing it.

    Without this, store_decision_snapshot would return None unconditionally,
    so node_log_decisions would log every built decision as "logged to
    cultural memory" even when the write itself failed.
    """
    manager = _manager_with_metadatas([])
    manager.collection.upsert.side_effect = RuntimeError("chroma write failed")
    decision = ARGUSDecision(ticker="AAPL", session_timestamp=datetime.now())

    assert manager.store_decision_snapshot(decision) is False


def test_retrieve_wisdom_and_retrieve_warnings_apply_a_symmetric_regime_filter():
    """retrieve_wisdom filters on regime exactly like retrieve_warnings does.

    Successes shouldn't be drawn from every regime while failures are
    confined to the current one.
    """
    macro = _macro(regime=Regime.CONTRACTION)

    wisdom_manager = _manager_with_query_collection()
    wisdom_manager.retrieve_wisdom(macro, "mixed technicals")
    wisdom_where = wisdom_manager.collection.query.call_args.kwargs["where"]
    assert wisdom_where == {"$and": [{"outcome": "SUCCESSFUL"}, {"regime": "CONTRACTION"}]}

    warnings_manager = _manager_with_query_collection()
    warnings_manager.retrieve_warnings(macro)
    warnings_where = warnings_manager.collection.query.call_args.kwargs["where"]
    assert warnings_where == {"$and": [{"outcome": "FAILED"}, {"regime": "CONTRACTION"}]}


def test_retrieve_wisdom_as_of_filters_on_timestamp():
    """as_of adds a timestamp upper bound to retrieve_wisdom's where clause."""
    manager = _manager_with_query_collection()
    as_of = datetime(2026, 1, 1)

    manager.retrieve_wisdom(_macro(regime=Regime.EXPANSION), "mixed technicals", as_of=as_of)

    where = manager.collection.query.call_args.kwargs["where"]
    assert where == {
        "$and": [
            {"outcome": "SUCCESSFUL"},
            {"regime": "EXPANSION"},
            {"timestamp": {"$lte": as_of.isoformat()}},
        ]
    }


def test_get_cultural_memory_ignores_a_later_persist_dir_and_warns(monkeypatch, caplog, tmp_path):
    """A later persist_dir cannot reconfigure the already-constructed singleton.

    The second call logs a warning instead of silently ignoring the new
    directory.
    """
    fake_manager = mock.Mock(persist_dir=str(tmp_path / "first"))
    monkeypatch.setattr(cultural_module, "_cultural_memory", fake_manager)

    with caplog.at_level("WARNING", logger="argus.cultural_memory"):
        returned = get_cultural_memory(str(tmp_path / "second"))

    assert returned is fake_manager
    assert any("ignored" in r.message for r in caplog.records)
