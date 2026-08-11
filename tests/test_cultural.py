"""
tests/test_cultural.py

Unit tests for CulturalMemoryManager.get_agent_accuracy's shrinkage-toward-prior
behavior (see docs/adr/0011). Builds the manager via object.__new__ plus a
stub .collection rather than the real constructor, which pulls in chromadb +
sentence-transformers (the optional [models] extra, see ADR 0007) just to
exercise arithmetic over already-stored metadata.
"""

from unittest import mock

from argus.memory.cultural import CulturalMemoryManager
from argus.params import MEMORY


def _manager_with_metadatas(metadatas: list[dict]) -> CulturalMemoryManager:
    manager = object.__new__(CulturalMemoryManager)
    manager.collection = mock.Mock()
    manager.collection.count.return_value = len(metadatas)
    manager.collection.get.return_value = {"metadatas": metadatas}
    return manager


def test_zero_observations_returns_prior():
    manager = _manager_with_metadatas([])
    assert manager.get_agent_accuracy("technical") == 0.5


def test_small_sample_shrinks_toward_prior_instead_of_reporting_the_raw_rate():
    # 2/2 wins -> raw win rate 1.0, which shrinkage should pull well below.
    metadatas = [{"outcome": "SUCCESSFUL", "primary_driver": "technical"}] * 2
    manager = _manager_with_metadatas(metadatas)

    accuracy = manager.get_agent_accuracy("technical")
    k = MEMORY.accuracy_shrinkage_k
    expected = (2 + k * 0.5) / (2 + k)

    assert accuracy == expected
    assert 0.5 < accuracy < 1.0


def test_large_sample_converges_to_the_raw_win_rate():
    metadatas = (
        [{"outcome": "SUCCESSFUL", "primary_driver": "technical"}] * 900
        + [{"outcome": "FAILED", "primary_driver": "technical"}] * 100
    )
    manager = _manager_with_metadatas(metadatas)

    assert abs(manager.get_agent_accuracy("technical") - 0.9) < 0.01
