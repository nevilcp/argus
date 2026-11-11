"""Tests the helpers in argus/orchestration/nodes.py that the DAG tests reach only indirectly."""

from unittest import mock

import pytest
from chromadb.errors import ChromaError

from argus.orchestration.nodes import (
    GraphDeps,
    _cultural_memory_or_none,
    _session_reliability,
    node_log_decisions,
)
from tests.helpers.builders import macro_context, neutral_cultural_memory


def _deps(provider) -> GraphDeps:
    """Builds a GraphDeps whose agents are unused placeholders and whose memory is `provider`."""
    return GraphDeps(
        market_data=mock.Mock(),
        macro_agent=mock.Mock(),
        tech_agent=mock.Mock(),
        fund_agent=mock.Mock(),
        sent_agent=mock.Mock(),
        risk_agent=mock.Mock(),
        aggregator=mock.Mock(),
        portfolio_agent=mock.Mock(),
        cultural_memory_provider=provider,
    )


def _raising(exc: Exception):
    def provider():
        raise exc

    return provider


@pytest.mark.parametrize("exc", [ImportError("no torch"), OSError("read-only"), ChromaError("bad")])
def test_cultural_memory_or_none_degrades_on_each_documented_failure(exc):
    """Each way the store can fail to open yields None instead of raising."""
    assert _cultural_memory_or_none(_deps(_raising(exc))) is None


def test_cultural_memory_or_none_does_not_swallow_an_unexpected_error():
    """Only the documented failures degrade; a bug in the provider still surfaces."""
    with pytest.raises(RuntimeError):
        _cultural_memory_or_none(_deps(_raising(RuntimeError("bug"))))


def test_session_reliability_is_unhealthy_only_when_the_store_is_unavailable():
    """No macro regime is a healthy neutral prior; an unopenable store is not."""
    macro = macro_context()

    _, _, healthy_without_regime = _session_reliability(None, None, _deps(_raising(OSError())))
    _, counts, healthy_without_store = _session_reliability(macro, None, _deps(_raising(OSError())))
    _, _, healthy_with_store = _session_reliability(
        macro, None, _deps(lambda: neutral_cultural_memory())
    )

    assert healthy_without_regime is True
    assert healthy_without_store is False
    assert set(counts.values()) == {0}
    assert healthy_with_store is True


def test_log_decisions_still_returns_decisions_when_the_store_is_unavailable():
    """An unavailable store skips persistence but not the decisions the node returns."""
    allocation = mock.Mock(portfolio=[])
    state = {"portfolio_allocation": allocation, "universe": []}

    result = node_log_decisions(state, _deps(_raising(ImportError("no torch"))))

    assert result == {"decisions": [], "errors": []}
