"""Tests for argus.infra (clock, files, backoff) and the shared config/state builders."""

import random
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from argus.config import settings, web_concurrency
from argus.infra.backoff import capped_backoff, full_jitter_delay
from argus.infra.clock import MARKET_TZ, as_naive_timestamp, now_local, now_market
from argus.infra.files import atomic_write_text
from argus.orchestration.state import ARGUSState, initial_state


def test_now_local_is_naive_and_now_market_is_aware_in_market_tz():
    assert now_local().tzinfo is None
    assert now_market().tzinfo is MARKET_TZ


def test_as_naive_timestamp_converts_aware_values_to_utc_before_dropping_tz():
    aware = datetime(2024, 1, 2, 9, 30, tzinfo=MARKET_TZ)  # 14:30 UTC

    assert as_naive_timestamp(aware) == pd.Timestamp("2024-01-02 14:30")
    assert as_naive_timestamp(aware.astimezone(UTC)) == pd.Timestamp("2024-01-02 14:30")


def test_as_naive_timestamp_passes_naive_values_through():
    assert as_naive_timestamp(datetime(2024, 1, 2, 9, 30)) == pd.Timestamp("2024-01-02 09:30")


def test_atomic_write_text_replaces_contents_and_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old")

    atomic_write_text(target, "new")

    assert target.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_capped_backoff_doubles_then_caps():
    assert [capped_backoff(a, 1.5, 10.0) for a in range(4)] == [1.5, 3.0, 6.0, 10.0]


def test_full_jitter_delay_prefers_retry_after():
    assert full_jitter_delay(0, 1.5, 10.0, retry_after=7.0) == 7.0


def test_full_jitter_delay_draws_from_zero_to_capped_backoff():
    rng = random.Random(0)
    delays = [full_jitter_delay(3, 1.5, 10.0, rng=rng) for _ in range(200)]

    assert all(0.0 <= d <= 10.0 for d in delays)
    assert max(delays) > 5.0  # the cap, not the base, bounds the draw


def test_data_path_follows_the_current_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ARGUS_DATA_DIR", str(tmp_path))

    assert settings.data_path("decisions.jsonl") == Path(tmp_path) / "decisions.jsonl"


def test_web_concurrency_reads_the_environment_at_call_time(monkeypatch):
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    assert web_concurrency() is None

    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    assert web_concurrency() == "4"


def test_initial_state_fills_every_key_and_leaves_node_outputs_empty():
    state = initial_state(
        ["AAPL", "MSFT"],
        session_states={"AAPL": {}},
        total_wealth=1000.0,
        invest_pct=0.5,
        risk_tolerance="MODERATE",
    )

    assert set(state) == set(ARGUSState.__annotations__)
    assert state["ticker"] == "AAPL"
    assert state["universe"] == ["AAPL", "MSFT"]
    assert state["as_of"] is None
    assert state["session_seed"] is None
    assert state["backtest_mode"] is False
    assert state["macro_context"] is None
    assert state["portfolio_allocation"] is None
    assert state["errors"] == []
    assert state["decisions"] == []
