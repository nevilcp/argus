"""Persisted drawdown halt restoration regressions."""

from __future__ import annotations

import threading
from unittest import mock

import argus.risk.kill_switch as kill_switch_module
from argus.config import settings
from argus.risk.kill_switch import KillSwitch, initialize_kill_switch
from tests.helpers.kill_switch import FETCH_VIX, make_ks, seed_persisted_halt


def test_halt_restores_from_a_persisted_dump(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed = seed_persisted_halt()

    halt_file = kill_switch_module._find_latest_halt_file()
    assert halt_file is not None

    restored = KillSwitch("MODERATE")
    restored._restore_halt_from_file(halt_file)

    assert restored.is_halted
    assert restored._halt_reason == seed._halt_reason


def test_halt_persisted_is_true_after_a_successful_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ARGUS_RUNS_DIR", str(tmp_path / "runs"))
    ks = make_ks("MODERATE", inception=100_000.0)
    ks._current_portfolio_value = 80_000.0

    with mock.patch(FETCH_VIX, return_value=15.0):
        ks._check()

    assert ks.is_halted
    assert ks.status.halt_persisted is True


def test_halt_stays_engaged_but_flags_an_unpersisted_dump(tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("")
    monkeypatch.setattr(settings, "ARGUS_RUNS_DIR", str(blocker / "runs"))
    ks = make_ks("MODERATE", inception=100_000.0)
    ks._current_portfolio_value = 80_000.0

    with (
        caplog.at_level("CRITICAL", logger="argus.risk.kill_switch"),
        mock.patch(FETCH_VIX, return_value=15.0),
    ):
        ks._check()

    assert ks.is_halted
    assert ks.status.halt_persisted is False
    assert any(r.levelname == "CRITICAL" and "persist" in r.message.lower() for r in caplog.records)

    ks.reset(100_000.0)
    assert ks.status.halt_persisted is True


def test_update_portfolio_value_advances_the_high_water_mark_under_the_state_lock():
    ks = make_ks("MODERATE", inception=100_000.0)

    with ks._state_lock:
        writer = threading.Thread(target=ks.update_portfolio_value, args=(150_000.0,))
        writer.start()
        writer.join(timeout=0.3)
        assert writer.is_alive(), "update must wait for the state lock"
        assert ks._high_water_mark == 100_000.0

    writer.join(timeout=5)
    assert ks._high_water_mark == 150_000.0


def test_concurrent_updates_never_lower_the_high_water_mark():
    ks = make_ks("MODERATE", inception=100_000.0)
    values = [100_000.0 + i for i in range(2_000)]

    threads = [threading.Thread(target=ks.update_portfolio_value, args=(v,)) for v in values]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ks._high_water_mark == max(values)


def test_halt_restores_from_a_corrupt_dump(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed_persisted_halt()
    halt_file = kill_switch_module._find_latest_halt_file()
    assert halt_file is not None
    halt_file.write_text("{not valid json")

    restored = KillSwitch("MODERATE")
    restored._restore_halt_from_file(halt_file)

    assert restored.is_halted


def test_halt_restores_from_a_truncated_dump(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed_persisted_halt()
    halt_file = kill_switch_module._find_latest_halt_file()
    assert halt_file is not None
    original = halt_file.read_text()
    halt_file.write_text(original[: len(original) // 2])

    restored = KillSwitch("MODERATE")
    restored._restore_halt_from_file(halt_file)

    assert restored.is_halted


def test_initialize_kill_switch_restores_halt_on_reinit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed = seed_persisted_halt()

    with mock.patch.object(KillSwitch, "start"):
        ks = initialize_kill_switch("MODERATE", portfolio_value=100_000.0)

    assert ks.is_halted
    assert ks._halt_reason == seed._halt_reason
