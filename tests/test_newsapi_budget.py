"""Tests for argus/data/newsapi_budget.py."""

import json
import logging
import multiprocessing
from datetime import UTC, datetime, timedelta

from argus.data import newsapi_budget
from argus.data.newsapi_budget import NewsApiBudget


def test_newsapi_budget_exhausts_after_daily_limit():
    budget = NewsApiBudget(daily_limit=2)
    assert budget.try_reserve() is True
    assert budget.try_reserve() is True
    assert budget.try_reserve() is False


def test_newsapi_budget_resets_on_new_day_within_one_instance(monkeypatch):

    class _FixedDatetime(datetime):
        _now = datetime(2026, 1, 1, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls._now

    monkeypatch.setattr(newsapi_budget, "datetime", _FixedDatetime)
    budget = NewsApiBudget(daily_limit=1)
    assert budget.try_reserve() is True
    assert budget.try_reserve() is False

    _FixedDatetime._now = datetime(2026, 1, 2, tzinfo=UTC)
    assert budget.try_reserve() is True


def test_newsapi_budget_fails_closed_on_wrong_typed_persisted_fields():
    budget = NewsApiBudget(daily_limit=1)
    today = datetime.now(UTC).date()
    budget._persist_path().write_text(json.dumps({"date": today.isoformat(), "requests_today": None}))

    assert budget.try_reserve() is False


def test_newsapi_budget_fails_closed_on_a_corrupt_file_and_heals_it(caplog):
    budget = NewsApiBudget(daily_limit=3)
    path = budget._persist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    with caplog.at_level(logging.ERROR, logger="argus.data.newsapi_budget"):
        assert budget.try_reserve() is False

    assert any("corrupt" in r.message for r in caplog.records)
    assert json.loads(path.read_text()) == {
        "date": datetime.now(UTC).date().isoformat(),
        "requests_today": 3,
    }
    assert budget.try_reserve() is False


def test_newsapi_budget_corrupt_file_clears_on_the_next_utc_day(monkeypatch):
    budget = NewsApiBudget(daily_limit=1)
    path = budget._persist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    assert budget.try_reserve() is False

    class _Tomorrow(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(days=1)

    monkeypatch.setattr(newsapi_budget, "datetime", _Tomorrow)
    assert budget.try_reserve() is True


def test_newsapi_budget_ignores_a_stale_persisted_date():
    budget = NewsApiBudget(daily_limit=2)
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    budget._persist_path().write_text(json.dumps({"date": yesterday.isoformat(), "requests_today": 2}))

    assert budget.try_reserve() is True
    assert budget.try_reserve() is True
    assert budget.try_reserve() is False


def test_newsapi_budget_survives_a_fresh_process():
    first = NewsApiBudget(daily_limit=2)
    assert first.try_reserve() is True

    second = NewsApiBudget(daily_limit=2)
    assert second.try_reserve() is True
    assert second.try_reserve() is False


def _reserve_n_times(data_dir: str, daily_limit: int, attempts: int, result_queue: multiprocessing.Queue[int]) -> None:
    """Reserve repeatedly in a separate process for cross-process budget checks."""
    newsapi_budget.settings.ARGUS_DATA_DIR = data_dir
    budget = NewsApiBudget(daily_limit=daily_limit)
    result_queue.put(sum(1 for _ in range(attempts) if budget.try_reserve()))


def test_newsapi_budget_enforces_cap_across_processes(tmp_path):
    daily_limit = 20
    ctx = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue[int] = ctx.Queue()
    procs = [
        ctx.Process(target=_reserve_n_times, args=(str(tmp_path), daily_limit, daily_limit, result_queue))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    total_reserved = sum(result_queue.get() for _ in procs)
    assert total_reserved == daily_limit


def _raise_os_error(*_args: object, **_kwargs: object) -> None:
    raise OSError("simulated disk error")


def test_newsapi_budget_degrades_on_disk_error(monkeypatch):
    monkeypatch.setattr(newsapi_budget, "open", _raise_os_error, raising=False)

    budget = NewsApiBudget(daily_limit=5)
    assert budget.try_reserve() is False
