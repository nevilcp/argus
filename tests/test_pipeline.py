import asyncio
import dataclasses
import itertools
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

import argus.data.pipeline as pipeline_module
from argus.config import settings
from argus.data.cache import OHLCVBuffer
from argus.data.indicators import bars_per_day, required_raw_bars
from argus.data.pipeline import (
    _FETCH_PERIOD,
    _STEADY_STATE_FETCH_PERIOD,
    MFTDataPipeline,
    _derive_buffer_size,
    _parse_interval_minutes,
)
from argus.infra.clock import now_market
from argus.params import SYSTEM
from tests.helpers.candles import ET, dst_straddling_candles, ohlcv_frame
from tests.helpers.market_data import FakeMarketData


def _new_pipeline(tickers: list[str], market_data=None, **kwargs) -> MFTDataPipeline:
    """Builds a pipeline over a FakeMarketData that has no intraday response configured."""
    return MFTDataPipeline(tickers, market_data or FakeMarketData(), **kwargs)


def _override_system(monkeypatch, **fields) -> None:
    """Swaps pipeline_module's SYSTEM params for a copy carrying `fields`."""
    monkeypatch.setattr(pipeline_module, "SYSTEM", dataclasses.replace(SYSTEM, **fields))


def _insert_warm_candles(pipeline: MFTDataPipeline, ticker: str, df: pd.DataFrame) -> None:
    """Inserts every bar of `df` into `pipeline`'s buffer, with prices shifted off zero.

    et_intraday_candles' raw closes start at 0, which compress_candles would
    divide by whenever a momentum lookback lands on the earliest bar.
    """
    df[["open", "high", "low", "close"]] += 100
    for idx, row in df.iterrows():
        pipeline.buffer.insert_candle(ticker, {"timestamp": idx, **row.to_dict()})


def _seed_buffer(pipeline: MFTDataPipeline, ticker: str, n: int = 20) -> None:
    """Inserts `n` warm candles for `ticker`, all within a single trading day."""
    _insert_warm_candles(pipeline, ticker, ohlcv_frame(n, tz=ET))


def _seed_two_session_buffer(
    pipeline: MFTDataPipeline, ticker: str, day1_bars: int = 390, day2_bars: int = 50
) -> None:
    """Inserts two calendar days of warm 1m candles for `ticker`.

    Clears both the raw-bar readiness pre-filter and momentum_1d's
    cross-session lookback (see argus/data/pipeline.py's `_return_since` in data/indicators.py),
    unlike `_seed_buffer`'s single-day fixture.
    """
    day1 = ohlcv_frame(day1_bars, start="2024-01-02 09:30", tz=ET)
    day2 = ohlcv_frame(day2_bars, start="2024-01-03 09:30", tz=ET)
    _insert_warm_candles(pipeline, ticker, pd.concat([day1, day2]))


def _break_get_candles_for_bad(monkeypatch, pipeline: MFTDataPipeline) -> None:
    """Points the buffer at tickers BAD and GOOD, with get_candles raising only for BAD."""
    real_get_candles = pipeline.buffer.get_candles

    def flaky_get_candles(ticker):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return real_get_candles(ticker)

    monkeypatch.setattr(pipeline.buffer, "get_all_tickers", lambda: ["BAD", "GOOD"])
    monkeypatch.setattr(pipeline.buffer, "get_candles", flaky_get_candles)


def _capture_fetch_period() -> tuple[FakeMarketData, dict]:
    """Builds a provider whose ohlcv_intraday records the `period` it is called with.

    Returns:
        The provider, and a dict that gains a "period" key once it has been called.
    """
    captured: dict = {}

    def fake_fetch(ticker, interval, period):
        captured["period"] = period
        return ohlcv_frame(5, tz=ET)

    return FakeMarketData(ohlcv_intraday=fake_fetch), captured


async def _noop_callback(states: dict) -> None:
    """A no-op on_session_ready callback for _fetch_loop tests that don't care about publication."""


def test_register_tickers():
    """Registering tickers already present in the pipeline is a no-op for those tickers."""
    pipeline = _new_pipeline([])
    pipeline.register_tickers(["AAPL", "MSFT"])
    assert "AAPL" in pipeline.tickers
    assert "MSFT" in pipeline.tickers
    assert len(pipeline.tickers) == 2

    pipeline.register_tickers(["AAPL", "TSLA"])
    assert len(pipeline.tickers) == 3
    assert "TSLA" in pipeline.tickers


def test_register_tickers_rejects_malformed_symbols():
    """A malformed ticker is dropped; well-formed siblings in the same call still register."""
    pipeline = _new_pipeline([])
    pipeline.register_tickers(["AAPL", "", "'; DROP TABLE ohlcv; --", "../../etc/passwd", "aapl"])
    assert pipeline.tickers == ["AAPL"]


def test_register_tickers_caps_the_tracked_universe(monkeypatch):
    """Registrations beyond SYSTEM.max_tracked_tickers are dropped, not appended."""
    _override_system(monkeypatch, max_tracked_tickers=3)
    pipeline = _new_pipeline([])
    pipeline.register_tickers(["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"])
    assert pipeline.tickers == ["AAPL", "MSFT", "TSLA"]

    pipeline.register_tickers(["NVDA"])
    assert pipeline.tickers == ["AAPL", "MSFT", "TSLA"]


def test_init_routes_initial_universe_through_register_tickers(monkeypatch):
    """A malformed ticker in the constructor's initial universe is dropped, not stored raw."""
    _override_system(monkeypatch, max_tracked_tickers=2)
    pipeline = _new_pipeline(["AAPL", "not-a-ticker!", "MSFT", "TSLA"])
    assert pipeline.tickers == ["AAPL", "MSFT"]


def test_register_tickers_refreshes_last_requested_at_for_already_tracked_tickers():
    """A repeat request for an already-tracked ticker still bumps its last_requested_at."""
    pipeline = _new_pipeline(["AAPL"])
    stale = now_market() - timedelta(days=10)
    pipeline.last_requested_at["AAPL"] = stale

    pipeline.register_tickers(["AAPL"])

    assert pipeline.last_requested_at["AAPL"] > stale


def test_evict_stale_tickers_drops_a_non_seed_ticker_unused_beyond_the_ttl(monkeypatch):
    """A ticker registered outside the pipeline's seed universe expires after the TTL."""
    _override_system(monkeypatch, tracked_ticker_ttl_seconds=60)
    pipeline = _new_pipeline(["AAPL"])
    pipeline.register_tickers(["MSFT"])
    pipeline.last_requested_at["MSFT"] = now_market() - timedelta(seconds=120)

    pipeline._evict_stale_tickers()

    assert pipeline.tickers == ["AAPL"]
    assert "MSFT" not in pipeline.last_requested_at


def test_evict_stale_tickers_exempts_the_seed_universe(monkeypatch):
    """The pipeline's own seed tickers survive the TTL even if never re-requested."""
    _override_system(monkeypatch, tracked_ticker_ttl_seconds=60)
    pipeline = _new_pipeline(["AAPL"])
    pipeline.last_requested_at["AAPL"] = now_market() - timedelta(seconds=120)

    pipeline._evict_stale_tickers()

    assert pipeline.tickers == ["AAPL"]


def test_evict_stale_tickers_keeps_a_recently_requested_ticker(monkeypatch):
    """A ticker requested within the TTL window is not evicted."""
    _override_system(monkeypatch, tracked_ticker_ttl_seconds=3600)
    pipeline = _new_pipeline(["AAPL"])
    pipeline.register_tickers(["MSFT"])

    pipeline._evict_stale_tickers()

    assert pipeline.tickers == ["AAPL", "MSFT"]


def test_ohlcv_buffer_insertion():
    """Inserting a candle at an existing timestamp overwrites it rather than appending."""
    buffer = OHLCVBuffer(db_path=":memory:", buffer_size=20, interval="1m")

    # 15 candles clears the buffer's 14-candle minimum before it returns data
    for i in range(15):
        buffer.insert_candle("AAPL", {"timestamp": f"2024-01-01T10:{i:02d}:00", "close": 150.0 + i})

    df = buffer.get_candles("AAPL")
    assert df is not None
    assert len(df) == 15
    assert df["close"].iloc[-1] == 164.0

    buffer.insert_candle("AAPL", {"timestamp": "2024-01-01T10:14:00", "close": 165.0})
    df2 = buffer.get_candles("AAPL")
    assert len(df2) == 15
    assert df2["close"].iloc[-1] == 165.0


def test_parse_interval_minutes():
    """yfinance interval strings convert to their minute equivalents."""
    assert _parse_interval_minutes("1m") == 1
    assert _parse_interval_minutes("5m") == 5
    assert _parse_interval_minutes("30m") == 30
    assert _parse_interval_minutes("1h") == 60


def test_parse_interval_minutes_rejects_unsupported_shape():
    """An interval string outside the shape raises rather than silently misparsing."""
    with pytest.raises(ValueError, match="unsupported interval string"):
        _parse_interval_minutes("daily")


def test_derive_buffer_size_matches_fetch_period():
    """Buffer size holds a full fetch period's candles at the interval, plus a day and bar."""
    assert bars_per_day(1) == 390
    assert bars_per_day(5) == 78
    assert _derive_buffer_size(1) == 1173
    assert _derive_buffer_size(5) == 237


def test_pipeline_derives_buffer_size_from_interval():
    """MFTDataPipeline sizes its buffer from the interval when not overridden."""
    pipeline_1m = _new_pipeline([], interval="1m")
    assert pipeline_1m.buffer._buffer_size == 1173

    pipeline_5m = _new_pipeline([], interval="5m")
    assert pipeline_5m.buffer._buffer_size == 237


async def test_sweep_uses_a_constant_inter_request_gap_independent_of_universe_size(monkeypatch):
    """Per-ticker fetch pause is constant regardless of how many tickers."""

    async def fake_fetch_one(_ticker):
        return None

    small = _new_pipeline(["A", "B", "C"], interval="1m")
    monkeypatch.setattr(small, "_fetch_one_ticker", fake_fetch_one)
    small_start = time.monotonic()
    await small._sweep_once()
    small_elapsed = time.monotonic() - small_start

    big = _new_pipeline([f"T{chr(65 + i)}" for i in range(20)], interval="1m")
    monkeypatch.setattr(big, "_fetch_one_ticker", fake_fetch_one)
    big_start = time.monotonic()
    await big._sweep_once()
    big_elapsed = time.monotonic() - big_start

    # 2 gaps for 3 tickers, 19 gaps for 20 — a fixed per-gap cost, not a
    # fill-the-interval spread that would instead keep total elapsed time constant
    assert small_elapsed == pytest.approx(2 * SYSTEM.min_inter_request_seconds, abs=0.05)
    assert big_elapsed == pytest.approx(19 * SYSTEM.min_inter_request_seconds, abs=0.1)


async def test_fetch_loop_sleeps_the_remainder_of_the_interval_not_a_full_one(monkeypatch):
    """A fetch_loop iteration's wait is shortened by however long the sweep itself took."""
    pipeline = _new_pipeline([], interval="1m")
    waits = []

    async def fake_sweep():
        await asyncio.sleep(0.05)

    async def fake_wait_or_stop(timeout):
        waits.append(timeout)
        pipeline.running = False

    monkeypatch.setattr(pipeline, "_is_market_hours", lambda: True)
    monkeypatch.setattr(pipeline, "_sweep_once", fake_sweep)
    monkeypatch.setattr(pipeline, "compress_all", lambda: {})
    monkeypatch.setattr(pipeline, "_wait_or_stop", fake_wait_or_stop)
    pipeline.running = True

    await pipeline._fetch_loop(_noop_callback)

    assert len(waits) == 1
    assert waits[0] < SYSTEM.mft_fetch_interval_seconds
    assert waits[0] == pytest.approx(SYSTEM.mft_fetch_interval_seconds - 0.05, abs=0.05)


async def test_fetch_loop_cadence_holds_across_a_slow_sweep(monkeypatch):
    """Sweep start gap tracks mft_fetch_interval_seconds, not sweep duration plus interval."""
    _override_system(monkeypatch, mft_fetch_interval_seconds=0.2)
    pipeline = _new_pipeline([], interval="1m")
    monkeypatch.setattr(pipeline, "_is_market_hours", lambda: True)
    monkeypatch.setattr(pipeline, "compress_all", lambda: {})

    starts = []

    async def fake_sweep():
        starts.append(time.monotonic())
        await asyncio.sleep(0.05)

    monkeypatch.setattr(pipeline, "_sweep_once", fake_sweep)

    pipeline.running = True
    task = asyncio.create_task(pipeline._fetch_loop(_noop_callback))
    await asyncio.sleep(0.55)
    await pipeline.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert len(starts) >= 2
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert all(gap == pytest.approx(0.2, abs=0.05) for gap in gaps)


async def test_fetch_loop_publishes_compressed_state_after_each_sweep():
    """Publication is merged into the fetch loop: one sweep, one compress, one callback."""
    pipeline = _new_pipeline(["AAPL"], interval="1m")
    _seed_two_session_buffer(pipeline, "AAPL")
    pipeline._is_market_hours = lambda: True
    received = []

    async def fake_sweep():
        return None

    pipeline._sweep_once = fake_sweep

    async def stop_after_one(states: dict) -> None:
        received.append(states)
        await pipeline.stop()

    pipeline.running = True
    await pipeline._fetch_loop(stop_after_one)

    assert len(received) == 1
    assert "AAPL" in received[0]


def test_default_db_path_uses_isolated_data_dir(tmp_path):
    """A pipeline built without an explicit db_path never reaches the production buffer."""
    pipeline = _new_pipeline(["AAPL"])
    assert Path(pipeline.buffer._db_path).parent == tmp_path
    assert Path(pipeline.buffer._db_path).parent == Path(settings.ARGUS_DATA_DIR)


def test_et_intraday_candles_are_tz_aware_and_monotonic():
    """The shared candle builder produces a tz-aware, monotonically increasing ET frame."""
    df = ohlcv_frame(20, tz=ET)
    assert len(df) == 20
    assert str(df.index.tz) == "America/New_York"
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 19.0


def test_dst_straddling_candles_are_monotonic_in_utc_despite_repeated_wall_clock():
    """The DST fixture's local hour repeats but its underlying UTC instants strictly increase."""
    df = dst_straddling_candles(n_per_side=6)
    assert len(df) == 12
    assert df.index.tz_convert("UTC").is_monotonic_increasing
    # The repeated 01:xx ET wall-clock hour is exactly the reproduced hazard
    assert (df.index.hour == 1).all()


def test_compress_all_skips_a_ticker_whose_get_candles_raises(monkeypatch):
    """compress_all catches a get_candles failure for one ticker and still returns the rest."""
    pipeline = _new_pipeline(["BAD", "GOOD"], interval="1m")
    _seed_two_session_buffer(pipeline, "GOOD")
    _break_get_candles_for_bad(monkeypatch, pipeline)

    states = pipeline.compress_all()

    assert "GOOD" in states
    assert "BAD" not in states


def test_compress_all_ignores_an_untracked_ticker(monkeypatch):
    """compress_all only considers tickers still in self.tickers."""
    pipeline = _new_pipeline(["GOOD"], interval="1m")
    _seed_two_session_buffer(pipeline, "GOOD")
    _seed_two_session_buffer(pipeline, "ORPHAN")

    states = pipeline.compress_all()

    assert "GOOD" in states
    assert "ORPHAN" not in states


def test_compress_all_prunes_untracked_buffer_rows():
    """compress_all deletes buffered rows for a ticker no longer in self.tickers."""
    pipeline = _new_pipeline(["GOOD"], interval="1m")
    _seed_two_session_buffer(pipeline, "GOOD")
    _seed_two_session_buffer(pipeline, "ORPHAN")

    pipeline.compress_all()

    assert "ORPHAN" not in pipeline.buffer.get_all_tickers()
    assert "GOOD" in pipeline.buffer.get_all_tickers()


def test_compress_all_skips_a_ticker_below_the_readiness_floor():
    """compress_all omits a ticker whose bars don't clear MACD warmup floor."""
    pipeline = _new_pipeline(["COLD"], interval="1m")
    # 20 raw bars clear OHLCVBuffer's own 14-row floor but fall far short of
    # required_raw_bars(1) == 391
    _seed_buffer(pipeline, "COLD", n=20)

    states = pipeline.compress_all()

    assert "COLD" not in states


async def test_start_propagates_a_fetch_loop_exception(monkeypatch):
    """start() surfaces _fetch_loop's exception rather than swallowing it (single loop now)."""
    pipeline = _new_pipeline([], interval="1m")

    async def dying_fetch_loop(on_session_ready):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(pipeline, "_fetch_loop", dying_fetch_loop)

    with pytest.raises(RuntimeError):
        await pipeline.start(on_session_ready=_noop_callback)


async def test_stop_ends_the_loop_promptly(monkeypatch):
    """stop() wakes the loop immediately rather than waiting out its full interval."""
    pipeline = _new_pipeline([], interval="1m")
    monkeypatch.setattr(pipeline, "_is_market_hours", lambda: False)
    task = asyncio.create_task(pipeline.start(on_session_ready=_noop_callback))
    await asyncio.sleep(0.05)

    await pipeline.stop()

    # Without the stop_event-based wait, this would block for up to
    # mft_fetch_interval_seconds instead of returning almost immediately
    await asyncio.wait_for(task, timeout=2.0)


async def test_start_after_stop_clears_the_stop_event(monkeypatch):
    """A restarted pipeline paces its sweeps again instead of inheriting the old stop signal."""
    pipeline = _new_pipeline([], interval="1m")
    await pipeline.stop()
    assert pipeline._stop_event.is_set()
    seen: list[bool] = []

    async def recording_fetch_loop(on_session_ready):
        seen.append(pipeline._stop_event.is_set())

    monkeypatch.setattr(pipeline, "_fetch_loop", recording_fetch_loop)

    await pipeline.start(on_session_ready=_noop_callback)

    assert seen == [False]


async def test_close_buffer_waits_out_an_in_flight_buffer_worker():
    """close_buffer blocks until a worker holding _buffer_op_lock releases it."""
    pipeline = _new_pipeline([], interval="1m")
    pipeline._buffer_op_lock.acquire()

    close_task = asyncio.create_task(pipeline.close_buffer())
    await asyncio.sleep(0.05)
    assert not close_task.done(), "close_buffer must not close the buffer while the lock is held"

    pipeline._buffer_op_lock.release()
    await asyncio.wait_for(close_task, timeout=2.0)


def test_fetch_and_insert_bulk_inserts_in_one_call(monkeypatch):
    """`_fetch_and_insert` passes the frame to insert_candles in one call."""
    df = ohlcv_frame(5, tz=ET)
    df[["open", "high", "low", "close"]] += 100
    pipeline = _new_pipeline(
        [], FakeMarketData(ohlcv_intraday=lambda ticker, interval, period: df), interval="1m"
    )
    calls = []
    monkeypatch.setattr(
        pipeline.buffer, "insert_candles", lambda ticker, candles: calls.append((ticker, candles))
    )

    n_candles, latest_close = pipeline._fetch_and_insert("AAPL")

    assert n_candles == 5
    assert latest_close == pytest.approx(float(df["close"].iloc[-1]))
    assert len(calls) == 1
    assert calls[0][0] == "AAPL"
    assert len(calls[0][1]) == 5


def test_fetch_and_insert_returns_zero_on_an_empty_fetch():
    """An empty fetch result reports zero candles rather than raising."""
    pipeline = _new_pipeline(
        [],
        FakeMarketData(ohlcv_intraday=lambda ticker, interval, period: pd.DataFrame()),
        interval="1m",
    )

    n_candles, latest_close = pipeline._fetch_and_insert("AAPL")

    assert (n_candles, latest_close) == (0, 0.0)


def test_fetch_and_insert_uses_the_full_period_for_a_cold_buffer():
    """A ticker with no buffered rows yet gets the full `_FETCH_PERIOD` fetch."""
    market_data, captured = _capture_fetch_period()
    pipeline = _new_pipeline([], market_data, interval="1m")

    pipeline._fetch_and_insert("AAPL")

    assert captured["period"] == _FETCH_PERIOD


def test_fetch_and_insert_uses_the_steady_state_period_for_a_warm_buffer(monkeypatch):
    """A ticker already holding an indicator-ready depth gets the short trailing fetch."""
    market_data, captured = _capture_fetch_period()
    pipeline = _new_pipeline([], market_data, interval="1m")
    monkeypatch.setattr(pipeline.buffer, "row_count", lambda ticker: required_raw_bars(1))

    pipeline._fetch_and_insert("AAPL")

    assert captured["period"] == _STEADY_STATE_FETCH_PERIOD
