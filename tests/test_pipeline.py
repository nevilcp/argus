"""
Tests for the MFT Data Pipeline.
"""

import asyncio
import dataclasses
import importlib
import math
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import argus.data.pipeline as pipeline_module
from argus.config import settings
from argus.data.cache import OHLCVBuffer
from argus.data.pipeline import (
    _FETCH_INTERVAL,
    MFTDataPipeline,
    _bars_per_day,
    _derive_buffer_size,
    _finite_or_none,
    _parse_interval_minutes,
    _required_indicator_bars,
    _required_raw_bars,
    _resample_ohlcv,
    _return_since,
    max_bar_age_seconds,
    session_state_ttl_seconds,
)
from argus.params import SYSTEM
from tests.helpers.candles import dst_straddling_candles, et_intraday_candles


def test_register_tickers():
    """Registering tickers already present in the pipeline is a no-op for those tickers."""
    pipeline = MFTDataPipeline([])
    pipeline.register_tickers(["AAPL", "MSFT"])
    assert "AAPL" in pipeline.tickers
    assert "MSFT" in pipeline.tickers
    assert len(pipeline.tickers) == 2

    pipeline.register_tickers(["AAPL", "TSLA"])
    assert len(pipeline.tickers) == 3
    assert "TSLA" in pipeline.tickers


def test_register_tickers_rejects_malformed_symbols():
    """A malformed ticker is dropped; well-formed siblings in the same call still register."""
    pipeline = MFTDataPipeline([])
    pipeline.register_tickers(["AAPL", "", "'; DROP TABLE ohlcv; --", "../../etc/passwd", "aapl"])
    assert pipeline.tickers == ["AAPL"]


def test_register_tickers_caps_the_tracked_universe(monkeypatch):
    """Registrations beyond SYSTEM.max_tracked_tickers are dropped, not appended (API-1)."""
    monkeypatch.setattr(pipeline_module, "SYSTEM", dataclasses.replace(SYSTEM, max_tracked_tickers=3))
    pipeline = MFTDataPipeline([])
    pipeline.register_tickers(["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"])
    assert pipeline.tickers == ["AAPL", "MSFT", "TSLA"]

    pipeline.register_tickers(["NVDA"])
    assert pipeline.tickers == ["AAPL", "MSFT", "TSLA"]


def test_init_routes_initial_universe_through_register_tickers(monkeypatch):
    """A malformed ticker in the constructor's initial universe is dropped, not stored raw."""
    monkeypatch.setattr(pipeline_module, "SYSTEM", dataclasses.replace(SYSTEM, max_tracked_tickers=2))
    pipeline = MFTDataPipeline(["AAPL", "not-a-ticker!", "MSFT", "TSLA"])
    assert pipeline.tickers == ["AAPL", "MSFT"]


def test_compress_candles():
    """Compressing candles returns the full set of pandas-ta derived indicator keys."""
    pipeline = MFTDataPipeline([])

    # 400 minutes covers the 390-bar (1 trading day) momentum lookback at 1m resolution
    dates = pd.date_range("2024-01-01 09:30", periods=400, freq="1min")

    # Monotonic close series so RSI has a known, deterministic value to assert on
    df = pd.DataFrame({
        "open": range(400),
        "high": range(1, 401),
        "low": range(400),
        "close": range(400),
        "volume": [1000] * 400,
    }, index=dates)

    result = pipeline._compress_candles(df)

    assert "rsi_14" in result
    assert "macd_histogram" in result
    assert "bb_percent_b" in result
    assert "atr_pct" in result
    assert "adx_14" in result
    assert "vwap_distance" in result
    assert "volume_ratio" in result
    assert "momentum_30m" in result
    assert "momentum_1d" in result
    assert "close" in result
    assert "timestamp" in result

    assert result["close"] == 399.0
    # A monotonically rising close series saturates RSI at 100
    assert result["rsi_14"] == 100.0
    # 400 1m bars resample to ~80 5m bars, well clear of the 34-bar MACD warmup floor
    assert result["macd_histogram"] is not None


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
    """An interval string outside the '<n>m' / '<n>h' shape raises rather than silently misparsing."""
    with pytest.raises(ValueError):
        _parse_interval_minutes("daily")


def test_derive_buffer_size_matches_fetch_period():
    """Buffer size holds a full fetch period's candles at the given interval, plus a day and a bar of slack."""
    assert _bars_per_day(1) == 390
    assert _bars_per_day(5) == 78
    assert _derive_buffer_size(1) == 1173
    assert _derive_buffer_size(5) == 237


def test_pipeline_derives_buffer_size_from_interval():
    """MFTDataPipeline sizes its buffer from the candle interval when CANDLE_BUFFER_SIZE isn't overridden."""
    pipeline_1m = MFTDataPipeline([], interval="1m")
    assert pipeline_1m.buffer._buffer_size == 1173

    pipeline_5m = MFTDataPipeline([], interval="5m")
    assert pipeline_5m.buffer._buffer_size == 237


def test_resample_ohlcv_aggregates_correctly():
    """Resampling combines OHLCV bars with the standard first/max/min/last/sum rules."""
    dates = pd.date_range("2024-01-01 09:30", periods=10, freq="1min")
    df = pd.DataFrame({
        "open": range(10),
        "high": [x + 1 for x in range(10)],
        "low": range(10),
        "close": range(10),
        "volume": [100] * 10,
    }, index=dates)

    resampled = _resample_ohlcv(df, interval_minutes=1, target_minutes=5)

    assert len(resampled) == 2
    assert resampled["open"].iloc[0] == 0
    assert resampled["close"].iloc[0] == 4
    assert resampled["high"].iloc[0] == 5
    assert resampled["low"].iloc[0] == 0
    assert resampled["volume"].iloc[0] == 500


def test_resample_ohlcv_is_a_noop_when_target_not_coarser():
    """Resampling to a resolution no coarser than the input returns the input unchanged."""
    dates = pd.date_range("2024-01-01", periods=5, freq="5min")
    df = pd.DataFrame({
        "open": range(5), "high": range(5), "low": range(5), "close": range(5), "volume": [1] * 5,
    }, index=dates)

    result = _resample_ohlcv(df, interval_minutes=5, target_minutes=5)

    pd.testing.assert_frame_equal(result, df)


def test_momentum_windows_scale_with_interval():
    """momentum_30m locates the bar exactly 30 minutes back at both 1m and 5m resolution."""
    dates_1m = pd.date_range("2024-01-01 09:30", periods=400, freq="1min")
    closes_1m = list(range(400))
    df_1m = pd.DataFrame({
        "open": closes_1m, "high": [c + 1 for c in closes_1m], "low": closes_1m,
        "close": closes_1m, "volume": [1000] * 400,
    }, index=dates_1m)
    result_1m = MFTDataPipeline([], interval="1m")._compress_candles(df_1m)
    # Last bar is 09:30 + 399min = 16:09; 30 minutes back lands exactly on bar 369
    assert result_1m["momentum_30m"] == pytest.approx((399 / 369) - 1.0, abs=1e-6)
    # Every bar falls on 2024-01-01, so momentum_1d has no strictly-earlier session to reach
    assert result_1m["momentum_1d"] is None

    dates_5m = pd.date_range("2024-01-01 09:30", periods=100, freq="5min")
    closes_5m = list(range(100))
    df_5m = pd.DataFrame({
        "open": closes_5m, "high": [c + 1 for c in closes_5m], "low": closes_5m,
        "close": closes_5m, "volume": [1000] * 100,
    }, index=dates_5m)
    result_5m = MFTDataPipeline([], interval="5m")._compress_candles(df_5m)
    # Last bar is 09:30 + 99*5min = 17:45; 30 minutes back lands exactly on bar 93
    assert result_5m["momentum_30m"] == pytest.approx((99 / 93) - 1.0, abs=1e-6)
    assert result_5m["momentum_1d"] is None


def test_finite_or_none():
    """Passes a finite value through unchanged; collapses None/NaN/inf to None."""
    assert _finite_or_none(1.5) == 1.5
    assert _finite_or_none(None) is None
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(float("-inf")) is None


def test_required_indicator_bars_is_pinned_to_macd_warmup():
    """The resampled-bar readiness floor is pinned to MACD(12,26,9)'s measured warmup."""
    assert _required_indicator_bars() == 34


def test_required_raw_bars_matches_the_issue_table():
    """Raw-bar pre-filter: the larger of the indicator-resample floor and one day's momentum lookback."""
    assert _required_raw_bars(1) == 391
    assert _required_raw_bars(5) == 79
    assert _required_raw_bars(15) == 34


def test_compress_candles_readiness_cliff_at_macd_warmup_floor():
    """33 resampled bars publish nothing; 34 publishes a real (non-None) MACD histogram."""
    pipeline = MFTDataPipeline([], interval="5m")

    dates_33 = pd.date_range("2024-01-01 09:30", periods=33, freq="5min")
    df_33 = pd.DataFrame({
        "open": range(33), "high": [c + 1 for c in range(33)], "low": range(33),
        "close": range(33), "volume": [1000] * 33,
    }, index=dates_33)
    assert pipeline._compress_candles(df_33) is None

    dates_34 = pd.date_range("2024-01-01 09:30", periods=34, freq="5min")
    df_34 = pd.DataFrame({
        "open": range(34), "high": [c + 1 for c in range(34)], "low": range(34),
        "close": range(34), "volume": [1000] * 34,
    }, index=dates_34)
    result_34 = pipeline._compress_candles(df_34)
    assert result_34 is not None
    assert result_34["macd_histogram"] is not None


def test_compress_candles_never_emits_nan():
    """Every float field in a compressed dict is a real number or None, never NaN."""
    pipeline = MFTDataPipeline([], interval="1m")
    dates = pd.date_range("2024-01-01 09:30", periods=400, freq="1min")
    df = pd.DataFrame({
        "open": range(400), "high": [c + 1 for c in range(400)], "low": range(400),
        "close": range(400), "volume": [1000] * 400,
    }, index=dates)
    result = pipeline._compress_candles(df)
    assert result is not None
    for value in result.values():
        if isinstance(value, float):
            assert not math.isnan(value)


def test_return_since_momentum_1d_none_without_a_prior_session():
    """momentum_1d's lookup returns None when the series holds only one calendar day."""
    close_s = et_intraday_candles(50)["close"]
    assert _return_since(close_s, pd.Timedelta(days=1), same_session=False) is None


def test_return_since_momentum_30m_does_not_reach_into_the_prior_session():
    """At 09:34 with only 5 bars printed today, the 30-minute lookup doesn't fall back to yesterday's close."""
    day1 = et_intraday_candles(390, start="2024-01-02 09:30")
    day2 = et_intraday_candles(5, start="2024-01-03 09:30")
    combined_close = pd.concat([day1["close"], day2["close"]])
    assert _return_since(combined_close, pd.Timedelta(minutes=30), same_session=True) is None


@given(drop=st.integers(min_value=0, max_value=10))
def test_momentum_1d_is_stable_under_dropping_up_to_10_oldest_bars(drop: int):
    """Locating momentum_1d by time is unaffected by pruning the buffer's oldest rows.

    Args:
        drop: Number of oldest bars removed from the series before re-checking.
    """
    day1 = et_intraday_candles(390, start="2024-01-02 09:30")
    day2 = et_intraday_candles(50, start="2024-01-03 09:30")
    close_s = pd.concat([day1["close"], day2["close"]])

    baseline = _return_since(close_s, pd.Timedelta(days=1), same_session=False)
    trimmed = _return_since(close_s.iloc[drop:], pd.Timedelta(days=1), same_session=False)

    assert baseline is not None
    assert trimmed == baseline


def test_session_state_ttl_seconds_tracks_decision_interval(monkeypatch):
    """The write-age TTL is derived from the configured decision interval, not hardcoded."""
    monkeypatch.setattr(settings, "MFT_DECISION_INTERVAL_SECONDS", 1800)
    assert session_state_ttl_seconds() == 1800 + _FETCH_INTERVAL + SYSTEM.freshness_margin_seconds

    monkeypatch.setattr(settings, "MFT_DECISION_INTERVAL_SECONDS", 900)
    assert session_state_ttl_seconds() == 900 + _FETCH_INTERVAL + SYSTEM.freshness_margin_seconds


def test_max_bar_age_seconds_scales_with_interval():
    """The bar-age budget grows with the candle interval, matching the issue's 480s-at-1m figure."""
    assert max_bar_age_seconds(1) == _FETCH_INTERVAL + 2 * 1 * 60 + SYSTEM.freshness_margin_seconds
    assert max_bar_age_seconds(1) == 480
    assert max_bar_age_seconds(5) > max_bar_age_seconds(1)


@pytest.mark.asyncio
async def test_sweep_uses_a_constant_inter_request_gap_independent_of_universe_size(monkeypatch):
    """The pause between per-ticker fetches is SYSTEM.min_inter_request_seconds regardless of universe size."""

    async def fake_fetch_one(_ticker):
        return None

    small = MFTDataPipeline(["A", "B", "C"], interval="1m")
    monkeypatch.setattr(small, "_fetch_one_ticker", fake_fetch_one)
    small_start = time.monotonic()
    await small._sweep_once()
    small_elapsed = time.monotonic() - small_start

    big = MFTDataPipeline([f"T{chr(65 + i)}" for i in range(20)], interval="1m")
    monkeypatch.setattr(big, "_fetch_one_ticker", fake_fetch_one)
    big_start = time.monotonic()
    await big._sweep_once()
    big_elapsed = time.monotonic() - big_start

    # 2 gaps for 3 tickers, 19 gaps for 20 — a fixed per-gap cost, not a
    # fill-the-interval spread that would instead keep total elapsed time constant
    assert small_elapsed == pytest.approx(2 * SYSTEM.min_inter_request_seconds, abs=0.05)
    assert big_elapsed == pytest.approx(19 * SYSTEM.min_inter_request_seconds, abs=0.1)


@pytest.mark.asyncio
async def test_fetch_loop_sleeps_the_remainder_of_the_interval_not_a_full_one(monkeypatch):
    """A fetch_loop iteration's wait is shortened by however long the sweep itself took."""
    pipeline = MFTDataPipeline([], interval="1m")
    waits = []

    async def fake_sweep():
        await asyncio.sleep(0.05)

    async def fake_wait_or_stop(timeout):
        waits.append(timeout)
        pipeline.running = False

    monkeypatch.setattr(pipeline, "_is_market_hours", lambda: True)
    monkeypatch.setattr(pipeline, "_sweep_once", fake_sweep)
    monkeypatch.setattr(pipeline, "_wait_or_stop", fake_wait_or_stop)
    pipeline.running = True

    await pipeline._fetch_loop()

    assert len(waits) == 1
    assert waits[0] < _FETCH_INTERVAL
    assert waits[0] == pytest.approx(_FETCH_INTERVAL - 0.05, abs=0.05)


@pytest.mark.asyncio
async def test_fetch_loop_cadence_holds_across_a_slow_sweep(monkeypatch):
    """The measured gap between sweep starts tracks _FETCH_INTERVAL, not sweep_duration + _FETCH_INTERVAL."""
    monkeypatch.setattr(pipeline_module, "_FETCH_INTERVAL", 0.2)
    pipeline = MFTDataPipeline([], interval="1m")
    monkeypatch.setattr(pipeline, "_is_market_hours", lambda: True)

    starts = []

    async def fake_sweep():
        starts.append(time.monotonic())
        await asyncio.sleep(0.05)

    monkeypatch.setattr(pipeline, "_sweep_once", fake_sweep)

    pipeline.running = True
    task = asyncio.create_task(pipeline._fetch_loop())
    await asyncio.sleep(0.55)
    await pipeline.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert len(starts) >= 2
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(gap == pytest.approx(0.2, abs=0.05) for gap in gaps)


def test_default_db_path_uses_isolated_data_dir(tmp_path):
    """A pipeline built without an explicit db_path never reaches the production buffer."""
    pipeline = MFTDataPipeline(["AAPL"])
    assert Path(pipeline.buffer._db_path).parent == tmp_path
    assert Path(pipeline.buffer._db_path).parent == Path(settings.ARGUS_DATA_DIR)


def test_pandas_ta_import_failure_is_a_boot_failure(monkeypatch):
    """A broken pandas_ta install fails at module import time, not silently inside compress_all."""
    monkeypatch.setitem(sys.modules, "pandas_ta", None)
    try:
        with pytest.raises(ImportError):
            importlib.reload(pipeline_module)
    finally:
        monkeypatch.undo()
        importlib.reload(pipeline_module)


def test_et_intraday_candles_are_tz_aware_and_monotonic():
    """The shared candle builder produces a tz-aware, monotonically increasing ET frame."""
    df = et_intraday_candles(20)
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


def _seed_buffer(pipeline: MFTDataPipeline, ticker: str, n: int = 20) -> None:
    """Inserts `n` warm candles for `ticker` directly into a pipeline's buffer.

    Closes are shifted off zero: with n < the 30-bar momentum lookback,
    momentum_1d's index lands on the earliest bar, and et_intraday_candles'
    raw closes start at 0 — a zero denominator _compress_candles would raise on.
    """
    df = et_intraday_candles(n)
    df[["open", "high", "low", "close"]] += 100
    for idx, row in df.iterrows():
        pipeline.buffer.insert_candle(ticker, {"timestamp": idx, **row.to_dict()})


def _seed_two_session_buffer(
    pipeline: MFTDataPipeline, ticker: str, day1_bars: int = 390, day2_bars: int = 50
) -> None:
    """Inserts two calendar days of warm 1m candles for `ticker`.

    Clears both the raw-bar readiness pre-filter and momentum_1d's
    cross-session lookback (see argus/data/pipeline.py's `_return_since`),
    unlike `_seed_buffer`'s single-day fixture.
    """
    day1 = et_intraday_candles(day1_bars, start="2024-01-02 09:30")
    day2 = et_intraday_candles(day2_bars, start="2024-01-03 09:30")
    combined = pd.concat([day1, day2])
    combined[["open", "high", "low", "close"]] += 100
    for idx, row in combined.iterrows():
        pipeline.buffer.insert_candle(ticker, {"timestamp": idx, **row.to_dict()})


def test_compress_all_skips_a_ticker_whose_get_candles_raises(monkeypatch):
    """compress_all catches a get_candles failure for one ticker and still returns the rest."""
    pipeline = MFTDataPipeline(["BAD", "GOOD"], interval="1m")
    _seed_two_session_buffer(pipeline, "GOOD")

    real_get_candles = pipeline.buffer.get_candles

    def flaky_get_candles(ticker):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return real_get_candles(ticker)

    monkeypatch.setattr(pipeline.buffer, "get_all_tickers", lambda: ["BAD", "GOOD"])
    monkeypatch.setattr(pipeline.buffer, "get_candles", flaky_get_candles)

    states = pipeline.compress_all()

    assert "GOOD" in states
    assert "BAD" not in states


def test_compress_all_ignores_an_untracked_ticker(monkeypatch):
    """compress_all only considers tickers still in self.tickers (API-10)."""
    pipeline = MFTDataPipeline(["GOOD"], interval="1m")
    _seed_two_session_buffer(pipeline, "GOOD")
    _seed_two_session_buffer(pipeline, "ORPHAN")

    states = pipeline.compress_all()

    assert "GOOD" in states
    assert "ORPHAN" not in states


def test_compress_all_prunes_untracked_buffer_rows():
    """compress_all deletes buffered rows for a ticker no longer in self.tickers (API-9)."""
    pipeline = MFTDataPipeline(["GOOD"], interval="1m")
    _seed_two_session_buffer(pipeline, "GOOD")
    _seed_two_session_buffer(pipeline, "ORPHAN")

    pipeline.compress_all()

    assert "ORPHAN" not in pipeline.buffer.get_all_tickers()
    assert "GOOD" in pipeline.buffer.get_all_tickers()


def test_compress_all_skips_a_ticker_below_the_readiness_floor():
    """compress_all omits a ticker whose buffered bars don't clear the resampled MACD warmup floor."""
    pipeline = MFTDataPipeline(["COLD"], interval="1m")
    # 20 raw bars clear OHLCVBuffer's own 14-row floor but fall far short of
    # _required_raw_bars(1) == 391
    _seed_buffer(pipeline, "COLD", n=20)

    states = pipeline.compress_all()

    assert "COLD" not in states


def test_buffer_warm_skips_a_ticker_whose_get_candles_raises(monkeypatch):
    """_buffer_warm doesn't let one ticker's get_candles failure hide a genuinely warm ticker."""
    pipeline = MFTDataPipeline([], interval="1m")
    _seed_buffer(pipeline, "GOOD")

    real_get_candles = pipeline.buffer.get_candles

    def flaky_get_candles(ticker):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return real_get_candles(ticker)

    monkeypatch.setattr(pipeline.buffer, "get_all_tickers", lambda: ["BAD", "GOOD"])
    monkeypatch.setattr(pipeline.buffer, "get_candles", flaky_get_candles)

    assert pipeline._buffer_warm() is True


@pytest.mark.asyncio
async def test_start_cancels_the_sibling_loop_when_one_dies(monkeypatch):
    """If one loop raises out of start()'s gather, the other is cancelled rather than orphaned."""
    pipeline = MFTDataPipeline([], interval="1m")
    session_loop_cancelled = asyncio.Event()

    async def dying_fetch_loop():
        raise RuntimeError("simulated crash")

    async def long_session_loop(callback):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            session_loop_cancelled.set()
            raise

    monkeypatch.setattr(pipeline, "_fetch_loop", dying_fetch_loop)
    monkeypatch.setattr(pipeline, "_session_loop", long_session_loop)

    with pytest.raises(RuntimeError):
        await pipeline.start(on_session_ready=lambda states: None)

    assert session_loop_cancelled.is_set()


@pytest.mark.asyncio
async def test_stop_ends_both_loops_promptly():
    """stop() wakes both loops immediately rather than waiting out their full interval."""
    pipeline = MFTDataPipeline([], interval="1m")
    task = asyncio.create_task(pipeline.start(on_session_ready=lambda states: None))
    await asyncio.sleep(0.05)

    await pipeline.stop()

    # Without the stop_event-based wait, this would block for up to
    # _WARMUP_POLL_INTERVAL/_FETCH_INTERVAL seconds instead of returning almost immediately
    await asyncio.wait_for(task, timeout=2.0)


def test_fetch_and_insert_bulk_inserts_in_one_call(monkeypatch):
    """`_fetch_and_insert` hands the whole fetched frame to insert_candles in one call, not per-row."""
    pipeline = MFTDataPipeline([], interval="1m")
    df = et_intraday_candles(5)
    df[["open", "high", "low", "close"]] += 100

    monkeypatch.setattr(
        pipeline_module, "fetch_ohlcv_intraday", lambda ticker, interval, period: df
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


def test_fetch_and_insert_returns_zero_on_an_empty_fetch(monkeypatch):
    """An empty fetch result reports zero candles rather than raising."""
    pipeline = MFTDataPipeline([], interval="1m")
    monkeypatch.setattr(
        pipeline_module,
        "fetch_ohlcv_intraday",
        lambda ticker, interval, period: pd.DataFrame(),
    )

    n_candles, latest_close = pipeline._fetch_and_insert("AAPL")

    assert (n_candles, latest_close) == (0, 0.0)
