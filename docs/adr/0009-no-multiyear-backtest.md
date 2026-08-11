# ADR 0009: Why ARGUS cannot be backtested over multi-year windows

**Status:** Accepted.

## Context

`argus/backtesting/` contained a full walk-forward apparatus: `engine.py`
(a `backtrader`-based simulation loop), `walk_forward.py` (n-fold
out-of-sample validation), `phase1_calibration.py` (a 16-config grid search
over technical indicator weights), `phase2_validation.py` (a second
validation pass), `bias_auditor.py` (survivorship/look-ahead/data-quality
checks), and `pit_enforcer.py` (a runtime point-in-time guard).

Running it surfaced the root problem, not a bug in it: `_simulate_session_state`
(the function feeding `TechnicalStatisticalAgent` inside the backtest loop)
returned `{recent_prices, volume}` — two of the eight keys
`TechnicalStatisticalAgent.analyze` requires (`_REQUIRED_INDICATOR_KEYS` in
`argus/agents/technical.py`: `rsi_14`, `macd_histogram`, `bb_percent_b`,
`adx_14`, `vwap_distance`, `momentum_30m`, `momentum_1d`, `close`). Every
call returned `None`. The phase 1/2 calibration grid search then only ever
measured `_simulate_session_state`'s fixed output — all 16 weight configs
produced identical results, because the thing being "calibrated" never
varied.

This isn't fixable by writing a better `_simulate_session_state`. The eight
required keys are 5-minute-resolution technical indicators computed by
`argus/data/pipeline.py`'s `MFTDataPipeline` from live-streamed intraday
candles — the same architecture ARGUS runs in production. Reconstructing
them for an arbitrary historical date requires historical 5-minute OHLCV
data, and Yahoo Finance (`yfinance`, ARGUS's only price data source)
enforces a hard **60-day lookback window** on intraday intervals: a request
for `5m` data older than 60 days returns an empty frame, confirmed by
direct query (`5m/1mo` → 1,794 rows; `5m/3mo` → 0 rows; `5m/6mo` → 0 rows).
There is no historical 5-minute data to backtest against beyond ~60 days —
not a missing feature, a boundary on the data source itself.

The remaining option — fabricating `session_states` for older dates from
whatever daily data *is* available — is explicitly the thing ADR 0002
("return None rather than fabricate defaults") rules out. A multi-year
backtest built on fabricated intraday indicators would report numbers that
look like a real historical evaluation and are not one.

## Decision

Delete `engine.py`, `walk_forward.py`, `phase1_calibration.py`,
`phase2_validation.py`, `bias_auditor.py`, `pit_enforcer.py`, and the
`backtrader` dependency. `metrics.py` (Sharpe, Sortino, max drawdown, IC —
pure functions of a returns series, no data-fetching) is kept; PR 10's
pre-registered evaluation reuses it.

In their place, `argus/backtesting/replay.py` replays recorded fixture
*sessions* (the same `tests/fixtures/`-shaped snapshots ADR 0007/0008
introduced — one real point-in-time capture of `market_data/` +
`llm_responses/`) through the real, compiled `build_graph()`, in strict
order. Point-in-time correctness becomes **structural** instead of
runtime-enforced: each session's `FixtureMarketDataProvider` is scoped to
that session's own directory, so there is no code path by which a later
session's data could reach an earlier one — nothing plays the role
`PointInTimeEnforcer` used to play, because nothing needs to. The design
targets replaying a genuinely available window (~60 days of 5-minute MFT
snapshots × 30-minute decision cycles ≈ 780 decision points per ticker) —
today exactly **one** session exists (captured in PR 5), so
`scripts/replay_backtest.py` replays one. Scaling this up means running the
live MFT pipeline and capturing more sessions over time, not writing more
code; the module's job here is to prove the replay mechanism is correct,
not to pretend a 780-point history exists when it doesn't.

## Consequences

- No `/backtest` API endpoint. `api/main.py`'s `BacktestRequest` model,
  `_backtest_jobs` dict, and the `POST /backtest` / `GET /backtest/{job_id}`
  routes are removed along with the engine they called.
- `tests/test_integration.py::test_pit_enforcer_prevents_future_data` is
  deleted with `pit_enforcer.py`. No replacement test asserts point-in-time
  correctness at runtime, because there is no longer a runtime mechanism to
  test — the property now holds by construction in `replay.py`, and that
  construction is what `tests/test_backtesting.py` exercises instead.
- Phase 1/2 calibration and the bias auditor are gone with no replacement.
  Re-introducing calibration or bias auditing later needs the same
  real-data prerequisite this ADR describes — it cannot be a variant of
  what's being deleted here.
- PR 10's pre-registered evaluation (rank IC, hit-rate-with-dead-band) will
  run against whatever session history `replay_sessions()` is given at that
  point; the plan already commits to reporting confidence intervals and
  disclosing low statistical power at this sample size rather than treating
  a small replay window as a large one.
