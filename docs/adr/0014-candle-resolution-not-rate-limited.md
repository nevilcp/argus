# ADR 0014: The candle interval was never rate-limited

**Status:** Accepted.

## Context

`data/pipeline.py` hardcoded `fetch_ohlcv_intraday(ticker, "5m", "2d")` behind
a comment attributing the choice to "yfinance's 5 req/min informal rate
ceiling." That ceiling is uncheckable — yfinance is an unofficial client with
no published limit at all (see ADR 0013's sibling finding for Groq; the same
verification pass covered yfinance) — and, measured directly against the
installed package, false as a description of what interval granularity
costs:

| interval / period | bars returned | HTTP requests |
|---|---|---|
| `1m` / `2d` | 459 | 1 |
| `5m` / `2d` | 92 | 1 |
| `30m` / `2d` | 16 | 1 |
| `1m` / `8d` | 2,799 | 1 |

Yahoo's chart endpoint returns the entire requested period in one response no
matter the interval, so finer granularity is free against any rate limit.
Exposure is driven only by (tickers) × (sweeps per hour), which `fetch_loop`
already paces independently.

Two real bugs had been hiding behind the invented constraint:

- `OHLCVBuffer(buffer_size=78)` was sized as "~1 trading day of 5m candles,"
  but a `5m`/`2d` fetch returns ~92 bars — about 14 were evicted on arrival,
  every fetch, forever.
- `momentum_1d` read back 79 bars into that 78-capacity buffer. It was
  always clamped to the oldest available bar, so it measured roughly 6.5
  hours while a docstring and `SentimentSignal.momentum_1d`'s field
  description both called it a day. It fed the signal blend with
  `momentum_1d_weight = 0.6` — the majority of the momentum term was
  measuring the wrong window.
- `momentum_30m` used a 7-bar lookback, which at 5m resolution is 35
  minutes, not 30 — a discrepancy the code's own comment already conceded
  without correcting.

Separately, `_BATCH_SIZE = 4` / `_INTER_REQUEST_SLEEP = 13` meant a 20-ticker
sweep spent 19 × 13 ≈ 247s sleeping before `_fetch_loop`'s own 300s idle —
candles actually refreshed every ~9.4 minutes against a name (`_FETCH_INTERVAL
= 300`) implying 5. A 24-ticker universe would have consumed the entire
interval in sleep alone, silently.

## Decision

**Fetch the finest interval yfinance offers, resample locally for whatever
resolution an indicator wants.** `MFT_CANDLE_INTERVAL` (`config.py`) defaults
to `"1m"`; `_fetch_one_ticker` reads it instead of a hardcoded string, and
`_resample_ohlcv` (`data/pipeline.py`) rolls raw bars up to
`_INDICATOR_RESAMPLE_MINUTES = 5` before RSI/MACD/BBands/ATR/ADX/VWAP/
volume-ratio see them — one request now yields every resolution those
indicators could want, and none of them run on raw 1-minute noise. Momentum
is computed on the raw (unresampled) series instead, since a point-to-point
return doesn't benefit from smoothing and resampling would only throw away
precision on exactly the number that needs it.

**Size the buffer and the momentum windows from the interval, not from a
number that happened to be right once.** `_derive_buffer_size` computes
`(390 // interval_minutes) * candle_buffer_days_retained` — 390 being the
minutes in a regular session, and the retained-days figure
(`SystemParams.candle_buffer_days_retained`, tagged `CONVENTION` now that it
has a stated reason: it matches `data/pipeline.py`'s own fetch period, so a
sweep's candles always fit without evicting same-day history). `config.py`'s
`CANDLE_BUFFER_SIZE` becomes `Optional[int] = None`, meaning "derive it,"
overridable only if a future interval choice needs a manual push.
`momentum_30m`/`momentum_1d` convert their windows from minutes
(30 minutes; one trading day = `_bars_per_day(interval_minutes)`) to bars at
whatever interval is active, so both windows mean what their names — and
`SentimentSignal`'s field descriptions — already claimed.

**Derive the sweep's pacing from its own deadline instead of a fixed
per-batch sleep.** `_inter_request_sleep(n_tickers)` splits
`_FETCH_INTERVAL` minus an estimated per-ticker fetch cost evenly across the
universe, so the sweep finishes within its own interval by construction and
adapts automatically as the universe grows — logging a warning rather than
silently degrading if the estimate says it can't. `_BATCH_SIZE` and
`_INTER_REQUEST_SLEEP` are deleted along with the comment attributing them to
an uncheckable rate ceiling; the real, statable reason for pacing at all is
that yfinance is unofficial and undocumented, so conservative self-pacing is
a deliberate choice, not a measured requirement.

## Consequences

- Candle resolution is now a choice with a real trade-off (RSI-14 on 5m bars
  vs. 1m vs. 15m) rather than the accidental consequence of a rate limit that
  never existed. Changing `MFT_CANDLE_INTERVAL` no longer risks silently
  starving the buffer, since buffer size and momentum windows both scale with
  it.
- `momentum_1d` and `momentum_30m` change value under this change, even at
  the same nominal 5m interval, because the buffer no longer evicts bars
  their old lookbacks needed. Any downstream calibration that implicitly
  depended on the old (wrong) windows should be re-checked; none currently
  exists per `params.py`'s own `CALIBRATED` category being empty.
- At `1m`/2-day retention the buffer holds ~780 rows per ticker (~15.6k rows
  across a 20-ticker universe) — trivial for the in-memory SQLite backing
  `OHLCVBuffer`.
- The sweep now has an explicit, named assumption
  (`_ESTIMATED_FETCH_SECONDS_PER_TICKER`) instead of an implicit one baked
  into unexplained constants; it is unmeasured and flagged as such, a
  candidate for replacement once real fetch-latency data exists.
