# ADR 0003: 503 rather than fall back to daily-resolution data

**Status:** Accepted (retrospective).

## Context

`/analyze` (`api/main.py`) depends on the MFT pipeline's live intraday
session cache (`_live_session_cache`), populated every 30 minutes by
`_mft_session_callback`. That cache can legitimately be empty or stale: the
market is closed, a ticker was just registered and the pipeline hasn't
completed its first cycle, or an entry has aged past
`_SESSION_STATE_TTL_SECONDS`. Daily-resolution OHLCV for the same ticker is
almost always available from `yfinance`, so the pipeline could fall back to
computing indicators from daily bars instead of failing the request.

## Decision

`/analyze` does not fall back. If any requested ticker is missing from the
live cache or its entry has expired, the endpoint raises `HTTPException(503)`
(`api/main.py:206-219`) with a message distinguishing "market is closed" from
"cache not yet warm," and asks the caller to retry.

The reason is calibration, not availability: `TechnicalStatisticalAgent`'s
scoring functions (`_score_rsi`, `_score_macd`, `_score_bollinger`, `_score_
adx_amplified`, `_score_vwap`, `_score_momentum`) and their weights were
designed around 5-minute-bar-derived intraday indicators (RSI-14 over 5-minute
bars, VWAP distance, 30-minute momentum). A same-named indicator computed
from daily bars is a different statistical object with different variance
and a different signal-to-noise ratio. Serving it under the same field name
would produce a technical signal that is silently wrong in a way nothing
downstream could detect — the schema and the code would both look correct.

## Consequences

- ARGUS refuses to produce a live signal outside market hours or during the
  pipeline's ~30-minute warm-up window after a new ticker is registered.
  This is a real capability gap, not hidden behind a fallback.
- This is the same reasoning, applied prospectively, that produced the
  finding behind PR 7: `_simulate_session_state`'s backtest path *did*
  invent placeholder values for these same keys, which is exactly the
  fabrication this ADR argues against. PR 7 deletes that path rather than
  fix it, because Yahoo's intraday history is capped at 60 days
  (`5m`/`3mo` returns 0 rows — see issue #1) and there is no historical
  source that satisfies the same calibration requirement over a multi-year
  window. Where live traffic gets a 503, the backtest gets deletion; neither
  gets a quieter wrong answer.
