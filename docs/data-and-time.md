# Data readiness and time boundaries

The [system guide](system-guide.md) describes ownership and end-to-end flow.
These details explain why a stored candle, a published feature, and an admissible
analysis input are separate states.

## Candle ingestion and readiness

`MFTDataPipeline` fetches one-minute candles and resamples locally for technical
indicators. Fetch granularity does not change the number of chart requests.
Five-minute indicator bars give RSI and other rolling measures their intended
horizon instead of changing that horizon with the raw interval setting.

A cold ticker fetches the full configured history; a warm ticker fetches a
shorter overlapping window. SQLite timestamp upserts make that overlap safe.
Batch insertion commits once per fetched frame. An untimestamped batch takes
one clock sample and offsets each row by a microsecond so a coarse clock cannot
collapse several candles onto one primary key.

`required_raw_bars` is only a cheap prefilter. Gaps mean enough raw rows do not
guarantee enough resampled bars. `compress_candles` checks the resampled depth,
including the MACD histogram warmup at `slow + signal - 1` bars.
`missing_session_state_keys` then rejects absent, null, and non-finite indicators.
The timestamp key must exist but is not interpreted as a numeric indicator.
The technical agent uses the same readiness contract and omits unusable tickers.

Indicator column lookup uses parameter-independent prefixes because pandas-ta
includes its parameters in generated column names. Zero-volume and near-zero
volume-mean cases remain unavailable instead of becoming invented neutral values
or extreme division results. Momentum locates bars by elapsed time, preserving
the horizon across gaps and interval changes; conflicting directional horizons
cancel instead of creating a directional allocation signal.

## Universe lifetime and request pacing

Ticker registration validates shape and capacity at one shared boundary used by
constructors, HTTP requests, collectors, and CLI overrides. Repeated requests
refresh existing ticker activity. Seed tickers remain pinned because unattended
collection does not refresh their request timestamps. Expired non-seed tickers
leave both tracking maps and persistent candle storage.

Each sweep iterates a snapshot of the tracked universe. Pacing imposes a fixed
minimum gap between provider requests; it does not spread requests to occupy the
whole collection interval. The fetch loop publishes features after the same
completed sweep and sleeps only the remaining cadence budget. This keeps
publication freshness aligned with fetch cadence.

## Publication age and bar age

`LiveSessionCache` accepts the tracked universe as an argument instead of holding
a pipeline reference. Publishing evicts untracked tickers and bounds capacity.
Admission distinguishes missing features, stalled publication, and old candle
timestamps. A newly published old buffer after restart must still be rejected.
Missing, malformed, or naive bar timestamps fail closed as stale evidence.

The API checks market hours before age admission and registers requested tickers
only after earlier gates pass. Closed-market requests do not consume universe
slots. Status reporting exposes both ages without initiating analysis.

## Timestamp meanings

| Boundary | Treatment |
|---|---|
| Persisted decision and session timestamps | Naive local values from `infra.clock.now_local` |
| Market hours and publication clocks | Aware Eastern values using `MARKET_TZ` |
| Mixed-offset candle storage | Canonical UTC ISO timestamps, converted to Eastern before resampling |
| Checkpoint retention conversion | Convert aware timestamps to UTC before removing timezone information |
| Replay session capture | Convert aware capture times to naive Eastern wall time |
| Daily refresh and news budget dates | UTC dates |

These conversions serve different contracts. Dropping timezone information
without converting first can shift a cutoff by the original offset. Subtracting
two aware datetimes with the same `ZoneInfo` object can measure wall-clock
distance across DST; publication ages use epoch seconds, and reconciliation
scheduling compares UTC instants.

## Cache and provider failure policy

TTL caches require a positive capacity independently of expiry. On insertion,
expired entries leave first; the oldest survivor leaves when a new key would
exceed capacity. Daily bars keep both bounded history and refresh markers,
pruning tickers outside the requested universe. Their lazy construction honors
the current `ARGUS_DATA_DIR` instead of capturing a working directory at import.
Stored candle counts use SQL aggregation without applying indicator-readiness
floors or constructing full frames.

Sector caching is live-provider-only so separate fixture sessions cannot leak
sectors through a process-global cache. A successful response with no sector may
be cached; a transient failure is left uncached. Fundamental ratios remain
provider measurements: debt-to-equity percentages are normalized to decimals,
and Yahoo's return-on-assets value is explicitly only a ROIC proxy.

Data retries distinguish terminal credentials or missing resources from transient
rate limits and server failures. Unknown exceptions receive the bounded retry
budget. Wrapping a `DataFetchError` again would multiply attempts, so it propagates
immediately. Shared helpers honor a provider retry delay, otherwise use capped
full jitter.

Yahoo uses an anonymous cookie and crumb rather than an API key. An invalid-crumb
401 can recover after clearing both the in-memory handshake and the persisted
`curlCffi` cookie. Clearing only memory can reload the rejected cookie immediately.
Exhausted handshake retries have their own error type. Other providers' credential
401s remain terminal.

NewsAPI reserves its UTC daily allowance before making each request. A sidecar
file lock covers the whole read, check, increment, and atomic replacement cycle
across threads and processes sharing that file. A corrupt ledger counts as spent
for today; resetting it would bypass the allowance. Disk errors reject the
reservation with diagnostics. The article page uses the maximum supported size
so each scarce request carries more evidence. See [limitations](limitations.md)
for news delays and historical coverage.
