# Model transport, measurements, and enforcement

The model proposes judgments; the provider and deterministic code own measured
facts and constraints. [Limitations](limitations.md) describes the assumptions
behind these judgments and the evidence needed to evaluate them.

## Ports and attempt ownership

`MarketDataProvider` supplies market and economic data. `LLMClient` supplies one
text completion and its remaining capacity. Agents depend on these protocols;
`seams.py` implements live and captured-fixture adapters. A missing fixture ticker
raises instead of producing empty or zero measurements. Fixture model responses
are selected from prompt content rather than guessed call order and never use
the live governor.

`GroqLLMClient.complete` makes one governed invocation. SDK retries are disabled
so the shared decoder controls the attempt budget and every response can inform
quota accounting. The adapter classifies transient provider failures into
`RetryableTransportError`, carrying a server delay when present. Terminal errors
propagate immediately. The response hook reads headers without forcing streaming
bodies to buffer. Response content may be plain text or text-tagged content blocks.

`structured_output.decode` strips only outer Markdown fences, parses JSON, and
validates the requested Pydantic schema. Closed, JSON-tagged, and unterminated
outer fences are supported; fenced text inside a JSON value survives.

JSON failures, schema failures, and retryable transport failures share one bounded
attempt budget. Transport retries use provider hints or full jitter; content
failures use deterministic capped exponential backoff. With repair enabled, the
latest content error stays appended across a subsequent transport failure. A
transport error supplies no replacement content feedback. Terminal errors and
governor rejections propagate without restarting their already bounded waits.

`decode_or_degrade` records distinct parse/validation, rate-limit, and other API
failure diagnostics before returning `None`. Callers can identify the failed
boundary without substituting a fabricated signal or allocation.

## Quota reservation and reconciliation

The governor estimates prompt tokens from word count and template overhead, then
reserves the completion budget as well. A request larger than an entire minute
quota cannot be admitted by waiting for rollover. Minute waits are bounded;
observed daily exhaustion raises immediately instead of blocking for a day.

| Quota axis | Before headers | After headers |
|---|---|---|
| Requests per minute | Configured conservative bootstrap | Still bootstrap; no provider RPM header |
| Tokens per minute | Bootstrap estimate | Observed limit, remaining allowance, and reset |
| Requests per day | No invented daily allowance | Observed daily request allowance |

Request and token headers update independently. Go-style reset durations accept
composite units and milliseconds; matching `ms` before `m` avoids interpreting
milliseconds as minutes. Once a reported deadline passes, the local observer
assumes reset until the next response corrects it.

Successful calls reconcile actual tokens against local estimates without
subtracting from provider remaining-token headers again. Headers already describe
the post-call allowance and can differ from billed token counts because cached
tokens affect accounting. Failed invocations release their pre-flight reservation.
The governor is process-local; separate scheduled runners do not share its
observations even when they use the same account.

## Measured data and model judgments

Fundamental verdicts contain judgment fields rather than financial ratios.
Provider ratios are merged after decoding and still undergo final schema
validation. Ticker anonymization uses a deterministic short MD5 identifier scoped
to ticker and session; it is obfuscation, not cryptography. A zero session seed
is valid and must be checked with `is not None`. Current Yahoo ratios remain
current-day data even when an anonymized session has a historical date.

Sentiment verdicts likewise exclude locally measured FinBERT metrics. The fetched
order determines the bounded headline subset; that subset is then sorted oldest
first because decay is positional rather than date-aware. Signed scores map
positive to positive confidence, negative to negative confidence, and neutral
to zero. An unavailable news fetch carries availability flags and null measured
sentiment instead of resembling a neutral observation. Earnings calendars accept
both provider dict and DataFrame forms.

Sentiment and portfolio decoding disable repair to bound token spend on their
large prompts. Fundamental decoding permits repair. Live sentiment batches pace
their scraper requests while parallel fundamental calls consume the longer graph
branch; fixture providers skip that live pacing. Local classifiers and embedding
models load lazily from the optional `models` extra.

Prompt schemas render from `PromptText` markers on the same verdict and proposal
models used for validation, including nested positions. External text is escaped
inside a named untrusted block; embedded closing tags are broken and the caller
appends `UNTRUSTED_GUARD` separately. The guard remains necessary after escaping.

## Consensus, risk, and allocation

Aggregation normalizes conviction against the vote capacity of all specialists,
excluding reliability from the denominator. Agreement among a lone voter cannot
claim full certainty. Direction needs a strict majority; exact ties become neutral.
Macro multipliers act once, during aggregation, without gating independent
specialist analysis. Reliability is read once per regime and session.

Risk first checks structural constraints, then solves joint portfolio ceilings,
then checks measured statistical thresholds. Raw asset returns receive weights
once. A single-ticker risk call normalizes to a full book so VaR and CVaR are not
diluted by its proposed allocation. Thin overlap or non-finite covariance fails
before optimization; missing beta skips that gate with diagnostics. A failed
direct VIX lookup blocks approval. Signed conviction encourages long exposure for
bullish evidence, discourages it for bearish evidence, and gives pure variance
minimization for neutral evidence.

The joint verdict propagates monotonically to individual tickers: it cannot
upgrade a veto, and a portfolio veto suppresses every individual allocation.
Uncomputed correlation, stops, and marginal VaR remain null on paths that never
measured them. Stop distances use mean absolute close-to-close changes; borrowing
ATR's period and multiplier does not make them true high/low-based ATR.

Portfolio proposals are advisory. Code drops unknown tickers, zeroes vetoed
positions, clamps caps, and recomputes dollar amounts, cash residual, and measured
stops. A small universe may have an achievable deployment ceiling below the
requested invest fraction. Capital is total wealth, with the invest fraction
applied once. No risk-approved tickers can produce an all-cash allocation; a failed
model call returns no allocation and diagnostics instead of an invented cash trade.

Half-Kelly anchors use historical win rate and sample count, including prior
shrinkage, with assumed win/loss magnitudes disclosed in parameter provenance.
Negative Kelly floors at zero. Schema validation rejects duplicate portfolio
tickers and risk-approved weights above their proposal. Reconstruction must call
validation; `model_copy` bypasses those validators.

## Parameter evidence

`params.p` stores the value normally and attaches provenance and a reason in
dataclass field metadata. Frozen module-level dataclass instances are discovered
as parameter groups; they do not need a second registration list.

| Provenance | Meaning |
|---|---|
| `LITERATURE` | Standard value with a citable source |
| `CONVENTION` | Widely used default with no single source |
| `CALIBRATED` | Tuned against project data, with reproducible evidence |
| `ARBITRARY` | Assumption requiring future evaluation |

No current parameter is supported as `CALIBRATED`. Bump `PARAMS_VERSION` when
values change. Changing pre-registered evaluation thresholds invalidates
comparisons with earlier reports. HMM fit and inference details remain in the
[macro walkthrough](macro_hmm.md), including why BIC is diagnostic rather than
the fixed three-state contract's selection rule.
