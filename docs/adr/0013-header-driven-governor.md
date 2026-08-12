# ADR 0013: Header-driven rate-limit governance

**Status:** Accepted.

## Context

`argus/orchestration/governor.py`'s `MODEL_LIMITS` table claimed to be
"tuned against each provider's free-tier daily caps." None of the Groq
numbers were correct: the two Llama rows were off by roughly 50x on
tokens-per-minute, and `requests_per_day`/`tokens_per_day` were invented —
Groq publishes no daily figure for either model, so those fabricated caps
were the *only* quotas that hard-failed a run (`RateLimitExceeded`) rather
than throttling it. A correcting commit fixed the published TPM/RPM values
and dropped the fabricated daily keys, but a static table is wrong by
construction the moment the account's tier differs from whichever plan the
published table describes — the Developer table read on 2026-08-12 is not
necessarily this key's tier, and Groq itself has no API for asking.

Groq's own answer to "what are my real limits" is on every response: Free
and Developer keys alike get `x-ratelimit-*` headers reporting the account's
actual budget as of that call. Two counterintuitive facts about those
headers, both verified against the installed SDKs, drove the design below:

1. `x-ratelimit-limit-requests`/`-remaining-requests` are a **daily**
   figure; `x-ratelimit-limit-tokens`/`-remaining-tokens` are a
   **per-minute** figure. Groq publishes no header for RPM or TPD at all —
   this asymmetry, unremarked anywhere in Groq's docs, is the likely origin
   of the original drift.
2. `langchain_groq` 1.1.2's `ChatGroq` discards these headers entirely;
   `response_metadata` is built from `token_usage`/`model_name`/
   `finish_reason` only. `ChatGroq` does accept `http_client=`, so an
   `httpx.Client` response event hook observes headers without forking
   anything, and without calling `response.read()` (which would break
   streaming by forcing it to buffer).

Three call sites (`fundamental.py`, `sentiment.py`, `portfolio.py`) each
independently estimated tokens and called `governor.wait_if_needed`
directly, immediately after `GroqLLMClient` construction (ADR 0007) had
already made this duplication removable, and each with a different,
undocumented multiplier — a flat `450` in `sentiment.py`, a
`len(prompt.split())*1.3 + 600` in `fundamental.py` (with `600` matching
that agent's `max_tokens` by coincidence, not by design), and a
`params.py`-configurable-but-unrelated `token_estimate_multiplier`/
`token_estimate_overhead` pair in `portfolio.py`. None counted the system
prompt.

## Decision

**Bootstrap, not a claimed fact.** `REGISTERED_MODELS` (a `frozenset`, pure
membership) replaces `MODEL_LIMITS`. `BOOTSTRAP_LIMITS` holds the same
Developer-table TPM/RPM values, but the module docstring is explicit that
they are a conservative floor used only until the first response header
corrects them for the account's real tier — not a claim about this key.

**Governance moves to the seam.** `GroqLLMClient` (`argus/seams.py`) is now
the sole choke point: it builds its own `httpx.Client` with a response event
hook (`_on_response` → `governor.observe_headers`), passes `max_retries=0`
to `ChatGroq` (the groq SDK's own default retry would otherwise silently
double-retry underneath this client's retry loop and hide 429s from the
governor), and `complete()` sequences
`estimate_tokens → wait_if_needed → invoke → record_usage`. The three
agents' per-call estimates and `wait_if_needed` calls are deleted;
`fundamental.py` keeps its `governor` import because its pre-flight
`get_remaining_capacity(...) < 20` skip-ahead check is unrelated to the
per-call reservation and becomes more meaningful once capacity is
header-derived. `FixtureLLMClient` is untouched and still never imports the
governor, so ADR 0008's zero-network guarantee holds — fixture-backed
graphs (`test_golden_dag.py`, `replay.py`, `test_reconciliation.py`) no
longer need to patch `wait_if_needed` at all, since nothing on that path
ever calls it; those now-dead patches were deleted as the direct evidence
that governance really did move.

**One estimator.** `governor.estimate_tokens(system_prompt, user_prompt,
max_tokens)` replaces the three divergent guesses: word count across both
prompts (not just the user prompt, unlike the deleted `fundamental.py`
estimate) times 1.3, plus `max_tokens` standing in for the unknown
completion length. `PortfolioParams.token_estimate_multiplier`/
`token_estimate_overhead` are retired (`PARAMS_VERSION` bumped to 2).

**Three independent axes, two enforcement sources.** RPM has no header,
ever — it stays `BOOTSTRAP_LIMITS`-enforced permanently, not just until the
first header. TPM and RPD are header-enforced once `observe_headers` sets
`limits_observed`, falling back to the bootstrap TPM figure (no bootstrap
RPD exists, matching the earlier commit's refusal to invent one) before
that. A window past its reported reset deadline is assumed to have rolled
over to full even without a confirming response
(`_refresh_observed_windows`), the same trust model the pre-header daily/
minute counters already used.

**Daily exhaustion fails fast, it does not sleep.** The existing per-minute
retry loop releases its lock and sleeps out the remainder of the minute
before re-checking (unchanged from the prior lock-release fix). A
header-reported daily budget of zero is different in kind: its reset window
is on the order of a day, and blocking a request thread for that long
defeats "cooperative back-pressure." `wait_if_needed` raises
`RateLimitExceeded` immediately in that case rather than entering the sleep
loop.

**Retry classification lives with the client, not the account.**
`_RETRYABLE_GROQ_ERRORS` (`RateLimitError`, `APIConnectionError`,
`APITimeoutError`, `InternalServerError`) get a second attempt;
`_TERMINAL_GROQ_ERRORS` (`AuthenticationError`, `PermissionDeniedError`,
`BadRequestError`, `NotFoundError`) propagate immediately — retrying a bad
key wastes nothing but time. A `RateLimitError`'s `Retry-After` header is
honored verbatim (Groq is telling us exactly how long its own limit takes
to clear); anything else backs off with full jitter, capped at 30s. This is
independent of each agent's own three-attempt retry loop around
`llm_client.complete(...)`, which exists for a different failure mode
entirely — a response that parses as invalid JSON or fails schema
validation, not a transport failure — and is left as-is.

**`record_usage` reconciles local bookkeeping only, never the header-derived
remaining.** By the time `complete()` calls it, `observe_headers` has
already run (from the same response's event hook) and overwritten
`remaining_tokens` with the provider's own post-call figure — applying a
second delta on top of that would double-adjust. `record_usage` corrects
only the informational `tokens_today`/`tokens_this_minute` counters, which
matter before the first header arrives and stay accurate afterward without
interfering with the authoritative header path. It also takes
`estimated_tokens` as an explicit fourth argument rather than having the
governor remember a single "last reservation" per model — the governor is
a process-wide singleton and two agents share the `llama-3.3-70b-versatile`
model ID (`fundamental.py`, `portfolio.py`), so a remembered pointer would
race under concurrent `/analyze` requests.

## Consequences

- A key's real tier is no longer guessed at from a published table: the
  first Groq response for each model sets `limits_observed=True` and
  `get_usage_report()` surfaces it, so `/health` and `/governor/report` can
  distinguish an observed fact from a bootstrap floor.
- RPM remains permanently un-verifiable from headers — if the account is
  actually on a lower-RPM tier than `BOOTSTRAP_LIMITS` assumes, this module
  cannot detect that. This is a known, disclosed gap, not an oversight: no
  Groq header reports RPM at all.
- The daily request budget is enforced for the first time since this system
  existed, using the provider's own live number rather than a guess — but
  only after at least one successful call has reported it. A cold process
  making its very first call of the day has no daily figure to check yet
  and is bootstrap-permissive on that axis until then.
- `GroqLLMClient.__init__` now performs no network I/O itself (constructing
  `httpx.Client` and `ChatGroq` is local), so existing tests that build one
  with a dummy API key and mock only `._llm.invoke` continue to exercise
  `complete()`'s real retry/governance logic without a live key
  (`tests/test_seams.py`).
