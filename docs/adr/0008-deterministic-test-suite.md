# ADR 0008: Deterministic test suite

**Status:** Accepted.

## Context

ADR 0007's injection seam made it *possible* to run agents and the full
graph without network or LLM calls, but PR 5 only proved this for
`FundamentalAgent`/`SentimentAgent` in isolation (`tests/test_seams.py`).
Nothing exercised the compiled `build_graph()` DAG end-to-end, and the
existing `tests/test_integration.py::test_full_graph_smoke` mocks
`FundamentalAgent.analyze`/`SentimentAgent.analyze` directly rather than
using the seam — it still calls real FRED/yfinance endpoints through
`MacroStatisticalAgent` and `node_fetch_price_history`, so it isn't
deterministic and can't run offline.

Two things this codebase does not have before this PR: a test that proves
the *whole* graph is replayable from fixtures, and property-based tests on
the pure scoring functions the plan calls out (`_score_rsi`, `_score_bollinger`,
aggregator conviction cap, `RiskAssessment.approved_weight ≤ proposed_weight`).

## Decision

**`tests/test_golden_dag.py`** calls `build_graph()` with `FixtureMarketDataProvider`
and three `FixtureLLMClient`s (fundamental, sentiment, portfolio), asserts the
DAG produces a valid `PortfolioAllocation`, and asserts two independent
invocations produce identical output once nondeterministic fields (timestamps,
`session_id`) are stripped.

Two things needed to become fixtures that PR 5 didn't already capture, both
pulled from the same `langsmith/run-*.json` traces `scripts/capture_fixtures.py`
already reads: `session_states` (technical indicator values — not computed by
`node_fetch_price_history` itself; the API layer normally supplies them from
the MFT live cache) and the portfolio LLM's raw response (one call for the
whole universe, unlike fundamental/sentiment's per-ticker calls).

Two boundaries turned out not to be reachable through the existing seam, and
had to be handled explicitly rather than silently working around them:

- **Cultural memory** stays mocked, not fixture-backed — consistent with
  ADR 0007's explicit position that it "isn't a candidate for fixture
  replay." `test_full_graph_smoke`'s existing mock pattern is reused as-is.
- **The rate-limit governor** is a real, process-wide singleton
  (`argus/orchestration/governor.py`) that every agent calls before its LLM
  invocation, independent of whether that invocation is real or
  fixture-backed. Left alone, `governor.wait_if_needed` throttles the
  fixture-backed graph exactly like the live one — multiple 60s sleeps per
  test run, since this test invokes the graph twice. `test_golden_dag.py`
  patches `governor.wait_if_needed` to a no-op; the throttling behavior
  itself is `test_governor.py`'s job, not this test's.
- **FinBERT** (`argus/agents/sentiment.py:score_headlines_with_finbert`)
  called `get_finbert()` — and therefore imported `transformers`/`torch` —
  before checking whether there were any headlines to score. With the
  `news.json` fixture placeholder empty (PR 5 never captured real news
  data; see `scripts/capture_fixtures.py`'s comment on why), every fixture
  run hit this unconditionally. Fixed by moving the empty-headlines check
  before the model load — a genuine bug independent of this test (a
  ticker with no news should never pay FinBERT's ~30s first-load cost),
  not a workaround. This is what makes the "no torch import" claim below
  literal rather than aspirational for the golden-path case; it is not a
  general seam over FinBERT (a ticker with real headlines still needs it).

**Property tests** (`tests/test_technical_properties.py`,
`tests/test_aggregator_properties.py`, `tests/test_risk_properties.py`, using
Hypothesis) check the four invariants the rebuild plan named directly.
Three held on first run. The fourth — `_score_rsi` monotonically
non-increasing — did not: RSI 45 scored −0.20 while RSI 45.5 scored −0.18,
a real bug. The neutral-zone formula (`(rsi - 50) / span`, momentum-following:
higher RSI near 50 scores more bullish) and the outer transition-band
formulas (mean-reversion: higher RSI scores more bearish) matched in value at
their shared boundary but disagreed in *slope*, producing a local hump on
each side of 50 instead of a single monotonic curve. Fixed by replacing the
sloped neutral zone with a flat `0.0` dead-zone and re-deriving both
transition bands to interpolate linearly toward that zero rather than toward
the old ±0.2 endpoint — same anchor scores at the oversold/overbought/transition
thresholds, genuinely monotonic in between. `rsi_neutral_span`
(`argus/params.py`), the normalization divisor the old formula needed, is
now unread and was deleted rather than left as a dead parameter — exactly
the kind of thing this rebuild is trying to stop happening (see the plan's
"dead-seam regression" verification step).

**CI gates** (`.github/workflows/ci.yml`): `ruff check .` and `mypy argus/`
run before the test suite, which now runs in full (previously four hand-picked
files). `[project.optional-dependencies].dev` gained `hypothesis`. mypy is
configured with `plugins = ["pydantic.mypy"]` (without it, mypy can't see
that `Field(None, ...)` on a pydantic model *is* a default, and flags every
such field as a missing required constructor argument) and
`ignore_missing_imports = true` (pandas/yfinance/scipy/hmmlearn/sklearn/
fredapi/newsapi/pytrends ship no stubs). `argus/backtesting/{engine,
walk_forward,phase1_calibration,phase2_validation,bias_auditor,pit_enforcer}.py`
are excluded from mypy's scope: PR 7 deletes all six, so precisely typing
code that's leaving next isn't worth doing; `metrics.py`, which PR 7 keeps,
stays in scope. The remaining real errors surfaced by turning mypy on for
the first time — a missing type annotation, a `None`-narrowing gap in
`graph.py`, a stale `ChatGroq` keyword argument (`groq_api_key=` is an
alias; `model_name=` is the actual field, and mypy without the pydantic
plugin only trusts the latter), and several chromadb/sentence-transformers
stub mismatches in `argus/memory/cultural.py` — were fixed or, where the
mismatch is genuinely in the third-party stubs rather than this code,
suppressed with a `# type: ignore[...]` naming the specific error code and a
one-line reason.

## Consequences

- Determinism is now checked, not just claimed: `test_golden_dag.py` passes
  with `torch`/`transformers`/`sentence-transformers` import-blocked (verified
  manually per this PR's exit criteria — see the plan's "Verification"
  section) and produces byte-identical output (modulo timestamps/UUIDs)
  across repeated runs with no network access.
- `test_full_graph_smoke` in `tests/test_integration.py` is now redundant
  with `test_golden_dag.py` in everything except that it still hits real
  network for macro/price data — left as-is rather than folded in, since
  deleting the only test that exercises the graph against live data would
  trade one honest gap for another. Worth revisiting once PR 7 removes the
  backtesting modules this file also imports.
- The RSI fix changes `_score_rsi`'s output for any RSI in roughly [30, 70]
  — this is a real behavior change, not a refactor, and downstream
  aggregate conviction/allocation numbers for tickers with mid-range RSI
  will shift accordingly. No calibration exists to compare against (PR 7
  deletes the vacuous backtest that would have measured this), so this is
  disclosed here rather than silently absorbed.
