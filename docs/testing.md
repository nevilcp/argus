# Test boundaries and documentation checks

[CONTRIBUTING](../CONTRIBUTING.md) owns setup, commands, and snapshot updates.

## Fixture isolation

Shared builders return valid contracts with keyword overrides so tests name only
relevant values. Reuse `FakeMarketData`, candle builders, and `FIXTURES_DIR` from
`tests/helpers/`; do not import another test module. Most route tests set runtime
singletons directly without starting lifespan. Lifecycle tests exercise actual
startup wiring with injected pipeline and graph boundaries.

Memory arithmetic tests attach a fake collection without constructing embedding
models. Kill-switch tests drive `_check` with mocked VIX instead of sleeping on
real intervals. Shared singleton and logger state is reset between tests. Capture
real sleep before patching the shared `time` module to avoid mock recursion.
No test calls Groq. Live-provider checks use `live_network`; an offline pass
establishes no claim about live availability.

## Replay and evaluation

A captured session has `market_data/` and `llm_responses/` directories matching
`tests/fixtures/`. Providers see only their own capture, and ordered replay finishes
one session before starting the next. Replay uses temporary checkpoints so simulated
decisions cannot enter production settlement. It cannot reconstruct unrecorded
intraday history.

Open-loop replay substitutes neutral memory. Closed-loop replay filters memory by
capture time but also depends on mutable store state. Determinism compares two
independent open-loop runs. Golden snapshots separately pin committed behavior,
remove volatile fields, and round serialized floats to tolerate insignificant
library differences. Fixture proposals include vetoed positions; zeroed weights
and residual-cash corrections are expected enforcement diagnostics.

Evaluation pairs signed conviction and measured forward returns through production
outcome logic. Missing signals, positions, and immature horizons are excluded;
fetch errors propagate so a partial evaluation cannot appear complete. Rank IC
is Spearman correlation. Directional hit rate excludes neutral calls and the same
flat-return band used by outcome classification. Confidence intervals resample
whole paired observations rather than assume large-sample normality.

The pre-registered bar requires closed-loop IC above open-loop and its confidence
interval excluding zero. Undefined values cannot clear it. Trade-level win/loss
and profit-factor statistics do not imply a daily equity curve or Sharpe ratio.
Non-finite metric results become JSON nulls. System behavior metrics remain separate
from predictive evidence; uninstrumented retries and per-decision tokens stay gaps.

## Structural evidence

Layer checks inspect imports at every AST depth, including deferred imports and
`TYPE_CHECKING`, resolving full names so `agents.risk` is not confused with the
risk package. Every top-level module needs a layer in `test_layering.py`.
Property tests check technical direction, aggregation monotonicity, participation,
and conviction caps. Covariance regressions expose double weighting. Beta fixtures
isolate the lookback from a different earlier relationship. DST fixtures repeat
Eastern wall-clock minutes at distinct UTC instants and check elapsed-time schedules.

## Source convention checks

`test_source_conventions.py` checks single-line ASCII docstrings and comments with
a strict 120-character physical limit, including indentation and URL lines that
Ruff may exempt. Static log text must be ASCII; interpolated provider data keeps
its original value. Assigned application strings such as prompts, CLI descriptions,
and fixtures are excluded. Non-Python hash comments preserve quoted hashes and
required directives. Commented environment assignments are copyable examples.

Ruff checks retained-docstring wording, punctuation, quotes, and spacing, plus
commented-out Python code, ambiguous Unicode, and 120-character code formatting.
Missing-docstring rules are disabled. Review still decides relevance, whether an
explanation belongs in a guide, and whether adjacent comments disguise a paragraph.
