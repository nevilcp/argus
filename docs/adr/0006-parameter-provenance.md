# ADR 0006: Parameter provenance

**Status:** Accepted.

## Context

ARGUS's statistical agents and risk engine are built from ~75 free
parameters — RSI bands, ADX dampening thresholds, agent-vote weights,
Kelly-sizing assumptions, VaR/CVaR limits, SLSQP tolerances, and more —
scattered as bare numeric literals across `config.py`, `agents/technical.py`,
`orchestration/aggregator.py`, `agents/portfolio.py`, and `agents/risk.py`.
Read in isolation, `if rsi < 25:` and `if adx > 40.0:` look equally
well-founded. They are not: RSI 30/70 is industry convention, ATR-14 and a
252-trading-day year are textbook, VaR at 99% confidence is standard — but
the 0.35/0.35/0.30 fundamental/technical/sentiment agent-vote split, the
ADX dampening multiplier of 0.6, and the 20-position diversification
ceiling have no basis beyond "seemed reasonable" at the time they were
written.

This distinction used to exist only in the original author's memory, not
in the code. The Phase 1/2 calibration harness referenced in several
comments (e.g. the old `config.py` docstring: "locked values are
persisted to `calibration_report.json` after Phase 1 completes") gave the
appearance of empirical tuning, but per the issue #1 audit it only ever
measured a constant — `_simulate_session_state` never produced the eight
keys the technical agent requires, so every grid-search configuration was
identical. Nothing in this codebase has actually been calibrated against
data yet (PR 7 deletes that harness). Presenting arbitrary and
literature-backed constants identically, next to a calibration apparatus
that wasn't really calibrating anything, is exactly the kind of quiet
overstatement this rebuild exists to remove.

## Decision

Every free numeric parameter is declared once, in `argus/params.py`, as a
field on a frozen dataclass, tagged with one of four provenance values via
a `p(value, provenance, note)` helper:

- `LITERATURE` — a value with a citable external source (ATR-14, a
  252-day trading year, 99% VaR confidence).
- `CONVENTION` — a widely-used default with no single citation, but
  common enough that deviating from it would need justification (RSI
  30/70, half-Kelly sizing, near-zero LLM temperature for structured
  output).
- `CALIBRATED` — tuned against ARGUS's own data or backtests. No value in
  the codebase honestly earns this tag today (see Context); it exists so
  a future value that *is* backed by PR 10's pre-registered evaluation has
  somewhere to go, distinct from a guess.
- `ARBITRARY` — a guess with no basis beyond "seemed reasonable." As of
  this PR, 57 of 76 declared parameters (75%) are tagged `ARBITRARY`,
  including all six technical-indicator weights, all seven aggregator
  weights/thresholds, and most of the risk engine's structural limits.

Call sites import the named group (`TECHNICAL`, `AGGREGATOR`, `PORTFOLIO`,
`RISK`, `SYSTEM`) and read fields directly — `TECHNICAL.rsi_oversold`, not
a wrapped value requiring unwrapping — so this is a pure rename at every
use site, not a behavior change. `config.py`'s `Settings` (pydantic
`BaseSettings`) sources its field defaults from `SYSTEM` and
`TECHNICAL_INDICATOR_WEIGHTS` instead of repeating the literals, so
env-var overridability is preserved. `tests/test_params.py` asserts every
declared field carries a `Provenance` tag, so a future parameter added
without going through `p(...)` fails CI rather than silently reverting to
an untagged literal. `PARAMS_VERSION` in the module exists so a specific
set of values can be pinned in a fixture or referenced from a future
evaluation run's metadata.

## Consequences

- The 75%-arbitrary figure is uncomfortable and is meant to be: it is a
  precise, honest count of how much of the system's behavior rests on
  unvalidated guesses, replacing a vague sense that "some of this was
  probably tuned." It is also a to-do list — anything a future PR
  actually validates against data moves from `ARBITRARY` to `CALIBRATED`
  with a note explaining the evidence, rather than staying an
  unmarked literal.
- This PR is a pure refactor: every migrated value is numerically
  identical to what it replaced, verified by the full test suite passing
  unchanged (40/40) before and after. It does not fix, retune, or
  second-guess any of the arbitrary values it surfaces — that is
  explicitly out of scope here and belongs to whatever future work uses
  PR 10's evaluation harness.
- Two `getattr(settings, "X", <stale-default>)` fallbacks in `risk.py` and
  `portfolio.py` were found and removed as a side effect of this audit
  (one, `MAX_SECTOR_CONCENTRATION`'s fallback of `0.35`, didn't even match
  the real default of `0.40`). The fallbacks were dead code — `settings`
  always has these attributes — so removing them changes no behavior, but
  leaving a wrong number sitting in a fallback that can never fire is its
  own small instance of the problem this ADR addresses.
