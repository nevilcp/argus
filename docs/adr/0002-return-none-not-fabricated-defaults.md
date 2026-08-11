# ADR 0002: Return None rather than fabricate defaults

**Status:** Accepted (retrospective).

## Context

Every agent occasionally lacks the inputs it needs: the MFT pipeline hasn't
warmed up a required indicator, `yfinance`/FRED returns an empty frame, the
LLM's structured output fails validation, or a required key is simply absent
from `session_state`. The tempting fallback is a hardcoded neutral value —
RSI 50.0, VIX regime "LOW", conviction 0.5 — so the pipeline always produces
a signal.

## Decision

Every `analyze()` in `argus/agents/` is typed `-> Optional[<Signal>]` and
returns `None` when it cannot produce a signal it's willing to stand behind,
instead of substituting a fabricated default:

- `TechnicalStatisticalAgent.analyze` (`technical.py:194`) checks
  `_REQUIRED_INDICATOR_KEYS` and returns `None` if any is missing, logging
  which ones — see the comment at `technical.py:35`: "None (instead of
  computing from fabricated defaults) when any key is absent."
- `MacroStatisticalAgent.analyze` (`macro.py:231`) returns `None` rather than
  "building a MacroContext from fabricated zero-defaults" when FRED/VIX data
  isn't available.
- `FundamentalAgent.analyze` and `SentimentAgent.analyze` return `None` on
  missing data, parse failure, or after exhausting retries — never a
  constant-conviction placeholder.
- `PortfolioManagerAgent.analyze` (`portfolio.py:190`) explicitly documents
  the alternative it rejected: "fabricated all-cash response," returning
  `None` instead "to allow graceful exit rather than a fabricated
  allocation" (`portfolio.py:340`).

Callers (`orchestration/graph.py`, `orchestration/aggregator.py`) are
required to treat `None` as "this agent abstained" and route around it —
excluding the ticker, down-weighting the aggregate conviction, or (for a
missing risk/portfolio result) failing the request with a 503 — rather than
treating a missing signal as a neutral one.

## Consequences

- A signal that exists is trustworthy; a `0.5` conviction score always means
  the agent computed 0.5, never "the agent had nothing to say." This is the
  precondition for the reliability-weighting work in PR 9 to mean anything —
  weighting agents by how often they're right is meaningless if some of
  their "signals" were actually silent fallbacks.
- Coverage gaps are visible as `None`s and log warnings rather than as
  quietly wrong numbers baked into an aggregate score. The cost is that the
  pipeline can produce fewer signals than tickers requested (see ADR 0003
  for the same trade-off at the API layer) — treated here as the honest
  outcome, not a defect to code around.
