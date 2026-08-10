# ADR 0001: Statistical vs. LLM agent split

**Status:** Accepted (retrospective — this ADR documents a decision already
implemented in the codebase, written after the fact as part of the issue #1
rebuild).

## Context

ARGUS runs six specialist agents. Three compute their signal deterministically
from numeric inputs with no model call: `TechnicalStatisticalAgent` (RSI, MACD,
Bollinger %B, ADX, VWAP, momentum), `MacroStatisticalAgent` (a Gaussian HMM
regime classifier over FRED series and VIX), and `RiskStatisticalEngine` (VaR,
CVaR, beta, an SLSQP portfolio optimizer, and hard structural gates). The other
three call an LLM to produce their signal: `FundamentalAgent`,
`SentimentAgent`, and `PortfolioManagerAgent`.

## Decision

Put an agent on the statistical side whenever its output is a well-defined
function of its inputs — the correct answer to "given these RSI/MACD/VaR
numbers, what's the score" doesn't require judgment, so a closed-form
calculation is strictly better than an LLM call: it's deterministic,
free, and instant, and its correctness can be unit-tested directly (see
ADR-driven Hypothesis property tests added in PR 6).

Put an agent on the LLM side only when the task is genuinely a synthesis or
judgment problem that doesn't reduce to a formula: turning a grab-bag of
valuation multiples and qualitative moat signals into a directional
fundamental thesis, weighing FinBERT scores against catalyst timing for
sentiment, or reconciling six independent signals plus risk ceilings into a
single portfolio allocation with a written rationale.

The dividing line is not "is finance involved" — it's whether the mapping
from inputs to output is specifiable in closed form. Risk enforcement
(`agents/risk.py`) looks like it could plausibly be "judgment," but its
thresholds (VaR 99% > 3%, beta > 1.5, correlation > 0.75) are hard limits by
design, not something we want an LLM exercising discretion over — see ADR
0005 for why risk verdicts are computed, not generated.

## Consequences

- The statistical agents are the cheapest and fastest part of the pipeline
  and require no external API availability to run (`api_calls_used == 0` on
  all three, exercised directly in `tests/test_integration.py`).
- The LLM agents are the only source of non-determinism in the graph. PR 5's
  fixture-backed `LLMClient` and PR 6's golden-DAG tests exist specifically to
  make that non-determinism testable rather than something the suite has to
  tolerate.
- This split is also why the risk engine can enforce a monotonic downgrade
  (ADR 0005) and the portfolio agent's output can be corrected server-side
  (ADR 0004): the statistical agents are trusted as ground truth precisely
  because they're formulas, and the LLM agents' outputs are treated as
  proposals subject to that ground truth, never the reverse.
