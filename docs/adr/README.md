# Architecture Decision Records

Records of decisions already implemented in the codebase before this ADR
directory existed are labelled **retrospective**: written to document and
justify choices the code already makes, not to propose new ones. Records
written before their corresponding code lands are **predictive** and say so.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-statistical-vs-llm-agent-split.md) | Statistical vs. LLM agent split | Accepted (retrospective) |
| [0002](0002-return-none-not-fabricated-defaults.md) | Return None rather than fabricate defaults | Accepted (retrospective) |
| [0003](0003-503-not-daily-fallback.md) | 503 rather than fall back to daily-resolution data | Accepted (retrospective) |
| [0004](0004-server-side-cash-reserve-recompute.md) | Recompute cash_reserve_pct server-side rather than trust the LLM | Accepted (retrospective) |
| [0005](0005-monotonic-risk-verdict-downgrade.md) | Monotonic risk-verdict downgrade | Accepted (retrospective) |
| [0006](0006-parameter-provenance.md) | Parameter provenance | Accepted |
| [0007](0007-injection-seam.md) | Injection seam for market data and LLM calls | Accepted (predictive) |
| [0008](0008-deterministic-test-suite.md) | Deterministic test suite | Accepted |
| [0009](0009-no-multiyear-backtest.md) | Why ARGUS cannot be backtested over multi-year windows | Accepted |
| [0010](0010-closing-the-decision-outcome-loop.md) | Closing the decision→outcome loop | Accepted (predictive) |
| [0011](0011-reliability-weighting.md) | Reliability weighting consumes the outcome loop | Accepted |
| [0012](0012-pre-registered-evaluation.md) | Pre-registered evaluation | Accepted (predictive) |
| [0013](0013-header-driven-governor.md) | Header-driven rate-limit governance | Accepted |

See [issue #1](https://github.com/nevilcp/argus/issues/1) for the rebuild
plan these were written as part of.
