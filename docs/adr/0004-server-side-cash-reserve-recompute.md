# ADR 0004: Recompute cash_reserve_pct server-side rather than trust the LLM

**Status:** Accepted (retrospective).

## Context

`PortfolioAllocation` requires `cash_reserve_pct + sum(allocation_pct) == 1.0`
(enforced by Pydantic validation on the schema). The synthesis prompt tells
the LLM this explicitly and in triplicate — as a numbered rule
("`cash_reserve_pct = 1.0 − sum(all allocation_pct values)`",
`portfolio.py:136`), as a pre-return checklist item
("verify cash_reserve_pct = 1.0 − sum(allocation_pct)", `portfolio.py:152`),
and as an inline step ("Do not set it independently", `portfolio.py:266`).
Despite that, an LLM asked to emit several numbers that must satisfy an
arithmetic identity will sometimes emit numbers that don't, because next-
token generation has no built-in commitment to global consistency across a
JSON object it's producing incrementally.

## Decision

Never depend on the LLM having gotten the arithmetic right. After parsing
the model's JSON response, `portfolio.py:323-325` recomputes the field
directly from the values the model *did* commit to per-position:

```python
total_equity = sum(p["allocation_pct"] for p in data.get("portfolio", []))
data["cash_reserve_pct"] = round(max(0.0, 1.0 - total_equity), 6)
```

The comment at `portfolio.py:322` states the reason plainly: this exists
"to prevent Pydantic sum validation failure when the LLM sets
cash_reserve_pct independently instead of as 1 − equity." The prompt asks
for the derived field anyway, both as a documented contract the model is
expected to honor and as a redundancy check — if the model's own value and
the recomputed value diverge widely, that's a signal (currently unused —
logging or surfacing that divergence to the eventual outcome-loop work in
PR 8/PR 9 would sharpen this).

## Consequences

- The response returned to the caller always satisfies its own schema
  invariant; it is not possible for `/analyze` to fail *at this step* due to
  the model's arithmetic, only due to the model producing invalid or
  incomplete allocations upstream (still an open failure mode, handled by
  the None-not-fabricated convention in ADR 0002).
- This is a narrow, specific instance of a general principle: any output
  the graph can compute deterministically from the LLM's other stated
  values should be computed, not requested from the LLM a second time.
  `stop_loss` follows the same pattern one block earlier
  (`portfolio.py:317-319`) — overwritten from the risk engine's `RiskAssessment`
  when available, rather than trusting whatever number the LLM wrote.
