---
trigger: always_on
---

# Code Commenting & Documentation

You are an expert in writing and maintaining code comments that serve as living documentation for both human developers and AI agents navigating a codebase.

## Key Principles
- Comment the **why**, not the **what** — the code shows what; comments explain intent, constraints, and decisions
- Treat comments as first-class code: stale comments are bugs
- Prefer self-documenting code over heavy commenting — good naming eliminates the need for most inline comments
- Comments must add signal, never noise — if a comment restates the code verbatim, delete it
- Every public API surface (function, class, module) must have a docstring; internal logic comments are earned, not mandatory

## Comment Types & When to Use Them

### Docstrings (Functions, Classes, Modules)
- Required on all public and exported symbols
- Use the language-native format: JSDoc (`/** */`) for JS/TS, Google-style or NumPy-style for Python, XML docs for C#
- Include: purpose, parameters with types/constraints, return value, raised exceptions, and usage example for non-trivial APIs
- Omit when the function name and signature are already fully self-describing (`def is_empty(lst: list) -> bool`)

### Inline Comments
- Use sparingly — only for non-obvious logic, algorithmic choices, or business rule enforcement
- Place on the line above (not trailing) for multi-word explanations
- Maximum one sentence; if it needs more, extract the logic into a named function instead

### Block / Section Comments
- Use to delineate logical phases within a long function (e.g., `# --- Phase 1: Validate input ---`)
- Signals that a refactor into smaller functions may be overdue — treat as a code smell worth noting

### TODO / FIXME / HACK Tags
- Use a standardized, searchable format: `# TODO(author, YYYY-MM-DD): description — TICKET-123`
- `TODO` = planned work not yet started
- `FIXME` = known defect requiring correction
- `HACK` = intentional temporary workaround, must link to a ticket for removal
- `NOTE` = non-obvious context a reader needs; not actionable
- Never merge `HACK` or `FIXME` without a linked ticket — stale hacks are silent liabilities

## Docstring Patterns

### Python (Google Style)
```python
def rerank_candidates(
    query: str,
    candidates: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """Re-rank retrieved documents using a cross-encoder model.

    Uses pairwise scoring rather than bi-encoder similarity to capture
    query-document interaction. More accurate than cosine similarity but
    ~10x slower — only call after coarse retrieval has reduced candidates.

    Args:
        query: Raw user query string, not embedded.
        candidates: Documents from initial retrieval stage. Assumes
            they have already been deduplicated.
        top_k: Maximum number of documents to return. Actual count
            may be lower if candidates < top_k.

    Returns:
        Documents sorted by cross-encoder score, highest first.

    Raises:
        ModelNotLoadedError: If the reranker model failed to initialize.
        ValueError: If top_k < 1 or candidates is empty.

    Example:
        docs = retriever.get(query, top_k=50)
        top_docs = rerank_candidates(query, docs, top_k=5)
    """
```

### TypeScript (JSDoc)
```typescript
/**
 * Applies Reciprocal Rank Fusion to merge results from multiple retrievers.
 *
 * Combines BM25 and dense retrieval rankings without requiring score
 * normalization. Uses k=60 constant per the original RRF paper (Cormack 2009),
 * which empirically outperforms tuned alternatives on most IR benchmarks.
 *
 * @param rankings - Array of ranked result lists, one per retriever.
 * @param k - RRF constant controlling rank influence (default: 60).
 * @returns Merged list sorted by combined RRF score, deduplicated by doc ID.
 *
 * @see https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf
 */
function reciprocalRankFusion(rankings: Document[][], k = 60): Document[] {
```

## What NOT to Comment

```python
# BAD — restates the code
counter += 1  # increment counter by 1

# BAD — states the obvious type/operation
user_list = []  # empty list to store users

# BAD — narrative for simple conditionals
if user.is_active:  # check if user is active
    send_email(user)

# GOOD — explains a non-obvious constraint
# Bypass cache for compliance audit trails; caching here would violate
# the 7-year immutability requirement under SOX Section 802.
record = fetch_direct(audit_id)

# GOOD — explains a counterintuitive workaround
# Float comparison intentional — value originates from a legacy CSV export
# that serializes as float32. Direct int cast silently truncates edge values.
if abs(value - expected) < 1e-6:
```

## Self-Documenting Code Checklist
Before writing a comment, exhaust these options first:
- Rename the variable/function to reveal its intent
- Extract the logic into a named helper function
- Replace a magic number with a named constant (`MAX_RETRY_ATTEMPTS = 3`)
- Add a type annotation to clarify what a value represents

## Comment Maintenance Rules
- **On every code change**: review comments in the modified scope — update or delete as needed
- **On code review**: stale or misleading comments are blocking issues, not style nits
- **On deletion**: if code is removed, its comments go with it — orphaned comments are misinformation
- **On refactoring**: renaming a function invalidates its docstring summary; rewrite, don't patch
- **At PR merge**: zero `HACK`/`FIXME` tags without a linked, open ticket

## File-Level Header Comments
Required for every module/file:
```python
"""
retrieval/reranker.py

Cross-encoder reranking stage for the RAG pipeline. Sits downstream of
hybrid retrieval (BM25 + dense) and upstream of the LangGraph agent.

Responsibilities:
  - Score query-document pairs using a cross-encoder model
  - Filter candidates below a configurable relevance threshold
  - Expose a LangChain-compatible reranker interface

Not responsible for:
  - Document chunking or embedding (see ingestion/chunker.py)
  - Semantic caching (see retrieval/cache.py)
  - Guardrail enforcement (see guardrails/pii.py)

Dependencies:
  - sentence-transformers >= 2.6.0
  - CROSS_ENCODER_MODEL env var must be set (see .env.example)
"""
```

## Consistency Rules
- All comments written in English
- Use active voice: "Retries on timeout" not "Retried on timeout"
- Use present tense: "Returns the top-k documents" not "Returned..."
- No trailing periods on single-line inline comments; full sentences get periods in docstrings
- Line length for comments matches the project's code line limit (default: 88 chars for Python, 100 for TS)

## Anti-Patterns to Reject
| Anti-Pattern | Why It's Harmful | Fix |
|---|---|---|
| Commented-out code | Pollutes history; version control exists for this | Delete it; Git remembers |
| "TODO: fix later" (no owner, no ticket) | Will never be fixed | Add owner + ticket or delete |
| Comments that explain a bad variable name | Symptom of poor naming | Rename the variable |
| Entire function wrapped in a description | Hides that the function is too long | Refactor into smaller functions |
| Auto-generated boilerplate left in | Adds zero signal | Delete unused template stubs |
| Lie-by-omission docstrings | Lists params but omits edge cases that cause bugs | Document the edges, not just the happy path |

## AI Agent–Specific Rules
When generating or modifying code in this codebase:
- Generate docstrings for every new function, class, and module — no exceptions
- Do not generate inline comments for logic that is self-evident from clean code
- When leaving a `TODO`, always include your reasoning and a concrete next step
- When you cannot fully implement something, use `# HACK` or `# FIXME` with a clear explanation — never silently write partial code without marking it
- Do not preserve or copy stale comments from surrounding context — verify before propagating
- Flag any comment you encounter that contradicts the current code; do not silently correct it — surface it in your response
