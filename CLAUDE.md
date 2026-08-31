# ARGUS

Multi-agent equity-research system: six specialist agents (technical, macro,
fundamental, sentiment, risk, portfolio) run as a LangGraph DAG behind a FastAPI
gateway, producing a thesis and an allocation. Research project only — it never
places a trade and its output is not advice.

Python ≥3.11, one package. `argus/` is the system, `api/` is the HTTP gateway,
`scripts/` holds the CLI entry points declared in `pyproject.toml`, `tests/`
mirrors `argus/` file-for-file.

## What you need to know before touching the code

Three statistical agents (technical, macro, risk) are deterministic math; three
(fundamental, sentiment, portfolio) call an LLM. The split is load-bearing — a
bad LLM call in one domain must not corrupt a statistically-grounded one.

The design principles the code actually enforces, in full in
[`docs/system-guide.md`](docs/system-guide.md) §2:

1. **Degrade, never fabricate** — a missing input yields a degraded, flagged
   result, never an invented number.
2. **The LLM proposes; code disposes** — every LLM output is Pydantic-validated
   and re-derived server-side before it reaches a response.
3. **Seams, not patches** — network and LLM boundaries are injected through
   `argus/seams.py`. Agent code never constructs `ChatGroq` or calls yfinance
   directly.
4. **Every number has a provenance** — magic constants live in
   `argus/params.py` with their source; env config in `argus/config.py`.
5. **One definition, shared by producer and consumer** — inter-agent contracts
   live in `argus/schemas/signals.py` and nowhere else.
6. **Safety fails closed and is independent of the graph** —
   `argus/risk/kill_switch.py` and `argus/orchestration/governor.py` gate the
   graph from outside it.
7. **Bound every store** — caches, buffers, and vaults all have an explicit cap.

## How to verify a change

```bash
pip install -e ".[dev]"
pytest tests/ -v      # ~150s, offline, no GROQ_API_KEY needed
ruff check .          # ruff and mypy are pinned exactly; do not float them
mypy argus/
```

CI (`.github/workflows/ci.yml`) runs all three, plus the suite on 3.11 and 3.12
and a no-cache clean install. Match it before pushing.

Every LLM boundary in tests is fixture-backed or mocked — never write a test that
calls Groq. `tests/test_integration.py::TestEndToEnd` is the one class that hits
the live network and needs `FRED_API_KEY`.

## Workflow

Enable the `/andrej-karpathy-skills:karpathy-guidelines` skill before writing any
code — new files, edits, or refactors alike.

Commit and PR messages: two concise sentences at most, written plainly the way a
human would. No AI references anywhere in them — no `Co-Authored-By: Claude`, no
mention of Claude, Anthropic, or AI assistance.

## Further reading — load only what the task needs

- [`docs/system-guide.md`](docs/system-guide.md) — the whole system: mental
  model, principles, one request end to end, layer-by-layer module reference,
  debugging, and change recipes. Read this before any non-trivial change.
- [`docs/limitations.md`](docs/limitations.md) — what the output cannot be
  trusted to do, and why. Check before claiming a behavior is a bug.
- [`docs/macro_hmm.md`](docs/macro_hmm.md) — the Gaussian HMM regime classifier:
  features, fit protocol, validation gate, retraining.
- [`README.md`](README.md) — install, deployment, and how to read an `/analyze`
  response and the evaluation metrics.
- [`.agents/rules/`](.agents/rules/) — standing rules for comments
  (`commenting-standards.md`), code review (`code-review.md`), LLM prompt design
  for the quant agents (`quant-prompt-engineering.md`), and the README
  (`readme-generation.md`).
