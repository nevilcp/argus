# Demo guide: what to run, when, and what it proves

This is a run sheet, not a tutorial. It exists because the two commands
that used to be the whole story in the README's Quick Start section don't
say what a presenter actually needs to know: which commands work without
API keys, which one silently needs network access, and that the live
`/analyze` endpoint is unreachable outside market hours and for roughly the
first half hour after open. Everything below has been run against this
repository to confirm the exact output shape; nothing here is aspirational.

## Choosing a track

| Condition | Track |
|---|---|
| No API keys configured, or presenting outside market hours | Track A — offline |
| `.env` populated and it's a weekday, 09:30–16:00 ET, market open ≥30 min | Track B — live |

Track A has no external dependency and is the reliable fallback for the
middle of a live presentation if Track B hits a 503 — see
[When the demo goes wrong](#when-the-demo-goes-wrong).

## Track A — offline (no API keys, any time)

### 1. Run the test suite

```bash
.venv/bin/python -m pytest tests/ -v
```

85 tests across 21 files, deterministic, no network calls by design (see
[ADR 0008](adr/0008-deterministic-test-suite.md)). Takes roughly 35 seconds.
Demonstrates: Pydantic validation boundaries, Half-Kelly sizing math,
cache TTL logic, and thread-safe rate limiting all pass without touching a
live API.

### 2. Run the CI gate locally

```bash
.venv/bin/ruff check .
.venv/bin/mypy argus/
```

Both pass clean at commit time; these are the same two commands
`.github/workflows/ci.yml` runs before pytest.

### 3. Replay a recorded session through the real graph

```bash
.venv/bin/python -m scripts.replay_backtest
```

This runs the actual LangGraph DAG — every node, every agent — over the
one captured fixture session (`tests/fixtures/`): 6 tickers (AAPL, MSFT,
NVDA, JPM, XOM, GOOGL), 251 daily bars per ticker spanning
`2025-05-23` → `2026-05-22`. Market data and LLM calls are served from
fixtures via the injection seam in [ADR 0007](adr/0007-injection-seam.md),
not fetched live — although the risk engine still attempts one live VIX
percentile and OLS-beta fetch, which fails gracefully offline and falls
back to defaults; you'll see a couple of harmless "fetch failed" log
lines. Output is:

```text
=== tests/fixtures ===
universe: ['AAPL', 'GOOGL', 'JPM', 'MSFT', 'NVDA', 'XOM']
cash_reserve_pct: 0.575
  AAPL   alloc=0.000 conviction=0.51  Skipped: bearish signal
  MSFT   alloc=0.123 conviction=0.61  Sentiment is bullish
  NVDA   alloc=0.075 conviction=0.54  Fundamental is bullish
  JPM    alloc=0.150 conviction=0.92  Neutral with high conviction
  XOM    alloc=0.000 conviction=0.50  Skipped: bearish signal
  GOOGL  alloc=0.077 conviction=0.67  Neutral with moderate conviction
```

Demonstrates: the full multi-agent pipeline — technical, fundamental,
sentiment, macro, risk, portfolio — producing a real allocation from
recorded inputs, with no external dependency.

### 4. Walk through the committed evaluation results

```bash
$EDITOR docs/evaluation-results.md
```

No command to run — this is the output of `scripts/run_evaluation.py`
(see Track B) already captured and committed, with the pre-registered
rank IC / hit-rate numbers and the negative finding stated outright:
closed-loop reliability weighting did not clear the pre-registered bar,
because `chroma_db` held 201 `PENDING` and 0 resolved outcomes at
evaluation time. Good material for explaining what the system measures
about itself, without needing to re-run anything.

Optional: `docs/case-study.md` and `docs/historical/` are the
before/after evidence for the rebuild, if there's time for that context.

## Track B — live (keys + market hours)

**Preconditions**, all of which are non-negotiable:
- `.env` populated with at least `GROQ_API_KEY`, `FRED_API_KEY`, `NEWSAPI_KEY`.
- Weekday, 09:30–16:00 ET.
- The MFT pipeline needs roughly 30 minutes after market open to complete
  its first session cycle and populate the live cache — `/analyze` will
  503 until it does, even during market hours.

### Start the API

```bash
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

or `docker compose up --build -d` (single `argus-api` service).

### Any-time endpoints (safe even if the market is closed)

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/governor/report
curl -s http://localhost:8000/memory/stats
```

`/health` returns model versions and whether the rate-limit governor
still has capacity; `/governor/report` returns per-model request/token
usage; `/memory/stats` summarizes the cultural memory vault. These are
the safe fallback to point at mid-presentation if `/analyze` 503s.

### Run a live analysis

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "NVDA"],
    "total_wealth": 100000,
    "invest_pct": 0.5,
    "risk_tolerance": "MODERATE"
  }'
```

Returns `session_id`, `portfolio` (per-ticker allocations), `cash_reserve_pct`,
`expected_sharpe`, `macro_regime`, `vix_level`, `governor_report`, and
`timestamp`. This drives the full live graph, not fixtures — expect real
LLM latency (several seconds to tens of seconds for 3 tickers).

### Run the pre-registered evaluation

```bash
.venv/bin/python -m scripts.run_evaluation
```

Requires network access — unlike everything in Track A, this fetches real
forward daily closes via `LiveMarketDataProvider` to score replayed
decisions against. Prints the same rank IC / hit-rate metrics committed to
`docs/evaluation-results.md`, plus a replay-determinism check.

### Reconcile outcomes (not really demo material)

```bash
.venv/bin/python -m scripts.reconcile_outcomes
```

Resolves decisions past their horizon against live prices and writes
outcomes back to cultural memory — the operational half of
[ADR 0010](adr/0010-closing-the-decision-outcome-loop.md). Requires
network access, defaults to reading `argus_graph.db` (~124 MB in this
repo), and is meant to run on a schedule rather than live in front of an
audience — mention it, don't run it mid-demo.

## When the demo goes wrong

`/analyze` returns one of four 503s, verbatim from `api/main.py`, in this
order of precedence:

| Message | Cause | Recovery |
|---|---|---|
| "System halted. Kill switch triggered. Manual reset required." | Drawdown breached the kill-switch threshold | `POST /kill-switch/reset?new_inception_value=...`, or fall back to Track A |
| "New positions blocked. VIX above threshold." | VIX ≥ the configured blackout level | Wait it out, or fall back to Track A |
| "US equity market is currently closed. MFT pipeline is idle. Retry between 09:30 and 16:00 ET on a weekday." | Outside market hours | Switch to Track A |
| "MFT live cache not yet populated for: [...]. The pipeline is warming up — retry after the next session cycle (~30 min after market open)." | Inside market hours, but too soon after open | Wait, or switch to Track A |

Other failure modes:
- **Missing API keys at startup**: the macro regime fit is wrapped in a
  try/except and logs a warning rather than aborting — the server still
  starts, but `/analyze` will fail downstream once it reaches an agent
  that needs the missing key.
- **`run_evaluation.py` run without network**: fails outright — it's the
  one command in this guide that has no offline fallback. If network is
  unreliable, skip it and use the committed numbers in
  `docs/evaluation-results.md` instead.

In every case, Track A has no external dependency and is the safe thing
to switch to without breaking stride.

## What each command is evidence for

| Command | Demonstrates |
|---|---|
| `pytest tests/ -v` | Deterministic, network-free correctness ([ADR 0008](adr/0008-deterministic-test-suite.md)) |
| `scripts.replay_backtest` | The real graph running end-to-end via the injection seam ([ADR 0007](adr/0007-injection-seam.md)) |
| `docs/evaluation-results.md` | The pre-registered evaluation bar and an honestly-reported negative result ([ADR 0012](adr/0012-pre-registered-evaluation.md)) |
| `POST /analyze` (live) | The full pipeline against real market data and live LLM calls |
| `scripts.run_evaluation` | Rank IC / hit-rate scoring against real forward returns |
| `scripts.reconcile_outcomes` | The decision→outcome loop closing in production ([ADR 0010](adr/0010-closing-the-decision-outcome-loop.md)) |
