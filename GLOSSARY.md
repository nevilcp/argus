# ARGUS Glossary

Vocabulary this project's design documents and code assume you already know. Where a
term is also a class or module, the reference is included so this file can be checked
against the code rather than trusted blindly.

## Verdict, signal, proposal

These three are the vocabulary introduced by the LLM seam work (issue #67) and are
easy to conflate because the codebase already overloads two of the words elsewhere —
see the notes under each entry.

- **Verdict** — what an LLM actually returns from a structured-output call. Narrower
  than a signal: it carries only the model's own judgement (direction, conviction, and
  the fields that support that judgement — moat score, decay risk, reasoning), never
  measured data the model was shown but does not own. *Not* `RiskVerdict`
  (`argus/schemas/signals.py`), which is the risk engine's deterministic
  APPROVE/VETO/REDUCE disposition and carries no model judgement at all.

- **Signal** — the domain object an agent produces by combining a verdict with
  measured data the agent itself owns. `FundamentalSignal` merges the fundamental
  verdict with ratios fetched from the market-data provider; `SentimentSignal` merges
  the sentiment verdict with FinBERT metrics computed locally. Neither is ever taken
  from the model's echo of data it was shown. Distinct from the `Signal` enum
  (`argus/schemas/signals.py`), which is just the BULLISH/BEARISH/NEUTRAL directional
  label carried inside one of these objects, not the object itself.

- **Proposal** — the portfolio agent's equivalent of a verdict: an advisory allocation
  the risk engine has not yet enforced against. A proposal becomes an allocation only
  after risk caps have been applied in code.

## Domain nouns

- **Session** — one invocation of the LangGraph DAG — a live `/analyze` request, an
  unattended collector cycle, or a replay run. Every decision the run produces shares
  one `session_timestamp` (`ARGUSDecision`, `argus/schemas/signals.py`), which is what
  later lets `paper_book.py` group them back into a single notional rebalance.

- **Universe** — the full set of tickers eligible for analysis in a session
  (`ARGUSState.universe`, `argus/orchestration/state.py`). Tracked by the MFT pipeline
  subject to a 100-ticker cap and 24h TTL eviction for non-seed tickers.

- **Session state** — the compressed technical-indicator dict for one ticker (RSI,
  MACD histogram, VWAP distance, etc. — see `SESSION_STATE_REQUIRED_KEYS`,
  `argus/schemas/signals.py`) that the MFT pipeline produces and the technical agent
  consumes. Not the same thing as a session: a session has many tickers, each with its
  own session state.

- **Sweep** — one pass of the MFT pipeline (`_sweep_once`, `argus/data/pipeline.py`)
  fetching the latest intraday candles for every tracked ticker in the universe. Its
  callers compress the buffer into session states afterward, via `compress_all`.

- **Decision** — an `ARGUSDecision` (`argus/schemas/signals.py`): a per-ticker snapshot
  of every agent's output plus the resulting allocation, logged once per session.

- **Allocation** — a `PositionAllocation` or `PortfolioAllocation`
  (`argus/schemas/signals.py`): the risk-enforced target weight(s) a session
  ultimately recommends. What a proposal becomes once risk caps have been applied.

- **Governor** — the per-process `RateLimitGovernor`
  (`argus/orchestration/governor.py`) that enforces Groq's per-model rate-limit quotas
  and provides cooperative back-pressure ahead of every LLM call.

- **Kill switch** — the background daemon (`argus/risk/kill_switch.py`) with two
  independent gates: it halts all activity on excessive portfolio drawdown, and
  separately blocks *new* positions during extreme VIX. It runs independently of the
  LangGraph DAG and knows nothing about it.

- **Cultural memory** — the ChromaDB-backed long-term memory
  (`argus/memory/cultural.py`) that stores trade outcomes and decision snapshots, and
  retrieves regime-scoped historical wisdom and warnings for the portfolio agent.

- **Reconciliation** — the slow-clock pass (`argus/orchestration/reconciliation.py`,
  composed as `run_reconciliation_pass`) that closes the decision-to-outcome loop:
  computing realized returns, assigning per-agent credit via leave-one-out ablation,
  persisting trade outcomes into cultural memory, compounding matured runs onto the
  paper equity curve, and bounding the decisions log, checkpoint database, PENDING
  snapshots, and applied-runs set so none of them grow forever. The API's background
  loop and the scheduled CLI script (`scripts/reconcile_outcomes.py`) both run it;
  only kill-switch sync differs between them, and stays at the API's call site.
