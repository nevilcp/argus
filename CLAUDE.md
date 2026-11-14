# ARGUS

ARGUS is an equity-research pipeline: six specialist agents run as a LangGraph
DAG behind FastAPI and produce a thesis and a paper allocation. It never places
a trade. Python 3.14 or newer is required.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, verification
commands, branch and message conventions, import layers, shared helpers, and
golden snapshot workflow. That is the single home for those conventions.

Read [docs/commenting-style.md](docs/commenting-style.md) before editing comments
or docstrings. Apply it to all source languages and configuration, with a
120-character code limit and ASCII static log text. Detailed design notes live
in the [documentation index](docs/README.md).

## Before changing the system

Read [docs/system-guide.md](docs/system-guide.md) for the architecture and module
ownership, and [docs/limitations.md](docs/limitations.md) before treating a
modeling assumption as a bug. [docs/macro_hmm.md](docs/macro_hmm.md) explains the
macro classifier's data, fit, validation and inference paths.

Three agents are deterministic statistics: technical, macro and risk. Three use
an LLM: fundamental, sentiment and portfolio. A failed model call must not
corrupt a statistical measurement.

The principles are:

1. Degrade with missing evidence and a note; never fabricate a measured value.
2. Validate model proposals and enforce allocations in code.
3. Inject market-data and model boundaries through `ports` and `seams`.
4. Keep numeric parameters and provenance in `params.py`.
5. Share contracts between producers and consumers in `schemas`.
6. Keep safety gates independent of graph execution and fail closed.
7. Give every store an explicit capacity or retention window.

## Module orientation

- `argus/infra/` owns clock, file, logging, backoff and diagnostics helpers.
- `argus/ports.py` defines protocols; `seams.py` holds live and fixture adapters;
  `governor.py` owns rate governance; `outcomes.py` computes realized outcomes.
- `argus/data/` owns fetches, retries, budgets, caches, indicator compression and
  the live MFT pipeline. `agents/macro_hmm.py` holds the classifier;
  `agents/macro_context.py` holds its pure context derivations.
- `argus/orchestration/graph.py` wires the DAG; `nodes.py` holds `GraphDeps` and
  node functions; `checkpoint.py`, `decision_log.py`, `attribution.py`,
  `reconcile.py`, `collector.py` and `scheduler.py` own their respective concerns.
- `api/main.py` holds routes, with contracts in `models.py`, startup guards in
  `guards.py`, and process state/startup/shutdown in `lifespan.py`.
- `scripts/` contains the five CLI entry points. Tests stay flat under `tests/`,
  with shared data builders and providers in `tests/helpers/`.

No test calls Groq. The `live_network` tests deliberately call live providers;
fixture-backed verification and replay require neither credentials nor models.
