# Contributing

ARGUS is a private research and portfolio project maintained by its author.
External pull requests are closed; see [LICENSE](LICENSE) for permitted use.

## Development and verification

Use Python 3.14 or newer and the pinned development tools in `pyproject.toml`:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy argus/ api/ scripts/
pytest tests/ -v
```

The `dev` extra includes `test`. The optional `models` extra supplies local
FinBERT and embedding models; fixture-backed tests do not need it. No test calls
Groq. Tests marked `live_network` exercise Yahoo/FRED and depend on provider
availability. To run the complete offline suite, use:

```bash
pytest tests/ -m "not live_network"
```

CI uses Python 3.14 in three jobs: lint/format/type checking, the test suite with
`.[test]`, and a fresh install of `.[test]` without a pip cache followed by the
same suite. The test jobs provide `FRED_API_KEY` and leave `GROQ_API_KEY` empty.
A successful offline run does not establish that the live providers work.

## Changes and messages

Start each PR on its own branch from current `main`. Keep commits and PR messages
plain and concise, with at most two sentences explaining the concrete change and
its effect. Follow the recent imperative commit style. Wait for every CI check
on the final commit to pass before merging, then delete the branch.

Retarget tests when a symbol moves. Keep one definition in its owning module;
avoid compatibility re-exports that obscure the import direction. Comments state
the current reason beside the code, without tracker IDs or change-history
narration. Use US spelling and explicit exceptions for runtime invariants.

## Shared conventions

- **Parameters:** put hand-set numbers in the appropriate frozen dataclass in
  `argus/params.py` using `p(value, Provenance.X, "reason")`. Explain arbitrary
  choices honestly and bump `PARAMS_VERSION` when parameter values change.
  Settings defaults derive from those parameters.
- **Paths:** use `settings.data_path(name)` for files under `ARGUS_DATA_DIR`;
  honor the explicit Chroma and runs-directory settings where provided.
- **Time:** use `infra.clock.now_local()` for naive local persisted timestamps,
  `now_market()` and `MARKET_TZ` for aware market time, and
  `as_naive_timestamp()` at timestamp boundaries. Do not redefine the timezone.
- **Files and retries:** use `infra.files.atomic_write_text()` for text-file
  replacement and `infra.backoff.full_jitter_delay()` for shared retry delays.
- **Logging:** use `logging.getLogger(__name__)`. Entry points call
  `infra.log.configure_logging()`; messages need no repeated module or function
  prefix. Use `Diagnostics` to collect and log degraded-path notes together.
- **Contracts:** schemas own inter-agent types, enums and prompt limits. Use
  `schemas.prompting.wrap_untrusted()` around external prompt content. Preserve
  unavailable measurements as `None` and report the degraded evidence.
- **Bounded storage:** every TTL cache needs a positive capacity. Daily and live
  caches follow the requested/tracked universe and their caps; settled cultural
  outcomes follow `MEMORY.max_trade_outcomes`.
- **Tests:** keep test files flat. Reuse `tests/helpers/builders.py`,
  `FakeMarketData`, candle builders and `FIXTURES_DIR`; do not import another
  test module. Inject `MarketDataProvider`, `LLMClient` and graph dependencies.

## Import direction

Imports may stay within a layer or point downward. The authoritative table is
`tests/test_layering.py::LAYERS`; its AST scan includes deferred imports and
`TYPE_CHECKING`. Register new top-level modules in that table.

| Layer, lowest first | Modules or packages |
|---|---|
| 1 | `params` |
| 2 | `schemas` |
| 3 | `config`, `infra` |
| 4 | `ports`, `governor` |
| 5 | `data`, `memory`, `outcomes` |
| 6 | `seams`, `structured_output` |
| 7 | `agents` |
| 8 | `risk` |
| 9 | `orchestration` |
| 10 | `backtesting` |
| 11 | `api`, `scripts` |

Ruff's `PLR0915` limit is 43 statements. Keep implementation bodies around 80
lines or less by extracting coherent helpers; docstrings can make total source
spans longer. Keep a longer function only when its single concern justifies it.

## Golden snapshots

`tests/test_golden_dag.py::test_golden_dag_matches_snapshot` compares the
fixture-backed graph with `tests/fixtures/golden/dag_snapshot.json`.
`tests/test_replay_backtest.py` pins the replay CLI output in
`tests/fixtures/golden/replay_stdout.txt`. These comparisons complement the test
that checks two independent graph runs for determinism.

A behavior-preserving PR leaves both snapshots unchanged. A deliberate behavior
change may regenerate them with:

```bash
pytest tests/test_golden_dag.py tests/test_replay_backtest.py --update-golden
```

Inspect and commit the resulting diff, show it in the PR description, and rerun
without `--update-golden` to verify the committed baseline. An empty diff should
be reported as unchanged. Do not regenerate a snapshot to conceal a regression.

See the [system guide](docs/system-guide.md) for module ownership and change
recipes, and [limitations](docs/limitations.md) for the output's practical limits.
