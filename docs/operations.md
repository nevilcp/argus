# Runtime lifecycle, settlement, and deployment

Use the [system guide](system-guide.md) for module ownership and request flow.
These contracts explain process boundaries and persistence across restarts.

## Startup and execution

The API validates registered model identifiers before serving traffic. Common
multiworker argv and environment spellings are checked, but the exclusive
`ARGUS_DATA_DIR` file lock is the actual guard against another API process on
that directory. Its file stays open for the process lifetime. Reload parents
that never enter application lifespan do not hold it.

Runtime state starts unset, allowing readiness errors before initialization.
Startup seeds the universe and shares a graph and concurrency lock with the
collector. Task references remain held and callbacks report failures. `/health`
distinguishes a dead enabled task from a task that was never enabled. Blank
API keys disable authentication and wildcard CORS permits all origins; startup
warns about these deployment defaults. Request ticker transformation precedes
pattern validation so case and whitespace normalization occur in the right order.

HTTP analysis and collection share one concurrency slot. Keep the lock check
adjacent to acquisition without an intervening await. Collection skips an occupied
slot; overlapping HTTP requests receive 429. Synchronous graph and checkpoint
work runs off the event loop. A request timeout cannot cancel an existing worker
thread. Safety is rechecked after execution because the independent monitor can
halt mid-run. Unexpected errors expose a correlation reference instead of details.

Shutdown stops loops before closing the candle store. Canceling an asyncio task
cannot stop its existing `to_thread` worker. Buffer workers hold a shared lock;
`close_buffer` waits on it after loop completion before closing SQLite.

## Graph state and durable decisions

Agents and graph dependencies are constructed once; invocation compiles against
a fresh SQLite connection. UUID-suffixed run identities remain unique within one
clock second. Checkpoint serialization allowlists nested schema models so reads
retain validated objects. Parallel macro and specialist errors use an append
reducer. The final decisions list has one writer and one timestamp per run.

Both entry points append decisions to JSONL with the running image identity.
A sidecar lock covers append or the entire compaction read and replacement cycle;
atomic replacement alone cannot stop a concurrent append from being overwritten.
Unreadable lines survive verbatim. Checkpoint fallback reads the newest checkpoint
per thread and must not create a new empty file when its input is absent.

## Settlement and retention

JSONL wins when both input paths are supplied. Reconciliation queries already
settled identities in a batch and fetches each ticker once per pass. Entry is the
technical signal's close; exit is the first available close on or after the horizon.
Missing allocation or an immature horizon defers settlement rather than inventing
an outcome. Leave-one-out attribution reuses baseline reliability, ranks direction
flips first, then baseline vote magnitude; an unweighted ablation would change the
comparison before any agent was removed. This remains heuristic attribution.

Successful, failed, and flat outcomes all contribute to the win-rate denominator;
flat outcomes are non-wins. Shrinkage returns sample count alongside its rate.
Pending snapshots do not dilute settled return averages, and settlement deletes
the pending copy. Success and failure retrieval use symmetric regime filters;
unknown-regime rows do not receive credit for a live regime.

Embedding provenance names the model that produced existing vectors.
`get_or_create_collection` does not update existing metadata; backfills omit
Chroma's immutable `hnsw:space`. Embedding dependencies load at construction,
not module import, so fixture-backed usage needs no local model installation.

A common retention cutoff bounds checkpoints and pending snapshots. Resolved
JSONL records become eligible for pruning; unresolved decisions survive to a
longer retirement cutoff. Applied paper-run identities survive at least as long
as unresolved records so late settlement cannot compound the same run twice.
Settled vector outcomes also have a capacity bound. Independent maintenance
steps report failure without skipping other stores. Collector housekeeping does
not turn a usable collection into a failed cycle. Checkpoint deletion depends on
the saver's checkpoint/write tables because it offers no public thread deletion;
keep that dependency isolated and covered by retention tests.

## Equity and safety recovery

Paper equity is synthetic: no positions or trades are executed. Runs average
allocation-weighted returns over matured members, leaving unmatured members out,
and compound at most once per horizon window. Missing history starts a new book;
corrupt existing history raises because a reset would erase losses and forget
applied runs. Atomic replacement protects writes from partial-file reads.

The kill switch advances its high-water mark under lock and measures drawdown
from that peak. Halt details are published before the flag. The configured
process risk tolerance governs every request. VIX blackout clears after a genuine
low reading; failed reads cannot clear it, and repeated failures engage it.
Drawdown halts persist until operator reset, including after restart. Even a
corrupt halt dump engages safety. Failed dump writes stay visible in status.
Dump ordering uses recorded halt time, falling back to file modification time.

Reset clears dumps and rebases persisted equity while retaining applied-run
identities. Otherwise reconciliation could restore old drawdown or apply returns
twice. Only the API caller syncs the daemon, and only after a successful equity
update; default report equity after failure is not a measured total loss.
Standalone scheduled collection has no shared daemon; see [limitations](limitations.md).

## Scheduled images and state

The builder installs native dependencies and prefetches CPU models. Runtime
copies the installed packages and model cache, runs as a non-root user, and uses
models offline to avoid writes to its read-only cache. Both stages share a pinned
Python digest. Weekly dependency refresh does not update that digest. Artifact
library major/minor changes require retraining and committing a compatible HMM.

`ARGUS_IMAGE_TAG` is embedded at build time so provenance survives latest tags
and rollbacks. Immutable pins live in `argus-deploy/image-tag.txt`; scheduled jobs
fall back to latest only before that branch exists. Rollback edits the pin without
rebuilding. Branch-backed pins work with repository-token permissions that cannot
write repository variables.

Collector cron approximates Eastern hours with March-October and November-February
offsets. March ticks can be an hour early before DST starts. Runtime market-hours
checks decide whether a tick works; the persisted news allowance controls spend.
Scheduled triggers can drift, skip, or stop after repository inactivity.
Collection and reconciliation share a concurrency group because both restore and
publish the complete `argus-data` tree. Malformed layouts fail before empty state
can overwrite history. Restored files permit the container UID to write.

Collection requires Groq for usable allocations and warns for missing market keys.
It deletes restored cycle output before invocation so an infrastructure failure
cannot republish old diagnostics with a fresh timestamp. No-op and success exit
zero; degraded cycles exit nonzero. After checkout, status still publishes for a
degraded cycle; a secret-preflight failure skips publication. Reconciliation uses
keyless prices and local embeddings without model credentials. CLI descriptions
are application data kept separately from short module docstrings.
