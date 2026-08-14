"""
scripts/remediate_macro_regime_tags.py

One-off remediation for the degenerate macro regime tag (issue #10, Stage 6):
every stored cultural-memory row was tagged EXPANSION by the old, always-picks-
a-state HMM, so the regime field currently carries zero information. Rewrites
it to "unknown" rather than deleting the rows, since the rest of each
document (technical/fundamental/sentiment signals, outcome, return) is still
valid evidence.

Not wired into any schedule — run manually, once:

    .venv/bin/python -m scripts.remediate_macro_regime_tags [--persist-dir chroma_db]

Backs up the sqlite file to <path>.bak-<timestamp> before writing.
"""

import argparse
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("argus.remediate_macro_regime_tags")

STALE_REGIME = "EXPANSION"
REPLACEMENT_REGIME = "unknown"


def main() -> None:
    """Backs up the ChromaDB sqlite file and rewrites stale regime tags to "unknown"."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persist-dir", default="./chroma_db")
    args = parser.parse_args()

    sqlite_path = Path(args.persist_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        print(f"No sqlite file at {sqlite_path}; nothing to remediate.")
        return

    backup_path = sqlite_path.with_name(
        f"{sqlite_path.name}.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    shutil.copy2(sqlite_path, backup_path)
    print(f"Backed up {sqlite_path} to {backup_path}")

    import chromadb

    client = chromadb.PersistentClient(
        path=args.persist_dir,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name="argus_wisdom")

    rows = collection.get(where={"regime": STALE_REGIME})
    ids = rows["ids"]
    if not ids:
        print(f"No rows tagged {STALE_REGIME}; nothing to remediate.")
        return

    metadatas = [dict(m, regime=REPLACEMENT_REGIME) for m in rows["metadatas"]]
    documents = [
        doc.replace(f"Macro regime: {STALE_REGIME}", f"Macro regime: {REPLACEMENT_REGIME}")
        for doc in rows["documents"]
    ]

    collection.update(ids=ids, metadatas=metadatas, documents=documents)
    print(f"Rewrote {len(ids)} row(s) from regime={STALE_REGIME!r} to regime={REPLACEMENT_REGIME!r}.")


if __name__ == "__main__":
    main()
