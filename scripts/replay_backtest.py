"""
scripts/replay_backtest.py

CLI entry point for argus/backtesting/replay.py — replays one or more
recorded fixture sessions through the real ARGUS graph and prints each
session's portfolio allocation. See that module's docstring, and
docs/adr/0009-no-multiyear-backtest.md, for what this is and isn't.

    .venv/bin/python scripts/replay_backtest.py [session_dir ...]

Defaults to tests/fixtures/ — the one session captured so far — when no
directories are given.
"""

from __future__ import annotations

import sys
from pathlib import Path

from argus.backtesting.replay import replay_sessions

DEFAULT_SESSION_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    session_dirs = [Path(p) for p in sys.argv[1:]] or [DEFAULT_SESSION_DIR]

    results = replay_sessions(session_dirs)

    for result in results:
        alloc = result.final_state.get("portfolio_allocation")
        print(f"\n=== {result.session_dir} ===")
        print(f"universe: {result.universe}")
        if alloc is None:
            print("portfolio_allocation: None (graph exited without an allocation)")
            continue
        print(f"cash_reserve_pct: {alloc.cash_reserve_pct:.3f}")
        for pos in alloc.portfolio:
            print(
                f"  {pos.ticker:6s} alloc={pos.allocation_pct:.3f} "
                f"conviction={pos.composite_conviction:.2f}  {pos.thesis}"
            )


if __name__ == "__main__":
    main()
