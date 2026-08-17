"""
scripts/run_evaluation.py

CLI entry point for the pre-registered evaluation. Replays a fixture session
open-loop and closed-loop, scores both against real forward returns, and
prints the report the pre-registered success threshold is judged against,
plus system-behavior metrics reported separately.

    .venv/bin/python scripts/run_evaluation.py [--fixtures-dir PATH]
        [--horizon-days N] [--deadband X]

Requires network access (LiveMarketDataProvider fetches real daily closes
for the session's tickers) — this is the one place in the test/eval stack
that's expected to, since the whole point is scoring against real forward
returns rather than fixture data.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from argus.backtesting.evaluation import (
    EvaluationResult,
    check_replay_determinism,
    evaluate_decisions,
    system_behavior_report,
    trade_level_win_loss_stats,
)
from argus.backtesting.replay import replay_session
from argus.params import RECONCILIATION
from argus.seams import LiveMarketDataProvider

logger = logging.getLogger("argus.run_evaluation")

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _print_evaluation(label: str, result: EvaluationResult) -> None:
    print(f"\n--- {label} (n={result.n}) ---")
    if result.n < 2:
        print("  Too few decisions to compute rank IC or hit-rate.")
        return
    ic_lo, ic_hi = result.rank_ic_ci
    ic_ci = f"[{ic_lo:+.4f}, {ic_hi:+.4f}]" if ic_lo is not None else "undefined"
    print(f"  rank IC:        {result.rank_ic:+.4f} (p={result.rank_ic_p_value:.4f}), "
          f"95% CI {ic_ci}")
    hit_lo, hit_hi = result.hit_rate_ci
    if result.hit_rate is not None:
        ci = f"[{hit_lo:.4f}, {hit_hi:.4f}]" if hit_lo is not None else "undefined"
        print(f"  hit-rate:       {result.hit_rate:.4f} (n={result.hit_rate_n}), 95% CI {ci}")
    else:
        print("  hit-rate:       undefined (every decision fell inside the dead band)")

    trade_stats = trade_level_win_loss_stats(result.pairs)
    if trade_stats:
        print(f"  win rate:       {trade_stats['win_rate']:.4f} "
              f"({trade_stats['total_trades']} decisions)")
        print(f"  profit factor:  {trade_stats['profit_factor']}")
        print(f"  avg win/loss:   {trade_stats['avg_win_pct']:+.4f} / "
              f"{trade_stats['avg_loss_pct']:+.4f} pct, "
              f"avg holding {trade_stats['avg_holding_days']:.1f}d")


def main() -> None:
    """Replays open-loop and closed-loop sessions and prints the evaluation report."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--horizon-days", type=int, default=RECONCILIATION.horizon_days)
    parser.add_argument("--deadband", type=float, default=RECONCILIATION.min_abs_return_for_storage)
    args = parser.parse_args()

    market_data = LiveMarketDataProvider()

    print("Replaying session open-loop (reliability fixed at 0.5 prior)...")
    open_loop = replay_session(args.fixtures_dir, closed_loop=False)
    open_decisions = open_loop.final_state.get("decisions") or []

    print("Replaying session closed-loop (reliability reads real chroma_db history)...")
    closed_loop = replay_session(args.fixtures_dir, closed_loop=True)
    closed_decisions = closed_loop.final_state.get("decisions") or []

    open_result = evaluate_decisions(open_decisions, market_data, args.horizon_days, args.deadband)
    closed_result = evaluate_decisions(closed_decisions, market_data, args.horizon_days, args.deadband)

    print(f"\n=== Pre-registered evaluation (H={args.horizon_days}d, deadband={args.deadband}) ===")
    _print_evaluation("Open-loop", open_result)
    _print_evaluation("Closed-loop", closed_result)

    if open_result.n < 6 or closed_result.n < 6:
        print(
            "\nNOTE: n < 6 (fewer than the full fixture universe) — statistical "
            "power is very low at this sample size."
        )

    if closed_result.rank_ic is not None and open_result.rank_ic is not None:
        ci_lo, _ci_hi = closed_result.rank_ic_ci
        helped = (
            ci_lo is not None
            and ci_lo > 0
            and closed_result.rank_ic > open_result.rank_ic
        )
        verdict = "closed-loop cleared the pre-registered bar" if helped else \
            "closed-loop did NOT clear the pre-registered bar"
        print(f"\nVerdict: {verdict}")

    print("\n=== System-behavior metrics (open-loop session; reported separately) ===")
    sys_report = system_behavior_report(open_loop)
    print(f"  schema validity:       {sys_report.decisions_built}/{sys_report.tickers_total} "
          f"({sys_report.schema_validity:.0%})")
    print(f"  errors recorded:       {len(sys_report.errors)}")
    print(f"  constraint violations: {sys_report.constraint_violations} "
          f"(enforced structurally by RiskAssessment's Pydantic validator; "
          f"{sys_report.reduce_verdicts} REDUCE verdicts exercised it)")
    print(f"  API calls/decision:    {sys_report.api_calls_per_decision:.2f} "
          f"(total {sys_report.total_api_calls})")
    print("  retries/decision:      not instrumented (disclosed gap)")
    print("  tokens/decision:       not instrumented (disclosed gap)")

    print("\n=== Replay determinism ===")
    deterministic = check_replay_determinism(args.fixtures_dir)
    print(f"  open-loop replay is deterministic: {deterministic}")


if __name__ == "__main__":
    main()
