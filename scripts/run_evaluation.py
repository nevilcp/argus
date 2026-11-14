"""CLI for pre-registered open-loop and closed-loop outcome evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.backtesting.evaluation import (
    EvaluationResult,
    check_replay_determinism,
    evaluate_decisions,
    pre_registered_verdict,
    system_behavior_report,
    trade_level_win_loss_stats,
)
from argus.backtesting.replay import replay_session
from argus.infra.log import configure_logging
from argus.params import EVALUATION, RECONCILIATION
from argus.ports import MarketDataProvider
from argus.seams import LiveMarketDataProvider

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


CLI_DESCRIPTION = """
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


def _format_ci(bounds: tuple[float | None, float | None], number_format: str) -> str:
    """Format a confidence interval for display."""
    lo, hi = bounds
    if lo is None:
        return "undefined"
    return f"[{lo:{number_format}}, {hi:{number_format}}]"


def _print_evaluation(label: str, result: EvaluationResult) -> None:
    """Print one evaluation section's rank IC, hit-rate, and trade-level stats."""
    print(f"\n--- {label} (n={result.n}) ---")
    if result.n < 2:
        print("  Too few decisions to compute rank IC or hit-rate.")
        return
    print(
        f"  rank IC:        {result.rank_ic:+.4f} (p={result.rank_ic_p_value:.4f}), "
        f"95% CI {_format_ci(result.rank_ic_ci, '+.4f')}"
    )
    if result.hit_rate is not None:
        print(
            f"  hit-rate:       {result.hit_rate:.4f} (n={result.hit_rate_n}), "
            f"95% CI {_format_ci(result.hit_rate_ci, '.4f')}"
        )
    else:
        print("  hit-rate:       undefined (every decision fell inside the dead band)")

    trade_stats = trade_level_win_loss_stats(result.pairs)
    if trade_stats:
        print(f"  win rate:       {trade_stats['win_rate']:.4f} ({trade_stats['total_trades']} decisions)")
        print(f"  profit factor:  {trade_stats['profit_factor']}")
        print(
            f"  avg win/loss:   {trade_stats['avg_win_pct']:+.4f} / "
            f"{trade_stats['avg_loss_pct']:+.4f} pct, "
            f"avg holding {trade_stats['avg_holding_days']:.1f}d"
        )


def main(*, market_data: MarketDataProvider | None = None) -> None:
    """Replay open-loop and closed-loop sessions and print the evaluation report."""
    configure_logging()

    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--horizon-days", type=int, default=RECONCILIATION.horizon_days)
    parser.add_argument("--deadband", type=float, default=RECONCILIATION.min_abs_return_for_storage)
    args = parser.parse_args()

    market_data = market_data or LiveMarketDataProvider()

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

    min_n = EVALUATION.min_n_for_power
    if open_result.n < min_n or closed_result.n < min_n:
        print(
            f"\nNOTE: n < {min_n} (fewer than the full fixture universe) — statistical "
            "power is very low at this sample size."
        )

    if closed_result.rank_ic is not None and open_result.rank_ic is not None:
        if pre_registered_verdict(open_result, closed_result):
            print("\nVerdict: closed-loop cleared the pre-registered bar")
        else:
            print("\nVerdict: closed-loop did NOT clear the pre-registered bar")

    print("\n=== System-behavior metrics (open-loop session; reported separately) ===")
    sys_report = system_behavior_report(open_loop)
    print(
        f"  schema validity:       {sys_report.decisions_built}/{sys_report.tickers_total} "
        f"({sys_report.schema_validity:.0%})"
    )
    print(f"  errors recorded:       {len(sys_report.errors)}")
    print(
        f"  constraint violations: {sys_report.constraint_violations} "
        f"(enforced structurally by RiskAssessment's Pydantic validator; "
        f"{sys_report.reduce_verdicts} REDUCE verdicts exercised it)"
    )
    print(f"  API calls/decision:    {sys_report.api_calls_per_decision:.2f} (total {sys_report.total_api_calls})")
    print("  retries/decision:      not instrumented (disclosed gap)")
    print("  tokens/decision:       not instrumented (disclosed gap)")

    print("\n=== Replay determinism ===")
    deterministic = check_replay_determinism(args.fixtures_dir)
    print(f"  open-loop replay is deterministic: {deterministic}")


if __name__ == "__main__":
    main()
