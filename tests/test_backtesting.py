"""
Tests for the Backtesting sub-package (argus/backtesting/).

Test plan (to be implemented):
  - test_pit_enforcer_blocks_future_data:
      Create a DataCache with rows spanning T-10 to T+5 days; confirm that
      PITEnforcer with as_of=T raises LookAheadViolationError for T+1 rows.

  - test_pit_enforcer_allows_past_data:
      Same setup — confirm that rows ≤ T pass through without error.

  - test_engine_equity_curve_starts_at_one:
      Run a minimal 30-day backtest on synthetic data and assert the equity
      curve's first value is 1.0 (fully-invested normalised basis).

  - test_sharpe_ratio_correct:
      Feed a constant daily-return series to metrics.compute_sharpe() and
      assert the result matches the analytical solution.

  - test_bias_auditor_detects_lookahead:
      Inject a deliberate look-ahead violation into a backtest run and confirm
      BiasReport.lookahead_violations > 0.

  - test_walk_forward_produces_n_folds:
      Run WalkForwardEngine with n_folds=3 on synthetic data and assert the
      WalkForwardReport contains exactly 3 out-of-sample PerformanceReport entries.
"""


def test_placeholder_backtesting() -> None:
    """Placeholder test — always passes until real tests are implemented."""
    assert True
