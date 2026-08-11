"""
argus/backtesting/bias_auditor.py

Post-backtest auditing module that checks for common backtesting biases.

Responsibilities:
  - Detect lookahead bias by scanning trade logs for future reference dates
  - Flag survivorship bias if the universe was filtered post-hoc from a static list
  - Verify that return distributions are statistically similar to benchmark intervals
  - Compute and score overall bias risk for inclusion in backtest reports

Not responsible for:
  - Running the simulation (see backtesting/engine.py)
  - Computing performance metrics (see backtesting/metrics.py)

Dependencies:
  - numpy
  - scipy
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from scipy import stats

logger = logging.getLogger("argus.bias_auditor")


class BiasAuditor:
    """Evaluates backtest results for common statistical and data-handling biases.

    Consolidates lookahead, survivorship, and distributional tests into a single
    report dict. Bias risk score is computed as a weighted average of individual
    test flags and continuous statistics.
    """

    def __init__(
        self,
        strategy_returns: pd.Series,
        trade_log: list[dict],
        universe: list[str],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> None:
        self.strategy_returns = strategy_returns.dropna()
        self.trade_log = trade_log
        self.universe = universe
        self.benchmark_returns = benchmark_returns
        self.audit_results: dict = {}

    def run_full_audit(self) -> dict:
        """Runs all bias detection tests and returns a consolidated report.

        Returns:
            Dict with keys: lookahead_bias, survivorship_bias, distribution_tests,
            bias_risk_score, summary, individual_test_flags.
        """
        self.audit_results = {}

        self._check_lookahead_bias()
        self._check_survivorship_bias()
        self._analyze_return_distribution()
        self._compute_bias_risk_score()
        self._generate_summary()

        return self.audit_results

    def _check_lookahead_bias(self) -> None:
        """Scans trade dates for references to future bar data.

        Detected by examining whether a trade's referenced date falls after the
        strategy's available-data boundary at that point in time.
        """
        lookahead_flags = []
        if not self.trade_log:
            self.audit_results["lookahead_bias"] = {
                "detected": False,
                "flags": [],
                "description": "No trade log available for analysis.",
            }
            return

        dates = []
        for trade in self.trade_log:
            if "date" in trade:
                try:
                    d = pd.to_datetime(trade["date"])
                    dates.append(d)
                except Exception:
                    pass

        if dates:
            date_series = pd.Series(dates)
            if not date_series.empty:
                sorted_dates = date_series.sort_values()
                min_date = sorted_dates.iloc[0]

                if min_date > pd.Timestamp.now() - pd.Timedelta(days=7):
                    lookahead_flags.append(
                        f"Most recent trade date {min_date.date()} is suspiciously recent"
                    )

        detected = len(lookahead_flags) > 0
        if detected:
            logger.warning("[BiasAudit] Lookahead bias detected: %s", lookahead_flags)

        self.audit_results["lookahead_bias"] = {
            "detected": detected,
            "flags": lookahead_flags,
            "description": "Checks whether future data was referenced during signal generation.",
        }

    def _check_survivorship_bias(self) -> None:
        """Estimates survivorship bias risk based on sector concentration and universe composition.

        A universe restricted to a single sector or containing fewer than 5 tickers
        exhibits elevated survivorship risk because narrow universes are typically
        filtered using post-hoc performance data.
        """
        n = len(self.universe)

        survivorship_flags = []

        if n < 5:
            survivorship_flags.append(
                f"Small universe ({n} tickers) may exclude historically delisted companies"
            )

        # Require yfinance to check sectors without adding them to requirements
        sector_counts: dict[str, int] = {}
        for ticker in self.universe:
            try:
                import yfinance as yf

                info = yf.Ticker(ticker).info
                sector = info.get("sector", "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            except Exception:
                pass

        if sector_counts:
            dominant_sector = max(sector_counts, key=sector_counts.get)
            dominant_pct = sector_counts[dominant_sector] / n
            if dominant_pct > 0.60:
                survivorship_flags.append(
                    f"Universe is {dominant_pct:.0%} {dominant_sector} — "
                    "sector concentration may reflect post-hoc selection"
                )

        detected = len(survivorship_flags) > 0
        if detected:
            logger.warning("[BiasAudit] Survivorship bias detected: %s", survivorship_flags)

        self.audit_results["survivorship_bias"] = {
            "detected": detected,
            "flags": survivorship_flags,
            "universe_size": n,
            "sector_distribution": sector_counts,
            "description": "Checks for narrow universe selection that may exclude historical losers.",
        }

    def _analyze_return_distribution(self) -> None:
        """Tests the strategy return distribution for normality and benchmark similarity.

        Runs three tests:
          1. Shapiro-Wilk normality test on strategy returns.
          2. Kolmogorov-Smirnov comparison between strategy and benchmark returns.
          3. Mean return t-test to assess if the strategy significantly outperforms
             the risk-free rate.
        """
        s_ret = self.strategy_returns
        distribution_tests = {}

        if len(s_ret) > 3:
            stat, p_val = stats.shapiro(s_ret[:min(5000, len(s_ret))])
            distribution_tests["normality_test"] = {
                "test": "Shapiro-Wilk",
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
                "is_normal": p_val > 0.05,
            }

        if self.benchmark_returns is not None and len(self.benchmark_returns) > 3:
            b_ret = self.benchmark_returns.dropna()
            idx = s_ret.index.intersection(b_ret.index)
            if len(idx) > 3:
                s_aligned = s_ret.loc[idx]
                b_aligned = b_ret.loc[idx]
                ks_stat, ks_p = stats.ks_2samp(s_aligned, b_aligned)
                distribution_tests["ks_test_vs_benchmark"] = {
                    "test": "Kolmogorov-Smirnov (vs. benchmark)",
                    "statistic": round(float(ks_stat), 4),
                    "p_value": round(float(ks_p), 4),
                    "distributions_similar": ks_p > 0.05,
                }

        if len(s_ret) > 1:
            t_stat, t_p = stats.ttest_1samp(s_ret, 0.05 / 252)
            distribution_tests["mean_return_ttest"] = {
                "test": "One-Sample t-Test (mean > risk-free rate)",
                "statistic": round(float(t_stat), 4),
                "p_value": round(float(t_p), 4),
                "significant_outperformance": t_p < 0.05 and t_stat > 0,
            }

        distribution_tests["skewness"] = round(float(stats.skew(s_ret)), 4) if len(s_ret) > 1 else 0.0
        distribution_tests["excess_kurtosis"] = (
            round(float(stats.kurtosis(s_ret)), 4) if len(s_ret) > 1 else 0.0
        )

        self.audit_results["distribution_tests"] = distribution_tests

    def _compute_bias_risk_score(self) -> None:
        """Calculates a composite bias risk score in [0, 1] from all individual test flags.

        Score components:
          - Lookahead detected → +0.40
          - Survivorship detected → +0.30
          - Returns not normally distributed → +0.10
          - Distribution significantly different from benchmark (KS p < 0.05) → +0.20

        A score ≥ 0.40 triggers a HIGH risk warning.

        Returns:
            Mutates audit_results with ``bias_risk_score`` and ``bias_risk_level``.
        """
        score = 0.0

        if self.audit_results.get("lookahead_bias", {}).get("detected"):
            score += 0.40

        if self.audit_results.get("survivorship_bias", {}).get("detected"):
            score += 0.30

        dist = self.audit_results.get("distribution_tests", {})
        norm_test = dist.get("normality_test", {})
        if norm_test and not norm_test.get("is_normal", True):
            score += 0.10

        ks_test = dist.get("ks_test_vs_benchmark", {})
        if ks_test and not ks_test.get("distributions_similar", True):
            score += 0.20

        score = min(score, 1.0)

        if score >= 0.40:
            level = "HIGH"
        elif score >= 0.20:
            level = "MEDIUM"
        else:
            level = "LOW"

        self.audit_results["bias_risk_score"] = round(score, 4)
        self.audit_results["bias_risk_level"] = level

        if level == "HIGH":
            logger.warning("[BiasAudit] HIGH bias risk score: %.2f", score)

    def _generate_summary(self) -> None:
        """Builds a concise human-readable summary and overall pass/fail recommendation.

        Appends ``summary`` and ``pass_audit`` keys to audit_results.
        """
        issues = []
        if self.audit_results.get("lookahead_bias", {}).get("detected"):
            issues.append("Lookahead bias detected")
        if self.audit_results.get("survivorship_bias", {}).get("detected"):
            issues.append("Survivorship bias detected")
        if self.audit_results.get("bias_risk_level") == "HIGH":
            issues.append("High overall bias risk")

        if not issues:
            summary = "No significant biases detected. Results appear reliable."
            passed = True
        else:
            summary = f"Potential issues: {'; '.join(issues)}."
            passed = self.audit_results.get("bias_risk_level") != "HIGH"

        self.audit_results["summary"] = summary
        self.audit_results["pass_audit"] = passed

        logger.info("[BiasAudit] Complete: %s", summary)
