"""
argus/backtesting/bias_auditor.py
=================================
Automated Bias Auditor for backtesting.
Checks for survivorship bias, lookahead bias, and data quality issues.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger("argus.bias_auditor")

class BiasAuditor:
    def run_full_audit(
        self,
        universe: list[str],
        backtest_start: date,
        strategy_returns: pd.Series,
        price_history: Optional[dict[str, pd.Series]] = None,
        prompt_log: Optional[list[str]] = None,
        well_known_events: Optional[dict[str, date]] = None
    ) -> dict:
        
        report = {}
        report["survivorship_bias"] = self.check_survivorship_bias(universe, backtest_start)
        report["lookahead_bias"] = self.check_lookahead_bias(strategy_returns)
        
        if prompt_log is not None:
            report["entity_anonymization"] = self.check_entity_anonymization(prompt_log, universe)
            
        if price_history is not None:
            report["data_quality"] = self.check_data_quality(price_history)
            
        return report

    def check_survivorship_bias(self, universe: list[str], backtest_start: date) -> dict:
        potentially_biased = []
        
        # Use yfinance to fetch a tiny window around backtest start
        start_str = (backtest_start - timedelta(days=5)).isoformat()
        end_str = (backtest_start + timedelta(days=5)).isoformat()
        
        for ticker in universe:
            try:
                df = yf.download(ticker, start=start_str, end=end_str, progress=False)
                if df.empty:
                    potentially_biased.append(ticker)
            except Exception:
                potentially_biased.append(ticker)
                
        total = len(universe)
        biased_count = len(potentially_biased)
        surv_pct = biased_count / total if total > 0 else 0.0
        
        status = "PASS" if surv_pct < 0.05 else "FAIL"
        
        return {
            "status": status,
            "total_tickers": total,
            "always_valid": total - biased_count,
            "potentially_biased": potentially_biased,
            "survivorship_bias_pct": round(surv_pct, 4),
            "recommendation": "Remove biased tickers or use historical constituents." if status == "FAIL" else "OK"
        }

    def check_lookahead_bias(self, strategy_returns: pd.Series) -> dict:
        if strategy_returns.empty or len(strategy_returns) < 2:
            return {"status": "PASS", "lag1_autocorr": 0.0, "pct_positive_days": 0.0, "suspicious": False}
            
        autocorr = float(strategy_returns.autocorr(lag=1))
        pct_positive = float((strategy_returns > 0).mean())
        
        suspicious = (abs(autocorr) > 0.2) or (pct_positive > 0.70) or (pct_positive < 0.40)
        
        return {
            "status": "WARN" if suspicious else "PASS",
            "lag1_autocorr": round(autocorr, 4),
            "pct_positive_days": round(pct_positive, 4),
            "suspicious": suspicious
        }

    def check_entity_anonymization(self, prompt_log: list[str], universe: list[str]) -> dict:
        flagged = 0
        for prompt in prompt_log:
            is_flagged = False
            for ticker in universe:
                # Basic substring check
                if f" {ticker} " in prompt or f"\n{ticker}" in prompt or prompt.startswith(f"{ticker} "):
                    is_flagged = True
                    break
            if is_flagged:
                flagged += 1
                
        total = len(prompt_log)
        anon_rate = 1.0 - (flagged / total) if total > 0 else 1.0
        
        return {
            "status": "PASS" if anon_rate == 1.0 else "FAIL",
            "anonymization_rate": round(anon_rate, 4),
            "flagged_prompts": flagged
        }

    def check_data_quality(self, price_history: dict[str, pd.Series]) -> dict:
        issues = []
        for ticker, series in price_history.items():
            if series.empty:
                continue
            
            # Check single day returns > 50%
            daily_returns = series.pct_change().dropna()
            if (daily_returns.abs() > 0.50).any():
                issues.append(f"{ticker}: Contains daily return > 50%")
                
            # Check for zero volume / stale prices (consecutive identical closes)
            rolling_std = series.rolling(5).std()
            if (rolling_std == 0).any():
                issues.append(f"{ticker}: Contains 5+ consecutive days of identical closes (stale data)")
                
        return {
            "status": "WARN" if issues else "PASS",
            "issues": issues
        }
