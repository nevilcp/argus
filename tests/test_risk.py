"""
Tests for the Risk Statistical Engine (argus/agents/risk.py).
"""

import numpy as np
import pandas as pd
import pytest

from argus.agents.risk import (
    RiskStatisticalEngine,
    compute_asset_returns,
    compute_portfolio_returns,
    ols_portfolio_beta,
)
from argus.params import RISK
from argus.schemas.signals import RiskVerdict


_TRADING_YEAR = pd.date_range(start="2023-01-01", periods=253, freq="B")


def _prices_from_returns(returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    """Compounds daily returns into a price series starting at 100."""
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)


def _random_prices(dates: pd.DatetimeIndex, mean: float, vol: float) -> pd.Series:
    """Builds a price series from N(mean, vol) daily returns drawn off the active numpy seed."""
    return _prices_from_returns(np.random.normal(mean, vol, len(dates)), dates)


@pytest.fixture
def price_history():
    """Create a simulated price history dictionary for tests.

    Returns:
        Mapping of ticker (plus "SPY" benchmark) to a one-year daily price series.
    """
    np.random.seed(42)  # Fixed seed keeps returns reproducible across runs

    # Moderate, normal volatility: 0.1% daily mean return, 1.5% daily vol
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC"]
    hist = {t: _random_prices(_TRADING_YEAR, 0.001, 0.015) for t in tickers}
    hist["SPY"] = _random_prices(_TRADING_YEAR, 0.0005, 0.01)

    return hist


@pytest.fixture
def risk_engine():
    """A fresh RiskStatisticalEngine instance."""
    return RiskStatisticalEngine()


def test_vix_blackout(risk_engine: RiskStatisticalEngine, price_history: dict) -> None:
    """VIX above the 35 blackout threshold vetoes the whole portfolio."""
    positions = [{"ticker": "AAPL", "weight": 0.1} for _ in range(5)]
    result = risk_engine.evaluate(positions, price_history, current_vix=40.0)

    assert result.verdict == RiskVerdict.VETO
    assert any("blackout" in r.lower() for r in result.veto_reasons)
    # RE-9: a structural-violation short-circuit never computed correlation, so
    # it must report None ("not measured"), not a fabricated 0.0
    assert result.avg_correlation is None


def test_overweight_position(risk_engine: RiskStatisticalEngine, price_history: dict) -> None:
    """A position above the 0.15 weight limit vetoes the portfolio."""
    positions = [
        {"ticker": "AAPL", "weight": 0.20},  # exceeds the 0.15 limit
        {"ticker": "MSFT", "weight": 0.10},
        {"ticker": "GOOGL", "weight": 0.10},
        {"ticker": "META", "weight": 0.10},
        {"ticker": "AMZN", "weight": 0.10},
    ]
    result = risk_engine.evaluate(positions, price_history, current_vix=20.0)

    assert result.verdict == RiskVerdict.VETO
    assert any("weight" in r.lower() for r in result.veto_reasons)


def test_high_var_reduce(risk_engine: RiskStatisticalEngine) -> None:
    """Extreme volatility pushes VaR high enough to trigger REDUCE rather than VETO."""
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]

    # 15% daily volatility is well above normal to force VaR past the REDUCE threshold
    hist = {t: _random_prices(_TRADING_YEAR, 0.0, 0.15) for t in tickers}
    hist["SPY"] = _random_prices(_TRADING_YEAR, 0.0, 0.01)

    positions = [{"ticker": t, "weight": 0.15} for t in tickers]

    result = risk_engine.evaluate(positions, hist, current_vix=20.0)

    assert result.verdict == RiskVerdict.REDUCE
    assert result.var_99 > 0.03


def test_approve_healthy_portfolio(risk_engine: RiskStatisticalEngine, price_history: dict) -> None:
    """A diversified, normally-weighted portfolio with normal vol is approved outright."""
    positions = [
        {"ticker": "AAPL", "weight": 0.1},
        {"ticker": "MSFT", "weight": 0.1},
        {"ticker": "GOOGL", "weight": 0.1},
        {"ticker": "META", "weight": 0.1},
        {"ticker": "AMZN", "weight": 0.1},
        {"ticker": "TSLA", "weight": 0.1},
        {"ticker": "JPM", "weight": 0.1},
        {"ticker": "BAC", "weight": 0.1},
    ]
    result = risk_engine.evaluate(positions, price_history, current_vix=20.0)

    assert result.verdict == RiskVerdict.APPROVE
    assert result.var_99 <= 0.03
    assert not result.veto_reasons


def test_slsqp_covariance_uses_unweighted_returns(price_history: dict) -> None:
    """The optimizer's w^T*cov*w term is built from raw returns, not pre-weighted ones.

    Regression test for RE-2: cov used to come from compute_portfolio_returns,
    whose series are already multiplied by weight, so w^T*cov*w applied the
    weight a second time — variance came out ~1/weight^2 too small.
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC"]
    weight = 0.125
    positions = [{"ticker": t, "weight": weight} for t in tickers]
    w = np.full(len(tickers), weight)

    asset_returns = compute_asset_returns(positions, price_history)
    cov = asset_returns.cov() * 252
    fixed_variance = float(np.dot(w, np.dot(cov, w)))

    # Var(sum_i w_i * r_i) is the ground truth this must match
    weighted_returns = compute_portfolio_returns(positions, price_history)
    true_portfolio_variance = float(weighted_returns.sum(axis=1).var() * 252)
    assert fixed_variance == pytest.approx(true_portfolio_variance, rel=0.05)

    # Reproduce the pre-fix computation (cov from already-weighted returns) to
    # prove this test would have failed against the old code
    buggy_cov = weighted_returns.cov() * 252
    buggy_variance = float(np.dot(w, np.dot(buggy_cov, w)))
    assert fixed_variance / buggy_variance == pytest.approx(1.0 / (weight**2), rel=0.05)


def test_zero_api_calls(risk_engine: RiskStatisticalEngine, price_history: dict) -> None:
    """The risk engine evaluates purely statistically, never calling an LLM."""
    positions = [{"ticker": "AAPL", "weight": 0.1} for _ in range(5)]
    result = risk_engine.evaluate(positions, price_history, current_vix=40.0)

    assert result.api_calls_used == 0


def test_small_universe_is_noted_not_vetoed(
    risk_engine: RiskStatisticalEngine, price_history: dict
) -> None:
    """RE-4 regression: a 2-4 ticker universe is informational, not a hard VETO.

    Before the fix, a book below the 5-position diversification floor was
    structurally VETOed regardless of its actual risk profile, returning
    all-cash with no reason a human could act on.
    """
    positions = [{"ticker": t, "weight": 0.1} for t in ["AAPL", "MSFT", "GOOGL"]]
    result = risk_engine.evaluate(positions, price_history, current_vix=20.0)

    assert result.verdict == RiskVerdict.APPROVE
    assert any("Below diversification floor" in r for r in result.veto_reasons)


def test_per_ticker_var_normalized_to_full_weight() -> None:
    """RE-7 regression: a volatile single ticker at weight=0.15 must REDUCE, not APPROVE.

    Before the fix, var99/cvar were computed from the weight-diluted return series
    (0.15x), so the same var_limit was ~6.7x looser for a per-ticker call than for
    the portfolio call — a stock volatile enough to clearly breach the limit at full
    weight passed the per-ticker gate outright.
    """
    engine = RiskStatisticalEngine()
    np.random.seed(3)
    hist = {
        "AAPL": _random_prices(_TRADING_YEAR, 0.0, 0.05),
        "SPY": _random_prices(_TRADING_YEAR, 0.0, 0.01),
    }

    result = engine.evaluate([{"ticker": "AAPL", "weight": 0.15}], hist, current_vix=20.0)

    assert result.verdict == RiskVerdict.REDUCE
    assert result.var_99 > 0.03


def test_ols_portfolio_beta_uses_lookback_window() -> None:
    """RE-14 regression: beta must come from RISK.returns_lookback_days, not the full series.

    Without .tail(lookback), a DailyBarCache holding more than a year of
    history would silently widen beta's window, diluting a recent, real
    change in co-movement with older data the risk gates were never
    calibrated against.
    """
    total_days = RISK.returns_lookback_days * 2 + 5
    dates = pd.date_range(start="2020-01-01", periods=total_days, freq="B")

    np.random.seed(7)
    spy_returns = np.random.normal(0.0005, 0.01, total_days)

    # Pre-lookback segment: unrelated to SPY (beta ~ 0). In-lookback segment: exactly
    # 2x SPY's return each day (beta == 2). Only .tail(lookback) should see the latter.
    split = total_days - RISK.returns_lookback_days - 1
    old_segment = np.random.normal(0.0, 0.02, split)
    recent_segment = spy_returns[split:] * 2.0
    asset_returns = np.concatenate([old_segment, recent_segment])

    price_history = {
        "AAPL": _prices_from_returns(asset_returns, dates),
        "SPY": _prices_from_returns(spy_returns, dates),
    }

    beta = ols_portfolio_beta([{"ticker": "AAPL", "weight": 0.1}], price_history)

    assert beta == pytest.approx(2.0, abs=0.1)


def test_compute_asset_returns_drops_short_history_ticker_without_shrinking_others() -> None:
    """RE-15 regression: one short-history ticker must not truncate the joint returns matrix.

    Before the fix, pd.DataFrame(returns_dict).dropna() intersected on the
    short ticker's handful of overlapping dates, shrinking every other
    ticker's return series down to that same handful of rows and wrecking
    the covariance SLSQP and the VaR/CVaR gates depend on.
    """
    short_dates = _TRADING_YEAR[-5:]

    hist = {
        "AAPL": pd.Series(np.linspace(100, 150, len(_TRADING_YEAR)), index=_TRADING_YEAR),
        "MSFT": pd.Series(np.linspace(200, 250, len(_TRADING_YEAR)), index=_TRADING_YEAR),
        "NEWCO": pd.Series(np.linspace(10, 11, len(short_dates)), index=short_dates),
    }
    positions = [{"ticker": t} for t in ("AAPL", "MSFT", "NEWCO")]

    dropped: list[str] = []
    df = compute_asset_returns(positions, hist, dropped=dropped)

    assert dropped == ["NEWCO"]
    assert "NEWCO" not in df.columns
    assert len(df) > 200


def test_evaluate_excludes_short_history_ticker_from_covariance(monkeypatch) -> None:
    """RE-15 regression: evaluate() surfaces the excluded ticker as a Covariance veto_reason
    note (picked up by graph.py's error-surfacing filter) rather than silently degrading
    every other ticker's covariance.
    """
    monkeypatch.setattr("argus.agents.risk.get_sector", lambda ticker: "Diversified")
    engine = RiskStatisticalEngine()

    np.random.seed(5)
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]
    hist = {t: _random_prices(_TRADING_YEAR, 0.001, 0.015) for t in tickers}
    hist["SPY"] = _random_prices(_TRADING_YEAR, 0.0005, 0.01)
    hist["NEWCO"] = pd.Series(np.linspace(10, 11, 5), index=_TRADING_YEAR[-5:])

    positions = [{"ticker": t, "weight": 0.1} for t in [*tickers, "NEWCO"]]
    result = engine.evaluate(positions, hist, current_vix=20.0)

    assert any(r.startswith("Covariance: excluded NEWCO") for r in result.veto_reasons)
    assert "NEWCO" not in result.optimal_weights


def test_evaluate_skips_optimizer_on_non_finite_covariance(monkeypatch) -> None:
    """RE-15 regression: a non-finite covariance (e.g. a zero-price data glitch producing an
    infinite return that survives dropna()) must not reach SLSQP silently.
    """
    monkeypatch.setattr("argus.agents.risk.get_sector", lambda ticker: "Diversified")
    engine = RiskStatisticalEngine()

    np.random.seed(11)
    aapl = _random_prices(_TRADING_YEAR, 0.0005, 0.01)
    aapl.iloc[100] = 0.0  # a zero print makes the next day's pct_change +inf, not NaN
    hist = {
        "AAPL": aapl,
        "MSFT": _random_prices(_TRADING_YEAR, 0.0005, 0.01),
        "SPY": _random_prices(_TRADING_YEAR, 0.0003, 0.008),
    }
    positions = [{"ticker": "AAPL", "weight": 0.1}, {"ticker": "MSFT", "weight": 0.1}]

    result = engine.evaluate(positions, hist, current_vix=20.0)

    assert result.optimal_weights == {}
    assert result.optimizer_converged is None
    assert any(r.startswith("Covariance: non-finite") for r in result.veto_reasons)
