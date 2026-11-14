from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from argus.agents.risk import (
    RiskStatisticalAgent,
    avg_pairwise_correlation,
    compute_asset_returns,
    compute_portfolio_returns,
    get_sector,
    ols_portfolio_beta,
)
from argus.data.cache import TTLCache
from argus.params import RISK
from argus.schemas.signals import RiskVerdict
from argus.seams import FixtureMarketDataProvider, LiveMarketDataProvider
from tests.helpers.market_data import FakeMarketData

_TRADING_YEAR = pd.date_range(start="2023-01-01", periods=253, freq="B")


def _prices_from_returns(returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    """Compound daily returns into a price series starting at 100."""
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)


def _random_prices(dates: pd.DatetimeIndex, mean: float, vol: float) -> pd.Series:
    """Build a price series from N(mean, vol) daily returns off the active numpy seed."""
    return _prices_from_returns(np.random.normal(mean, vol, len(dates)), dates)


@pytest.fixture
def price_history():
    """Build a simulated price history dictionary for tests."""
    np.random.seed(42)

    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC"]
    hist = {t: _random_prices(_TRADING_YEAR, 0.001, 0.015) for t in tickers}
    hist["SPY"] = _random_prices(_TRADING_YEAR, 0.0005, 0.01)

    return hist


@pytest.fixture
def risk_engine():
    """Return a fresh statistical risk agent."""
    return RiskStatisticalAgent(market_data=FixtureMarketDataProvider())


def test_vix_blackout(risk_engine: RiskStatisticalAgent, price_history: dict) -> None:
    positions = [{"ticker": "AAPL", "weight": 0.1} for _ in range(5)]
    result = risk_engine.evaluate(positions, price_history, current_vix=40.0)

    assert result.verdict == RiskVerdict.VETO
    assert any("blackout" in r.lower() for r in result.veto_reasons)
    # Keep uncomputed correlation as None when structural rejection skips statistical measurement.
    assert result.avg_correlation is None


def test_overweight_position(risk_engine: RiskStatisticalAgent, price_history: dict) -> None:
    positions = [
        {"ticker": "AAPL", "weight": 0.20},
        {"ticker": "MSFT", "weight": 0.10},
        {"ticker": "GOOGL", "weight": 0.10},
        {"ticker": "META", "weight": 0.10},
        {"ticker": "AMZN", "weight": 0.10},
    ]
    result = risk_engine.evaluate(positions, price_history, current_vix=20.0)

    assert result.verdict == RiskVerdict.VETO
    assert any("weight" in r.lower() for r in result.veto_reasons)


def test_high_var_reduce(risk_engine: RiskStatisticalAgent) -> None:
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]

    hist = {t: _random_prices(_TRADING_YEAR, 0.0, 0.15) for t in tickers}
    hist["SPY"] = _random_prices(_TRADING_YEAR, 0.0, 0.01)

    positions = [{"ticker": t, "weight": 0.15} for t in tickers]

    result = risk_engine.evaluate(positions, hist, current_vix=20.0)

    assert result.verdict == RiskVerdict.REDUCE
    assert result.var_99 > 0.03


def test_approve_healthy_portfolio(risk_engine: RiskStatisticalAgent, price_history: dict) -> None:
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
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC"]
    weight = 0.125
    positions = [{"ticker": t, "weight": weight} for t in tickers]
    w = np.full(len(tickers), weight)

    asset_returns = compute_asset_returns(positions, price_history)
    cov = asset_returns.cov() * 252
    fixed_variance = float(np.dot(w, np.dot(cov, w)))

    weighted_returns = compute_portfolio_returns(positions, price_history)
    true_portfolio_variance = float(weighted_returns.sum(axis=1).var() * 252)
    assert fixed_variance == pytest.approx(true_portfolio_variance, rel=0.05)

    # Compare against double-weighted covariance to detect accidental weighting before risk estimation.
    buggy_cov = weighted_returns.cov() * 252
    buggy_variance = float(np.dot(w, np.dot(buggy_cov, w)))
    assert fixed_variance / buggy_variance == pytest.approx(1.0 / (weight**2), rel=0.05)


def test_zero_api_calls(risk_engine: RiskStatisticalAgent, price_history: dict) -> None:
    positions = [{"ticker": "AAPL", "weight": 0.1} for _ in range(5)]
    result = risk_engine.evaluate(positions, price_history, current_vix=40.0)

    assert result.api_calls_used == 0


def test_small_universe_is_noted_not_vetoed(risk_engine: RiskStatisticalAgent, price_history: dict) -> None:
    positions = [{"ticker": t, "weight": 0.1} for t in ["AAPL", "MSFT", "GOOGL"]]
    result = risk_engine.evaluate(positions, price_history, current_vix=20.0)

    assert result.verdict == RiskVerdict.APPROVE
    assert any("Below diversification floor" in r for r in result.veto_reasons)


def test_per_ticker_var_normalized_to_full_weight() -> None:
    engine = RiskStatisticalAgent(market_data=FixtureMarketDataProvider())
    np.random.seed(3)
    hist = {
        "AAPL": _random_prices(_TRADING_YEAR, 0.0, 0.05),
        "SPY": _random_prices(_TRADING_YEAR, 0.0, 0.01),
    }

    result = engine.evaluate([{"ticker": "AAPL", "weight": 0.15}], hist, current_vix=20.0)

    assert result.verdict == RiskVerdict.REDUCE
    assert result.var_99 > 0.03


def test_ols_portfolio_beta_uses_lookback_window() -> None:
    total_days = RISK.returns_lookback_days * 2 + 5
    dates = pd.date_range(start="2020-01-01", periods=total_days, freq="B")

    np.random.seed(7)
    spy_returns = np.random.normal(0.0005, 0.01, total_days)

    # Use beta zero before the lookback and beta two within it to expose accidental whole-history fitting.
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
    monkeypatch.setattr("argus.agents.risk.get_sector", lambda ticker, market_data: "Diversified")
    engine = RiskStatisticalAgent(market_data=FixtureMarketDataProvider())

    np.random.seed(5)
    tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]
    hist = {t: _random_prices(_TRADING_YEAR, 0.001, 0.015) for t in tickers}
    hist["SPY"] = _random_prices(_TRADING_YEAR, 0.0005, 0.01)
    hist["NEWCO"] = pd.Series(np.linspace(10, 11, 5), index=_TRADING_YEAR[-5:])

    positions = [{"ticker": t, "weight": 0.1} for t in [*tickers, "NEWCO"]]
    result = engine.evaluate(positions, hist, current_vix=20.0)

    assert any(r.startswith("Covariance: excluded NEWCO") for r in result.veto_reasons)
    assert "NEWCO" not in result.optimal_weights


def test_avg_pairwise_correlation_returns_none_without_overlap() -> None:
    disjoint_dates = pd.date_range(start="2010-01-01", periods=len(_TRADING_YEAR), freq="B")
    np.random.seed(1)
    returns_matrix = pd.DataFrame(
        {
            "AAPL": pd.Series(np.random.normal(0.001, 0.01, len(_TRADING_YEAR)), index=_TRADING_YEAR),
            "NEWCO": pd.Series(np.random.normal(0.001, 0.01, len(disjoint_dates)), index=disjoint_dates),
        }
    )
    assert avg_pairwise_correlation(returns_matrix) is None


def test_evaluate_completes_when_a_ticker_has_no_price_history_overlap(monkeypatch) -> None:
    monkeypatch.setattr("argus.agents.risk.get_sector", lambda ticker, market_data: "Diversified")
    engine = RiskStatisticalAgent(market_data=FixtureMarketDataProvider())

    np.random.seed(9)
    disjoint_dates = pd.date_range(start="2010-01-01", periods=len(_TRADING_YEAR), freq="B")
    hist = {
        "AAPL": _random_prices(_TRADING_YEAR, 0.0005, 0.01),
        "SPY": _random_prices(_TRADING_YEAR, 0.0005, 0.01),
        "NEWCO": _random_prices(disjoint_dates, 0.0005, 0.01),
    }
    positions = [{"ticker": "AAPL", "weight": 0.1}, {"ticker": "NEWCO", "weight": 0.1}]

    result = engine.evaluate(positions, hist, current_vix=20.0)

    assert result.verdict in (RiskVerdict.APPROVE, RiskVerdict.REDUCE)
    assert result.avg_correlation is None
    assert result.var_99 is None
    assert result.cvar is None
    assert any("not measurable" in r for r in result.veto_reasons)


def test_evaluate_skips_optimizer_on_non_finite_covariance(monkeypatch) -> None:
    monkeypatch.setattr("argus.agents.risk.get_sector", lambda ticker, market_data: "Diversified")
    engine = RiskStatisticalAgent(market_data=FixtureMarketDataProvider())

    np.random.seed(11)
    aapl = _random_prices(_TRADING_YEAR, 0.0005, 0.01)
    aapl.iloc[100] = 0.0
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


def test_get_sector_does_not_cache_across_non_live_providers(monkeypatch) -> None:
    monkeypatch.setattr(
        "argus.agents.risk._SECTOR_CACHE",
        TTLCache(
            ttl=timedelta(seconds=RISK.sector_cache_ttl_seconds),
            max_entries=RISK.sector_cache_max_entries,
        ),
    )

    session_one = FakeMarketData(ticker_info={"sector": "Technology"})
    session_two = FakeMarketData(ticker_info={"sector": "Energy"})

    assert get_sector("AAPL", session_one) == "Technology"
    assert get_sector("AAPL", session_two) == "Energy"
    assert session_one.calls["ticker_info"] == 1
    assert session_two.calls["ticker_info"] == 1


class _StubLiveMarketData(LiveMarketDataProvider):
    """A LiveMarketDataProvider whose ticker_info is a counted stub, not a real yfinance call."""

    def __init__(self, sector: str) -> None:
        self._sector = sector
        self.calls = 0

    def ticker_info(self, ticker: str) -> dict:
        self.calls += 1
        return {"sector": self._sector}


class _FlakyLiveMarketData(LiveMarketDataProvider):
    """A LiveMarketDataProvider whose ticker_info fails once, then succeeds."""

    def __init__(self, sector: str) -> None:
        self._sector = sector
        self.calls = 0

    def ticker_info(self, ticker: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("yfinance blip")
        return {"sector": self._sector}


def test_get_sector_retries_after_a_failed_fetch_instead_of_caching_unknown(monkeypatch) -> None:
    monkeypatch.setattr(
        "argus.agents.risk._SECTOR_CACHE",
        TTLCache(ttl=timedelta(seconds=RISK.sector_cache_ttl_seconds), max_entries=10),
    )
    market_data = _FlakyLiveMarketData("Technology")

    assert get_sector("AAPL", market_data) == "Unknown"
    assert get_sector("AAPL", market_data) == "Technology"
    assert market_data.calls == 2


def test_get_sector_caches_a_successful_response_that_lacks_a_sector(monkeypatch) -> None:
    monkeypatch.setattr(
        "argus.agents.risk._SECTOR_CACHE",
        TTLCache(ttl=timedelta(seconds=RISK.sector_cache_ttl_seconds), max_entries=10),
    )

    class _NoSector(_StubLiveMarketData):
        def ticker_info(self, ticker: str) -> dict:
            self.calls += 1
            return {}

    market_data = _NoSector("unused")

    assert get_sector("AAPL", market_data) == "Unknown"
    assert get_sector("AAPL", market_data) == "Unknown"
    assert market_data.calls == 1


def test_get_sector_cache_evicts_the_oldest_entry_at_capacity(monkeypatch) -> None:
    monkeypatch.setattr(
        "argus.agents.risk._SECTOR_CACHE",
        TTLCache(ttl=timedelta(seconds=RISK.sector_cache_ttl_seconds), max_entries=2),
    )
    market_data = _StubLiveMarketData("Technology")

    get_sector("AAPL", market_data)
    get_sector("MSFT", market_data)
    get_sector("GOOG", market_data)
    assert market_data.calls == 3

    get_sector("AAPL", market_data)
    assert market_data.calls == 4, "AAPL was evicted to keep the cache at its cap"

    get_sector("GOOG", market_data)
    assert market_data.calls == 4, "GOOG is still cached"


def test_unmeasured_veto_statistics_are_null(risk_engine, price_history):
    result = risk_engine.evaluate([{"ticker": "AAPL", "weight": 0.1}], price_history, None)
    assert result.verdict == RiskVerdict.VETO
    for name in ("var_99", "cvar", "marginal_var", "portfolio_beta", "stop_loss"):
        assert getattr(result, name) is None


def test_missing_benchmark_skips_beta_gate_with_a_note(price_history):
    engine = RiskStatisticalAgent(
        market_data=FakeMarketData(ohlcv_daily=OSError(), ticker_info={"sector": "Technology"})
    )
    history = {"AAPL": price_history["AAPL"]}
    assert ols_portfolio_beta([{"ticker": "AAPL", "weight": 0.1}], history, market_data=engine.market_data) is None
    result = engine.evaluate([{"ticker": "AAPL", "weight": 0.1}], history, 20.0)
    assert result.portfolio_beta is None
    assert any("beta not measurable" in note for note in result.veto_reasons)
    assert not any(note.startswith("Beta ") for note in result.veto_reasons)
