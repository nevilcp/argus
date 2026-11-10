"""Stand-ins for api.main's module-level singletons."""

from unittest import mock

import pytest

import api.main as api_main


def api_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, market_hours: bool = True, interval_minutes: int = 1
) -> mock.Mock:
    """Installs a fake MFTDataPipeline as api.main's module-level `_mft_pipeline`.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
        market_hours: What the fake's `is_market_hours()` returns.
        interval_minutes: The fake's candle interval.

    Returns:
        The installed fake.
    """
    fake = mock.Mock()
    fake.is_market_hours.return_value = market_hours
    fake.interval_minutes = interval_minutes
    monkeypatch.setattr(api_main, "_mft_pipeline", fake)
    return fake
