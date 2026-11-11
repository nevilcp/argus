"""Stand-ins for the API's runtime singletons (api.lifespan.runtime)."""

from unittest import mock

import pytest

from api.lifespan import runtime


def api_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, market_hours: bool = True, interval_minutes: int = 1
) -> mock.Mock:
    """Installs a fake MFTDataPipeline as `runtime.pipeline`.

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
    monkeypatch.setattr(runtime, "pipeline", fake)
    return fake
