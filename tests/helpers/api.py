"""Stand-ins for the API's runtime singletons (api.lifespan.runtime)."""

from unittest import mock

import pytest

from api.lifespan import runtime


def api_pipeline(monkeypatch: pytest.MonkeyPatch, *, market_hours: bool = True, interval_minutes: int = 1) -> mock.Mock:
    """Install a fake MFTDataPipeline as `runtime.pipeline`."""
    fake = mock.Mock()
    fake.is_market_hours.return_value = market_hours
    fake.interval_minutes = interval_minutes
    monkeypatch.setattr(runtime, "pipeline", fake)
    return fake
