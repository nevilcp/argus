"""
tests/helpers/http_errors.py

Exceptions shaped like the ones the HTTP clients behind the fetchers raise.
"""

from __future__ import annotations

from unittest import mock

import requests


def http_error(status_code: int, headers: dict | None = None) -> requests.exceptions.HTTPError:
    """Builds an HTTPError whose .response mimics a real requests response."""
    response = mock.Mock(status_code=status_code, headers=headers or {})
    return requests.exceptions.HTTPError(response=response)


def crumb_error() -> requests.exceptions.HTTPError:
    """Builds a 401 HTTPError matching Yahoo's actual "Invalid Crumb" response body."""
    response = mock.Mock(status_code=401, headers={})
    return requests.exceptions.HTTPError(
        '{"finance":{"result":null,"error":{"code":"Unauthorized","description":"Invalid Crumb"}}}',
        response=response,
    )
