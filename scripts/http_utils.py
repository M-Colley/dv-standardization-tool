"""Shared HTTP helper constants and predicates."""

from __future__ import annotations

import httpx

TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def is_retryable_http_exception(exc: BaseException) -> bool:
    """Return whether an HTTP exception should trigger a retry."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in TRANSIENT_HTTP_STATUS_CODES
