"""HTTP fetch helpers shared by the remote source loaders.

This module bundles the URL retry/backoff client, the landing-page HTML
parser used to scrape dataset download links, the Cloudflare/login
challenge detector, and the small bag of constants (timeouts, hosts,
markers) that drive them.

Like ``scripts.archive_utils`` this module is deliberately stdlib-only so
it can be imported and unit-tested without paying the cost of
``run_batch_standardization``'s pandas/transformers initialisation.
``run_batch_standardization`` re-exports every public name (constants,
classes, helpers) so existing tests and callers continue to work.

Security / robustness notes:

* ``_read_url_response`` retries timeouts, 429s, and 5xx errors with a
  linear-backoff sleep, but never retries 4xx (other than 429). The
  retry count and per-call timeout default to ``NETWORK_TIMEOUTS`` —
  centralised here so corporate-network operators have a single dial.
* ``_normalize_remote_filename`` strips path components and forbids any
  character outside ``[A-Za-z0-9._-]`` to defuse Content-Disposition
  abuse (path traversal, hidden files).
* ``_looks_like_challenge_response`` detects Cloudflare/CAPTCHA
  interstitials by host, ``<title>`` markers, and well-known body
  fragments so the source-loader can surface a clear ``access_restricted``
  ``SourceAccessError`` instead of crashing later in the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request
from urllib.parse import unquote, urljoin, urlparse

from scripts.archive_utils import ARCHIVE_SUFFIXES, DATA_FILE_SUFFIXES, MAPPING_SUFFIXES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HTTP_USER_AGENT = "OpenDV-HCI/1.0"

# Centralised network tunables. Previously these magic numbers were sprinkled
# across _read_url_response, _read_url_bytes, and the OSF/web download helpers,
# which made it hard to reason about retry/timeout behaviour in tests or to
# adjust caps for slow corporate networks. Edit values here to retune.
NETWORK_TIMEOUTS: dict[str, int] = {
    # Long-lived bulk downloads (OSF blobs, archive payloads).
    "download_seconds": 180,
    # Web-dataset landing-page fetches (HTML or small JSON).
    "landing_seconds": 60,
    # Lightweight JSON API calls (OSF metadata).
    "api_seconds": 45,
    # Number of attempts for transient network failures (timeouts, 429, 5xx).
    "max_retry_attempts": 4,
}

WEB_LOGIN_MARKERS = (
    "login to access dataset files",
    "create a free account",
    "/saml_login",
    "login required",
    "sign in",
)

# CJK markers were duplicated in the original module — kept here for
# clarity. Both the unicode-escaped and literal variants are exposed.
WEB_NO_DATA_MARKERS_ASCII = (
    "no data",
    "no downloadable data",
    "\u3010\u7ed3\u675f\u3011",
    "\u5df2\u7ed3\u675f",
    "\u95ee\u5377\u5df2\u7ed3\u675f",
    "\u8c03\u67e5\u5df2\u7ed3\u675f",
    "\u6682\u65e0\u6570\u636e",
    "\u6ca1\u6709\u6570\u636e",
)
WEB_NO_DATA_MARKERS = WEB_NO_DATA_MARKERS_ASCII

# Hosts that are known to gate content behind interactive challenges
# (Cloudflare, bot management) and therefore cannot be fetched with a
# plain HTTP client.
WEB_CHALLENGE_HOSTS = {
    "dl.acm.org",
    "www.dl.acm.org",
}
WEB_CHALLENGE_TITLE_MARKERS = (
    "just a moment",
    "attention required",
    "access denied",
    "checking your browser",
)
WEB_CHALLENGE_BODY_MARKERS = (
    "challenges.cloudflare.com",
    "cf_chl_",
    "cf-mitigated",
    "enable javascript and cookies to continue",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SourceAccessError(RuntimeError):
    """Raised when a remote source is reachable but refuses public access.

    ``status`` is a short machine-readable tag (e.g. ``access_restricted``,
    ``login_required``, ``not_available``) consumed by the manifest runner
    to decide between WARN-and-skip vs. fail-the-batch behaviour.
    """

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Landing page HTML parser
# ---------------------------------------------------------------------------

class _LandingPageParser(HTMLParser):
    """Minimal HTML parser that captures ``<title>``, ``<a href=...>``, and
    ``<script type="application/ld+json">`` payloads from a dataset landing
    page. Stateful — instantiate one per page.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._capture_ldjson = False
        self._current_ldjson: list[str] = []
        self.title_parts: list[str] = []
        self.links: list[str] = []
        self.ldjson_blocks: list[str] = []

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() == "a":
            href = attr_map.get("href")
            if href:
                self.links.append(href)
        if tag.lower() == "script" and "ld+json" in str(attr_map.get("type", "")).lower():
            self._capture_ldjson = True
            self._current_ldjson = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        if tag.lower() == "script" and self._capture_ldjson:
            block = "".join(self._current_ldjson).strip()
            if block:
                self.ldjson_blocks.append(block)
            self._capture_ldjson = False
            self._current_ldjson = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._capture_ldjson:
            self._current_ldjson.append(data)


# ---------------------------------------------------------------------------
# Filename / response helpers
# ---------------------------------------------------------------------------

def _extract_content_disposition_filename(disposition: str | None) -> str | None:
    if not disposition:
        return None

    utf8_match = re.search(r"filename\*=UTF-8''([^;]+)", disposition, flags=re.IGNORECASE)
    if utf8_match:
        return unquote(utf8_match.group(1).strip().strip('"'))

    plain_match = re.search(r'filename="?([^";]+)"?', disposition, flags=re.IGNORECASE)
    if plain_match:
        return plain_match.group(1).strip()
    return None


def _infer_extension_from_content_type(content_type: str) -> str:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    return {
        "application/zip": ".zip",
        "text/csv": ".csv",
        "text/tab-separated-values": ".tsv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.ms-excel": ".xls",
        "application/x-yaml": ".yaml",
        "text/yaml": ".yaml",
        "text/x-yaml": ".yaml",
    }.get(normalized, "")


def _normalize_remote_filename(
    raw_name: str | None,
    fallback_prefix: str,
    final_url: str,
    content_type: str,
) -> str:
    candidates = [raw_name or ""]
    url_name = Path(unquote(urlparse(final_url).path)).name
    if url_name:
        candidates.append(url_name)

    for candidate in candidates:
        clean = Path(str(candidate)).name.strip().strip('"').strip("'")
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", clean).strip("._")
        if clean:
            return clean

    extension = _infer_extension_from_content_type(content_type)
    return f"{fallback_prefix}{extension}"


def _build_unique_destination(target_dir: Path, filename: str) -> Path:
    destination = target_dir / filename
    stem = destination.stem
    suffix = destination.suffix
    counter = 2
    while destination.exists():
        destination = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return destination


def _looks_like_supported_download_url(url: str) -> bool:
    parsed = urlparse(url)
    suffix = Path(unquote(parsed.path)).suffix.lower()
    return (
        suffix in DATA_FILE_SUFFIXES
        or suffix in ARCHIVE_SUFFIXES
        or suffix in MAPPING_SUFFIXES
        or "/ndownloader/" in parsed.path
    )


def _is_supported_download_response(final_url: str, headers: Any) -> bool:
    suffixes = {Path(unquote(urlparse(final_url).path)).suffix.lower()}
    disposition = headers.get("Content-Disposition") or headers.get("Content-disposition")
    disposition_name = _extract_content_disposition_filename(disposition)
    if disposition_name:
        suffixes.add(Path(disposition_name).suffix.lower())
    if any(suffix in DATA_FILE_SUFFIXES or suffix in ARCHIVE_SUFFIXES or suffix in MAPPING_SUFFIXES for suffix in suffixes):
        return True

    content_type = (
        headers.get_content_type()
        if hasattr(headers, "get_content_type")
        else str(headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
    )
    return content_type in {
        "application/zip",
        "text/csv",
        "text/tab-separated-values",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    }


# ---------------------------------------------------------------------------
# URL fetch with retry/backoff
# ---------------------------------------------------------------------------

def _read_url_response(
    url_or_request: Any,
    timeout: int,
    max_attempts: int | None = None,
) -> tuple[bytes, Any, str]:
    """Fetch a URL with linear-backoff retry on transient failures.

    Returns ``(body_bytes, headers, final_url)``. Retries timeouts, 429,
    and 5xx; never retries other 4xx. ``max_attempts`` defaults to
    ``NETWORK_TIMEOUTS["max_retry_attempts"]`` so callers can override per
    request without leaking the constant.
    """
    if max_attempts is None:
        max_attempts = int(NETWORK_TIMEOUTS["max_retry_attempts"])
    for attempt in range(max_attempts):
        try:
            with request.urlopen(url_or_request, timeout=timeout) as resp:
                return resp.read(), resp.headers, resp.geturl()
        except Exception as exc:  # noqa: BLE001
            is_timeout = isinstance(exc, (TimeoutError, socket.timeout))
            if isinstance(exc, urlerror.URLError):
                is_timeout = is_timeout or isinstance(exc.reason, socket.timeout)
            is_transient_http = (
                isinstance(exc, urlerror.HTTPError)
                and (exc.code == 429 or 500 <= exc.code < 600)
            )
            if attempt == max_attempts - 1 or not (is_timeout or is_transient_http):
                logger.debug(
                    "URL request failed permanently after %d attempt(s): %s",
                    attempt + 1, exc,
                )
                raise
            logger.debug(
                "URL request transient failure on attempt %d/%d: %s",
                attempt + 1, max_attempts, exc,
            )
            time.sleep(0.75 * (attempt + 1))


def _read_url_bytes(
    url_or_request: Any,
    timeout: int,
    max_attempts: int | None = None,
) -> bytes:
    """Read bytes from URL with retry/backoff for transient network errors."""
    payload, _, _ = _read_url_response(url_or_request, timeout=timeout, max_attempts=max_attempts)
    return payload


def _decode_http_text(payload: bytes, headers: Any) -> str:
    charset = headers.get_content_charset() if hasattr(headers, "get_content_charset") else None
    encoding = charset or "utf-8"
    return payload.decode(encoding, errors="replace")


# ---------------------------------------------------------------------------
# Landing page link extraction + challenge detection
# ---------------------------------------------------------------------------

def _iter_content_urls(payload: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "contentUrl" and isinstance(value, str):
                urls.append(value)
            else:
                urls.extend(_iter_content_urls(value))
    elif isinstance(payload, list):
        for item in payload:
            urls.extend(_iter_content_urls(item))
    return urls


def _extract_html_download_urls(html_text: str, base_url: str) -> tuple[str, list[str]]:
    parser = _LandingPageParser()
    parser.feed(html_text)

    urls: set[str] = set()
    for block in parser.ldjson_blocks:
        try:
            payload = json.loads(unescape(block))
        except json.JSONDecodeError:
            continue
        for content_url in _iter_content_urls(payload):
            absolute_url = urljoin(base_url, content_url)
            if _looks_like_supported_download_url(absolute_url):
                urls.add(absolute_url)

    for href in parser.links:
        absolute_url = urljoin(base_url, unescape(href))
        if _looks_like_supported_download_url(absolute_url):
            urls.add(absolute_url)

    return parser.title, sorted(urls)


def _looks_like_challenge_response(
    host: str,
    title: str,
    html_text: str,
) -> bool:
    normalized_host = host.lower()
    if normalized_host in WEB_CHALLENGE_HOSTS:
        return True
    normalized_title = title.lower()
    if any(marker in normalized_title for marker in WEB_CHALLENGE_TITLE_MARKERS):
        return True
    normalized_html = html_text.lower()
    return any(marker in normalized_html for marker in WEB_CHALLENGE_BODY_MARKERS)


def _build_web_dataset_access_error(
    location: str,
    final_url: str,
    html_text: str,
    title: str,
) -> SourceAccessError:
    resolved_url = final_url or location
    host = urlparse(resolved_url).netloc.lower()
    normalized_html = html_text.lower()
    normalized_title = title.lower()

    if _looks_like_challenge_response(host, title, html_text):
        return SourceAccessError(
            "access_restricted",
            (
                f"Dataset host {host or resolved_url} returned a browser/CAPTCHA challenge "
                f"(Cloudflare-style) for {resolved_url}. Automated download is not possible; "
                "download the file manually and re-ingest it with source_type: local_path."
            ),
        )

    if host.endswith("ieee-dataport.org") or any(marker in normalized_html for marker in WEB_LOGIN_MARKERS):
        return SourceAccessError(
            "login_required",
            f"Dataset files require a login or account at {resolved_url}. No public download links were detected.",
        )

    if host.endswith("wjx.cn") or any(
        marker in html_text or marker in title for marker in WEB_NO_DATA_MARKERS_ASCII
    ):
        return SourceAccessError(
            "not_available",
            f"Dataset is not available at {resolved_url}. The page indicates it is closed or does not expose downloadable data.",
        )

    if host.endswith("data.4tu.nl"):
        return SourceAccessError(
            "not_available",
            f"No downloadable dataset files were detected on the 4TU landing page: {resolved_url}",
        )

    if any(marker in normalized_title for marker in ("closed", "ended")):
        return SourceAccessError(
            "not_available",
            f"Dataset is not available at {resolved_url}. The landing page appears to be closed.",
        )

    return SourceAccessError(
        "not_available",
        f"No downloadable dataset files were detected for {resolved_url}",
    )


# ---------------------------------------------------------------------------
# Download payload writer
# ---------------------------------------------------------------------------

def _write_download_payload(
    target_dir: Path,
    payload: bytes,
    final_url: str,
    headers: Any,
    fallback_prefix: str,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    disposition = headers.get("Content-Disposition") or headers.get("Content-disposition")
    filename = _normalize_remote_filename(
        _extract_content_disposition_filename(disposition),
        fallback_prefix=fallback_prefix,
        final_url=final_url,
        content_type=(
            headers.get_content_type()
            if hasattr(headers, "get_content_type")
            else str(headers.get("Content-Type", ""))
        ),
    )
    destination = _build_unique_destination(target_dir, filename)
    destination.write_bytes(payload)
    return destination


__all__ = [
    # constants
    "HTTP_USER_AGENT",
    "NETWORK_TIMEOUTS",
    "WEB_LOGIN_MARKERS",
    "WEB_NO_DATA_MARKERS",
    "WEB_NO_DATA_MARKERS_ASCII",
    "WEB_CHALLENGE_HOSTS",
    "WEB_CHALLENGE_TITLE_MARKERS",
    "WEB_CHALLENGE_BODY_MARKERS",
    # types
    "SourceAccessError",
    # helpers
    "_LandingPageParser",
    "_extract_content_disposition_filename",
    "_infer_extension_from_content_type",
    "_normalize_remote_filename",
    "_build_unique_destination",
    "_looks_like_supported_download_url",
    "_is_supported_download_response",
    "_read_url_response",
    "_read_url_bytes",
    "_decode_http_text",
    "_iter_content_urls",
    "_extract_html_download_urls",
    "_looks_like_challenge_response",
    "_build_web_dataset_access_error",
    "_write_download_payload",
]
