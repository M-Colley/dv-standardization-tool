"""Open Science Framework (OSF) source-loader helpers.

Wraps the OSF v2 JSON API for the batch pipeline:

* ``_extract_osf_project_id`` — accept a 5-char project id or any
  ``osf.io/<id>`` URL and return the canonical lowercase id.
* ``_osf_json_get`` — GET JSON:API payload from an OSF endpoint with
  retry/backoff via ``http_utils._read_url_bytes``.
* ``_iter_osf_child_node_ids`` / ``_iter_osf_node_ids`` — recursively
  enumerate component nodes attached to a project.
* ``_iter_osf_provider_urls`` / ``_iter_osf_file_entries`` — enumerate
  storage providers and the files they contain (recurses into folders).
* ``_download_osf_file`` — fetch a single file URL into a destination
  path on disk.

This module imports only from stdlib + ``scripts.http_utils`` so it can
be unit-tested without booting the full pipeline.
``run_batch_standardization`` re-exports every public name for backwards
compatibility.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib import request

from scripts.http_utils import HTTP_USER_AGENT, NETWORK_TIMEOUTS, _read_url_bytes

OSF_API_BASE = "https://api.osf.io/v2"


# ---------------------------------------------------------------------------
# Project id resolution
# ---------------------------------------------------------------------------

def _extract_osf_project_id(location: str) -> str:
    """Resolve OSF project id from a raw id or osf.io URL."""
    location = (location or "").strip()
    if not location:
        raise ValueError("OSF location must be a non-empty project id or osf.io URL.")

    if re.fullmatch(r"[a-z0-9]{5}", location.lower()):
        return location.lower()

    # Accept 5-char alphanumeric IDs with mixed case.
    if re.fullmatch(r"[A-Za-z0-9]{5}", location):
        return location.lower()

    match = re.search(r"osf\.io/([a-z0-9]{5})(?:/|$)", location.lower())
    if not match:
        raise ValueError(
            f"Invalid OSF location: '{location}'. "
            "Use a 5-character alphanumeric project id (e.g., cwd6h) "
            "or an OSF URL such as https://osf.io/cwd6h/overview."
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# JSON:API helpers
# ---------------------------------------------------------------------------

def _osf_json_get(url: str) -> dict[str, Any]:
    """GET a JSON:API payload from the OSF v2 API with retry/backoff."""
    req = request.Request(
        url,
        headers={
            "Accept": "application/vnd.api+json",
            "User-Agent": HTTP_USER_AGENT,
        },
    )
    payload = _read_url_bytes(req, timeout=NETWORK_TIMEOUTS["api_seconds"]).decode("utf-8")
    return json.loads(payload)


# ---------------------------------------------------------------------------
# Node traversal
# ---------------------------------------------------------------------------

def _iter_osf_child_node_ids(node_id: str) -> list[str]:
    """List direct child component node IDs for an OSF node."""
    child_ids: list[str] = []
    page_url: str | None = f"{OSF_API_BASE}/nodes/{node_id}/children/"

    while page_url:
        payload = _osf_json_get(page_url)
        for item in payload.get("data", []):
            child_id = str(item.get("id", "")).strip()
            if child_id:
                child_ids.append(child_id)

        next_link = payload.get("links", {}).get("next")
        page_url = str(next_link) if next_link else None

    return child_ids


def _iter_osf_node_ids(project_id: str) -> list[str]:
    """List root+component node IDs reachable from an OSF project (BFS)."""
    ordered: list[str] = []
    seen: set[str] = set()
    queue: list[str] = [project_id]

    while queue:
        node_id = queue.pop(0)
        if node_id in seen:
            continue

        seen.add(node_id)
        ordered.append(node_id)
        queue.extend(_iter_osf_child_node_ids(node_id))

    return ordered


# ---------------------------------------------------------------------------
# Storage provider + file traversal
# ---------------------------------------------------------------------------

def _iter_osf_provider_urls(node_id: str) -> list[str]:
    """List all storage-provider listing URLs for an OSF node."""
    provider_urls: list[str] = []
    page_url: str | None = f"{OSF_API_BASE}/nodes/{node_id}/files/"

    while page_url:
        payload = _osf_json_get(page_url)
        for item in payload.get("data", []):
            related = (
                item.get("relationships", {})
                .get("files", {})
                .get("links", {})
                .get("related", {})
                .get("href")
            )
            if not related:
                related = item.get("links", {}).get("related", {}).get("href")
            if related:
                provider_urls.append(str(related))

        next_link = payload.get("links", {}).get("next")
        page_url = str(next_link) if next_link else None

    return provider_urls


def _iter_osf_file_entries(node_id: str) -> list[dict[str, Any]]:
    """List file entries for all storage providers under an OSF node (recursive)."""
    entries: list[dict[str, Any]] = []
    queue: list[str] = _iter_osf_provider_urls(node_id)
    seen_pages: set[str] = set()

    while queue:
        page_url = queue.pop(0)
        while page_url:
            if page_url in seen_pages:
                break
            seen_pages.add(page_url)

            payload = _osf_json_get(page_url)
            data = payload.get("data", [])
            for item in data:
                attrs = item.get("attributes", {}) or {}
                kind = attrs.get("kind")
                if kind == "file":
                    entries.append(item)
                elif kind == "folder":
                    related = (
                        item.get("relationships", {})
                        .get("files", {})
                        .get("links", {})
                        .get("related", {})
                        .get("href")
                    )
                    if related:
                        queue.append(str(related))

            next_link = payload.get("links", {}).get("next")
            page_url = str(next_link) if next_link else None

    return entries


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def _download_osf_file(download_url: str, destination: Path) -> None:
    """Download an OSF file URL into ``destination`` (parents created)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        _read_url_bytes(download_url, timeout=NETWORK_TIMEOUTS["download_seconds"])
    )


__all__ = [
    "OSF_API_BASE",
    "_extract_osf_project_id",
    "_osf_json_get",
    "_iter_osf_child_node_ids",
    "_iter_osf_node_ids",
    "_iter_osf_provider_urls",
    "_iter_osf_file_entries",
    "_download_osf_file",
]
