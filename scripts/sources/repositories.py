"""Research-data repositories as batch sources: Zenodo, figshare, 4TU.ResearchData, Dataverse.

Each repository exposes a REST API that lists a record's files with their
download URL, size and checksum. This module resolves a manifest
``location`` (record URL, DOI or bare id) to that list and downloads the data,
archive and mapping files, verifying every checksum the repository provides.

====================  =====================  ==============================================
source_type           location examples      API
====================  =====================  ==============================================
zenodo_record         zenodo.org/records/ID  ``https://zenodo.org/api/records/{id}``
                      10.5281/zenodo.ID
figshare_article      figshare.com/articles  ``https://api.figshare.com/v2/articles/{id}``
                      /<type>/<slug>/ID[/V]  (``/versions/{v}`` when a version is given)
                      10.6084/m9.figshare.ID
fourtu_dataset        data.4tu.nl/datasets/  ``https://data.4tu.nl/v2/articles/{uuid}``
                      UUID[/V]               (Djehuty, figshare-compatible)
dataverse_dataset     <server>/dataset.xhtml ``{server}/api/datasets/:persistentId``
                      ?persistentId=doi:...  files via ``/api/access/datafile/{id}``
====================  =====================  ==============================================

Dataverse converts uploaded tables to tab-separated ``.tab`` files; the
original upload is requested (``format=original``) so the original file
format and name are kept.

HTTP 401/403/404/410 become :class:`~scripts.http_utils.SourceAccessError`
(``login_required`` / ``access_restricted`` / ``not_available``) so the batch
skips the source with a warning; 5xx outages propagate and are reported as
temporarily unavailable by the runner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib import request
from urllib.parse import parse_qs, quote, urlparse

import httpx

from scripts.http_utils import (
    HTTP_USER_AGENT,
    NETWORK_TIMEOUTS,
    SourceAccessError,
    _read_url_bytes,
    _stream_url_to_path,
)

logger = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api"
FIGSHARE_API = "https://api.figshare.com/v2"
FOURTU_API = "https://data.4tu.nl/v2"
SUPPORTED_CHECKSUMS = ("md5", "sha1", "sha256")


@dataclass(frozen=True)
class RemoteFile:
    """One file of a repository record."""

    path: str  # relative path inside the record (POSIX)
    url: str
    size: int | None = None
    checksum: tuple[str, str] | None = None  # (algorithm, hex digest)


@dataclass(frozen=True)
class RemoteRecord:
    files: tuple[RemoteFile, ...]
    revision: str  # recorded as the dataset's provenance id ("commit")


# ── HTTP ─────────────────────────────────────────────────────────────────────

_ACCESS_STATUS = {401: "login_required", 403: "access_restricted", 404: "not_available", 410: "not_available"}


def _get_json(url: str) -> Any:
    """GET a JSON document with the shared retrying client."""
    req = request.Request(url, headers={"Accept": "application/json", "User-Agent": HTTP_USER_AGENT})
    try:
        payload = _read_url_bytes(req, timeout=NETWORK_TIMEOUTS["api_seconds"])
    except httpx.HTTPStatusError as exc:
        status = _ACCESS_STATUS.get(exc.response.status_code)
        if status is None:
            raise
        raise SourceAccessError(status, f"HTTP {exc.response.status_code} for {url}") from exc
    return json.loads(payload.decode("utf-8"))


def _download(url: str, destination: Path) -> None:
    _stream_url_to_path(url, destination, timeout=NETWORK_TIMEOUTS["download_seconds"])


def _parse_checksum(value: Any, default_algorithm: str = "md5") -> tuple[str, str] | None:
    """``"md5:abc"`` / ``{"type": "MD5", "value": "abc"}`` / ``"abc"`` -> ``("md5", "abc")``."""
    if not value:
        return None
    if isinstance(value, dict):
        algorithm, digest = str(value.get("type", "")), str(value.get("value", ""))
    elif ":" in str(value):
        algorithm, digest = str(value).split(":", 1)
    else:
        algorithm, digest = default_algorithm, str(value)
    algorithm = algorithm.lower().replace("-", "")
    if algorithm not in SUPPORTED_CHECKSUMS or not digest:
        return None
    return algorithm, digest.lower()


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── Location parsing ─────────────────────────────────────────────────────────

def parse_zenodo_location(location: str) -> str:
    """Record id from a Zenodo URL, DOI (10.5281/zenodo.N) or bare id."""
    text = location.strip()
    for pattern in (r"zenodo\.org/(?:api/)?records?/(\d+)", r"10\.5281/zenodo\.(\d+)", r"^(\d+)$"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot read a Zenodo record id from {location!r}")


def parse_figshare_location(location: str) -> tuple[str, str | None]:
    """(article id, version) from a figshare URL, DOI (10.6084/m9.figshare.N[.vV]) or bare id."""
    text = location.strip()
    match = re.search(r"10\.6084/m9\.figshare\.(\d+)(?:\.v(\d+))?", text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    parsed = urlparse(text)
    if "figshare" in parsed.netloc.lower():
        # .../articles/<type>/<slug>/<id>[/<version>]: the id is the last long
        # number, an optional short number after it is the version.
        segments = [seg for seg in parsed.path.split("/") if seg]
        numeric = [i for i, seg in enumerate(segments) if seg.isdigit()]
        ids = [i for i in numeric if len(segments[i]) >= 5]
        if ids:
            i = ids[-1]
            version = segments[i + 1] if i + 1 < len(segments) and segments[i + 1].isdigit() else None
            return segments[i], version
    if re.fullmatch(r"\d+", text):
        return text, None
    raise ValueError(f"Cannot read a figshare article id from {location!r}")


_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def parse_fourtu_location(location: str) -> tuple[str, str | None]:
    """(dataset id, version) from a 4TU.ResearchData URL, UUID, or the numeric
    article id of datasets published before the move to Djehuty."""
    text = location.strip().lower()
    match = re.search(rf"({_UUID})(?:/(\d+))?", text)
    if match:
        return match.group(1), match.group(2)
    match = re.search(r"4tu\.nl/(?:articles|datasets)/(?:[^/]+/)?(\d{5,})(?:/(\d+))?", text) or re.fullmatch(
        r"(\d{5,})()", text
    )
    if match:
        return match.group(1), match.group(2) or None
    raise ValueError(f"Cannot read a 4TU.ResearchData dataset id from {location!r}")


def parse_dataverse_location(location: str) -> tuple[str, str]:
    """(server base URL, persistent id) from a Dataverse dataset URL."""
    parsed = urlparse(location.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Dataverse location must be a dataset URL, got {location!r}")
    pid = (parse_qs(parsed.query).get("persistentId") or [None])[0]
    if not pid:
        match = re.search(r"(doi:10\.\d{4,}/\S+|hdl:\S+)", location)
        pid = match.group(1) if match else None
    if not pid:
        raise ValueError(f"No persistentId in Dataverse location {location!r}")
    return f"{parsed.scheme}://{parsed.netloc}", pid


# ── File listings ────────────────────────────────────────────────────────────

def zenodo_record(record_id: str, api_base: str = ZENODO_API) -> RemoteRecord:
    data = _get_json(f"{api_base}/records/{record_id}")
    files_field = data.get("files") or []
    if isinstance(files_field, dict):  # InvenioRDM style {"entries": {...}}
        files_field = list((files_field.get("entries") or {}).values())
    files = []
    for entry in files_field:
        key = str(entry.get("key") or entry.get("filename") or "")
        links = entry.get("links") or {}
        url = links.get("content") or links.get("download") or links.get("self")
        if not key or not url:
            continue
        files.append(RemoteFile(key, str(url), entry.get("size") or entry.get("filesize"),
                                _parse_checksum(entry.get("checksum"))))
    version = data.get("revision") or data.get("updated") or ""
    return RemoteRecord(tuple(files), f"zenodo:{data.get('id', record_id)}@{version}".rstrip("@"))


def _figshare_style_record(article_url: str, prefix: str) -> RemoteRecord:
    data = _get_json(article_url)
    files = []
    for entry in data.get("files") or []:
        name, url = str(entry.get("name") or ""), entry.get("download_url")
        if not name or not url:
            continue
        checksum = _parse_checksum(entry.get("computed_md5") or entry.get("supplied_md5"))
        files.append(RemoteFile(name, str(url), entry.get("size"), checksum))
    version = data.get("version")
    revision = f"{prefix}:{data.get('id') or data.get('uuid') or article_url.rsplit('/', 1)[-1]}"
    return RemoteRecord(tuple(files), f"{revision}@v{version}" if version else revision)


def figshare_record(article_id: str, version: str | None = None, api_base: str = FIGSHARE_API) -> RemoteRecord:
    url = f"{api_base}/articles/{article_id}" + (f"/versions/{version}" if version else "")
    return _figshare_style_record(url, "figshare")


def fourtu_record(uuid: str, version: str | None = None, api_base: str = FOURTU_API) -> RemoteRecord:
    url = f"{api_base}/articles/{uuid}" + (f"/versions/{version}" if version else "")
    record = _figshare_style_record(url, "4tu")
    if record.files:
        return record
    # Djehuty lists files on a separate endpoint for some datasets.
    listing = _get_json(f"{api_base}/articles/{uuid}/files")
    files = tuple(
        RemoteFile(str(e["name"]), str(e["download_url"]), e.get("size"), _parse_checksum(e.get("computed_md5")))
        for e in listing if e.get("name") and e.get("download_url")
    )
    return RemoteRecord(files, record.revision)


def dataverse_record(server: str, persistent_id: str) -> RemoteRecord:
    data = _get_json(f"{server}/api/datasets/:persistentId/?persistentId={quote(persistent_id, safe=':/')}")
    version = (data.get("data") or {}).get("latestVersion") or {}
    files = []
    for entry in version.get("files") or []:
        data_file = entry.get("dataFile") or {}
        file_id = data_file.get("id")
        # Original upload, not the .tab conversion of ingested tables.
        name = data_file.get("originalFileName") or data_file.get("filename") or entry.get("label")
        if file_id is None or not name:
            continue
        directory = str(entry.get("directoryLabel") or "").strip("/")
        path = f"{directory}/{name}" if directory else str(name)
        if data_file.get("originalFileName"):
            url = f"{server}/api/access/datafile/{file_id}?format=original"
            checksum = None  # the published checksum is of the .tab conversion
            size = data_file.get("originalFileSize")
        else:
            url = f"{server}/api/access/datafile/{file_id}"
            checksum = _parse_checksum(data_file.get("checksum") or data_file.get("md5"))
            size = data_file.get("filesize")
        files.append(RemoteFile(path, url, size, checksum))
    number = f"{version.get('versionNumber', '')}.{version.get('versionMinorNumber', '')}".strip(".")
    return RemoteRecord(tuple(files), f"{persistent_id}@v{number}" if number else persistent_id)


# ── Download ─────────────────────────────────────────────────────────────────

def download_record(
    record: RemoteRecord,
    target: Path,
    wanted_suffixes: Iterable[str],
    max_file_bytes: int | None = None,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[Path]:
    """Download the files whose suffix is wanted, verifying checksums.

    Paths that would escape ``target`` are skipped; a checksum mismatch
    deletes the file and raises ``ValueError``.
    """
    downloader = downloader or _download
    wanted = {s.lower() for s in wanted_suffixes}
    root = target.resolve()
    written = []
    for remote in record.files:
        relative = PurePosixPath(remote.path)
        if relative.suffix.lower() not in wanted:
            continue
        if relative.is_absolute() or ".." in relative.parts:
            logger.warning("Skipping repository file whose path escapes the cache root: %r", remote.path)
            continue
        if max_file_bytes is not None and remote.size and remote.size > max_file_bytes:
            logger.warning("Skipping %s (%d bytes > limit %d)", remote.path, remote.size, max_file_bytes)
            continue
        destination = (target / Path(*relative.parts)).resolve()
        if root not in destination.parents:
            logger.warning("Skipping repository file whose path escapes the cache root: %r", remote.path)
            continue
        downloader(remote.url, destination)
        if remote.size is not None and destination.stat().st_size != int(remote.size):
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"size mismatch for {remote.path}: expected {remote.size} bytes, "
                f"got a different size; the download is incomplete or the record changed"
            )
        if remote.checksum:
            algorithm, expected = remote.checksum
            actual = file_digest(destination, algorithm)
            if actual != expected:
                destination.unlink(missing_ok=True)
                raise ValueError(f"{algorithm} checksum mismatch for {remote.path}: expected {expected}, got {actual}")
        written.append(destination)
    return written


__all__ = [
    "FIGSHARE_API",
    "FOURTU_API",
    "RemoteFile",
    "RemoteRecord",
    "ZENODO_API",
    "dataverse_record",
    "download_record",
    "figshare_record",
    "file_digest",
    "fourtu_record",
    "parse_dataverse_location",
    "parse_figshare_location",
    "parse_fourtu_location",
    "parse_zenodo_location",
    "zenodo_record",
]
