#!/usr/bin/env python3
"""Batch standardization pipeline for multi-source DV harmonization.

Phase-1 orchestration that reads a source manifest, discovers tabular files,
standardizes columns, and emits a consolidated meta-view artifact.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import hashlib
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib import request
from urllib.parse import urlparse

import httpx
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Pydantic v2 manifest validation
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, field_validator, model_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

if _PYDANTIC_AVAILABLE:
    class SourceManifestEntry(BaseModel):
        model_config = {"extra": "allow"}  # unknown fields are warnings, not errors

        source_id: str
        source_type: str  # local_path | github_repo | osf_project | web_dataset
        location: str
        ref: str | None = None
        include_globs: list[str] = []
        exclude_globs: list[str] = []
        mapping_path: str | None = None
        publication_doi: str | None = None
        llm_context: str | list[str] | None = None  # str, list of strings, or None
        use_llm_deduction: bool = False
        extract_archives: bool = True
        archive_max_depth: int = 3
        repeated_measures: bool = False
        llm_models: list[str] | None = None

        @field_validator("source_type")
        @classmethod
        def check_source_type(cls, v: str) -> str:
            valid = {"local_path", "github_repo", "osf_project", "web_dataset"}
            if v not in valid:
                raise ValueError(f"source_type must be one of {valid}, got '{v}'")
            return v

        @field_validator("source_id")
        @classmethod
        def check_source_id(cls, v: str) -> str:
            import re as _re
            if not _re.match(r'^[a-zA-Z0-9_\-]+$', v):
                raise ValueError(
                    f"source_id '{v}' contains invalid characters. "
                    "Use only letters, digits, underscores, hyphens."
                )
            return v

    class SourcesManifest(BaseModel):
        sources: list[SourceManifestEntry]


def _validate_manifest(raw: dict) -> list[str]:
    """Validate manifest structure using Pydantic v2. Returns list of error strings."""
    errors: list[str] = []
    if not _PYDANTIC_AVAILABLE:
        logger.warning("pydantic not installed — manifest validation skipped")
        return errors
    try:
        SourcesManifest(**raw)
        logger.info(
            "Manifest validation passed (%d sources)",
            len(raw.get("sources", [])),
        )
    except Exception as exc:  # noqa: BLE001
        for err in getattr(exc, "errors", lambda: [{"msg": str(exc)}])():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            errors.append(f"Manifest error at [{loc}]: {err['msg']}")
    return errors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.batch_profiles import (
    DATASET_TYPE_DETECTION,
    DATASET_TYPE_PROCESS,
    DATASET_TYPE_QUESTIONNAIRE,
    DATASET_TYPE_RESULTS,
    DATASET_TYPE_SENSOR,
    MAPPING_DOMAIN_BLOCKED,
    MAPPING_DOMAIN_DETECTION,
    MAPPING_DOMAIN_DV,
    MAPPING_DOMAIN_METADATA,
    MAPPING_DOMAIN_SENSOR,
    classify_dataset_type,
    get_schema_for_dataset_type,
    infer_mapping_domain,
)
from scripts.batch_reporting import build_mapping_debug_summary, build_unknown_alias_summary
from scripts.convert_dv import (
    _FORMAT_REGISTRY as _CDV_FORMAT_REGISTRY,
    build_original_column_lookup,
    identify_unmapped_columns,
    load_input_file,
    load_schema,
    save_output_file,
    standardize_columns,
    strip_duplicate_suffix,
)
from scripts.item_composition import (
    compose_item_groups,
    is_item_suffixed,
    strip_item_suffix,
)
from scripts.llm_utils import collect_repository_context, deduce_standard_name_with_local_llm
# llm_coordinator owns the alias-scoring, candidate-shortlisting, and
# LLM-deduction aggregation helpers. Re-exported here so existing tests
# (e.g. ``from scripts.run_batch_standardization import _augment_mapping_with_llm_deductions``)
# keep working after the Wave C extraction.
from scripts.llm_coordinator import (
    _augment_mapping_with_llm_deductions,
    _build_llm_deduction_log_lines,
    _collect_llm_deduction,
    _finalize_llm_deductions,
    _score_alias_match,
    _select_llm_candidate_shortlist,
)
# scripts.sources owns the per-source-type loaders (currently github URL
# parsing + OSF JSON-API traversal). Re-exported here so existing imports
# continue to work.
from scripts.sources.github import GITHUB_HOSTS, GitHubLocation, _parse_github_location
from scripts.sources.osf import (
    OSF_API_BASE,
    _download_osf_file,
    _extract_osf_project_id,
    _iter_osf_child_node_ids,
    _iter_osf_file_entries,
    _iter_osf_node_ids,
    _iter_osf_provider_urls,
    _osf_json_get,
)
# archive_utils owns the suffix sets and the recursive zip extraction helpers.
# We re-export them so existing test imports
# (e.g. ``from scripts.run_batch_standardization import _extract_zip_files_recursive``)
# continue to work after the Wave C extraction.
from scripts.archive_utils import (
    ARCHIVE_SUFFIXES,
    DATA_FILE_SUFFIXES,
    DEFAULT_ARCHIVE_MAX_DEPTH,
    MAPPING_SUFFIXES,
    _extract_archives_in_tree,
    _extract_zip_files_recursive,
    _is_within_directory,
    _resolve_archive_max_depth,
    _should_skip_archive_member,
)
# http_utils owns the URL retry/backoff client, the landing-page parser, and
# the Cloudflare/login challenge detector. Re-exported here so existing
# tests (e.g. ``from scripts.run_batch_standardization import _read_url_response``)
# continue to import them from the same location.
from scripts.http_utils import (
    HTTP_USER_AGENT,
    NETWORK_TIMEOUTS,
    SourceAccessError,
    WEB_CHALLENGE_BODY_MARKERS,
    WEB_CHALLENGE_HOSTS,
    WEB_CHALLENGE_TITLE_MARKERS,
    WEB_LOGIN_MARKERS,
    WEB_NO_DATA_MARKERS,
    WEB_NO_DATA_MARKERS_ASCII,
    _LandingPageParser,
    _build_unique_destination,
    _build_web_dataset_access_error,
    _decode_http_text,
    _extract_content_disposition_filename,
    _extract_html_download_urls,
    _infer_extension_from_content_type,
    _is_supported_download_response,
    _iter_content_urls,
    _looks_like_challenge_response,
    _looks_like_supported_download_url,
    _normalize_remote_filename,
    _read_url_bytes,
    _read_url_response,
    _write_download_payload,
)

# TABULAR_SUFFIXES is the historical alias for DATA_FILE_SUFFIXES used by
# discovery code in this module. Kept as a re-export for backwards compat.
TABULAR_SUFFIXES = DATA_FILE_SUFFIXES
PICKLE_SUFFIXES = {".pkl", ".pickle"}
DEFAULT_SENSOR_SCHEMA_PATH = REPO_ROOT / "schemas" / "standard_sensor_mapping.yaml"
DEFAULT_DETECTION_SCHEMA_PATH = REPO_ROOT / "schemas" / "standard_detection_mapping.yaml"
DEFAULT_METADATA_SCHEMA_PATH = REPO_ROOT / "schemas" / "standard_metadata_mapping.yaml"
SUPPORTED_SOURCE_TYPES = {"local_path", "github_repo", "osf_project", "web_dataset"}
WEB_DATASET_HOSTS = {
    "data.4tu.nl",
    "www.data.4tu.nl",
    "ieee-dataport.org",
    "www.ieee-dataport.org",
    "wjx.cn",
    "www.wjx.cn",
}
WINDOWS_PATH_SOFT_LIMIT = 240
MIN_ARTIFACT_PREFIX_LENGTH = 48
ARTIFACT_HASH_LENGTH = 12
# Survey/admin/identifier fields that should never be mapped to DVs.
NEVER_MAP_NORMALIZED_COLUMNS = {
    "id",
    "userid",
    "user_id",
    "participantid",
    "participant_id",
    "subjectid",
    "subject_id",
    "prolificid",
    "prolific_id",
    "responseid",
    "response_id",
    "condition",
    "conditionid",
    "condition_id",
    "group",
    "groupid",
    "group_id",
    "treatment",
    "treatmentid",
    "treatment_id",
    "seed",
    "lastpage",
    "startlanguage",
    "submitdate",
    "startdate",
    "enddate",
    "recordeddate",
    "status",
    "finished",
    "durationinseconds",
    "progress",
    "distributionchannel",
    "ipaddress",
    "recipientlastname",
    "recipientfirstname",
    "recipientemail",
    "externalreference",
    "locationlatitude",
    "locationlongitude",
}

# Technical/admin columns that should never be inferred by the LLM unless an
# explicit schema mapping exists. This keeps metadata streams out of the DV path.
LLM_EXCLUDED_NORMALIZED_COLUMNS = {
    "timestamp",
    "time_stamp",
    "datetime",
    "date",
    "logtime",
    "capturetime",
    "videoframe",
    "framenumber",
    "frame",
    "video",
    "objectid",
    "object_id",
    "class",
    "bboxclass",
    "confidence",
    "phase",
    "run",
    "trialindex",
    "sessionindex",
    "blockindex",
    "ispareto",
    "pareto",
    "x",
    "y",
    "z",
    "width",
    "height",
}

@dataclass
class SourceRunResult:
    source_id: str
    status: str
    discovered_files: int
    processed_files: int
    failed_files: int
    unknown_columns: int
    total_columns: int
    mapped_columns: int
    mapped_ratio: float
    output_dir: str
    message: str | None = None


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse manifest YAML at '{manifest_path}': {exc}"
        ) from exc

    if not isinstance(manifest, dict) or "sources" not in manifest:
        raise ValueError("Manifest must be a YAML object with a top-level 'sources' list.")

    # Pydantic v2 validation (if available).
    validation_errors = _validate_manifest(manifest)
    if validation_errors:
        for err_msg in validation_errors:
            print(f"[ERROR] {err_msg}", file=sys.stderr)
        raise SystemExit(1)

    sources = manifest["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("Manifest 'sources' must be a non-empty list.")

    seen_source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every source entry must be a dictionary.")
        for key in ("source_id", "source_type", "location"):
            if key not in source or not source[key]:
                raise ValueError(f"Source entries must include a non-empty '{key}'.")
        if source["source_type"] not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(
                f"Unsupported source_type '{source['source_type']}'. Supported values: {sorted(SUPPORTED_SOURCE_TYPES)}"
            )
        if source["source_id"] in seen_source_ids:
            raise ValueError(f"Duplicate source_id '{source['source_id']}' in manifest.")
        seen_source_ids.add(source["source_id"])

    return sources


def _download_remote_file(download_url: str, target_dir: Path, fallback_prefix: str) -> Path:
    req = request.Request(download_url, headers={"User-Agent": HTTP_USER_AGENT, "Accept": "*/*"})
    payload, headers, final_url = _read_url_response(
        req, timeout=NETWORK_TIMEOUTS["download_seconds"]
    )
    return _write_download_payload(target_dir, payload, final_url, headers, fallback_prefix=fallback_prefix)


def _materialize_web_dataset_source(location: str, target: Path) -> str:
    req = request.Request(
        location,
        headers={
            "User-Agent": HTTP_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        payload, headers, final_url = _read_url_response(
            req, timeout=NETWORK_TIMEOUTS["landing_seconds"]
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403, 503):
            # Some hosts (Cloudflare, IEEE login walls) return the challenge
            # / login page in the body of a 4xx. Read it through the same
            # parser/marker pipeline we use for 200-OK landing pages so we
            # can classify the access failure cleanly.
            error_payload = exc.response.content or b""
            try:
                error_html = error_payload.decode("utf-8", errors="replace")
            except Exception as decode_exc:  # noqa: BLE001
                logger.debug(
                    "Failed to decode challenge response body for %s: %s",
                    location, decode_exc,
                )
                error_html = ""
            parser = _LandingPageParser()
            try:
                parser.feed(error_html)
            except Exception as parse_exc:  # noqa: BLE001
                logger.debug(
                    "Failed to parse challenge response HTML for %s: %s",
                    location, parse_exc,
                )
            error_title = parser.title
            host = urlparse(location).netloc.lower()
            if _looks_like_challenge_response(host, error_title, error_html):
                raise _build_web_dataset_access_error(location, location, error_html, error_title)
        raise

    if _is_supported_download_response(final_url, headers):
        _write_download_payload(target, payload, final_url, headers, fallback_prefix="dataset")
        return final_url

    html_text = _decode_http_text(payload, headers)
    title, download_urls = _extract_html_download_urls(html_text, final_url)
    if not download_urls or _looks_like_challenge_response(
        urlparse(final_url).netloc.lower(), title, html_text
    ):
        raise _build_web_dataset_access_error(location, final_url, html_text, title)

    for index, download_url in enumerate(download_urls, start=1):
        _download_remote_file(download_url, target, fallback_prefix=f"download_{index}")

    return final_url


def _coerce_manifest_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    return []


def _on_rmtree_error(func: Any, path: str, exc_info: tuple[type[BaseException], BaseException, Any]) -> None:
    """Best-effort handler for Windows read-only files during recursive delete."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as chmod_exc:  # noqa: BLE001
        logger.debug(
            "rmtree retry failed for %s after chmod: %s; re-raising original error.",
            path, chmod_exc,
        )
        _, exc, _ = exc_info
        raise exc


def _safe_rmtree(path: Path, max_attempts: int = 4) -> None:
    """Recursively delete a directory with Windows-friendly retry behavior."""
    for attempt in range(max_attempts):
        try:
            shutil.rmtree(path, onerror=_on_rmtree_error)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))
        except OSError as exc:
            if getattr(exc, "winerror", None) == 5 and attempt < max_attempts - 1:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise

def _match_files(base_dir: Path, include_globs: list[str] | None, exclude_globs: list[str] | None) -> list[Path]:
    include_globs = include_globs or ["**/*"]
    exclude_globs = exclude_globs or []

    candidates: list[Path] = []
    for pattern in include_globs:
        for path in base_dir.glob(pattern):
            if not path.is_file() or path.suffix.lower() not in DATA_FILE_SUFFIXES:
                continue
            if _should_skip_archive_member(path.relative_to(base_dir)):
                continue
            candidates.append(path)

    unique_candidates = sorted(set(candidates))
    if not exclude_globs:
        return unique_candidates

    def _is_excluded(path: Path) -> bool:
        relative_posix = path.relative_to(base_dir).as_posix()
        for pattern in exclude_globs:
            normalized_pattern = str(pattern).replace("\\", "/")
            candidate_patterns = {normalized_pattern}
            trimmed_pattern = normalized_pattern
            while trimmed_pattern.startswith("**/"):
                trimmed_pattern = trimmed_pattern[3:]
                candidate_patterns.add(trimmed_pattern)
            if any(fnmatch.fnmatch(relative_posix, candidate) for candidate in candidate_patterns):
                return True
        return False

    return [path for path in unique_candidates if not _is_excluded(path)]

def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(value).strip().lower())


def _is_never_map_column(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    return normalized in NEVER_MAP_NORMALIZED_COLUMNS


def _filter_never_map_columns(column_names: list[str]) -> list[str]:
    return [column for column in column_names if not _is_never_map_column(column)]


def _is_llm_excluded_column(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    return normalized in LLM_EXCLUDED_NORMALIZED_COLUMNS


def _filter_llm_eligible_columns(column_names: list[str]) -> list[str]:
    return [
        column
        for column in column_names
        if not _is_never_map_column(column) and not _is_llm_excluded_column(column)
    ]


def _remove_never_map_aliases(mapping: dict[str, str]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for alias, canonical in mapping.items():
        if isinstance(alias, str) and _is_never_map_column(alias):
            continue
        filtered[alias] = canonical
    return filtered

def _load_optional_schema_mapping(
    schema_path: Path | None,
) -> tuple[dict[str, str], set[str], str | None]:
    path = schema_path
    if path is None:
        return {}, set(), None
    if not path.exists():
        return {}, set(), None

    try:
        schema_data = load_schema(str(path))
    except (yaml.YAMLError, ValueError) as exc:
        logger.warning("Failed to load optional schema '%s': %s", path, exc)
        return {}, set(), None

    mapping = schema_data["mapping"]
    aliases_ci = {
        str(alias).lower()
        for alias in mapping.keys()
        if isinstance(alias, str)
    }
    return mapping, aliases_ci, str(path)


def _source_cache_key(source: dict[str, Any]) -> str:
    source_type = str(source.get("source_type", ""))
    location = str(source.get("location", ""))
    ref = str(source.get("ref", "HEAD"))
    fingerprint = f"{source_type}|{location}|{ref}"
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]

def _ensure_cache_dir(cache_root: Path | None, source: dict[str, Any], working_dir: Path) -> Path:
    if cache_root is None:
        return working_dir / source["source_id"]
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{source['source_id']}_{_source_cache_key(source)}"

def discover_source_files(
    source: dict[str, Any],
    working_dir: Path,
    cache_root: Path | None = None,
    refresh_remote_cache: bool = False,
) -> tuple[Path, list[Path], str | None]:
    source_type = source["source_type"]
    extract_archives = bool(source.get("extract_archives", True))
    archive_max_depth = _resolve_archive_max_depth(source) if extract_archives else DEFAULT_ARCHIVE_MAX_DEPTH

    if source_type == "local_path":
        base_dir = Path(source["location"]).expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"local_path does not exist: {base_dir}")
        if extract_archives:
            _extract_archives_in_tree(base_dir, archive_max_depth)
        files = _match_files(
            base_dir,
            source.get("include_globs"),
            source.get("exclude_globs"),
        )
        return base_dir, files, None

    target = _ensure_cache_dir(cache_root, source, working_dir)
    include_globs = source.get("include_globs")
    exclude_globs = source.get("exclude_globs")

    if source_type == "osf_project":
        project_id = _extract_osf_project_id(source["location"])
        marker = target / ".source_ready"
        expected_marker = f"{project_id}|layout=v2"

        if refresh_remote_cache and target.exists():
            _safe_rmtree(target)

        marker_text = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
        if marker_text != expected_marker:
            if target.exists():
                _safe_rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

            node_ids = _iter_osf_node_ids(project_id)
            for node_id in node_ids:
                file_entries = _iter_osf_file_entries(node_id)
                for entry in file_entries:
                    attrs = entry.get("attributes", {}) or {}
                    materialized_path = str(attrs.get("materialized_path", "")).lstrip("/")
                    raw_path = str(attrs.get("path", "")).lstrip("/")
                    file_name = str(attrs.get("name", "")).strip()
                    path = materialized_path or raw_path or file_name
                    if not path:
                        continue

                    suffix = Path(file_name or path).suffix.lower()
                    if (
                        suffix not in DATA_FILE_SUFFIXES
                        and suffix not in ARCHIVE_SUFFIXES
                        and suffix not in MAPPING_SUFFIXES
                    ):
                        continue

                    download_url = entry.get("links", {}).get("download")
                    if not download_url:
                        continue

                    relative_path = Path(path)
                    if relative_path.name == "" and file_name:
                        relative_path = relative_path / file_name
                    if node_id != project_id:
                        relative_path = Path(f"node_{node_id}") / relative_path

                    local_path = target / relative_path
                    # Guard against server-controlled materialized_path values
                    # containing ".." segments (lstrip("/") above only strips
                    # leading slashes). Mirrors the zip-slip protection used by
                    # the archive extractor.
                    if not _is_within_directory(local_path.resolve(), target.resolve()):
                        logger.warning(
                            "[%s] Skipping OSF file whose path escapes the cache root: %r",
                            project_id, path,
                        )
                        continue
                    _download_osf_file(str(download_url), local_path)

            marker.write_text(expected_marker, encoding="utf-8")

        if extract_archives:
            _extract_archives_in_tree(target, archive_max_depth)
        files = _match_files(target, include_globs, exclude_globs)
        return target, files, project_id

    if source_type == "web_dataset":
        marker = target / ".source_ready"
        expected_marker = f"{source['location']}|layout=v1"

        if refresh_remote_cache and target.exists():
            _safe_rmtree(target)

        marker_text = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
        if marker_text != expected_marker:
            if target.exists():
                _safe_rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            _materialize_web_dataset_source(str(source["location"]), target)
            marker.write_text(expected_marker, encoding="utf-8")

        if extract_archives:
            _extract_archives_in_tree(target, archive_max_depth)
        files = _match_files(target, include_globs, exclude_globs)
        return target, files, str(source["location"])

    github_location = _parse_github_location(str(source["location"]))
    repo_url = github_location.clone_url
    pinned_ref = str(source.get("ref") or github_location.ref or "HEAD")
    marker = target / ".source_ready"

    if refresh_remote_cache and target.exists():
        _safe_rmtree(target)

    if not marker.exists():
        if target.exists():
            _safe_rmtree(target)
        clone_cmd = ["git", "clone", "--depth", "1"]
        if pinned_ref != "HEAD" and not re.fullmatch(r"[0-9a-f]{7,40}", pinned_ref):
            clone_cmd.extend(["--branch", pinned_ref])
        clone_cmd.extend([repo_url, str(target)])
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
        if pinned_ref != "HEAD":
            subprocess.run(
                ["git", "-C", str(target), "checkout", pinned_ref],
                check=True,
                capture_output=True,
                text=True,
            )
        marker.write_text("ready", encoding="utf-8")

    commit_sha = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    base_dir = target
    if github_location.subpath is not None:
        base_dir = (target / github_location.subpath).resolve()
        if not base_dir.exists():
            raise FileNotFoundError(
                f"GitHub location points to a missing repository subpath: {github_location.subpath}"
            )

    if base_dir.is_file():
        files = [base_dir] if base_dir.suffix.lower() in DATA_FILE_SUFFIXES else []
        return base_dir.parent, files, commit_sha

    if extract_archives:
        _extract_archives_in_tree(base_dir, archive_max_depth)

    files = _match_files(
        base_dir,
        include_globs,
        exclude_globs,
    )
    return base_dir, files, commit_sha

def _load_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = load_input_file(str(path))
    # Strip whitespace from headers — many real CSVs have " scenario", "id ",
    # etc. and the schema mapping is whitespace-sensitive.
    df.columns = [str(col).strip() for col in df.columns]
    return df

def _shorten_artifact_prefix(prefix: str, max_length: int) -> str:
    if len(prefix) <= max_length:
        return prefix

    digest = hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:ARTIFACT_HASH_LENGTH]
    if max_length <= len(digest):
        return digest[:max_length]

    if max_length <= len(digest) + 2:
        return f"{prefix[:1]}_{digest[: max_length - 2]}"

    remaining = max_length - len(digest) - 2
    head_length = max(remaining // 2, 1)
    tail_length = max(remaining - head_length, 1)
    shortened = f"{prefix[:head_length]}_{digest}_{prefix[-tail_length:]}"
    shortened = shortened[:max_length].strip("_")
    return shortened or digest[:max_length]


def _build_artifact_prefix(
    relative_path: Path,
    output_dir: Path | None = None,
    artifact_suffix_length: int = 0,
) -> str:
    """Create a collision-safe file prefix from a path relative to the source root."""
    normalized = relative_path.as_posix()
    safe_prefix = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")
    prefix = safe_prefix or "dataset"
    if output_dir is None:
        return prefix

    max_prefix_length = WINDOWS_PATH_SOFT_LIMIT - len(str(output_dir)) - 1 - max(artifact_suffix_length, 0)
    max_prefix_length = max(max_prefix_length, MIN_ARTIFACT_PREFIX_LENGTH)
    return _shorten_artifact_prefix(prefix, max_prefix_length)


def _find_repository_mapping_candidates(source_root: Path) -> list[Path]:
    mapping_patterns = [
        "*mapping*.yaml",
        "*mapping*.yml",
        "*dv*.yaml",
        "*dv*.yml",
    ]
    candidates: list[Path] = []
    for pattern in mapping_patterns:
        candidates.extend(path for path in source_root.rglob(pattern) if path.is_file())

    return sorted(set(candidates))


def _resolve_repository_mapping_path(
    source_root: Path,
    mapping_candidates: list[Path],
    requested_mapping_path: str | None = None,
    dataset_path: Path | None = None,
) -> Path | None:
    if requested_mapping_path:
        candidate = Path(requested_mapping_path).expanduser()
        if not candidate.is_absolute():
            source_relative = (source_root / candidate).resolve()
            if source_relative.exists() and source_relative.is_file():
                return source_relative
            repo_relative = (REPO_ROOT / candidate).resolve()
            if repo_relative.exists() and repo_relative.is_file():
                logger.debug(
                    "mapping_path '%s' not found relative to source root; resolved via project root: %s",
                    requested_mapping_path, repo_relative,
                )
                return repo_relative
            raise FileNotFoundError(
                f"Requested mapping_path does not exist: tried '{source_relative}' and '{repo_relative}'"
            )
        else:
            candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Requested mapping_path does not exist: {candidate}")
        return candidate

    if len(mapping_candidates) == 1:
        return mapping_candidates[0]

    if dataset_path is not None:
        resolved_dataset_parent = dataset_path.resolve().parent
        scoped_candidates: list[tuple[int, Path]] = []
        for candidate in mapping_candidates:
            try:
                relative = resolved_dataset_parent.relative_to(candidate.parent.resolve())
            except ValueError:
                continue
            scoped_candidates.append((len(relative.parts), candidate))

        if scoped_candidates:
            nearest_depth = min(depth for depth, _ in scoped_candidates)
            nearest_candidates = sorted(
                {candidate for depth, candidate in scoped_candidates if depth == nearest_depth}
            )
            if len(nearest_candidates) == 1:
                return nearest_candidates[0]

    resolved_source_root = source_root.resolve()
    top_level_candidates = [
        path for path in mapping_candidates
        if path.parent.resolve() == resolved_source_root
    ]
    if len(top_level_candidates) == 1:
        return top_level_candidates[0]

    return None


def _load_repository_mapping(
    source_root: Path,
    requested_mapping_path: str | None = None,
    dataset_path: Path | None = None,
) -> tuple[dict[str, str], str | None]:
    """Load a source-local mapping YAML when a concrete mapping path can be resolved."""
    mapping_candidates = _find_repository_mapping_candidates(source_root)
    mapping_path = _resolve_repository_mapping_path(
        source_root,
        mapping_candidates,
        requested_mapping_path=requested_mapping_path,
        dataset_path=dataset_path,
    )
    if mapping_path is None:
        return {}, None

    schema_data = load_schema(
        str(mapping_path),
        standard_schema_path=str(REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"),
    )
    return schema_data["mapping"], str(mapping_path)

def _merge_with_standard_precedence(
    source_mapping: dict[str, str],
    standard_mapping: dict[str, str],
    source_overrides_standard: bool = False,
) -> dict[str, str]:
    """Merge source and standard alias dicts.

    Default behaviour: the standard schema wins on case-insensitive conflicts —
    source mappings only extend coverage for aliases the standard does not
    know about. This protects well-known canonicals from accidental
    redefinition.

    When ``source_overrides_standard=True`` (per-source manifest opt-in via
    ``override_standard_aliases: true``), source-specific mappings win. Use
    this for legitimate dataset-specific overrides such as fact_av's
    ``understanding_rating -> sart_understanding`` rename where the standard's
    self-mapping must be displaced.
    """
    if source_overrides_standard:
        merged = {**standard_mapping, **source_mapping}
        source_ci = {
            alias.lower(): canonical
            for alias, canonical in source_mapping.items()
            if isinstance(alias, str)
        }
        for alias, canonical in list(merged.items()):
            if isinstance(alias, str):
                merged[alias] = source_ci.get(alias.lower(), canonical)
        return merged

    merged = {**source_mapping, **standard_mapping}
    standard_ci = {alias.lower(): canonical for alias, canonical in standard_mapping.items() if isinstance(alias, str)}

    for alias, canonical in list(merged.items()):
        if isinstance(alias, str):
            merged[alias] = standard_ci.get(alias.lower(), canonical)

    return merged


def _summarize_dataset(
    source_id: str,
    dataset_id: str,
    original_df: pd.DataFrame,
    standardized_df: pd.DataFrame,
    mapping: dict[str, str],
    provenance: dict[str, Any],
    mapping_debug_records: list[dict[str, Any]],
    dataset_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_lookup = build_original_column_lookup(original_df.copy(), mapping)
    unknown_columns = _filter_never_map_columns(
        identify_unmapped_columns([str(c) for c in original_df.columns], mapping)
    )
    debug_lookup = {
        str(row["original_column"]): row
        for row in mapping_debug_records
    }

    records: list[dict[str, Any]] = []
    # When ``standardize_columns`` suffixed duplicate canonicals (e.g.
    # ``trust_rating__dup_2``) we still want to record the base canonical so
    # meta-view rows pool as the same construct. We walk the columns by
    # position because pandas indexing on a duplicate name returns a frame.
    # ``__item_N`` columns are sub-items of a composite that already lives in
    # the DataFrame under the bare canonical — skip them so meta-view treats
    # the construct as one observation per row, not (k items × N rows).
    position_by_base: dict[str, int] = {}
    for column_position, suffixed_canonical in enumerate(standardized_df.columns):
        if is_item_suffixed(str(suffixed_canonical)):
            continue
        base_canonical = strip_duplicate_suffix(str(suffixed_canonical))
        idx_in_base = position_by_base.get(base_canonical, 0)
        position_by_base[base_canonical] = idx_in_base + 1

        series = standardized_df.iloc[:, column_position]
        aliases = original_lookup.get(base_canonical, [base_canonical])
        if 0 <= idx_in_base < len(aliases):
            original_alias = aliases[idx_in_base]
        elif aliases:
            original_alias = aliases[-1]
        else:
            original_alias = base_canonical
        mapping_record = debug_lookup.get(str(original_alias), {})
        numeric_series = pd.to_numeric(series, errors="coerce")

        records.append(
            {
                "source_id": source_id,
                "dataset_id": dataset_id,
                "dataset_type": dataset_type,
                "canonical_dv": base_canonical,
                "original_alias": original_alias,
                "mapping_domain": mapping_record.get("mapping_domain"),
                "mapping_method": mapping_record.get("mapping_method"),
                "mapping_source": mapping_record.get("mapping_source"),
                "unit": None,
                "scale": None,
                "n": int(numeric_series.notna().sum()),
                "mean": float(numeric_series.mean()) if numeric_series.notna().any() else None,
                "sd": float(numeric_series.std(ddof=1)) if numeric_series.notna().sum() > 1 else None,
                "review_flag": base_canonical in unknown_columns,
                "mapping_confidence": "high" if base_canonical not in unknown_columns else "unknown_alias",
                **provenance,
            }
        )

    quality = {
        "dataset_type": dataset_type,
        "total_columns": len(original_df.columns),
        "unknown_columns": len(unknown_columns),
        "mapped_columns": len(original_df.columns) - len(unknown_columns),
        "mapped_ratio": (
            (len(original_df.columns) - len(unknown_columns)) / len(original_df.columns)
            if len(original_df.columns) else 0.0
        ),
        "unknown_aliases": unknown_columns,
    }
    return records, quality

def _build_dataset_mapping_debug_records(
    source_mapping: dict[str, str],
    dataset_mapping: dict[str, str],
    original_columns: list[str],
    unknown_before_llm: list[str],
    source_custom_aliases_ci: set[str],
    source_mapping_path: str | None,
    schema_family_aliases_ci: dict[str, set[str]],
    schema_family_paths: dict[str, str | None],
    canonical_domain_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    unknown_before_llm_ci = {str(col).lower() for col in unknown_before_llm}
    debug_records: list[dict[str, Any]] = []

    for original_column in original_columns:
        column_name = str(original_column)
        normalized = _normalize_column_name(column_name)
        mapped = dataset_mapping.get(column_name)
        if mapped is None:
            mapped = dataset_mapping.get(column_name.lower())
        mapped_column = mapped or column_name

        was_blocked = _is_never_map_column(column_name)
        in_source_mapping = (
            column_name in source_mapping
            or column_name.lower() in source_mapping
        )
        inferred_by_llm = (
            not was_blocked
            and (column_name.lower() in unknown_before_llm_ci)
            and mapped is not None
        )

        if was_blocked:
            mapping_origin = "blocked_never_map"
            mapping_status = "blocked"
            mapping_method = "blocked"
            mapping_source = "never_map_blocklist"
        elif inferred_by_llm:
            mapping_origin = "llm"
            mapping_status = "mapped"
            mapping_method = "llm"
            mapping_source = "llm_deduction"
        elif in_source_mapping and mapped is not None:
            mapping_origin = "schema"
            mapping_status = "mapped"
            mapping_method = "mapping"
            if source_mapping_path and column_name.lower() in source_custom_aliases_ci:
                mapping_source = source_mapping_path
            else:
                mapping_source = "in_memory_mapping"
                for family_name in ("metadata", "detection", "sensor", "dv"):
                    family_aliases = schema_family_aliases_ci.get(family_name, set())
                    family_path = schema_family_paths.get(family_name)
                    if family_path and column_name.lower() in family_aliases:
                        mapping_source = family_path
                        break
        elif mapped is not None:
            mapping_origin = "mapping"
            mapping_status = "mapped"
            mapping_method = "mapping"
            mapping_source = "in_memory_mapping"
        else:
            mapping_origin = "none"
            mapping_status = "unmapped"
            mapping_method = "unmapped"
            mapping_source = None

        # Safety net: if a renamed output slipped through as "unmapped",
        # force it to a generic mapping classification.
        if mapping_method == "unmapped" and mapped_column != column_name:
            mapping_origin = "mapping"
            mapping_status = "mapped"
            mapping_method = "mapping"
            mapping_source = "in_memory_mapping"

        mapping_domain = infer_mapping_domain(
            mapped_column,
            mapping_status,
            canonical_domain_lookup,
            mapping_source=mapping_source,
        )

        debug_records.append(
            {
                "original_column": column_name,
                "normalized_column": normalized,
                "mapped_column": mapped_column,
                "mapping_origin": mapping_origin,
                "mapping_status": mapping_status,
                "mapping_method": mapping_method,
                "mapping_source": mapping_source,
                "mapping_domain": mapping_domain,
            }
        )

    return debug_records


def run_batch(
    manifest_path: Path,
    output_dir: Path,
    schema_path: Path,
    llm_deduction_enabled: bool = True,
    cache_root: Path | None = None,
    refresh_remote_cache: bool = False,
    preferred_models: list[str] | None = None,
    debug_mappings: bool = False,
) -> dict[str, Any]:
    schema_data = load_schema(str(schema_path))
    standard_dv_mapping = schema_data["mapping"]
    sensor_mapping, sensor_aliases_ci, sensor_schema_path = _load_optional_schema_mapping(
        DEFAULT_SENSOR_SCHEMA_PATH
    )
    detection_mapping, detection_aliases_ci, detection_schema_path = _load_optional_schema_mapping(
        DEFAULT_DETECTION_SCHEMA_PATH
    )
    metadata_mapping, metadata_aliases_ci, metadata_schema_path = _load_optional_schema_mapping(
        DEFAULT_METADATA_SCHEMA_PATH
    )
    schema_family_aliases_ci = {
        "metadata": metadata_aliases_ci,
        "detection": detection_aliases_ci,
        "sensor": sensor_aliases_ci,
        "dv": {
            str(alias).lower()
            for alias in standard_dv_mapping.keys()
            if isinstance(alias, str)
        },
    }
    schema_family_paths = {
        "metadata": metadata_schema_path,
        "detection": detection_schema_path,
        "sensor": sensor_schema_path,
        "dv": str(schema_path),
    }
    # Mapping from schema filename (as returned by get_schema_for_dataset_type)
    # to the pre-loaded domain-specific mapping dict.  This lets the YAML-driven
    # profiles drive schema selection without re-loading schemas on every file.
    _schema_filename_to_mapping: dict[str, dict[str, str]] = {
        "standard_dv_mapping.yaml": standard_dv_mapping,
        "standard_sensor_mapping.yaml": sensor_mapping,
        "standard_detection_mapping.yaml": detection_mapping,
        "standard_metadata_mapping.yaml": metadata_mapping,
    }

    canonical_domain_lookup: dict[str, str] = {}
    for canonical_name in metadata_mapping.values():
        canonical_domain_lookup[str(canonical_name)] = MAPPING_DOMAIN_METADATA
    for canonical_name in detection_mapping.values():
        canonical_domain_lookup[str(canonical_name)] = MAPPING_DOMAIN_DETECTION
    for canonical_name in sensor_mapping.values():
        canonical_domain_lookup[str(canonical_name)] = MAPPING_DOMAIN_SENSOR
    for canonical_name in standard_dv_mapping.values():
        canonical_domain_lookup[str(canonical_name)] = MAPPING_DOMAIN_DV

    output_dir.mkdir(parents=True, exist_ok=True)
    standardized_root = output_dir / "standardized"
    standardized_root.mkdir(parents=True, exist_ok=True)
    resolved_cache_root = cache_root or (output_dir / ".cache" / "sources")
    resolved_cache_root.mkdir(parents=True, exist_ok=True)

    sources = load_manifest(manifest_path)
    run_results: list[SourceRunResult] = []
    meta_rows: list[dict[str, Any]] = []
    llm_deductions_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    unknown_alias_events: list[dict[str, Any]] = []
    all_mapping_debug_records: list[dict[str, Any]] = []
    all_composite_records: list[dict[str, Any]] = []
    global_mapping_metrics = {
        "mapping": 0,
        "llm": 0,
        "blocked": 0,
        "unmapped": 0,
        "total_columns_seen": 0,
    }
    global_mapping_domains: dict[str, int] = {}
    source_mapping_metrics: dict[str, dict[str, int]] = {}
    source_mapping_domains: dict[str, dict[str, int]] = {}

    with TemporaryDirectory(prefix="opendv_batch_") as temp_dir:
        checkout_root = Path(temp_dir)

        source_progress = tqdm(sources, desc="Sources", unit="source")
        for source in source_progress:
            source_id = source["source_id"]
            source_progress.set_postfix(source_id=source_id)
            source_output_dir = standardized_root / source_id
            source_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                base_dir, files, commit_sha = discover_source_files(
                    source,
                    checkout_root,
                    cache_root=resolved_cache_root,
                    refresh_remote_cache=refresh_remote_cache,
                )
            except SourceAccessError as exc:
                logger.warning("[%s] %s: %s", exc.status.upper(), source_id, exc)
                run_results.append(
                    SourceRunResult(
                        source_id=source_id,
                        status=exc.status,
                        discovered_files=0,
                        processed_files=0,
                        failed_files=0,
                        unknown_columns=0,
                        total_columns=0,
                        mapped_columns=0,
                        mapped_ratio=0.0,
                        output_dir=str(source_output_dir),
                        message=str(exc),
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001
                # logger.exception preserves traceback so post-mortem debugging
                # can recover the original failure point even though we keep
                # the loop running across remaining sources.
                logger.exception("[FAILED] %s: %s", source_id, exc)
                run_results.append(
                    SourceRunResult(
                        source_id=source_id,
                        status="failed",
                        discovered_files=0,
                        processed_files=0,
                        failed_files=0,
                        unknown_columns=0,
                        total_columns=0,
                        mapped_columns=0,
                        mapped_ratio=0.0,
                        output_dir=str(source_output_dir),
                        message=str(exc),
                    )
                )
                continue

            if not files:
                message = "No supported dataset files were discovered for this source."
                logger.warning("[NOT_AVAILABLE] %s: %s", source_id, message)
                run_results.append(
                    SourceRunResult(
                        source_id=source_id,
                        status="not_available",
                        discovered_files=0,
                        processed_files=0,
                        failed_files=0,
                        unknown_columns=0,
                        total_columns=0,
                        mapped_columns=0,
                        mapped_ratio=0.0,
                        output_dir=str(source_output_dir),
                        message=message,
                    )
                )
                continue

            processed = 0
            failed = 0
            total_unknown = 0
            total_columns = 0
            total_mapped_columns = 0
            # CLI --disable-llm-deduction is a hard off-switch: when LLM deduction is
            # disabled at the run level, no per-source manifest flag can re-enable it.
            # When enabled at run level, the per-source flag still controls opt-in.
            source_llm_enabled = llm_deduction_enabled and bool(
                source.get("use_llm_deduction", llm_deduction_enabled)
            )
            source_preferred_models = source.get("llm_models") if isinstance(source.get("llm_models"), list) else preferred_models
            source_overrides_standard = bool(source.get("override_standard_aliases", False))
            source_llm_cache: dict[str, str | None] = {}
            source_column_sig_cache: dict[tuple[str, ...], dict[str, str]] = {}
            source_repository_context: str | None = None
            source_metrics = {
                "mapping": 0,
                "llm": 0,
                "blocked": 0,
                "unmapped": 0,
                "total_columns_seen": 0,
            }
            source_domain_metrics: dict[str, int] = {}
            dataset_errors: list[str] = []
            requested_mapping_path = (
                str(source.get("mapping_path")).strip()
                if source.get("mapping_path")
                else None
            )
            mapping_candidates = _find_repository_mapping_candidates(base_dir)
            mapping_cache: dict[
                str, tuple[dict[str, str], set[str], list[dict[str, Any]]]
            ] = {}
            if requested_mapping_path:
                try:
                    _resolve_repository_mapping_path(
                        base_dir,
                        mapping_candidates,
                        requested_mapping_path=requested_mapping_path,
                    )
                except Exception as exc:  # noqa: BLE001
                    # The mapping path resolution failure is the actual cause
                    # we want surfaced; logger.exception keeps the traceback
                    # in the run log even though we recover and continue.
                    logger.exception(
                        "[FAILED] %s: requested mapping_path '%s' could not be resolved: %s",
                        source_id, requested_mapping_path, exc,
                    )
                    run_results.append(
                        SourceRunResult(
                            source_id=source_id,
                            status="failed",
                            discovered_files=len(files),
                            processed_files=0,
                            failed_files=0,
                            unknown_columns=0,
                            total_columns=0,
                            mapped_columns=0,
                            mapped_ratio=0.0,
                            output_dir=str(source_output_dir),
                            message=str(exc),
                        )
                    )
                    continue

            dataset_progress = tqdm(
                files,
                desc=f"Datasets ({source_id})",
                unit="file",
                leave=False,
            )
            for file_path in dataset_progress:
                relative_file = file_path.relative_to(base_dir)
                dataset_id = relative_file.as_posix()
                artifact_suffix_length = max(
                    len(f"-standardized{file_path.suffix}"),
                    len("-mapping-debug.json"),
                    len("-quality.json"),
                    len("-error.log"),
                )
                artifact_prefix = _build_artifact_prefix(
                    relative_file,
                    output_dir=source_output_dir,
                    artifact_suffix_length=artifact_suffix_length,
                )
                dataset_progress.set_postfix(dataset=dataset_id)
                destination = source_output_dir / f"{artifact_prefix}-standardized{file_path.suffix}"
                relative_path = str(relative_file)

                try:
                    original_df = _load_any_table(file_path)
                    raw_columns = [str(column) for column in original_df.columns]
                    dataset_type = classify_dataset_type(relative_file, raw_columns)
                    if dataset_type in {DATASET_TYPE_RESULTS, DATASET_TYPE_QUESTIONNAIRE}:
                        standard_mapping = {**metadata_mapping, **standard_dv_mapping}
                    elif dataset_type == DATASET_TYPE_SENSOR:
                        standard_mapping = {**metadata_mapping, **sensor_mapping}
                    elif dataset_type == DATASET_TYPE_DETECTION:
                        standard_mapping = {**metadata_mapping, **detection_mapping}
                    elif dataset_type == DATASET_TYPE_PROCESS:
                        standard_mapping = dict(metadata_mapping)
                    else:
                        standard_mapping = {**metadata_mapping, **standard_dv_mapping}

                    source_mapping = dict(standard_mapping)
                    source_mapping_path = None
                    source_custom_aliases_ci: set[str] = set()
                    source_declared_scales: list[dict[str, Any]] = []
                    resolved_mapping_path = _resolve_repository_mapping_path(
                        base_dir,
                        mapping_candidates,
                        requested_mapping_path=requested_mapping_path,
                        dataset_path=file_path,
                    )
                    if resolved_mapping_path is not None:
                        cache_key = str(resolved_mapping_path)
                        cached_mapping = mapping_cache.get(cache_key)
                        if cached_mapping is None:
                            try:
                                custom_schema_data = load_schema(str(resolved_mapping_path))
                            except (yaml.YAMLError, ValueError) as exc:
                                logger.warning(
                                    "Skipping malformed mapping YAML '%s' for dataset '%s': %s",
                                    resolved_mapping_path, dataset_id, exc,
                                )
                                resolved_mapping_path = None
                                custom_schema_data = None
                            if custom_schema_data is not None:
                                source_specific_mapping = custom_schema_data["mapping"]
                                source_custom_aliases_ci = {
                                    str(alias).lower()
                                    for alias in custom_schema_data["mapping"].keys()
                                    if isinstance(alias, str)
                                }
                                source_declared_scales = list(
                                    custom_schema_data.get("scales") or []
                                )
                                cached_mapping = (
                                    source_specific_mapping,
                                    source_custom_aliases_ci,
                                    source_declared_scales,
                                )
                                mapping_cache[cache_key] = cached_mapping
                        else:
                            (
                                source_specific_mapping,
                                source_custom_aliases_ci,
                                source_declared_scales,
                            ) = cached_mapping

                        # Set source_mapping_path consistently across the cache-hit
                        # and cache-miss paths: only once a non-empty source mapping
                        # is actually merged in.
                        if resolved_mapping_path is not None and source_specific_mapping:
                            source_mapping = _merge_with_standard_precedence(
                                source_specific_mapping,
                                standard_mapping,
                                source_overrides_standard=source_overrides_standard,
                            )
                            source_mapping_path = cache_key
                    source_mapping = _remove_never_map_aliases(source_mapping)
                    dataset_mapping = source_mapping
                    unknown_before_llm = _filter_llm_eligible_columns(
                        identify_unmapped_columns(
                            raw_columns,
                            source_mapping,
                        )
                    )
                    should_apply_llm = (
                        source_llm_enabled
                        and dataset_type in {DATASET_TYPE_RESULTS, DATASET_TYPE_QUESTIONNAIRE}
                        and bool(unknown_before_llm)
                    )
                    if should_apply_llm:
                        # Column-signature dedup: reuse mapping from a
                        # prior file with the exact same column set and
                        # dataset type to avoid redundant LLM calls.
                        col_sig = tuple(sorted(str(c).lower() for c in raw_columns)) + (dataset_type,)
                        cached_augmented = source_column_sig_cache.get(col_sig)
                        if cached_augmented is not None:
                            dataset_mapping = cached_augmented
                            logger.info(
                                "Reusing cached column-signature mapping for '%s' "
                                "(%d columns, type=%s) — skipping LLM.",
                                dataset_id, len(raw_columns), dataset_type,
                            )
                        else:
                            if source_repository_context is None:
                                extra_context = _coerce_manifest_text_list(source.get("llm_context"))
                                extra_context.extend(
                                    _coerce_manifest_text_list(source.get("publication_context"))
                                )
                                source_repository_context = collect_repository_context(
                                    base_dir,
                                    explicit_dois=(
                                        _coerce_manifest_text_list(source.get("publication_doi"))
                                        + _coerce_manifest_text_list(source.get("doi"))
                                    ),
                                    explicit_pdf_urls=(
                                        _coerce_manifest_text_list(source.get("publication_pdf_url"))
                                        + _coerce_manifest_text_list(source.get("pdf_url"))
                                    ),
                                    extra_context=extra_context,
                                )
                            dataset_mapping = _augment_mapping_with_llm_deductions(
                                source_mapping,
                                unknown_before_llm,
                                base_dir,
                                preferred_models=source_preferred_models,
                                inference_cache=source_llm_cache,
                                repository_context=source_repository_context,
                            )
                            source_column_sig_cache[col_sig] = dataset_mapping
                        for alias in unknown_before_llm:
                            alias_text = str(alias)
                            inferred = dataset_mapping.get(alias_text)
                            if not inferred:
                                inferred = dataset_mapping.get(alias_text.lower())
                            if inferred:
                                _collect_llm_deduction(
                                    llm_deductions_by_key,
                                    source_id=source_id,
                                    dataset_id=dataset_id,
                                    alias=alias_text,
                                    canonical_dv=str(inferred),
                                )

                    mapping_debug_records = _build_dataset_mapping_debug_records(
                        source_mapping=source_mapping,
                        dataset_mapping=dataset_mapping,
                        original_columns=raw_columns,
                        unknown_before_llm=unknown_before_llm,
                        source_custom_aliases_ci=source_custom_aliases_ci,
                        source_mapping_path=source_mapping_path,
                        schema_family_aliases_ci=schema_family_aliases_ci,
                        schema_family_paths=schema_family_paths,
                        canonical_domain_lookup=canonical_domain_lookup,
                    )
                    for row in mapping_debug_records:
                        method = str(row.get("mapping_method", "unmapped"))
                        if method not in source_metrics:
                            method = "unmapped"
                        source_metrics[method] += 1
                        source_metrics["total_columns_seen"] += 1
                        global_mapping_metrics[method] += 1
                        global_mapping_metrics["total_columns_seen"] += 1
                        domain = str(row.get("mapping_domain", "unmapped"))
                        source_domain_metrics[domain] = source_domain_metrics.get(domain, 0) + 1
                        global_mapping_domains[domain] = global_mapping_domains.get(domain, 0) + 1
                        all_mapping_debug_records.append(
                            {
                                "source_id": source_id,
                                "dataset_id": dataset_id,
                                "dataset_type": dataset_type,
                                **row,
                            }
                        )
                        if method == "unmapped":
                            unknown_alias_events.append(
                                {
                                    "source_id": source_id,
                                    "dataset_id": dataset_id,
                                    "dataset_type": dataset_type,
                                    "alias": str(row["original_column"]),
                                }
                            )

                    if debug_mappings:
                        debug_path = source_output_dir / f"{artifact_prefix}-mapping-debug.json"
                        with open(debug_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "source_id": source_id,
                                    "dataset_id": dataset_id,
                                    "dataset_type": dataset_type,
                                    "path": relative_path,
                                    "summary": build_mapping_debug_summary(mapping_debug_records),
                                    "debug_mappings": mapping_debug_records,
                                },
                                f,
                                indent=2,
                            )
                        # Per-row traces are already persisted to the
                        # per-dataset JSON above; only emit a single debug
                        # breadcrumb here so stdout stays readable on large
                        # batches.
                        logger.debug(
                            "[DEBUG] Mapping trace (%s/%s) -> %s",
                            source_id, dataset_id, debug_path,
                        )

                    standardized_df = standardize_columns(original_df.copy(), dataset_mapping)
                    # Build canonical -> ordered originals for declared-scale
                    # resolution. Items match the canonical-group order (first
                    # occurrence -> bare canonical, subsequent -> __dup_N) so
                    # the scale's reverse_aliases can be applied by name.
                    canonical_to_originals_ordered: dict[str, list[str]] = {}
                    for orig_col in raw_columns:
                        target = dataset_mapping.get(orig_col)
                        if target is None and isinstance(orig_col, str):
                            target = dataset_mapping.get(orig_col.lower())
                        target = target or orig_col
                        canonical_to_originals_ordered.setdefault(target, []).append(orig_col)
                    standardized_df, composite_records = compose_item_groups(
                        standardized_df,
                        dataset_type,
                        declared_scales=source_declared_scales,
                        canonical_to_originals=canonical_to_originals_ordered,
                    )
                    for record in composite_records:
                        all_composite_records.append(
                            {
                                "source_id": source_id,
                                "dataset_id": dataset_id,
                                "dataset_type": dataset_type,
                                **record,
                            }
                        )
                    if destination.suffix.lower() == ".tsv":
                        standardized_df.to_csv(destination, sep="\t", index=False)
                    else:
                        save_output_file(standardized_df, str(destination))

                    provenance = {
                        "source_type": source["source_type"],
                        "location": source["location"],
                        "path": relative_path,
                        "commit": commit_sha,
                        "source_mapping": source_mapping_path,
                        "run_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    rows, quality = _summarize_dataset(
                        source_id,
                        dataset_id,
                        original_df,
                        standardized_df,
                        dataset_mapping,
                        provenance,
                        mapping_debug_records,
                        dataset_type,
                    )
                    total_unknown += quality["unknown_columns"]
                    total_columns += quality["total_columns"]
                    total_mapped_columns += quality["mapped_columns"]
                    meta_rows.extend(rows)

                    with open(source_output_dir / f"{artifact_prefix}-quality.json", "w", encoding="utf-8") as f:
                        json.dump(quality, f, indent=2)

                    processed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    dataset_errors.append(f"{dataset_id}: {exc}")
                    # Persist the full traceback alongside the per-dataset
                    # error log so post-run triage doesn't lose the stack frame
                    # information when many sources fail in the same batch.
                    logger.exception(
                        "Dataset processing failed for %s/%s: %s",
                        source_id, dataset_id, exc,
                    )
                    with open(source_output_dir / f"{artifact_prefix}-error.log", "w", encoding="utf-8") as f:
                        f.write(str(exc))

            source_mapping_metrics[source_id] = source_metrics
            source_mapping_domains[source_id] = dict(sorted(source_domain_metrics.items()))
            status = "completed" if failed == 0 else ("partial" if processed else "failed")
            message = None
            if dataset_errors:
                preview = "; ".join(dataset_errors[:3])
                if len(dataset_errors) > 3:
                    preview += f"; ... and {len(dataset_errors) - 3} more"
                message = preview
                if status == "failed":
                    logger.error("[FAILED] %s: %s", source_id, message)
            run_results.append(
                SourceRunResult(
                    source_id=source_id,
                    status=status,
                    discovered_files=len(files),
                    processed_files=processed,
                    failed_files=failed,
                    unknown_columns=total_unknown,
                    total_columns=total_columns,
                    mapped_columns=total_mapped_columns,
                    mapped_ratio=(total_mapped_columns / total_columns) if total_columns else 0.0,
                    output_dir=str(source_output_dir),
                    message=message,
                )
            )

    meta_df = pd.DataFrame(meta_rows)
    meta_csv = output_dir / "meta_view.csv"
    meta_json = output_dir / "meta_view.json"
    meta_df.to_csv(meta_csv, index=False)
    meta_df.to_json(meta_json, orient="records", indent=2)
    llm_deductions = _finalize_llm_deductions(llm_deductions_by_key)
    llm_log_path = output_dir / "llm_deductions.log"
    llm_log_path.write_text(
        "\n".join(_build_llm_deduction_log_lines(llm_deductions)) + "\n",
        encoding="utf-8",
    )
    llm_json_path = output_dir / "llm_deductions.json"
    with open(llm_json_path, "w", encoding="utf-8") as f:
        json.dump(llm_deductions, f, indent=2)
    unknown_alias_summary = build_unknown_alias_summary(unknown_alias_events)
    unknown_alias_summary_path = output_dir / "unknown_alias_summary.json"
    with open(unknown_alias_summary_path, "w", encoding="utf-8") as f:
        json.dump(unknown_alias_summary, f, indent=2)
    mapping_debug_summary = build_mapping_debug_summary(all_mapping_debug_records)
    mapping_debug_summary_path = output_dir / "mapping_debug_summary.json"
    with open(mapping_debug_summary_path, "w", encoding="utf-8") as f:
        json.dump(mapping_debug_summary, f, indent=2)
    composites_path = output_dir / "composites.json"
    composed_count = sum(1 for r in all_composite_records if r.get("decision") == "composed")
    skipped_low_alpha = sum(
        1 for r in all_composite_records if r.get("decision") == "skip_low_alpha"
    )
    skipped_insufficient = sum(
        1 for r in all_composite_records if r.get("decision") == "skip_insufficient_rows"
    )
    composite_summary = {
        "total_groups_evaluated": len(all_composite_records),
        "composed": composed_count,
        "skipped_low_alpha": skipped_low_alpha,
        "skipped_insufficient_rows": skipped_insufficient,
        "records": all_composite_records,
    }
    with open(composites_path, "w", encoding="utf-8") as f:
        json.dump(composite_summary, f, indent=2)
    mapped_total = int(global_mapping_metrics["mapping"] + global_mapping_metrics["llm"])
    mappable_total = int(global_mapping_metrics["total_columns_seen"] - global_mapping_metrics["blocked"])
    global_mapping_metrics["mapped_total"] = mapped_total
    global_mapping_metrics["mappable_total"] = mappable_total
    global_mapping_metrics["mapped_rate_all_columns"] = (
        (mapped_total / global_mapping_metrics["total_columns_seen"])
        if global_mapping_metrics["total_columns_seen"]
        else 0.0
    )
    global_mapping_metrics["mapped_rate_excluding_blocked"] = (
        (mapped_total / mappable_total)
        if mappable_total
        else 0.0
    )

    failed_sources = [r.source_id for r in run_results if r.status == "failed"]
    partial_sources = [r.source_id for r in run_results if r.status == "partial"]
    unavailable_sources = [
        r.source_id for r in run_results
        if r.status in {"not_available", "login_required"}
    ]

    summary = {
        "manifest": str(manifest_path),
        "schema": str(schema_path),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_sources": len(sources),
        "successful_sources": sum(1 for r in run_results if r.status in {"completed", "partial"}),
        "failed_sources": failed_sources,
        "partial_sources": partial_sources,
        "unavailable_sources": unavailable_sources,
        "run_complete": len(failed_sources) == 0 and len(partial_sources) == 0,
        "meta_view_rows": int(meta_df.shape[0]),
        "llm_deduction_enabled": llm_deduction_enabled,
        "llm_deductions_count": len(llm_deductions),
        "llm_deductions_log": str(llm_log_path),
        "llm_deductions_json": str(llm_json_path),
        "metadata_schema": metadata_schema_path,
        "metadata_mapping_aliases": len(metadata_mapping),
        "sensor_schema": sensor_schema_path,
        "sensor_mapping_aliases": len(sensor_mapping),
        "detection_schema": detection_schema_path,
        "detection_mapping_aliases": len(detection_mapping),
        "mapping_metrics": global_mapping_metrics,
        "mapping_metrics_by_source": source_mapping_metrics,
        "mapping_metrics_by_domain": dict(sorted(global_mapping_domains.items())),
        "mapping_metrics_by_domain_source": source_mapping_domains,
        "unknown_alias_summary": str(unknown_alias_summary_path),
        "unknown_alias_events": int(unknown_alias_summary["total_unknown_alias_events"]),
        "mapping_debug_summary": str(mapping_debug_summary_path),
        "composites_path": str(composites_path),
        "composites_composed": composed_count,
        "composites_skipped_low_alpha": skipped_low_alpha,
        "composites_skipped_insufficient_rows": skipped_insufficient,
        "debug_mappings_enabled": debug_mappings,
        "cache_root": str(resolved_cache_root),
        "results": [r.__dict__ for r in run_results],
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def main() -> None:
    # Ensure runtime status messages and warnings actually surface when the
    # script is invoked from the CLI. Library callers can override this by
    # calling logging.basicConfig themselves before importing this module.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    parser = argparse.ArgumentParser(description="Batch DV standardization from a manifest.")
    parser.add_argument("--manifest", required=True, help="Path to source manifest YAML.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/batch_runs/latest",
        help="Directory for standardized outputs, logs, and consolidated meta-view artifacts.",
    )
    parser.add_argument(
        "--schema",
        default=str(Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"),
        help="Path to standard mapping schema YAML.",
    )
    parser.add_argument(
        "--snapshot-manifest",
        action="store_true",
        help="Copy the input manifest into the output directory for provenance.",
    )
    parser.add_argument(
        "--disable-llm-deduction",
        action="store_true",
        help="Disable local LLM-based alias deduction for deterministic runs.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory for caching downloaded remote sources (OSF/GitHub/web datasets).",
    )
    parser.add_argument(
        "--refresh-remote-cache",
        action="store_true",
        help="Force refresh of cached remote sources before processing.",
    )
    parser.add_argument(
        "--llm-models",
        default=None,
        help="Comma-separated local model ids for LLM deduction priority.",
    )
    parser.add_argument(
        "--debug-mappings",
        action="store_true",
        help="Print and save per-dataset column mapping traces (can be very verbose).",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    schema_path = Path(args.schema).resolve()
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else None

    preferred_models = None
    if args.llm_models:
        preferred_models = [m.strip() for m in args.llm_models.split(",") if m.strip()] or None

    summary = run_batch(
        manifest_path,
        output_dir,
        schema_path,
        llm_deduction_enabled=not args.disable_llm_deduction,
        cache_root=cache_root,
        refresh_remote_cache=args.refresh_remote_cache,
        preferred_models=preferred_models,
        debug_mappings=args.debug_mappings,
    )

    if args.snapshot_manifest:
        shutil.copy2(manifest_path, output_dir / "manifest_snapshot.yaml")

    print(json.dumps(summary, indent=2))
    if summary.get("llm_deductions_count", 0):
        llm_log_path = Path(str(summary["llm_deductions_log"]))
        try:
            print(llm_log_path.read_text(encoding="utf-8").rstrip())
        except OSError:
            print(
                f"LLM-derived mappings were applied. See log: {summary['llm_deductions_log']}"
            )

if __name__ == "__main__":
    main()
