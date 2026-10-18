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
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
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
        # local_path | github_repo | osf_project | web_dataset | zenodo_record |
        # figshare_article | fourtu_dataset | dataverse_dataset (+ registered types)
        source_type: str
        location: str
        ref: str | None = None
        include_globs: list[str] = []
        exclude_globs: list[str] = []
        mapping_path: str | None = None
        publication_doi: str | None = None
        llm_context: str | list[str] | None = None  # str, list of strings, or None
        # None means "inherit the run-level LLM switch"; an explicit true/false
        # in the manifest overrides it (but --disable-llm-deduction always wins).
        use_llm_deduction: bool | None = None
        extract_archives: bool = True
        archive_max_depth: int = 3
        repeated_measures: bool = False
        llm_models: list[str] | None = None
        # Repository sources: skip files larger than this (MB).
        max_file_mb: float | None = None

        @field_validator("source_type")
        @classmethod
        def check_source_type(cls, v: str) -> str:
            # Resolved at validation time so source types registered via
            # ``register_source_type`` are accepted without editing this model.
            valid = SUPPORTED_SOURCE_TYPES
            if v not in valid:
                raise ValueError(f"source_type must be one of {sorted(valid)}, got '{v}'")
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

    # Resolve the postponed ``list[SourceManifestEntry]`` annotation now, while
    # this module's globals are in scope. Without it pydantic resolves lazily
    # via ``sys.modules[__module__]``, which breaks when the script runs as
    # ``__main__`` under a wrapper such as ``python -m cProfile -m ...``.
    SourcesManifest.model_rebuild()


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

# DATASET_TYPE_* are re-exported for callers that import them from here;
# dataset-type behaviour itself is driven by schemas/dataset_type_profiles.yaml.
from scripts.fair_export import frictionless_type
from scripts.batch_profiles import (  # noqa: F401
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
    item_composition_dataset_types,
    llm_eligible_dataset_types,
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
from scripts.llm_plausibility import load_rules as load_plausibility_rules
from scripts.value_labels import apply_value_labels, recode_negative_answer_codes
from scripts.llm_plausibility import (
    ColumnProfile,
    canonical_ranges_from_schema,
    check_llm_suggestion,
    profile_columns,
    screen_llm_candidates,
)
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
from scripts.sources import repositories
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
DV_SCHEMA_FILENAME = "standard_dv_mapping.yaml"
METADATA_SCHEMA_FILENAME = DEFAULT_METADATA_SCHEMA_PATH.name
# Populated by ``register_source_type`` further down; the manifest validators
# read it at validation time.
SUPPORTED_SOURCE_TYPES: set[str] = set()
# Remote failures that mean "try again later" rather than "this source is
# broken" (e.g. OSF's scheduled-maintenance page is served as a 503).
TEMPORARY_UNAVAILABLE_STATUS_CODES = {502, 503, 504}
UNAVAILABLE_SOURCE_STATUSES = {
    "not_available",
    "login_required",
    "access_restricted",
    "temporarily_unavailable",
}
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
    # Simulator/log admin + condition fields (e.g. ROADS driving logs): these
    # are identifiers or independent variables, never dependent measures.
    "scenarioid",
    "requestid",
    "controlmode",
    "sideofconstructionsite",
    # eHMI-for-All independent variables / conditions
    "scenario",
    "distraction",
    "ehmi",
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
    # "mapped_columns" counts every schema family (dv + sensor + metadata +
    # detection) plus blocked admin fields, so a telemetry-heavy source can show
    # high coverage while contributing zero dependent variables. These fields
    # surface that distinction without digging into mapping_metrics_by_domain.
    mapped_dv_columns: int = 0
    mapping_domains: dict[str, int] | None = None


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
        raise ValueError(
            "Manifest validation failed:\n" + "\n".join(validation_errors)
        )

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

# Directory names never worth descending into during dataset/mapping
# discovery. Cloned GitHub sources carry a full .git object store that can
# dwarf the actual data tree.
_DISCOVERY_PRUNED_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".hg", ".svn"}

# Non-tabular files that get discovered because their suffix (e.g. .json) is a
# valid data format, but which are never study data: game-engine/build/IDE asset
# trees and package/config manifests (Unity projects, Node, .NET) and ML-pipeline
# bookkeeping. Discovering these as "datasets" inflates the column count with
# ungappable keys and depresses the mapped ratio (see 4TU Unity ZIP, FACT-AV ML
# artifacts). Applied in addition to any per-source exclude_globs.
DEFAULT_EXCLUDE_GLOBS = [
    # Engine / build / IDE asset trees
    "**/Library/**", "**/PackageCache/**", "**/ProjectSettings/**",
    "**/Packages/**", "**/obj/**", "**/bin/**", "**/Temp/**",
    "**/Build/**", "**/Builds/**", "**/.vs/**", "**/.idea/**",
    "**/.gradle/**", "**/target/**",
    # Package / build / config manifests (descriptors, not tabular data)
    "**/package.json", "**/package-lock.json", "**/manifest.json",
    "**/tsconfig*.json", "**/jsconfig*.json", "**/composer.json",
    "**/launchSettings.json", "**/appsettings*.json",
    "**/ValidationExceptions.json", "**/*.tests.json", "**/*.sample.json",
    # ML training / experiment bookkeeping
    "**/catboost_info/**", "**/wandb/**", "**/mlruns/**",
    "**/.ipynb_checkpoints/**",
]


def _iter_discovery_files(base_dir: Path) -> list[Path]:
    """Walk ``base_dir`` once, pruning VCS/cache directories."""
    found: list[Path] = []
    for current_root, dir_names, file_names in os.walk(base_dir):
        dir_names[:] = [name for name in dir_names if name not in _DISCOVERY_PRUNED_DIRS]
        root_path = Path(current_root)
        for file_name in file_names:
            found.append(root_path / file_name)
    return found


def _glob_pattern_variants(pattern: str) -> set[str]:
    """Expand a pathlib-style glob into fnmatch-compatible variants.

    ``fnmatch`` has no ``**`` notion (``*`` already crosses separators), but
    pathlib's ``**/`` also matches *zero* directories. Emitting the pattern
    both with and without each ``**/`` segment reproduces that behaviour.
    """
    normalized = str(pattern).replace("\\", "/")
    variants = {normalized}
    while True:
        additions = {
            variant.replace("**/", "", 1)
            for variant in variants
            if "**/" in variant
        }
        if additions <= variants:
            break
        variants |= additions
    return variants


def _matches_any_glob(relative_posix: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if any(fnmatch.fnmatch(relative_posix, variant) for variant in _glob_pattern_variants(pattern)):
            return True
    return False


def _match_files(base_dir: Path, include_globs: list[str] | None, exclude_globs: list[str] | None) -> list[Path]:
    include_globs = [str(p) for p in (include_globs or ["**/*"])]
    # Per-source excludes take effect on top of the global non-tabular defaults.
    exclude_globs = [str(p) for p in (exclude_globs or [])] + DEFAULT_EXCLUDE_GLOBS

    candidates: list[Path] = []
    for path in _iter_discovery_files(base_dir):
        if path.suffix.lower() not in DATA_FILE_SUFFIXES:
            continue
        relative = path.relative_to(base_dir)
        if _should_skip_archive_member(relative):
            continue
        relative_posix = relative.as_posix()
        if not _matches_any_glob(relative_posix, include_globs):
            continue
        if exclude_globs and _matches_any_glob(relative_posix, exclude_globs):
            continue
        candidates.append(path)

    return sorted(set(candidates))

@lru_cache(maxsize=65536)
def _normalize_column_name(value: str) -> str:
    # Memoized: the never-map / LLM-exclusion checks call this for every
    # column of every dataset, and wide telemetry exports repeat the same
    # few hundred headers across thousands of files.
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

# ---------------------------------------------------------------------------
# Source-type registry
# ---------------------------------------------------------------------------
# A materializer makes one manifest source available on local disk and returns
# ``(base_dir, provenance_id)``. ``base_dir`` may point at a single file (e.g. a
# GitHub blob URL); ``provenance_id`` is recorded per dataset as ``commit``
# (git SHA, OSF project id, landing-page URL, ...). Archive extraction and
# include/exclude matching are shared by every source type, so a new source
# type only needs to fetch files:
#
#     @register_source_type("my_repository")
#     def _materialize_my_repository(source, target, refresh_remote_cache):
#         ...download into target...
#         return target, record_version
#
# Repository APIs that list files with checksums (Zenodo, figshare, 4TU,
# Dataverse) share ``_materialize_repository_record``.
SourceMaterializer = Callable[[dict[str, Any], Path, bool], tuple[Path, str | None]]
_SOURCE_MATERIALIZERS: dict[str, SourceMaterializer] = {}


def register_source_type(name: str) -> Callable[[SourceMaterializer], SourceMaterializer]:
    """Register a materializer for manifest entries with ``source_type: name``."""

    def decorator(fn: SourceMaterializer) -> SourceMaterializer:
        _SOURCE_MATERIALIZERS[name] = fn
        SUPPORTED_SOURCE_TYPES.add(name)
        return fn

    return decorator


def _reuse_marked_cache(target: Path, expected_marker: str, refresh_remote_cache: bool) -> bool:
    """Return True when ``target`` already holds a complete download.

    The ``.source_ready`` marker is written only after a download finishes,
    so an interrupted fetch (or a layout-version bump) triggers a clean
    re-download instead of reusing a partial tree. When this returns False,
    ``target`` exists and is empty.
    """
    marker = target / ".source_ready"
    if refresh_remote_cache and target.exists():
        _safe_rmtree(target)
    marker_text = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
    if marker_text == expected_marker:
        return True
    if target.exists():
        _safe_rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return False


@register_source_type("local_path")
def _materialize_local_path(
    source: dict[str, Any], target: Path, refresh_remote_cache: bool
) -> tuple[Path, str | None]:
    base_dir = Path(source["location"]).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"local_path does not exist: {base_dir}")
    return base_dir, None


@register_source_type("osf_project")
def _materialize_osf_project(
    source: dict[str, Any], target: Path, refresh_remote_cache: bool
) -> tuple[Path, str | None]:
    project_id = _extract_osf_project_id(source["location"])
    expected_marker = f"{project_id}|layout=v2"
    if _reuse_marked_cache(target, expected_marker, refresh_remote_cache):
        return target, project_id

    for node_id in _iter_osf_node_ids(project_id):
        for entry in _iter_osf_file_entries(node_id):
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

    (target / ".source_ready").write_text(expected_marker, encoding="utf-8")
    return target, project_id


@register_source_type("web_dataset")
def _materialize_web_dataset(
    source: dict[str, Any], target: Path, refresh_remote_cache: bool
) -> tuple[Path, str | None]:
    expected_marker = f"{source['location']}|layout=v1"
    if not _reuse_marked_cache(target, expected_marker, refresh_remote_cache):
        _materialize_web_dataset_source(str(source["location"]), target)
        (target / ".source_ready").write_text(expected_marker, encoding="utf-8")
    return target, str(source["location"])


def _materialize_repository_record(
    source: dict[str, Any],
    target: Path,
    refresh_remote_cache: bool,
    key: str,
    fetch: Callable[[], repositories.RemoteRecord],
) -> tuple[Path, str | None]:
    """Shared download/caching for repository records (Zenodo, figshare, ...).

    The record's revision is stored next to the cache marker so a cached
    source still reports which version it holds.
    """
    expected_marker = f"{key}|layout=v1"
    revision_file = target / ".source_revision"
    if _reuse_marked_cache(target, expected_marker, refresh_remote_cache):
        revision = revision_file.read_text(encoding="utf-8").strip() if revision_file.exists() else key
        return target, revision
    record = fetch()
    max_mb = source.get("max_file_mb")
    written = repositories.download_record(
        record,
        target,
        DATA_FILE_SUFFIXES | ARCHIVE_SUFFIXES | MAPPING_SUFFIXES,
        max_file_bytes=int(float(max_mb) * 1024 * 1024) if max_mb else None,
    )
    if not written:
        raise SourceAccessError("not_available", f"{key}: the record has no data, archive or mapping files")
    revision_file.write_text(record.revision, encoding="utf-8")
    (target / ".source_ready").write_text(expected_marker, encoding="utf-8")
    return target, record.revision


@register_source_type("zenodo_record")
def _materialize_zenodo_record(
    source: dict[str, Any], target: Path, refresh_remote_cache: bool
) -> tuple[Path, str | None]:
    record_id = repositories.parse_zenodo_location(str(source["location"]))
    return _materialize_repository_record(
        source, target, refresh_remote_cache, f"zenodo:{record_id}",
        lambda: repositories.zenodo_record(record_id),
    )


@register_source_type("figshare_article")
def _materialize_figshare_article(
    source: dict[str, Any], target: Path, refresh_remote_cache: bool
) -> tuple[Path, str | None]:
    article_id, version = repositories.parse_figshare_location(str(source["location"]))
    version = str(source.get("ref") or version or "") or None
    return _materialize_repository_record(
        source, target, refresh_remote_cache, f"figshare:{article_id}@{version or 'latest'}",
        lambda: repositories.figshare_record(article_id, version),
    )


@register_source_type("fourtu_dataset")
def _materialize_fourtu_dataset(
    source: dict[str, Any], target: Path, refresh_remote_cache: bool
) -> tuple[Path, str | None]:
    uuid, version = repositories.parse_fourtu_location(str(source["location"]))
    version = str(source.get("ref") or version or "") or None
    return _materialize_repository_record(
        source, target, refresh_remote_cache, f"4tu:{uuid}@{version or 'latest'}",
        lambda: repositories.fourtu_record(uuid, version),
    )


@register_source_type("dataverse_dataset")
def _materialize_dataverse_dataset(
    source: dict[str, Any], target: Path, refresh_remote_cache: bool
) -> tuple[Path, str | None]:
    server, persistent_id = repositories.parse_dataverse_location(str(source["location"]))
    return _materialize_repository_record(
        source, target, refresh_remote_cache, f"dataverse:{persistent_id}",
        lambda: repositories.dataverse_record(server, persistent_id),
    )


@register_source_type("github_repo")
def _materialize_github_repo(
    source: dict[str, Any], target: Path, refresh_remote_cache: bool
) -> tuple[Path, str | None]:
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
    return base_dir, commit_sha


def discover_source_files(
    source: dict[str, Any],
    working_dir: Path,
    cache_root: Path | None = None,
    refresh_remote_cache: bool = False,
) -> tuple[Path, list[Path], str | None]:
    source_type = source["source_type"]
    materializer = _SOURCE_MATERIALIZERS.get(source_type)
    if materializer is None:
        raise ValueError(
            f"Unsupported source_type '{source_type}'. Supported values: {sorted(SUPPORTED_SOURCE_TYPES)}"
        )
    extract_archives = bool(source.get("extract_archives", True))
    archive_max_depth = _resolve_archive_max_depth(source) if extract_archives else DEFAULT_ARCHIVE_MAX_DEPTH

    target = _ensure_cache_dir(cache_root, source, working_dir)
    base_dir, provenance_id = materializer(source, target, refresh_remote_cache)

    if base_dir.is_file():
        files = [base_dir] if base_dir.suffix.lower() in DATA_FILE_SUFFIXES else []
        return base_dir.parent, files, provenance_id

    if extract_archives:
        _extract_archives_in_tree(base_dir, archive_max_depth)
    files = _match_files(
        base_dir,
        source.get("include_globs"),
        source.get("exclude_globs"),
    )
    return base_dir, files, provenance_id

def _load_any_table(path: Path) -> pd.DataFrame:
    # The format registry handles .tsv (with encoding + delimiter detection),
    # so every discovered file goes through the same robust loader.
    df = load_input_file(str(path))
    # Strip whitespace from headers — many real CSVs have " scenario", "id ",
    # etc. and the schema mapping is whitespace-sensitive.
    df.columns = [str(col).strip() for col in df.columns]
    return df


# Output suffixes ``save_output_file`` (or the dedicated .tsv branch) can
# actually serialize. Inputs in any other discovered format (.parquet, .json,
# .sav, .dta, .feather, .ods, .xlsm, ...) are written back as CSV; .xls is
# upgraded to .xlsx because pandas 2.x dropped the xlwt writer.
_WRITABLE_OUTPUT_SUFFIXES = {".csv", ".tsv", ".xlsx", ".pkl", ".pickle", ".parquet"}


def _resolve_output_suffix(input_suffix: str, output_format: str = "native") -> str:
    """Suffix of a standardized file: the input's own format by default, or
    Parquet for every table with ``output_format="parquet"``."""
    if output_format == "parquet":
        return ".parquet"
    suffix = input_suffix.lower()
    if suffix in _WRITABLE_OUTPUT_SUFFIXES:
        return suffix
    if suffix == ".xls":
        return ".xlsx"
    return ".csv"

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
    # Single pruned walk instead of four full rglob passes (which would also
    # descend into .git object stores of cloned sources).
    candidates = {
        path
        for path in _iter_discovery_files(source_root)
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in mapping_patterns)
    }
    return sorted(candidates)


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
    # build_original_column_lookup only reads df.columns — no copy needed.
    original_lookup = build_original_column_lookup(original_df, mapping)
    unknown_columns = _filter_never_map_columns(
        identify_unmapped_columns([str(c) for c in original_df.columns], mapping)
    )
    unknown_column_set = set(unknown_columns)
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
        # Numeric columns are already what ``to_numeric`` would return; only
        # coerce object/string/bool columns (the per-column ``to_numeric``
        # call dominated summarization time on wide telemetry exports).
        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
            numeric_series = series
        else:
            numeric_series = pd.to_numeric(series, errors="coerce")
        n_valid = int(numeric_series.count())

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
                "n": n_valid,
                "mean": float(numeric_series.mean()) if n_valid else None,
                "sd": float(numeric_series.std(ddof=1)) if n_valid > 1 else None,
                "review_flag": base_canonical in unknown_column_set,
                "mapping_confidence": "high" if base_canonical not in unknown_column_set else "unknown_alias",
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


# ---------------------------------------------------------------------------
# Built-in schema families
# ---------------------------------------------------------------------------
# (mapping domain, schema filename). The filename is what
# ``schemas/dataset_type_profiles.yaml`` references in each profile's
# ``schema`` field. Order matters for the canonical -> domain lookup: later
# families win when two families define the same canonical name.
BUILTIN_SCHEMA_FAMILIES: tuple[tuple[str, Path], ...] = (
    (MAPPING_DOMAIN_METADATA, DEFAULT_METADATA_SCHEMA_PATH),
    (MAPPING_DOMAIN_DETECTION, DEFAULT_DETECTION_SCHEMA_PATH),
    (MAPPING_DOMAIN_SENSOR, DEFAULT_SENSOR_SCHEMA_PATH),
)


@dataclass
class SchemaFamilies:
    """Built-in alias families, loaded once per run and shared with workers.

    ``mappings`` is keyed by schema filename so dataset-type profiles select
    their family declaratively: adding a profile whose ``schema`` names a new
    YAML file under ``schemas/`` needs no code change (the file is loaded on
    first use).
    """

    mappings: dict[str, dict[str, str]]
    paths: dict[str, str | None]
    aliases_ci: dict[str, set[str]]
    canonical_domain_lookup: dict[str, str]
    # Canonical value range per DV id; used to reject implausible LLM suggestions.
    canonical_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    schema_version: str = ""

    @classmethod
    def load(cls, dv_schema_path: Path) -> "SchemaFamilies":
        mappings: dict[str, dict[str, str]] = {}
        paths: dict[str, str | None] = {}
        aliases_ci: dict[str, set[str]] = {}
        canonical_domain_lookup: dict[str, str] = {}
        for domain, schema_path in BUILTIN_SCHEMA_FAMILIES:
            mapping, family_aliases_ci, loaded_path = _load_optional_schema_mapping(schema_path)
            mappings[schema_path.name] = mapping
            paths[domain] = loaded_path
            aliases_ci[domain] = family_aliases_ci
            for canonical_name in mapping.values():
                canonical_domain_lookup[str(canonical_name)] = domain

        # The DV family comes from the run's --schema (which may be a custom
        # file) but is registered under the standard filename profiles use.
        dv_schema = load_schema(str(dv_schema_path))
        dv_mapping = dv_schema["mapping"]
        mappings[DV_SCHEMA_FILENAME] = dv_mapping
        paths[MAPPING_DOMAIN_DV] = str(dv_schema_path)
        aliases_ci[MAPPING_DOMAIN_DV] = {
            str(alias).lower() for alias in dv_mapping.keys() if isinstance(alias, str)
        }
        for canonical_name in dv_mapping.values():
            canonical_domain_lookup[str(canonical_name)] = MAPPING_DOMAIN_DV
        return cls(
            mappings, paths, aliases_ci, canonical_domain_lookup,
            canonical_ranges=canonical_ranges_from_schema(dv_schema.get("schema") or {}),
            schema_version=str((dv_schema.get("schema") or {}).get("version", "")),
        )

    def mapping_for_schema(self, schema_filename: str) -> dict[str, str]:
        mapping = self.mappings.get(schema_filename)
        if mapping is None:
            mapping, _, _ = _load_optional_schema_mapping(REPO_ROOT / "schemas" / schema_filename)
            if not mapping:
                logger.warning(
                    "Dataset-type profile references schema '%s', which is missing or empty.",
                    schema_filename,
                )
            self.mappings[schema_filename] = mapping
        return mapping

    def standard_mapping_for(self, dataset_type: str) -> dict[str, str]:
        """Metadata aliases overlaid with the dataset type's schema family."""
        family = self.mapping_for_schema(get_schema_for_dataset_type(dataset_type))
        return {**self.mappings.get(METADATA_SCHEMA_FILENAME, {}), **family}


@dataclass(frozen=True)
class _ResolvedMapping:
    """Alias mapping applied to one dataset before any LLM deduction."""

    mapping: dict[str, str]
    mapping_path: str | None  # repository mapping YAML actually merged in
    custom_aliases_ci: frozenset[str]
    declared_scales: tuple[dict[str, Any], ...]
    value_labels: tuple[dict[str, Any], ...] = ()


class _MappingResolver:
    """Build and memoize per-dataset alias mappings for one source.

    A dataset's mapping depends only on its schema family and the repository
    mapping YAML that applies to it, so it is built once per
    ``(family, mapping path)`` pair rather than once per file (thousands of
    telemetry files share a handful of mappings).
    """

    def __init__(self, families: SchemaFamilies, overrides_standard: bool) -> None:
        self._families = families
        self._overrides_standard = overrides_standard
        self._custom: dict[str, tuple[Any, ...] | None] = {}
        self._resolved: dict[tuple[str, str | None], _ResolvedMapping] = {}

    def _load_custom(self, mapping_path: str, dataset_id: str):
        if mapping_path not in self._custom:
            try:
                custom_schema_data = load_schema(mapping_path)
            except (yaml.YAMLError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed mapping YAML '%s' (first seen for dataset '%s'): %s",
                    mapping_path, dataset_id, exc,
                )
                self._custom[mapping_path] = None
            else:
                custom_mapping = custom_schema_data["mapping"]
                raw_schema = custom_schema_data.get("schema") or {}
                self._custom[mapping_path] = (
                    custom_mapping,
                    frozenset(str(alias).lower() for alias in custom_mapping.keys() if isinstance(alias, str)),
                    tuple(custom_schema_data.get("scales") or []),
                    tuple(raw_schema.get("value_labels") or []) if isinstance(raw_schema, dict) else (),
                )
        return self._custom[mapping_path]

    def resolve(self, dataset_type: str, mapping_path: Path | str | None, dataset_id: str) -> _ResolvedMapping:
        schema_filename = get_schema_for_dataset_type(dataset_type)
        path_key = str(mapping_path) if mapping_path is not None else None
        key = (schema_filename, path_key)
        resolved = self._resolved.get(key)
        if resolved is not None:
            return resolved

        standard_mapping = self._families.standard_mapping_for(dataset_type)
        source_mapping = standard_mapping
        merged_path: str | None = None
        custom_aliases_ci: frozenset[str] = frozenset()
        declared_scales: tuple[dict[str, Any], ...] = ()
        value_labels: tuple[dict[str, Any], ...] = ()
        custom = self._load_custom(path_key, dataset_id) if path_key is not None else None
        if custom is not None:
            custom_mapping, custom_aliases_ci, declared_scales, value_labels = custom
            # Only report a mapping path once a non-empty source mapping is
            # actually merged in.
            if custom_mapping:
                source_mapping = _merge_with_standard_precedence(
                    custom_mapping,
                    standard_mapping,
                    source_overrides_standard=self._overrides_standard,
                )
                merged_path = path_key
        resolved = _ResolvedMapping(
            mapping=_remove_never_map_aliases(source_mapping),
            mapping_path=merged_path,
            custom_aliases_ci=custom_aliases_ci,
            declared_scales=declared_scales,
            value_labels=value_labels,
        )
        self._resolved[key] = resolved
        return resolved


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SourceJob:
    """Settings shared by every dataset task of one source (picklable)."""

    source_id: str
    source_type: str
    location: str
    commit_sha: str | None
    base_dir: Path
    output_dir: Path
    requested_mapping_path: str | None
    mapping_candidates: tuple[Path, ...]
    overrides_standard: bool
    llm_enabled: bool
    debug_mappings: bool
    llm_plausibility_check: bool = True
    output_format: str = "native"


@dataclass
class _DatasetTask:
    index: int
    source: _SourceJob
    file_path: Path
    # LLM-deduced aliases computed by the parent process (second pass).
    llm_delta: dict[str, str] | None = None


@dataclass
class _DatasetOutcome:
    """Everything the run-level aggregation needs from one dataset."""

    index: int
    dataset_id: str
    status: str  # "processed" | "failed" | "needs_llm"
    dataset_type: str | None = None
    mapping_path: str | None = None  # repository mapping YAML resolved for the file
    raw_columns: list[str] = field(default_factory=list)
    unknown_before_llm: list[str] = field(default_factory=list)
    llm_inferred: list[tuple[str, str]] = field(default_factory=list)
    mapping_debug_records: list[dict[str, Any]] = field(default_factory=list)
    meta_rows: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] | None = None
    composite_records: list[dict[str, Any]] = field(default_factory=list)
    llm_rejections: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


# (resolved mapping, dataset_id, dataset_type, raw_columns, unknown aliases) -> LLM alias delta
LlmResolver = Callable[[_ResolvedMapping, str, str, list[str], list[str]], dict[str, str]]


def _dataset_artifact_prefix(task: _DatasetTask) -> tuple[Path, str]:
    relative_file = task.file_path.relative_to(task.source.base_dir)
    output_suffix = _resolve_output_suffix(task.file_path.suffix, task.source.output_format)
    artifact_suffix_length = max(
        len(f"-standardized{output_suffix}"),
        len("-mapping-debug.json"),
        len("-quality.json"),
        len("-error.log"),
    )
    prefix = _build_artifact_prefix(
        relative_file,
        output_dir=task.source.output_dir,
        artifact_suffix_length=artifact_suffix_length,
    )
    return relative_file, prefix


def _write_dataset_error(task: _DatasetTask, message: str) -> None:
    _, artifact_prefix = _dataset_artifact_prefix(task)
    with open(task.source.output_dir / f"{artifact_prefix}-error.log", "w", encoding="utf-8") as f:
        f.write(message)


def _process_dataset(
    task: _DatasetTask,
    families: SchemaFamilies,
    resolver: _MappingResolver,
    llm_resolver: LlmResolver | None = None,
) -> _DatasetOutcome:
    """Classify, map, standardize, compose and summarize one dataset file.

    Runs in the parent process or in a worker. When the dataset needs LLM
    deduction and neither ``task.llm_delta`` nor ``llm_resolver`` is
    available (worker first pass), it returns ``status="needs_llm"`` without
    writing anything; the parent then runs the model and resubmits the task
    with ``llm_delta``. The local model therefore only ever lives in the
    parent process.
    """
    source = task.source
    relative_file, artifact_prefix = _dataset_artifact_prefix(task)
    dataset_id = relative_file.as_posix()
    relative_path = str(relative_file)
    output_suffix = _resolve_output_suffix(task.file_path.suffix, source.output_format)
    destination = source.output_dir / f"{artifact_prefix}-standardized{output_suffix}"

    try:
        original_df = _load_any_table(task.file_path)
        raw_columns = [str(column) for column in original_df.columns]
        dataset_type = classify_dataset_type(relative_file, raw_columns)
        repository_mapping_path = _resolve_repository_mapping_path(
            source.base_dir,
            list(source.mapping_candidates),
            requested_mapping_path=source.requested_mapping_path,
            dataset_path=task.file_path,
        )
        resolved = resolver.resolve(dataset_type, repository_mapping_path, dataset_id)
        # Text-labelled responses declared in the repository mapping become
        # numbers before anything numeric (profiles, summaries) looks at them.
        original_df, value_label_report = apply_value_labels(original_df, resolved.value_labels)
        answer_code_report: dict[str, int] = {}
        if dataset_type in llm_eligible_dataset_types():
            # Survey-like tables only: LimeSurvey exports "neg1"/"neg2" for
            # negative scale points, which numeric coercion would drop.
            original_df, answer_code_report = recode_negative_answer_codes(original_df)
        source_mapping = resolved.mapping
        dataset_mapping = source_mapping
        unknown_before_llm = _filter_llm_eligible_columns(
            identify_unmapped_columns(raw_columns, source_mapping)
        )

        # Columns the LLM may see. With the plausibility check on, columns whose
        # name or values rule them out as DVs are dropped (and logged) first.
        llm_candidates: list[str] = []
        llm_rejections: list[dict[str, Any]] = []
        profiles: dict[str, ColumnProfile] = {}
        if source.llm_enabled and dataset_type in llm_eligible_dataset_types() and unknown_before_llm:
            if source.llm_plausibility_check:
                profiles = profile_columns(original_df, unknown_before_llm)
                llm_candidates, llm_rejections = screen_llm_candidates(
                    unknown_before_llm, raw_columns, profiles
                )
            else:
                llm_candidates = list(unknown_before_llm)

        llm_delta = task.llm_delta
        if llm_delta is None and llm_candidates:
            if llm_resolver is None:
                return _DatasetOutcome(
                    index=task.index,
                    dataset_id=dataset_id,
                    status="needs_llm",
                    dataset_type=dataset_type,
                    mapping_path=str(repository_mapping_path) if repository_mapping_path else None,
                    raw_columns=raw_columns,
                    unknown_before_llm=llm_candidates,
                )
            llm_delta = llm_resolver(resolved, dataset_id, dataset_type, raw_columns, llm_candidates)

        llm_inferred: list[tuple[str, str]] = []
        if llm_delta is not None:
            accepted: dict[str, str] = {}
            for alias in llm_candidates:
                alias_text = str(alias)
                inferred = llm_delta.get(alias_text) or llm_delta.get(alias_text.lower())
                if not inferred:
                    continue
                if source.llm_plausibility_check:
                    rejection = check_llm_suggestion(
                        alias_text, str(inferred), profiles.get(alias_text), families.canonical_ranges
                    )
                    if rejection is not None:
                        llm_rejections.append(rejection)
                        continue
                accepted[alias_text] = str(inferred)
                accepted[alias_text.lower()] = str(inferred)
                llm_inferred.append((alias_text, str(inferred)))
            dataset_mapping = {**source_mapping, **accepted}

        mapping_debug_records = _build_dataset_mapping_debug_records(
            source_mapping=source_mapping,
            dataset_mapping=dataset_mapping,
            original_columns=raw_columns,
            unknown_before_llm=unknown_before_llm,
            source_custom_aliases_ci=set(resolved.custom_aliases_ci),
            source_mapping_path=resolved.mapping_path,
            schema_family_aliases_ci=families.aliases_ci,
            schema_family_paths=families.paths,
            canonical_domain_lookup=families.canonical_domain_lookup,
        )

        if source.debug_mappings:
            debug_path = source.output_dir / f"{artifact_prefix}-mapping-debug.json"
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_id": source.source_id,
                        "dataset_id": dataset_id,
                        "dataset_type": dataset_type,
                        "path": relative_path,
                        "summary": build_mapping_debug_summary(mapping_debug_records),
                        "debug_mappings": mapping_debug_records,
                    },
                    f,
                    indent=2,
                )
            # Per-row traces are already persisted to the per-dataset JSON
            # above; only emit a single debug breadcrumb here so stdout stays
            # readable on large batches.
            logger.debug(
                "[DEBUG] Mapping trace (%s/%s) -> %s",
                source.source_id, dataset_id, debug_path,
            )

        # Shallow copy: standardize_columns only reassigns .columns, so
        # sharing the data blocks avoids duplicating large sensor-stream
        # frames in memory.
        standardized_df = standardize_columns(original_df.copy(deep=False), dataset_mapping)
        # Build canonical -> ordered originals for declared-scale resolution.
        # Items match the canonical-group order (first occurrence -> bare
        # canonical, subsequent -> __dup_N) so the scale's reverse_aliases can
        # be applied by name.
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
            eligible_dataset_types=item_composition_dataset_types(),
            declared_scales=list(resolved.declared_scales),
            canonical_to_originals=canonical_to_originals_ordered,
        )
        if destination.suffix.lower() == ".tsv":
            standardized_df.to_csv(destination, sep="\t", index=False)
        else:
            save_output_file(standardized_df, str(destination))

        provenance = {
            "source_type": source.source_type,
            "location": source.location,
            "path": relative_path,
            "commit": source.commit_sha,
            "source_mapping": resolved.mapping_path,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        rows, quality = _summarize_dataset(
            source.source_id,
            dataset_id,
            original_df,
            standardized_df,
            dataset_mapping,
            provenance,
            mapping_debug_records,
            dataset_type,
        )
        if value_label_report:
            quality["value_labels_applied"] = value_label_report
        if answer_code_report:
            quality["negative_answer_codes_recoded"] = answer_code_report
        quality["dataset_id"] = dataset_id
        quality["standardized_file"] = destination.name
        # Column types and row count let the data-package export describe the
        # table without reading it back (scripts/fair_export.py).
        quality["standardized_rows"] = int(len(standardized_df))
        quality["standardized_column_types"] = {
            str(col): frictionless_type(standardized_df.iloc[:, i])
            for i, col in enumerate(standardized_df.columns)
        }
        with open(source.output_dir / f"{artifact_prefix}-quality.json", "w", encoding="utf-8") as f:
            json.dump(quality, f, indent=2)

        return _DatasetOutcome(
            index=task.index,
            dataset_id=dataset_id,
            status="processed",
            dataset_type=dataset_type,
            raw_columns=raw_columns,
            unknown_before_llm=unknown_before_llm,
            llm_inferred=llm_inferred,
            llm_rejections=llm_rejections,
            mapping_debug_records=mapping_debug_records,
            meta_rows=rows,
            quality=quality,
            composite_records=composite_records,
        )
    except Exception as exc:  # noqa: BLE001
        # Persist the full traceback in the log and the message alongside the
        # dataset so post-run triage doesn't lose it when many files fail.
        logger.exception(
            "Dataset processing failed for %s/%s: %s",
            source.source_id, dataset_id, exc,
        )
        _write_dataset_error(task, str(exc))
        return _DatasetOutcome(index=task.index, dataset_id=dataset_id, status="failed", error=str(exc))


# Worker-process state, set once per worker by ``_init_dataset_worker`` so the
# schema families are pickled once per worker instead of once per task.
_WORKER_FAMILIES: SchemaFamilies | None = None
_WORKER_RESOLVERS: dict[tuple[str, bool], _MappingResolver] = {}


def _init_dataset_worker(families: SchemaFamilies, log_level: int) -> None:
    global _WORKER_FAMILIES
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _WORKER_FAMILIES = families
    _WORKER_RESOLVERS.clear()


def _run_dataset_task_in_worker(task: _DatasetTask) -> _DatasetOutcome:
    assert _WORKER_FAMILIES is not None, "worker not initialised"
    key = (task.source.source_id, task.source.overrides_standard)
    resolver = _WORKER_RESOLVERS.get(key)
    if resolver is None:
        resolver = _WORKER_RESOLVERS[key] = _MappingResolver(_WORKER_FAMILIES, task.source.overrides_standard)
    return _process_dataset(task, _WORKER_FAMILIES, resolver)


def default_worker_count() -> int:
    """Worker processes used by the CLI when ``--workers`` is not given."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


class _SourceLlmState:
    """Per-source LLM caches; lives in the parent process only."""

    def __init__(
        self,
        source: dict[str, Any],
        base_dir: Path,
        preferred_models: list[str] | None,
        shortlist: str = "definition",
    ) -> None:
        self._source = source
        self._shortlist = shortlist
        self._base_dir = base_dir
        self._preferred_models = preferred_models
        self._inference_cache: dict[str, str | None] = {}
        self._signature_cache: dict[tuple[Any, ...], dict[str, str]] = {}
        self._repository_context: str | None = None

    def deduce(
        self,
        resolved: _ResolvedMapping,
        dataset_id: str,
        dataset_type: str,
        raw_columns: list[str],
        unknown_before_llm: list[str],
    ) -> dict[str, str]:
        """Return the aliases the LLM adds on top of ``resolved.mapping``."""
        # Column-signature dedup: reuse the deduction of a prior file with the
        # exact same column set, dataset type and mapping to avoid redundant
        # LLM calls.
        signature = (
            tuple(sorted(str(c).lower() for c in raw_columns)),
            tuple(sorted(str(c) for c in unknown_before_llm)),
            dataset_type,
            resolved.mapping_path,
        )
        cached = self._signature_cache.get(signature)
        if cached is not None:
            logger.info(
                "Reusing cached column-signature mapping for '%s' "
                "(%d columns, type=%s) — skipping LLM.",
                dataset_id, len(raw_columns), dataset_type,
            )
            return cached
        if self._repository_context is None:
            source = self._source
            extra_context = _coerce_manifest_text_list(source.get("llm_context"))
            extra_context.extend(_coerce_manifest_text_list(source.get("publication_context")))
            self._repository_context = collect_repository_context(
                self._base_dir,
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
        augmented = _augment_mapping_with_llm_deductions(
            resolved.mapping,
            unknown_before_llm,
            self._base_dir,
            preferred_models=self._preferred_models,
            inference_cache=self._inference_cache,
            repository_context=self._repository_context,
            shortlist=self._shortlist,
        )
        delta = {
            alias: canonical
            for alias, canonical in augmented.items()
            if resolved.mapping.get(alias) != canonical
        }
        self._signature_cache[signature] = delta
        return delta


@dataclass
class _PreparedSource:
    """A materialized source whose datasets are ready to be processed."""

    source: dict[str, Any]
    source_job: _SourceJob
    tasks: list[_DatasetTask]
    preferred_models: list[str] | None
    futures: list[Future] | None = None  # first-pass futures when a pool is used


def _run_source_datasets(
    tasks: list[_DatasetTask],
    families: SchemaFamilies,
    resolver: _MappingResolver,
    llm_state: _SourceLlmState,
    pool: ProcessPoolExecutor | None,
    progress_desc: str,
    futures: list[Future] | None = None,
) -> list[_DatasetOutcome]:
    """Process a source's datasets, returning outcomes in file order.

    Without a pool every dataset is processed inline (LLM deduction happens
    in place). With a pool, datasets are fanned out to worker processes; the
    few that need LLM deduction come back as ``needs_llm`` and are handled
    here **in file order** — so the LLM caches see exactly the same call
    sequence as a sequential run — before being resubmitted.
    """
    progress = tqdm(total=len(tasks), desc=progress_desc, unit="file", leave=False)
    try:
        if pool is None:
            outcomes = []
            for task in tasks:
                progress.set_postfix(dataset=task.file_path.name)
                outcomes.append(_process_dataset(task, families, resolver, llm_resolver=llm_state.deduce))
                progress.update()
            return outcomes

        def _collect(task: _DatasetTask, future: Future) -> _DatasetOutcome:
            try:
                return future.result()
            except Exception as exc:  # noqa: BLE001 — e.g. a worker died
                logger.exception("Worker failed on %s: %s", task.file_path, exc)
                _write_dataset_error(task, str(exc))
                return _DatasetOutcome(
                    index=task.index,
                    dataset_id=task.file_path.relative_to(task.source.base_dir).as_posix(),
                    status="failed",
                    error=str(exc),
                )

        if futures is None:
            futures = [pool.submit(_run_dataset_task_in_worker, task) for task in tasks]
        outcomes: list[_DatasetOutcome] = []
        second_pass: dict[int, Future] = {}
        for task, future in zip(tasks, futures, strict=True):
            outcome = _collect(task, future)
            if outcome.status == "needs_llm":
                assert outcome.dataset_type is not None
                try:
                    resolved = resolver.resolve(outcome.dataset_type, outcome.mapping_path, outcome.dataset_id)
                    task.llm_delta = llm_state.deduce(
                        resolved,
                        outcome.dataset_id,
                        outcome.dataset_type,
                        outcome.raw_columns,
                        outcome.unknown_before_llm,
                    )
                except Exception as exc:  # noqa: BLE001 — same contract as the sequential path
                    logger.exception(
                        "Dataset processing failed for %s/%s: %s",
                        task.source.source_id, outcome.dataset_id, exc,
                    )
                    _write_dataset_error(task, str(exc))
                    outcomes.append(_DatasetOutcome(
                        index=task.index, dataset_id=outcome.dataset_id, status="failed", error=str(exc)
                    ))
                    progress.update()
                    continue
                second_pass[task.index] = pool.submit(_run_dataset_task_in_worker, task)
            else:
                progress.update()
            outcomes.append(outcome)
        for task in tasks:
            if task.index in second_pass:
                outcomes[task.index] = _collect(task, second_pass[task.index])
                progress.update()
        return outcomes
    finally:
        progress.close()


def _collect_llm_rejection(
    rejections_by_key: dict[tuple[Any, ...], dict[str, Any]],
    source_id: str,
    dataset_id: str,
    rejection: dict[str, Any],
) -> None:
    """Group a plausibility rejection with identical ones from other datasets."""
    key = (source_id, rejection["alias"], rejection.get("proposed_canonical"), rejection["stage"], rejection["reason"])
    entry = rejections_by_key.get(key)
    if entry is None:
        entry = rejections_by_key[key] = {"source_id": source_id, **rejection, "datasets": []}
    if dataset_id not in entry["datasets"]:
        entry["datasets"].append(dataset_id)


def _finalize_llm_rejections(rejections_by_key: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    entries = [{**entry, "datasets": sorted(entry["datasets"])} for entry in rejections_by_key.values()]
    return sorted(entries, key=lambda e: (e["source_id"], e["stage"], e["reason"], e["alias"].lower()))


def _build_llm_rejection_log_lines(rejections: list[dict[str, Any]]) -> list[str]:
    if not rejections:
        return []
    lines = ["", f"Rejected by the LLM plausibility check ({len(rejections)}):"]
    for entry in rejections:
        proposal = f" -> {entry['proposed_canonical']}" if entry.get("proposed_canonical") else ""
        lines.append(
            f"[{entry['source_id']}] {entry['alias']}{proposal}: {entry['reason']} "
            f"({entry['stage']}; datasets: {', '.join(entry['datasets'])})"
        )
    return lines


def _source_result(
    source_id: str,
    status: str,
    output_dir: Path,
    message: str | None,
    discovered_files: int = 0,
) -> SourceRunResult:
    """Result row for a source that produced no datasets."""
    return SourceRunResult(
        source_id=source_id,
        status=status,
        discovered_files=discovered_files,
        processed_files=0,
        failed_files=0,
        unknown_columns=0,
        total_columns=0,
        mapped_columns=0,
        mapped_ratio=0.0,
        output_dir=str(output_dir),
        message=message,
    )


def run_batch(
    manifest_path: Path,
    output_dir: Path,
    schema_path: Path,
    llm_deduction_enabled: bool = True,
    cache_root: Path | None = None,
    refresh_remote_cache: bool = False,
    preferred_models: list[str] | None = None,
    debug_mappings: bool = False,
    workers: int = 1,
    llm_plausibility_check: bool = True,
    llm_shortlist: str = "definition",
    output_format: str = "native",
    fair_export: bool = True,
) -> dict[str, Any]:
    """Standardize every source in the manifest and write run-level artefacts.

    ``output_format="parquet"`` writes every standardized table as Parquet
    (default: the input's own format). ``fair_export`` adds a Frictionless
    ``datapackage.json`` per source and an RO-Crate for the run
    (``scripts/fair_export.py``).

    ``llm_plausibility_check`` screens LLM input and output with
    ``schemas/llm_plausibility_rules.yaml`` (see ``scripts/llm_plausibility``);
    rejections are written to ``llm_rejected_suggestions.json``. Turn it off
    only to measure what the check changes. ``llm_shortlist`` chooses how
    candidate DVs are retrieved for the LLM (``"fuzzy"`` alias strings or
    ``"definition"`` TF-IDF over DV definitions, ``scripts/dv_retrieval.py``).

    ``workers > 1`` processes each source's datasets in that many worker
    processes. Outputs are aggregated in file order, so they match a
    sequential run (apart from per-dataset ``run_timestamp`` values).
    """
    families = SchemaFamilies.load(schema_path)
    sensor_mapping = families.mappings[DEFAULT_SENSOR_SCHEMA_PATH.name]
    detection_mapping = families.mappings[DEFAULT_DETECTION_SCHEMA_PATH.name]
    metadata_mapping = families.mappings[METADATA_SCHEMA_FILENAME]

    output_dir.mkdir(parents=True, exist_ok=True)
    standardized_root = output_dir / "standardized"
    standardized_root.mkdir(parents=True, exist_ok=True)
    resolved_cache_root = cache_root or (output_dir / ".cache" / "sources")
    resolved_cache_root.mkdir(parents=True, exist_ok=True)

    sources = load_manifest(manifest_path)
    run_results: list[SourceRunResult] = []
    meta_rows: list[dict[str, Any]] = []
    llm_deductions_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    llm_rejections_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
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
    pool: ProcessPoolExecutor | None = None

    try:
        with TemporaryDirectory(prefix="opendv_batch_") as temp_dir:
            checkout_root = Path(temp_dir)

            # Phase 1: materialize every source (network-bound, sequential).
            prepared: list[SourceRunResult | _PreparedSource] = []
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
                    prepared.append(_source_result(source_id, exc.status, source_output_dir, str(exc)))
                    continue
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in TEMPORARY_UNAVAILABLE_STATUS_CODES:
                        logger.exception("[FAILED] %s: %s", source_id, exc)
                        prepared.append(_source_result(source_id, "failed", source_output_dir, str(exc)))
                        continue
                    # Retries are exhausted but the host reports a transient
                    # outage (OSF serves its maintenance page as a 503): not a
                    # pipeline failure, so report it as unavailable for now.
                    message = (
                        f"Remote host temporarily unavailable (HTTP {exc.response.status_code}) "
                        f"for {exc.request.url}; re-run later to include this source."
                    )
                    logger.warning("[TEMPORARILY_UNAVAILABLE] %s: %s", source_id, message)
                    prepared.append(
                        _source_result(source_id, "temporarily_unavailable", source_output_dir, message)
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    # logger.exception preserves traceback so post-mortem debugging
                    # can recover the original failure point even though we keep
                    # the loop running across remaining sources.
                    logger.exception("[FAILED] %s: %s", source_id, exc)
                    prepared.append(_source_result(source_id, "failed", source_output_dir, str(exc)))
                    continue

                if not files:
                    message = "No supported dataset files were discovered for this source."
                    logger.warning("[NOT_AVAILABLE] %s: %s", source_id, message)
                    prepared.append(_source_result(source_id, "not_available", source_output_dir, message))
                    continue

                # CLI --disable-llm-deduction is a hard off-switch: when LLM deduction is
                # disabled at the run level, no per-source manifest flag can re-enable it.
                # When enabled at run level, an absent/None per-source flag inherits the
                # run-level setting; an explicit per-source false opts that source out.
                source_llm_flag = source.get("use_llm_deduction")
                if source_llm_flag is None:
                    source_llm_flag = llm_deduction_enabled
                source_llm_enabled = llm_deduction_enabled and bool(source_llm_flag)
                source_preferred_models = (
                    source.get("llm_models") if isinstance(source.get("llm_models"), list) else preferred_models
                )
                source_overrides_standard = bool(source.get("override_standard_aliases", False))
                requested_mapping_path = (
                    str(source.get("mapping_path")).strip()
                    if source.get("mapping_path")
                    else None
                )
                mapping_candidates = _find_repository_mapping_candidates(base_dir)
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
                        prepared.append(
                            _source_result(
                                source_id, "failed", source_output_dir, str(exc), discovered_files=len(files)
                            )
                        )
                        continue

                source_job = _SourceJob(
                    source_id=source_id,
                    source_type=source["source_type"],
                    location=source["location"],
                    commit_sha=commit_sha,
                    base_dir=base_dir,
                    output_dir=source_output_dir,
                    requested_mapping_path=requested_mapping_path,
                    mapping_candidates=tuple(mapping_candidates),
                    overrides_standard=source_overrides_standard,
                    llm_enabled=source_llm_enabled,
                    debug_mappings=debug_mappings,
                    llm_plausibility_check=llm_plausibility_check,
                    output_format=output_format,
                )
                tasks = [
                    _DatasetTask(index=index, source=source_job, file_path=file_path)
                    for index, file_path in enumerate(files)
                ]
                prepared.append(_PreparedSource(source, source_job, tasks, source_preferred_models))

            # Phase 2: process. With a pool, every dataset of every source is
            # submitted up front so small/single-file sources overlap with
            # large ones; outcomes are still consumed source by source, in
            # manifest and file order.
            n_tasks = sum(len(entry.tasks) for entry in prepared if isinstance(entry, _PreparedSource))
            if workers > 1 and n_tasks > 1:
                pool = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_dataset_worker,
                    initargs=(families, logging.getLogger().getEffectiveLevel()),
                )
                for entry in prepared:
                    if isinstance(entry, _PreparedSource):
                        entry.futures = [pool.submit(_run_dataset_task_in_worker, task) for task in entry.tasks]

            for entry in prepared:
                if isinstance(entry, SourceRunResult):
                    run_results.append(entry)
                    continue
                source = entry.source
                source_id = entry.source_job.source_id
                source_output_dir = entry.source_job.output_dir
                files = [task.file_path for task in entry.tasks]
                outcomes = _run_source_datasets(
                    entry.tasks,
                    families,
                    _MappingResolver(families, entry.source_job.overrides_standard),
                    _SourceLlmState(source, entry.source_job.base_dir, entry.preferred_models, llm_shortlist),
                    pool,
                    progress_desc=f"Datasets ({source_id})",
                    futures=entry.futures,
                )

                processed = 0
                failed = 0
                total_unknown = 0
                total_columns = 0
                total_mapped_columns = 0
                source_metrics = {
                    "mapping": 0,
                    "llm": 0,
                    "blocked": 0,
                    "unmapped": 0,
                    "total_columns_seen": 0,
                }
                source_domain_metrics: dict[str, int] = {}
                dataset_errors: list[str] = []
                for outcome in outcomes:
                    if outcome.status != "processed":
                        failed += 1
                        dataset_errors.append(f"{outcome.dataset_id}: {outcome.error}")
                        continue
                    dataset_id = outcome.dataset_id
                    dataset_type = outcome.dataset_type
                    for rejection in outcome.llm_rejections:
                        _collect_llm_rejection(llm_rejections_by_key, source_id, dataset_id, rejection)
                    for alias, canonical in outcome.llm_inferred:
                        _collect_llm_deduction(
                            llm_deductions_by_key,
                            source_id=source_id,
                            dataset_id=dataset_id,
                            alias=alias,
                            canonical_dv=canonical,
                        )
                    for row in outcome.mapping_debug_records:
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
                    for record in outcome.composite_records:
                        all_composite_records.append(
                            {
                                "source_id": source_id,
                                "dataset_id": dataset_id,
                                "dataset_type": dataset_type,
                                **record,
                            }
                        )
                    quality = outcome.quality or {}
                    total_unknown += quality.get("unknown_columns", 0)
                    total_columns += quality.get("total_columns", 0)
                    total_mapped_columns += quality.get("mapped_columns", 0)
                    meta_rows.extend(outcome.meta_rows)
                    processed += 1

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
                        mapped_dv_columns=int(source_domain_metrics.get("dv", 0)),
                        mapping_domains=dict(sorted(source_domain_metrics.items())),
                    )
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    meta_df = pd.DataFrame(meta_rows)
    meta_csv = output_dir / "meta_view.csv"
    meta_json = output_dir / "meta_view.json"
    meta_df.to_csv(meta_csv, index=False)
    meta_df.to_json(meta_json, orient="records", indent=2)
    llm_deductions = _finalize_llm_deductions(llm_deductions_by_key)
    llm_rejections = _finalize_llm_rejections(llm_rejections_by_key)
    llm_log_path = output_dir / "llm_deductions.log"
    llm_log_path.write_text(
        "\n".join(
            _build_llm_deduction_log_lines(llm_deductions) + _build_llm_rejection_log_lines(llm_rejections)
        ) + "\n",
        encoding="utf-8",
    )
    llm_json_path = output_dir / "llm_deductions.json"
    with open(llm_json_path, "w", encoding="utf-8") as f:
        json.dump(llm_deductions, f, indent=2)
    llm_rejections_path = output_dir / "llm_rejected_suggestions.json"
    with open(llm_rejections_path, "w", encoding="utf-8") as f:
        json.dump(llm_rejections, f, indent=2)
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
        if r.status in UNAVAILABLE_SOURCE_STATUSES
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
        "llm_plausibility_check": llm_plausibility_check,
        "llm_shortlist": llm_shortlist,
        "llm_plausibility_rules_version": load_plausibility_rules().version if llm_plausibility_check else None,
        "llm_rejected_count": len(llm_rejections),
        "llm_rejected_by_reason": dict(sorted(
            pd.Series([r["reason"] for r in llm_rejections], dtype=object).value_counts().items()
        )) if llm_rejections else {},
        "llm_rejected_suggestions_json": str(llm_rejections_path),
        "metadata_schema": families.paths.get(MAPPING_DOMAIN_METADATA),
        "metadata_mapping_aliases": len(metadata_mapping),
        "sensor_schema": families.paths.get(MAPPING_DOMAIN_SENSOR),
        "sensor_mapping_aliases": len(sensor_mapping),
        "detection_schema": families.paths.get(MAPPING_DOMAIN_DETECTION),
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
        "workers": workers,
        "output_format": output_format,
        "schema_version": families.schema_version,
        "llm_models_used": _llm_models_used() if llm_deduction_enabled else [],
        "cache_root": str(resolved_cache_root),
        "results": [r.__dict__ for r in run_results],
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if fair_export:
        # Frictionless data package per source + RO-Crate for the run.
        try:
            from scripts.fair_export import export_run

            summary["fair_export"] = export_run(output_dir, schema_path)
        except Exception as exc:  # noqa: BLE001 - metadata must not fail a finished run
            logger.exception("FAIR export failed: %s", exc)
            summary["fair_export"] = {"error": str(exc)}
        with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def _llm_models_used() -> list[str]:
    """Models the local LLM fallback actually loaded in this process."""
    try:
        from scripts.llm_utils import models_used
    except ImportError:
        return []
    return models_used()

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
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help=(
            "Worker processes for per-dataset standardization (default: %(default)s). "
            "Use 1 for a fully sequential run. LLM deduction always runs in the main process."
        ),
    )
    parser.add_argument(
        "--no-llm-plausibility-check",
        dest="llm_plausibility_check",
        action="store_false",
        help=(
            "Send every unmapped column to the LLM and accept every suggestion "
            "(disables schemas/llm_plausibility_rules.yaml; for ablation only)."
        ),
    )
    parser.add_argument(
        "--llm-shortlist",
        choices=["fuzzy", "definition"],
        default="definition",
        help="How candidate DVs are retrieved for the LLM: TF-IDF over DV definitions (definition; "
             "default, more than doubles mapping precision in analyses/evaluate_llm_mapping.py) or "
             "alias-string similarity (fuzzy, the earlier behaviour).",
    )
    parser.add_argument(
        "--output-format",
        choices=["native", "parquet"],
        default="native",
        help="Format of standardized tables: the input's own format (native, default) or Parquet.",
    )
    parser.add_argument(
        "--no-fair-export",
        action="store_true",
        help="Skip the Frictionless data packages and the RO-Crate written after the run.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

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
        workers=args.workers,
        llm_plausibility_check=args.llm_plausibility_check,
        llm_shortlist=args.llm_shortlist,
        output_format=args.output_format,
        fair_export=not args.no_fair_export,
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
