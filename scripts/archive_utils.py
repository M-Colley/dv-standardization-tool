"""Archive extraction helpers for the batch standardization pipeline.

This module is deliberately self-contained so it can be unit-tested without
booting the full ``run_batch_standardization`` import graph (which pulls in
pandas, transformers, etc.). The five public helpers — ``_is_within_directory``,
``_resolve_archive_max_depth``, ``_should_skip_archive_member``,
``_extract_zip_files_recursive``, and ``_extract_archives_in_tree`` — are
re-exported from ``scripts.run_batch_standardization`` for backwards
compatibility with existing tests and callers.

Security notes:

* Zip-slip is mitigated by resolving each member destination and checking it
  is within the per-archive extract root before writing.
* ``__MACOSX`` directories and macOS metadata files (``.DS_Store``, ``._*``)
  are skipped to avoid emitting confusing forks during downstream discovery.
* Recursion is depth-limited (default 3) to avoid pathological zip-bombs of
  the "archive containing an archive containing..." variety.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Suffix sets used during archive walk + member filtering
# ---------------------------------------------------------------------------

# .txt and .dat are loadable by the converter but excluded from auto-discovery
# because they create too many false positives (README.txt, notes.dat, etc.).
_AMBIGUOUS_SUFFIXES: set[str] = {".txt", ".dat"}

try:  # pragma: no cover - exercised indirectly via convert_dv import
    from scripts.convert_dv import _FORMAT_REGISTRY as _CDV_FORMAT_REGISTRY

    DATA_FILE_SUFFIXES: set[str] = set(_CDV_FORMAT_REGISTRY.keys()) - _AMBIGUOUS_SUFFIXES
except Exception:  # noqa: BLE001 - convert_dv may fail to import in minimal envs
    DATA_FILE_SUFFIXES = {
        ".csv", ".tsv",
        ".xlsx", ".xls", ".xlsm", ".ods",
        ".pkl", ".pickle",
        ".parquet",
        ".sav", ".zsav",
        ".dta",
        ".feather", ".arrow",
        ".json", ".jsonl", ".ndjson",
    }

ARCHIVE_SUFFIXES: set[str] = {".zip"}
MAPPING_SUFFIXES: set[str] = {".yaml", ".yml"}
DEFAULT_ARCHIVE_MAX_DEPTH: int = 3


# ---------------------------------------------------------------------------
# Path / depth helpers
# ---------------------------------------------------------------------------

def _is_within_directory(path: Path, root: Path) -> bool:
    """Return True if ``path`` is within ``root`` (zip-slip guard)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_archive_max_depth(source: dict[str, Any]) -> int:
    """Validate and coerce the ``archive_max_depth`` field on a source spec."""
    value = source.get("archive_max_depth", DEFAULT_ARCHIVE_MAX_DEPTH)
    try:
        depth = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"archive_max_depth must be an integer, got {value!r}") from exc
    if depth < 1:
        raise ValueError("archive_max_depth must be >= 1.")
    return depth


def _should_skip_archive_member(member_path: Path) -> bool:
    """Return True for macOS metadata members that should not be extracted."""
    normalized_parts = [part for part in member_path.parts if part not in {"", "."}]
    if any(part == "__MACOSX" for part in normalized_parts):
        return True

    filename = member_path.name
    if filename == ".DS_Store" or filename.startswith("._"):
        return True

    return False


# ---------------------------------------------------------------------------
# Recursive extraction
# ---------------------------------------------------------------------------

def _extract_zip_files_recursive(
    zip_path: Path,
    source_root: Path,
    depth: int,
    max_depth: int,
) -> None:
    """Extract ``zip_path`` under ``source_root/__extracted_archives/<id>``.

    Recurses into nested archives until ``depth`` exceeds ``max_depth``. Members
    outside the destination root (zip-slip) and macOS metadata are skipped.
    """
    if depth > max_depth:
        return

    try:
        relative_zip = zip_path.relative_to(source_root)
    except ValueError:
        relative_zip = Path(zip_path.name)

    archive_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(relative_zip.with_suffix(""))).strip("_")
    if not archive_id:
        archive_id = "archive"

    extract_root = source_root / "__extracted_archives" / archive_id
    extract_root.mkdir(parents=True, exist_ok=True)
    resolved_extract_root = extract_root.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            member_path = Path(member.filename)
            if _should_skip_archive_member(member_path):
                continue

            suffix = member_path.suffix.lower()
            if (
                suffix not in DATA_FILE_SUFFIXES
                and suffix not in ARCHIVE_SUFFIXES
                and suffix not in MAPPING_SUFFIXES
            ):
                continue

            destination = (extract_root / member.filename).resolve()
            if not _is_within_directory(destination, resolved_extract_root):
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if suffix in ARCHIVE_SUFFIXES:
                _extract_zip_files_recursive(destination, source_root, depth + 1, max_depth)


def _extract_archives_in_tree(source_root: Path, max_depth: int) -> None:
    """Walk ``source_root`` and recursively extract every archive found.

    The dedicated ``__extracted_archives`` sibling directory is skipped so
    re-runs do not double-process previously extracted payloads. Empty
    sub-directories created during extraction are pruned at the end.
    """
    extracted_root = source_root / "__extracted_archives"
    processed_archives: set[Path] = set()

    for archive_path in sorted(source_root.rglob("*")):
        if not archive_path.is_file() or archive_path.suffix.lower() not in ARCHIVE_SUFFIXES:
            continue

        try:
            relative_path = archive_path.relative_to(source_root)
        except ValueError:
            continue
        if relative_path.parts and relative_path.parts[0] == "__extracted_archives":
            continue

        resolved_archive = archive_path.resolve()
        if resolved_archive in processed_archives:
            continue
        processed_archives.add(resolved_archive)
        _extract_zip_files_recursive(archive_path, source_root, depth=1, max_depth=max_depth)

    if not extracted_root.exists():
        return

    for path in sorted(extracted_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


__all__ = [
    "DATA_FILE_SUFFIXES",
    "ARCHIVE_SUFFIXES",
    "MAPPING_SUFFIXES",
    "DEFAULT_ARCHIVE_MAX_DEPTH",
    "_is_within_directory",
    "_resolve_archive_max_depth",
    "_should_skip_archive_member",
    "_extract_zip_files_recursive",
    "_extract_archives_in_tree",
]
