#!/usr/bin/env python3
"""Batch standardization pipeline for multi-source DV harmonization.

Phase-1 orchestration that reads a source manifest, discovers tabular files,
standardizes columns, and emits a consolidated meta-view artifact.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib import request

import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_dv import (
    build_original_column_lookup,
    identify_unmapped_columns,
    load_input_file,
    load_schema,
    save_output_file,
    standardize_columns,
)
from scripts.llm_utils import deduce_standard_name_with_local_llm

SUPPORTED_SOURCE_TYPES = {"local_path", "github_repo", "osf_project"}
TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls", ".tsv"}
ARCHIVE_SUFFIXES = {".zip"}
OSF_API_BASE = "https://api.osf.io/v2"

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
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    if not isinstance(manifest, dict) or "sources" not in manifest:
        raise ValueError("Manifest must be a YAML object with a top-level 'sources' list.")

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

def _match_files(base_dir: Path, include_globs: list[str] | None, exclude_globs: list[str] | None) -> list[Path]:
    include_globs = include_globs or ["**/*"]
    exclude_globs = exclude_globs or []

    candidates: list[Path] = []
    for pattern in include_globs:
        for path in base_dir.glob(pattern):
            if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES:
                candidates.append(path)

    unique_candidates = sorted(set(candidates))
    if not exclude_globs:
        return unique_candidates

    excluded: set[Path] = set()
    for pattern in exclude_globs:
        excluded.update(path for path in base_dir.glob(pattern) if path.is_file())

    return [path for path in unique_candidates if path not in excluded]

def _extract_osf_project_id(location: str) -> str:
    """Resolve OSF project id from a raw id or osf.io URL."""
    location = (location or "").strip()
    if not location:
        raise ValueError("OSF location must be a non-empty project id or osf.io URL.")

    if re.fullmatch(r"[a-z0-9]{5}", location.lower()):
        return location.lower()

    match = re.search(r"osf\.io/([a-z0-9]{5})(?:/|$)", location.lower())
    if not match:
        raise ValueError(
            "Invalid OSF location. Use a 5-character project id (e.g., cwd6h) "
            "or an OSF URL such as https://osf.io/cwd6h/overview."
        )
    return match.group(1)

def _osf_json_get(url: str) -> dict[str, Any]:
    req = request.Request(
        url,
        headers={
            "Accept": "application/vnd.api+json",
            "User-Agent": "OpenDV-HCI/1.0",
        },
    )
    with request.urlopen(req, timeout=20) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)

def _iter_osf_file_entries(project_id: str) -> list[dict[str, Any]]:
    """List file entries under a project's osfstorage provider (recursive)."""
    entries: list[dict[str, Any]] = []
    queue: list[str] = [f"{OSF_API_BASE}/nodes/{project_id}/files/osfstorage/"]

    while queue:
        page_url = queue.pop(0)
        while page_url:
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

            page_url = payload.get("links", {}).get("next")
            if page_url is not None:
                page_url = str(page_url)

    return entries

def _download_osf_file(download_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(download_url, timeout=60) as resp:
        destination.write_bytes(resp.read())


def _extract_zip_tabular_files(zip_path: Path, source_root: Path) -> None:
    """Extract supported tabular files from a ZIP archive into a cache subtree."""
    relative_zip = zip_path.relative_to(source_root)
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
            if member_path.suffix.lower() not in TABULAR_SUFFIXES:
                continue

            destination = (extract_root / member.filename).resolve()
            if not str(destination).startswith(str(resolved_extract_root)):
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)

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

    if source_type == "local_path":
        base_dir = Path(source["location"]).expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"local_path does not exist: {base_dir}")
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

        if refresh_remote_cache and target.exists():
            shutil.rmtree(target)

        if not marker.exists():
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

            file_entries = _iter_osf_file_entries(project_id)
            for entry in file_entries:
                attrs = entry.get("attributes", {}) or {}
                path = str(attrs.get("path", "")).lstrip("/")
                if not path:
                    continue

                suffix = Path(path).suffix.lower()
                if suffix not in TABULAR_SUFFIXES and suffix not in ARCHIVE_SUFFIXES:
                    continue

                download_url = entry.get("links", {}).get("download")
                if not download_url:
                    continue

                local_path = target / Path(path)
                _download_osf_file(str(download_url), local_path)

                if suffix == ".zip":
                    _extract_zip_tabular_files(local_path, target)

            marker.write_text(project_id, encoding="utf-8")

        files = _match_files(target, include_globs, exclude_globs)
        return target, files, project_id

    repo_url = source["location"]
    pinned_ref = source.get("ref", "HEAD")
    marker = target / ".source_ready"

    if refresh_remote_cache and target.exists():
        shutil.rmtree(target)

    if not marker.exists():
        if target.exists():
            shutil.rmtree(target)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
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

    files = _match_files(
        target,
        include_globs,
        exclude_globs,
    )
    return target, files, commit_sha

def _load_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return load_input_file(str(path))

def _build_artifact_prefix(relative_path: Path) -> str:
    """Create a collision-safe file prefix from a path relative to the source root."""
    normalized = relative_path.as_posix()
    safe_prefix = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")
    return safe_prefix or "dataset"

def _load_repository_mapping(source_root: Path) -> tuple[dict[str, str], str | None]:
    """Load a source-local mapping YAML if one is unambiguously available."""
    mapping_patterns = [
        "*mapping*.yaml",
        "*mapping*.yml",
        "*dv*.yaml",
        "*dv*.yml",
    ]
    candidates: list[Path] = []
    for pattern in mapping_patterns:
        candidates.extend(path for path in source_root.glob(pattern) if path.is_file())

    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) != 1:
        return {}, None

    mapping_path = unique_candidates[0]
    schema_data = load_schema(
        str(mapping_path),
        standard_schema_path=str(REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"),
    )
    return schema_data["mapping"], str(mapping_path)

def _merge_with_standard_precedence(
    source_mapping: dict[str, str],
    standard_mapping: dict[str, str],
) -> dict[str, str]:
    """Merge mapping dictionaries while preserving standard aliases on case-insensitive conflicts."""
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_lookup = build_original_column_lookup(original_df.copy(), mapping)
    unknown_columns = identify_unmapped_columns([str(c) for c in original_df.columns], mapping)

    records: list[dict[str, Any]] = []
    for canonical_dv in standardized_df.columns:
        series = standardized_df[canonical_dv]
        aliases = original_lookup.get(canonical_dv, [canonical_dv])
        original_alias = aliases[0] if aliases else canonical_dv
        numeric_series = pd.to_numeric(series, errors="coerce")

        records.append(
            {
                "source_id": source_id,
                "dataset_id": dataset_id,
                "canonical_dv": canonical_dv,
                "original_alias": original_alias,
                "unit": None,
                "scale": None,
                "n": int(numeric_series.notna().sum()),
                "mean": float(numeric_series.mean()) if numeric_series.notna().any() else None,
                "sd": float(numeric_series.std(ddof=1)) if numeric_series.notna().sum() > 1 else None,
                "review_flag": canonical_dv in unknown_columns,
                "mapping_confidence": "high" if canonical_dv not in unknown_columns else "unknown_alias",
                **provenance,
            }
        )

    quality = {
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

def _augment_mapping_with_llm_deductions(
    mapping: dict[str, str],
    columns: list[str],
    source_root: Path,
    preferred_models: list[str] | None = None,
) -> dict[str, str]:
    """Attempt local-LLM alias deduction for unknown columns.

    This is especially useful for repositories that do not provide a custom
    mapping YAML file.
    """
    augmented = dict(mapping)
    canonical_candidates = sorted({value for value in mapping.values() if isinstance(value, str)})
    if not canonical_candidates:
        return augmented

    unknown = identify_unmapped_columns(columns, augmented)
    for alias in unknown:
        inferred = deduce_standard_name_with_local_llm(
            raw_column_name=str(alias),
            canonical_candidates=canonical_candidates,
            source_root=source_root,
            preferred_models=preferred_models,
        )
        if inferred:
            augmented[str(alias)] = inferred
            augmented[str(alias).lower()] = inferred

    return augmented

def run_batch(
    manifest_path: Path,
    output_dir: Path,
    schema_path: Path,
    llm_deduction_enabled: bool = True,
    cache_root: Path | None = None,
    refresh_remote_cache: bool = False,
    preferred_models: list[str] | None = None,
) -> dict[str, Any]:
    schema_data = load_schema(str(schema_path))
    standard_mapping = schema_data["mapping"]

    output_dir.mkdir(parents=True, exist_ok=True)
    standardized_root = output_dir / "standardized"
    standardized_root.mkdir(parents=True, exist_ok=True)
    resolved_cache_root = cache_root or (output_dir / ".cache" / "sources")
    resolved_cache_root.mkdir(parents=True, exist_ok=True)

    sources = load_manifest(manifest_path)
    run_results: list[SourceRunResult] = []
    meta_rows: list[dict[str, Any]] = []

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
            except Exception as exc:  # noqa: BLE001
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

            processed = 0
            failed = 0
            total_unknown = 0
            total_columns = 0
            total_mapped_columns = 0
            source_mapping = dict(standard_mapping)
            source_mapping_path = None
            source_llm_enabled = bool(source.get("use_llm_deduction", llm_deduction_enabled))
            source_preferred_models = source.get("llm_models") if isinstance(source.get("llm_models"), list) else preferred_models

            if source["source_type"] in {"local_path", "github_repo"}:
                source_specific_mapping, detected_mapping_path = _load_repository_mapping(base_dir)
                if source_specific_mapping:
                    source_mapping = _merge_with_standard_precedence(source_specific_mapping, standard_mapping)
                    source_mapping_path = detected_mapping_path

            dataset_progress = tqdm(
                files,
                desc=f"Datasets ({source_id})",
                unit="file",
                leave=False,
            )
            for file_path in dataset_progress:
                relative_file = file_path.relative_to(base_dir)
                dataset_id = relative_file.as_posix()
                artifact_prefix = _build_artifact_prefix(relative_file)
                dataset_progress.set_postfix(dataset=dataset_id)
                destination = source_output_dir / f"{artifact_prefix}-standardized{file_path.suffix}"
                relative_path = str(relative_file)

                try:
                    original_df = _load_any_table(file_path)
                    dataset_mapping = source_mapping
                    should_apply_llm = source_llm_enabled and (
                        source["source_type"] == "osf_project" or source_mapping_path is None
                    )
                    if should_apply_llm:
                        dataset_mapping = _augment_mapping_with_llm_deductions(
                            source_mapping,
                            [str(column) for column in original_df.columns],
                            base_dir,
                            preferred_models=source_preferred_models,
                        )

                    standardized_df = standardize_columns(original_df.copy(), dataset_mapping)
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
                    with open(source_output_dir / f"{artifact_prefix}-error.log", "w", encoding="utf-8") as f:
                        f.write(str(exc))

            status = "completed" if failed == 0 else ("partial" if processed else "failed")
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
                    message=None,
                )
            )

    meta_df = pd.DataFrame(meta_rows)
    meta_csv = output_dir / "meta_view.csv"
    meta_json = output_dir / "meta_view.json"
    meta_df.to_csv(meta_csv, index=False)
    meta_df.to_json(meta_json, orient="records", indent=2)

    summary = {
        "manifest": str(manifest_path),
        "schema": str(schema_path),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_sources": len(sources),
        "successful_sources": sum(1 for r in run_results if r.status in {"completed", "partial"}),
        "meta_view_rows": int(meta_df.shape[0]),
        "llm_deduction_enabled": llm_deduction_enabled,
        "cache_root": str(resolved_cache_root),
        "results": [r.__dict__ for r in run_results],
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def main() -> None:
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
        help="Optional directory for caching downloaded remote sources (OSF/GitHub).",
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
    )

    if args.snapshot_manifest:
        shutil.copy2(manifest_path, output_dir / "manifest_snapshot.yaml")

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

