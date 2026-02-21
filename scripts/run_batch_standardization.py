#!/usr/bin/env python3
"""Batch standardization pipeline for multi-source DV harmonization.

Phase-1 orchestration that reads a source manifest, discovers tabular files,
standardizes columns, and emits a consolidated meta-view artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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

SUPPORTED_SOURCE_TYPES = {"local_path", "github_repo"}
TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls", ".tsv"}


@dataclass
class SourceRunResult:
    source_id: str
    status: str
    discovered_files: int
    processed_files: int
    failed_files: int
    unknown_columns: int
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


def discover_source_files(source: dict[str, Any], working_dir: Path) -> tuple[Path, list[Path], str | None]:
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

    repo_url = source["location"]
    pinned_ref = source.get("ref", "HEAD")
    target = working_dir / source["source_id"]
    subprocess.run([
        "git", "clone", "--depth", "1", repo_url, str(target)
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        "git", "-C", str(target), "checkout", pinned_ref
    ], check=True, capture_output=True, text=True)
    commit_sha = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    files = _match_files(
        target,
        source.get("include_globs"),
        source.get("exclude_globs"),
    )
    return target, files, commit_sha


def _load_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return load_input_file(str(path))


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


def run_batch(manifest_path: Path, output_dir: Path, schema_path: Path) -> dict[str, Any]:
    schema_data = load_schema(str(schema_path))
    standard_mapping = schema_data["mapping"]

    output_dir.mkdir(parents=True, exist_ok=True)
    standardized_root = output_dir / "standardized"
    standardized_root.mkdir(parents=True, exist_ok=True)

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
                base_dir, files, commit_sha = discover_source_files(source, checkout_root)
            except Exception as exc:  # noqa: BLE001
                run_results.append(
                    SourceRunResult(
                        source_id=source_id,
                        status="failed",
                        discovered_files=0,
                        processed_files=0,
                        failed_files=0,
                        unknown_columns=0,
                        output_dir=str(source_output_dir),
                        message=str(exc),
                    )
                )
                continue

            processed = 0
            failed = 0
            total_unknown = 0
            source_mapping = dict(standard_mapping)
            source_mapping_path = None
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
                dataset_id = file_path.stem
                dataset_progress.set_postfix(dataset=dataset_id)
                destination = source_output_dir / f"{dataset_id}-standardized{file_path.suffix}"
                relative_path = str(file_path.relative_to(base_dir))

                try:
                    original_df = _load_any_table(file_path)
                    standardized_df = standardize_columns(original_df.copy(), source_mapping)
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
                        source_mapping,
                        provenance,
                    )
                    total_unknown += quality["unknown_columns"]
                    meta_rows.extend(rows)

                    with open(source_output_dir / f"{dataset_id}-quality.json", "w", encoding="utf-8") as f:
                        json.dump(quality, f, indent=2)

                    processed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    with open(source_output_dir / f"{dataset_id}-error.log", "w", encoding="utf-8") as f:
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
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    schema_path = Path(args.schema).resolve()

    summary = run_batch(manifest_path, output_dir, schema_path)

    if args.snapshot_manifest:
        shutil.copy2(manifest_path, output_dir / "manifest_snapshot.yaml")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
