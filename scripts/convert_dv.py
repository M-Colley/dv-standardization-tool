"""
convert_dv.py

This script standardizes dependent variable (DV) names in a dataset by applying
a canonical mapping defined in a YAML schema. It also infers measurement metadata
(category, units, scale type) for each column and exports this as a JSON sidecar file.

Part of the OpenDV-HCI project for promoting reproducibility and interoperability
in HCI research.

Usage:
    python convert_dv.py --input path/to/input.csv --output path/to/output.csv
    python convert_dv.py --input data.csv --output standardized.csv --with-metadata
    python convert_dv.py --input path/to/folder
"""

import argparse
import json
import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional rapidfuzz import for fuzzy column matching
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz import process as _fuzz_process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning(
        "rapidfuzz not installed — fuzzy column matching will be disabled. "
        "Install it with: pip install rapidfuzz"
    )

# Import the new inference module
try:
    from scripts.dv_inference import batch_infer, get_measurement_from_schema
except ImportError:
    try:
        from dv_inference import batch_infer, get_measurement_from_schema
    except ImportError:
        logger.warning("dv_inference module not found. Metadata inference will be disabled.")
        batch_infer = None
        get_measurement_from_schema = lambda *_args, **_kwargs: None

# data_loaders owns the per-format loaders, the format registry, encoding /
# delimiter detection, and the inverse save_output_file. Re-exported here so
# existing imports (e.g. ``from scripts.convert_dv import load_input_file``,
# ``from scripts.convert_dv import _FORMAT_REGISTRY``) keep working after the
# Wave C extraction.
from scripts.data_loaders import (
    _FORMAT_REGISTRY,
    _MAGIC_BYTES,
    _coerce_pickled_payload_to_dataframe,
    _detect_delimiter,
    _detect_encoding,
    _load_excel,
    _load_feather,
    _load_json,
    _load_jsonlines,
    _load_parquet,
    _load_pickle,
    _load_spss,
    _load_stata,
    _load_text_table,
    _register_format,
    _sniff_format,
    load_input_file,
    save_output_file,
)


def _normalize_colname(raw: str) -> str:
    """Normalise a column name to lowercase alphanumeric + underscores for fuzzy comparison."""
    return re.sub(r"[^a-z0-9_]+", "", str(raw).strip().lower())


def _fuzzy_match_column(
    raw_col: str,
    alias_lookup: dict[str, str],
    threshold: float = 85.0,
) -> tuple[str | None, float]:
    """Return (canonical_dv_id, score) if fuzzy match exceeds threshold, else (None, 0).

    Uses rapidfuzz token_sort_ratio so that column name tokens can be matched
    regardless of ordering (e.g. "DurationTask" vs "TaskDuration").
    """
    if not _RAPIDFUZZ_AVAILABLE or not alias_lookup:
        return None, 0.0
    normalized = _normalize_colname(raw_col)
    match = _fuzz_process.extractOne(
        normalized,
        alias_lookup.keys(),
        scorer=_fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )
    if match:
        matched_alias, score, _ = match
        return alias_lookup[matched_alias], float(score)
    return None, 0.0


def _build_alias_mapping(schema: Dict) -> Dict[str, str]:
    """Build a case-insensitive alias -> standard_name mapping for a schema dictionary."""
    alias_to_standard: Dict[str, str] = {}

    if schema is None:
        return alias_to_standard
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a YAML object (mapping) at the top level.")

    # Handle new schema format (version 2.1+) with 'dvs' list
    if 'dvs' in schema:
        dvs = schema.get('dvs')
        if dvs is None:
            return alias_to_standard
        if not isinstance(dvs, list):
            raise ValueError("Schema field 'dvs' must be a list when provided.")

        for dv in dvs:
            if not isinstance(dv, dict):
                continue
            standard_name = dv.get('id')
            aliases = dv.get('aliases', [])
            if aliases is None:
                aliases = []
            if not isinstance(aliases, list):
                raise ValueError(
                    f"Schema field 'aliases' must be a list for DV '{standard_name or '<missing id>'}'."
                )
            if not standard_name:
                continue
            for alias in aliases:
                if not isinstance(alias, str):
                    continue
                alias_to_standard[alias] = standard_name
                # Case-insensitive matching improves robustness for common
                # real-world formatting inconsistencies.
                alias_to_standard[alias.lower()] = standard_name
            # Also map the id itself
            alias_to_standard[standard_name] = standard_name
            alias_to_standard[standard_name.lower()] = standard_name
    else:
        # Legacy formats — two flavours, both supported:
        #   1. {canonical_id: [alias, alias, ...]}        — list of aliases
        #   2. {alias: canonical_id}                       — single alias per line
        #
        # Source-specific override files (e.g. schemas/fact_av_mapping.yaml,
        # schemas/running_into_traffic_mapping.yaml) typically use form (2)
        # because it reads naturally as "rename this raw column to that
        # canonical". The two flavours are distinguished by the value type.
        #
        # The reserved top-level key ``scales:`` declares multi-item scales
        # and is handled separately by item composition — skip it here.
        for key, value in schema.items():
            if key == "scales":
                continue
            if isinstance(value, list):
                # Form (1): {canonical: [aliases]}
                for alias in value:
                    if not isinstance(alias, str):
                        continue
                    alias_to_standard[alias] = key
                    alias_to_standard[alias.lower()] = key
                alias_to_standard[key] = key
                alias_to_standard[key.lower()] = key
            elif isinstance(value, str):
                # Form (2): {alias: canonical}
                alias_to_standard[key] = value
                alias_to_standard[key.lower()] = value

    return alias_to_standard


def _collect_alias_conflicts(
    custom_mapping: Dict[str, str],
    standard_mapping: Dict[str, str],
) -> List[Dict[str, str]]:
    """Return alias collisions where custom and standard map to different DV ids."""
    conflicts: Dict[str, Dict[str, str]] = {}

    for alias, custom_target in custom_mapping.items():
        standard_target = standard_mapping.get(alias)
        if standard_target is None or standard_target == custom_target:
            continue

        normalized = alias.lower() if isinstance(alias, str) else str(alias)
        existing = conflicts.get(normalized)

        if existing is None:
            conflicts[normalized] = {
                "alias": str(alias),
                "custom_id": custom_target,
                "standard_id": standard_target,
            }
        else:
            # Prefer preserving the original-case alias when available.
            if existing["alias"] == normalized and alias != normalized:
                existing["alias"] = str(alias)

    return sorted(conflicts.values(), key=lambda row: row["alias"].lower())


def load_schema(
    schema_path: str,
    standard_schema_path: str | None = None,
    alias_conflict_policy: Literal["prefer_standard", "prefer_custom", "error"] = "prefer_standard",
) -> Dict:
    """
    Load the DV mapping schema from YAML.

    Args:
        schema_path: Path to standard_dv_mapping.yaml

    Returns:
        Dictionary with alias_to_standard mapping and full schema
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    if schema is None:
        raise ValueError(
            f"Schema file '{schema_path}' is empty or only contains null values. "
            "Provide a YAML object with either a 'dvs' list or legacy alias mapping entries."
        )
    if not isinstance(schema, dict):
        raise ValueError(
            f"Schema file '{schema_path}' must contain a YAML object (mapping) at the top level."
        )

    # If a custom schema is selected, combine it with the standard schema.
    # Standard mappings take precedence on conflicts; custom schema extends
    # coverage for unmapped/novel aliases.
    if standard_schema_path:
        standard_path = Path(standard_schema_path).resolve()
        selected_path = Path(schema_path).resolve()
        if selected_path != standard_path:
            with open(standard_path, "r", encoding="utf-8") as f:
                standard_schema = yaml.safe_load(f)
            if standard_schema is None or not isinstance(standard_schema, dict):
                raise ValueError(
                    f"Standard schema file '{standard_path}' must contain a non-empty YAML object."
                )

            custom_mapping = _build_alias_mapping(schema)
            standard_mapping = _build_alias_mapping(standard_schema)
            conflicts = _collect_alias_conflicts(custom_mapping, standard_mapping)

            if conflicts and alias_conflict_policy == "error":
                preview = ", ".join(
                    f"{row['alias']} (custom={row['custom_id']} vs standard={row['standard_id']})"
                    for row in conflicts[:5]
                )
                raise ValueError(
                    "Alias conflicts detected between custom and standard mappings. "
                    f"Set --alias-conflict-policy to prefer_standard or prefer_custom to proceed. Conflicts: {preview}"
                )

            if alias_conflict_policy == "prefer_custom":
                merged_mapping = {**standard_mapping, **custom_mapping}
            else:
                merged_mapping = {**custom_mapping, **standard_mapping}

            return {
                "mapping": merged_mapping,
                "schema": schema,
                "scales": schema.get("scales", []) if isinstance(schema, dict) else [],
                "standard_mappings_applied": True,
                "standard_mapping_count": len(standard_mapping),
                "custom_mapping_count": len(custom_mapping),
                "alias_conflicts": conflicts,
                "alias_conflict_policy": alias_conflict_policy,
            }

    schema_version = schema.get("version")
    if schema_version is not None:
        try:
            major = int(str(schema_version).split(".")[0])
            if major > 3:
                logger.warning(
                    "Schema version %s may be newer than this tool supports. "
                    "Consider updating the tool if mappings behave unexpectedly.",
                    schema_version,
                )
        except (ValueError, IndexError):
            pass

    return {
        "mapping": _build_alias_mapping(schema),
        "schema": schema,
        "scales": schema.get("scales", []) if isinstance(schema, dict) else [],
        "standard_mappings_applied": False,
        "alias_conflicts": [],
        "alias_conflict_policy": alias_conflict_policy,
    }


def detect_single_file(folder: Path, patterns: Tuple[str, ...], kind: str) -> Path | None:
    """
    Detect exactly one file in a folder by suffix pattern.

    Returns:
        The file path if exactly one candidate is found, else None.
    """
    candidates = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in patterns
    )

    if len(candidates) > 1:
        warnings.warn(
            (
                f"Multiple {kind} files detected in '{folder}': "
                f"{', '.join(path.name for path in candidates)}"
            ),
            stacklevel=2,
        )
        return None

    if len(candidates) == 1:
        return candidates[0]

    return None


def resolve_io_paths(input_arg: str, output_arg: str | None, schema_arg: str | None) -> Tuple[Path, Path, Path]:
    """Resolve input/output/schema paths from file or folder-oriented CLI arguments."""
    input_path = Path(input_arg)
    default_schema = Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"

    if input_path.is_file():
        resolved_input = input_path
        resolved_schema = Path(schema_arg) if schema_arg else default_schema
    elif input_path.is_dir():
        data_candidates = [
            path for path in sorted(input_path.iterdir())
            if path.is_file() and path.suffix.lower() in _FORMAT_REGISTRY
        ]
        if len(data_candidates) > 1:
            warnings.warn(
                (
                    f"Multiple input data files detected in '{input_path}': "
                    f"{', '.join(path.name for path in data_candidates)}"
                ),
                stacklevel=2,
            )
            raise ValueError(
                "Multiple input files found. Please provide --input with a specific file path."
            )
        if not data_candidates:
            supported = ", ".join(sorted(_FORMAT_REGISTRY.keys()))
            raise ValueError(
                f"No supported data file found in the provided folder "
                f"(supported formats: {supported}). "
                "Provide --input with a specific file or add a single input file to the folder."
            )
        resolved_input = data_candidates[0]

        if schema_arg:
            resolved_schema = Path(schema_arg)
        else:
            schema_candidates = [
                path for path in sorted(input_path.iterdir())
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            ]
            if len(schema_candidates) > 1:
                warnings.warn(
                    (
                        f"Multiple schema files detected in '{input_path}': "
                        f"{', '.join(path.name for path in schema_candidates)}"
                    ),
                    stacklevel=2,
                )
                raise ValueError(
                    "Multiple schema files found. Please pass --schema with the YAML file to use."
                )
            resolved_schema = schema_candidates[0] if schema_candidates else default_schema
    else:
        raise ValueError(f"Input path does not exist: {input_arg}")

    if output_arg:
        resolved_output = Path(output_arg)
    else:
        resolved_output = resolved_input.with_name(f"{resolved_input.stem}-standardized{resolved_input.suffix}")

    return resolved_input, resolved_output, resolved_schema


def _augment_mapping_with_fuzzy(
    raw_columns: List[str],
    alias_lookup: Dict[str, str],
    threshold: float = 85.0,
    fuzzy_metadata: Dict[str, Dict] | None = None,
) -> Dict[str, str]:
    """Return a copy of alias_lookup extended with fuzzy matches for unresolved columns.

    For each raw column that has no exact match in alias_lookup, try
    _fuzzy_match_column. Successful fuzzy matches are added to the returned
    mapping and, when fuzzy_metadata is not None, their match info is stored
    there for downstream JSON sidecar output.

    Args:
        raw_columns: Column names as they appear in the input file.
        alias_lookup: The canonical alias→standard-id mapping.
        threshold: rapidfuzz score threshold (0–100).
        fuzzy_metadata: Optional dict updated in-place with fuzzy match info
                        keyed by raw column name.

    Returns:
        A new mapping dict that includes the fuzzy-derived entries.
    """
    if not _RAPIDFUZZ_AVAILABLE:
        return dict(alias_lookup)

    augmented = dict(alias_lookup)
    # Build a normalised-key version of the alias lookup for fuzzy search so
    # that _fuzzy_match_column can compare against normalised forms.
    norm_lookup: Dict[str, str] = {
        _normalize_colname(k): v
        for k, v in alias_lookup.items()
    }

    for col in raw_columns:
        # Skip columns that already resolve exactly.
        if alias_lookup.get(col) is not None:
            continue
        if isinstance(col, str) and alias_lookup.get(col.lower()) is not None:
            continue

        canonical, score = _fuzzy_match_column(col, norm_lookup, threshold=threshold)
        if canonical is not None:
            augmented[col] = canonical
            logger.debug(
                "Fuzzy match: '%s' → '%s' (score=%.0f%%)",
                col,
                canonical,
                score,
            )
            if fuzzy_metadata is not None:
                fuzzy_metadata[col] = {
                    "match_type": "fuzzy",
                    "fuzzy_score": score,
                    "needs_review": True,
                    "canonical_dv": canonical,
                }
    return augmented


DUPLICATE_COLUMN_SUFFIX = "__dup_"


def standardize_columns(df: pd.DataFrame, mapping: Dict) -> pd.DataFrame:
    """Rename DataFrame columns using the alias mapping.

    When several raw columns map to the same canonical (common when LLM
    deduction collapses item-level columns like ``Trust1``/``Trust2`` into
    ``trust_rating``), the second and subsequent occurrences are suffixed
    with ``__dup_N`` so the DataFrame retains unique column names. Downstream
    summarization strips this suffix when emitting the canonical DV name so
    pooling still treats them as the same construct.
    """
    new_columns: List[str] = []
    seen_counts: Dict[str, int] = {}
    for col in df.columns:
        mapped_col = mapping.get(col)
        if mapped_col is None and isinstance(col, str):
            mapped_col = mapping.get(col.lower())
        target = mapped_col or col
        count = seen_counts.get(target, 0)
        if count == 0:
            new_columns.append(target)
        else:
            new_columns.append(f"{target}{DUPLICATE_COLUMN_SUFFIX}{count + 1}")
        seen_counts[target] = count + 1
    df.columns = new_columns
    return df


def strip_duplicate_suffix(column_name: str) -> str:
    """Remove the ``__dup_N`` suffix used by :func:`standardize_columns`."""
    if not isinstance(column_name, str):
        return column_name
    idx = column_name.rfind(DUPLICATE_COLUMN_SUFFIX)
    if idx == -1:
        return column_name
    tail = column_name[idx + len(DUPLICATE_COLUMN_SUFFIX):]
    if tail.isdigit():
        return column_name[:idx]
    return column_name


def build_original_column_lookup(df: pd.DataFrame, mapping: Dict) -> Dict[str, list]:
    """Build standardized column -> original input columns lookup."""
    original_names: Dict[str, list] = {}
    for col in df.columns:
        mapped_col = mapping.get(col)
        if mapped_col is None and isinstance(col, str):
            mapped_col = mapping.get(col.lower())
        standardized = mapped_col or col
        original_names.setdefault(standardized, []).append(col)
    return original_names


def identify_unmapped_columns(column_names: List[str], mapping: Dict[str, str]) -> List[str]:
    """Return input columns that do not resolve to a known standard DV mapping."""
    unknown_columns: List[str] = []
    for col in column_names:
        col_str = str(col)
        mapped_col = mapping.get(col_str)
        if mapped_col is None:
            mapped_col = mapping.get(col_str.lower())
        if mapped_col is None:
            unknown_columns.append(col_str)
    return unknown_columns



def _suggest_dv_id(alias: str) -> str:
    """Create a conservative snake_case DV id suggestion from an unknown alias."""
    normalized = re.sub(r"[^a-z0-9]+", "_", alias.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "proposed_dv"


def build_schema_suggestion_template(unknown_columns: List[str]) -> Dict:
    """Build a schema-ready suggestion payload for unknown aliases."""
    suggestions = []
    for alias in unknown_columns:
        suggestions.append({
            "id": _suggest_dv_id(str(alias)),
            "aliases": [str(alias)],
            "notes": "TODO: assign to an existing standard DV id or add a new DV definition.",
        })
    return {"dvs": suggestions}


def write_schema_suggestion_file(output_path: str, unknown_columns: List[str]) -> Path:
    """Write unknown-alias suggestions to a YAML file next to the output artifact."""
    suggestion_template = build_schema_suggestion_template(unknown_columns)
    suggestion_path = Path(output_path).with_suffix('')
    suggestion_path = Path(str(suggestion_path) + "_schema_suggestions.yaml")

    with open(suggestion_path, "w") as f:
        yaml.safe_dump(suggestion_template, f, sort_keys=False, allow_unicode=True)

    return suggestion_path


def standardize_with_metadata(
    df: pd.DataFrame,
    mapping: Dict,
    schema: Dict,
    confidence_threshold: float = 0.7
) -> Tuple[pd.DataFrame, Dict]:
    """
    Standardize columns and infer measurement types.

    Args:
        df: Input DataFrame with raw column names
        mapping: Alias -> standard name mapping
        schema: Full schema dictionary
        confidence_threshold: Minimum confidence for auto-accept

    Returns:
        (standardized_df, column_metadata)
    """
    # Step 1: Rename columns and track source names for metadata lineage.
    original_names = build_original_column_lookup(df, mapping)
    standardized_df = standardize_columns(df.copy(), mapping)

    # Step 2: Infer measurement types for all columns
    column_meta = {}

    if batch_infer is None:
        # If inference module not available, return basic metadata with
        # all expected fields so downstream JSON serialization succeeds.
        logger.warning("Metadata inference unavailable; emitting placeholder metadata for all columns.")
        for col in standardized_df.columns:
            column_meta[col] = {
                "category": "Unknown",
                "primary_unit": "unknown",
                "allowed_units": [],
                "scale_type": "ratio",
                "direction": "neutral",
                "confidence": 0.0,
                "inferred": False,
                "matched_rules": [],
                "needs_review": True,
                "original_name": original_names.get(col, [col]),
                "inference_available": False,
            }
        return standardized_df, column_meta

    # Try to get metadata from schema first (ground truth)
    for col in standardized_df.columns:
        # Check if this column has schema-defined metadata
        schema_meta = get_measurement_from_schema(col)

        if schema_meta:
            # Use schema-defined metadata (highest confidence)
            column_meta[col] = {
                **schema_meta.to_dict(),
                "original_name": original_names.get(col, [col])
            }
        else:
            # Infer metadata for columns not in schema
            inferences = batch_infer([col], confidence_threshold)
            _, meta, _ = inferences[0]
            column_meta[col] = {
                **meta.to_dict(),
                "original_name": original_names.get(col, [col])
            }

    return standardized_df, column_meta


def export_with_metadata(
    df: pd.DataFrame,
    column_meta: Dict,
    output_path: str,
    schema_version: str = "2.1",
    unknown_columns: List[str] | None = None,
) -> Dict[str, str | None]:
    """
    Export CSV with sidecar metadata JSON.

    Creates:
        - output_path: Standardized CSV
        - output_path_metadata.json: Column metadata
    """
    # Export standardized dataset with extension-aware serializer.
    save_output_file(df, output_path)
    print(f"[OK] Standardized file saved to: {output_path}")

    # Export metadata JSON
    meta_path = str(Path(output_path).with_suffix('')) + "_metadata.json"
    metadata = {
        "schema_version": schema_version,
        "inference_timestamp": datetime.now().isoformat(),
        "columns": column_meta,
        "summary": {
            "total_columns": len(column_meta),
            "needs_review": sum(
                1 for m in column_meta.values()
                if m.get("needs_review", False)
            ),
            "categories": {},
            "unknown_columns": unknown_columns or [],
        }
    }

    suggestion_path: Path | None = None
    if unknown_columns:
        metadata["summary"]["recommendation"] = (
            "Consider proposing these unknown aliases to schemas/standard_dv_mapping.yaml "
            "in a pull request so future runs can standardize them automatically."
        )
        metadata["summary"]["schema_suggestion_template"] = build_schema_suggestion_template(unknown_columns)
        suggestion_path = write_schema_suggestion_file(output_path, unknown_columns)
        metadata["summary"]["schema_suggestion_file"] = str(suggestion_path)

    # Count categories
    for col, meta in column_meta.items():
        cat = meta.get("category", "Unknown")
        metadata["summary"]["categories"][cat] = metadata["summary"]["categories"].get(cat, 0) + 1

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Metadata JSON saved to: {meta_path}")

    # Warn if columns need review
    if metadata["summary"]["needs_review"] > 0:
        print(f"[WARN] {metadata['summary']['needs_review']} column(s) flagged for manual review (low confidence)")

    if unknown_columns:
        print(
            "[WARN] Unknown columns detected: "
            f"{', '.join(str(col) for col in unknown_columns)}"
        )
        print(
            "  -> Consider adding these aliases to schemas/standard_dv_mapping.yaml "
            "via a pull request."
        )
        if suggestion_path is not None:
            print(f"  -> Schema suggestion template written to: {suggestion_path}")

    return {
        "metadata_path": meta_path,
        "schema_suggestion_file": str(suggestion_path) if suggestion_path else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standardize DV column names and infer measurement metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (column renaming only):
  python convert_dv.py --input data.csv --output standardized.csv

  # Folder mode (auto-detect one input file and optional schema):
  python convert_dv.py --input ./dataset_folder

  # With measurement metadata inference:
  python convert_dv.py --input data.csv --output standardized.csv --with-metadata

  # Custom schema file:
  python convert_dv.py --input data.csv --output out.csv --schema custom_schema.yaml
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV/Excel file OR a folder containing exactly one CSV/Excel file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save standardized CSV/Excel output (default: '<input>-standardized.<ext>')"
    )
    default_schema = Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help=(
            "Path to YAML schema file. In folder mode, if omitted, the tool auto-uses "
            "exactly one .yaml/.yml in the folder, else falls back to "
            f"{default_schema}."
        )
    )
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help="Infer and export measurement metadata (category, units, scale type)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for auto-accepting inferences (default: 0.7)"
    )
    parser.add_argument(
        "--alias-conflict-policy",
        type=str,
        default="prefer_standard",
        choices=["prefer_standard", "prefer_custom", "error"],
        help=(
            "How to resolve alias collisions when combining --schema with the standard mapping. "
            "prefer_standard keeps canonical IDs from schemas/standard_dv_mapping.yaml (default), "
            "prefer_custom keeps custom IDs, and error aborts on the first conflict set."
        ),
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=85.0,
        metavar="SCORE",
        help=(
            "Minimum rapidfuzz token_sort_ratio score (0–100) required for a fuzzy column "
            "match to be accepted (default: 85.0). Ignored when --no-fuzzy is set."
        ),
    )
    parser.add_argument(
        "--no-fuzzy",
        action="store_true",
        default=False,
        help=(
            "Disable fuzzy column matching entirely. "
            "Only exact alias lookups will be performed (strict/reproducible mode)."
        ),
    )

    args = parser.parse_args()

    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("--confidence-threshold must be between 0 and 1.")

    resolved_input, resolved_output, resolved_schema = resolve_io_paths(
        args.input,
        args.output,
        args.schema,
    )

    print("=" * 60)
    print("OpenDV-HCI: DV Standardization Tool")
    print("=" * 60)

    # Load input file
    print(f"Loading input file: {resolved_input}")
    df = load_input_file(str(resolved_input))
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Auto-detect and reshape long-format data to wide format
    try:
        from scripts.reshape_utils import detect_data_shape, auto_reshape_to_wide
    except ImportError:
        try:
            from reshape_utils import detect_data_shape, auto_reshape_to_wide
        except ImportError:
            detect_data_shape = None
            auto_reshape_to_wide = None

    if detect_data_shape is not None:
        shape = detect_data_shape(df)
        if shape == "long":
            print("  [INFO] Detected long-format data -- reshaping to wide format...")
            df = auto_reshape_to_wide(df)
            print(f"  Reshaped to {len(df)} rows, {len(df.columns)} columns")
        elif shape == "ambiguous":
            print("  [INFO] Data shape is ambiguous (could be long or wide). Proceeding as-is.")

    # Load schema and build mapping
    print(f"Loading schema: {resolved_schema}")
    schema_data = load_schema(
        str(resolved_schema),
        str(default_schema),
        alias_conflict_policy=args.alias_conflict_policy,
    )
    mapping = schema_data["mapping"]
    schema = schema_data["schema"]
    print(f"  [OK] Loaded {len(mapping)} alias mappings")
    if schema_data.get("standard_mappings_applied"):
        print(
            "  [OK] Combined selected schema with standard mapping "
            f"(policy={schema_data['alias_conflict_policy']}; extensible custom aliases)."
        )
        alias_conflicts = schema_data.get("alias_conflicts", [])
        if alias_conflicts:
            print(
                "  âš  Detected alias conflicts between selected and standard schema: "
                f"{len(alias_conflicts)}"
            )
            for conflict in alias_conflicts[:10]:
                print(
                    "    - "
                    f"{conflict['alias']}: custom={conflict['custom_id']}, "
                    f"standard={conflict['standard_id']}"
                )
            if len(alias_conflicts) > 10:
                print(f"    ... and {len(alias_conflicts) - 10} more conflicts")

    # Optionally augment mapping with fuzzy matches for unresolved columns.
    fuzzy_match_metadata: Dict[str, Dict] = {}
    if not args.no_fuzzy:
        if not _RAPIDFUZZ_AVAILABLE:
            print(
                "  [WARN] --fuzzy-threshold set but rapidfuzz is not installed; "
                "fuzzy matching skipped."
            )
        else:
            raw_col_names = [str(col) for col in df.columns]
            mapping = _augment_mapping_with_fuzzy(
                raw_col_names,
                mapping,
                threshold=args.fuzzy_threshold,
                fuzzy_metadata=fuzzy_match_metadata,
            )
            if fuzzy_match_metadata:
                print(
                    f"  ~ Fuzzy matched {len(fuzzy_match_metadata)} column(s) "
                    f"(threshold={args.fuzzy_threshold:.0f}%): "
                    + ", ".join(
                        f"'{c}' -> '{m['canonical_dv']}' ({m['fuzzy_score']:.0f}%)"
                        for c, m in fuzzy_match_metadata.items()
                    )
                )

    unknown_columns = identify_unmapped_columns([str(col) for col in df.columns], mapping)

    # Apply standardization
    if args.with_metadata:
        print("Standardizing columns and inferring measurement metadata...")
        df_standardized, column_meta = standardize_with_metadata(
            df,
            mapping,
            schema,
            args.confidence_threshold
        )
        # Merge fuzzy-match flags into column_meta for sidecar JSON output.
        # The canonical column name after renaming is the target key.
        for raw_col, fmeta in fuzzy_match_metadata.items():
            canonical = mapping.get(raw_col, raw_col)
            if canonical in column_meta:
                column_meta[canonical].update({
                    "match_type": fmeta["match_type"],
                    "fuzzy_score": fmeta["fuzzy_score"],
                    "needs_review": True,
                })
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        _ = export_with_metadata(
            df_standardized,
            column_meta,
            str(resolved_output),
            schema.get('version', '2.1'),
            unknown_columns=unknown_columns,
        )
    else:
        print("Standardizing columns (metadata inference disabled)...")
        df_standardized = standardize_columns(df, mapping)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        save_output_file(df_standardized, str(resolved_output))
        print(f"[OK] Standardized file saved to: {resolved_output}")

        if unknown_columns:
            print(
                "[WARN] Unknown columns detected: "
                f"{', '.join(str(col) for col in unknown_columns)}"
            )
            print(
                "  -> Consider adding these aliases to schemas/standard_dv_mapping.yaml "
                "via a pull request."
            )
            suggestion_path = write_schema_suggestion_file(str(resolved_output), unknown_columns)
            print(f"  -> Schema suggestion template written to: {suggestion_path}")

    print("=" * 60)
    print("[OK] Standardization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

