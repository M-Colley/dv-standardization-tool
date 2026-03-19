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
import csv
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
        # Legacy format: flat dict {standard_name: [aliases]}
        for standard, aliases in schema.items():
            if isinstance(aliases, list):
                for alias in aliases:
                    alias_to_standard[alias] = standard
                    alias_to_standard[alias.lower()] = standard
                alias_to_standard[standard] = standard
                alias_to_standard[standard.lower()] = standard

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
        "standard_mappings_applied": False,
        "alias_conflicts": [],
        "alias_conflict_policy": alias_conflict_policy,
    }


def load_input_file(input_path: str) -> pd.DataFrame:
    """Load tabular input data from CSV, Excel, or pickle files."""
    input_file = Path(input_path)
    suffix = input_file.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(input_file)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel files requires the optional dependency 'openpyxl'. "
                "Install it with: pip install openpyxl"
            ) from exc
    if suffix == ".csv":
        # Use a constrained delimiter sniff so alphabetic headers are never
        # misdetected as delimiters (e.g., splitting "DurationX" on "u").
        with open(input_file, "r", encoding="utf-8", errors="ignore", newline="") as f:
            sample = f.read(8192)

        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        decode_errors: list[UnicodeDecodeError] = []
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(input_file, sep=delimiter, encoding=encoding)
            except UnicodeDecodeError as exc:
                decode_errors.append(exc)
            except pd.errors.ParserError as exc:
                raise ValueError(
                    f"Failed to parse CSV file '{input_file}': {exc}. "
                    "The file may be malformed or use an unsupported format."
                ) from exc
            except pd.errors.EmptyDataError as exc:
                raise ValueError(
                    f"CSV file '{input_file}' is empty or contains no parseable data."
                ) from exc

        # Preserve the original exception type for callers/tests.
        if decode_errors:
            raise decode_errors[-1]
        return pd.read_csv(input_file, sep=delimiter)
    if suffix in {".pkl", ".pickle"}:
        payload = pd.read_pickle(input_file)
        return _coerce_pickled_payload_to_dataframe(payload)

    raise ValueError(
        f"Unsupported input format: '{suffix or 'no extension'}'. "
        "Supported formats are .csv, .xlsx, .xls, .pkl, and .pickle."
    )


def _coerce_pickled_payload_to_dataframe(payload: Any) -> pd.DataFrame:
    """Normalize common pickle payload shapes into a DataFrame."""
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, pd.Series):
        return payload.to_frame()
    if isinstance(payload, dict):
        if not payload:
            return pd.DataFrame()
        try:
            return pd.DataFrame(payload)
        except ValueError:
            return pd.json_normalize(payload)
    if isinstance(payload, (list, tuple)):
        if not payload:
            return pd.DataFrame()
        return pd.DataFrame(payload)
    if hasattr(payload, "shape"):
        try:
            return pd.DataFrame(payload)
        except ValueError:
            pass

    raise ValueError(
        f"Unsupported pickle payload type '{type(payload).__name__}'. "
        "Expected a DataFrame-like object such as a pandas DataFrame, Series, dict, list, or array."
    )


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
            if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
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
                "Multiple input files found. Please provide --input with a specific CSV/Excel file."
            )
        if not data_candidates:
            raise ValueError(
                "No CSV/Excel file found in the provided folder. "
                "Provide --input with a valid file or add a single input file to the folder."
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


def save_output_file(df: pd.DataFrame, output_path: str) -> None:
    """Save standardized data to CSV, Excel, or pickle based on file extension."""
    output_file = Path(output_path)
    suffix = output_file.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        try:
            df.to_excel(output_file, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing Excel files requires the optional dependency 'openpyxl'. "
                "Install it with: pip install openpyxl"
            ) from exc
        return
    if suffix == ".csv":
        df.to_csv(output_file, index=False)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(output_file)
        return

    raise ValueError(
        f"Unsupported output format: '{suffix or 'no extension'}'. "
        "Supported formats are .csv, .xlsx, .xls, .pkl, and .pickle."
    )


def standardize_columns(df: pd.DataFrame, mapping: Dict) -> pd.DataFrame:
    """
    Rename DataFrame columns using the alias mapping.

    Args:
        df: Input DataFrame with raw column names
        mapping: Alias -> standard name dictionary

    Returns:
        DataFrame with standardized column names
    """
    new_columns = []
    for col in df.columns:
        mapped_col = mapping.get(col)
        if mapped_col is None and isinstance(col, str):
            mapped_col = mapping.get(col.lower())
        new_columns.append(mapped_col or col)  # Default to original if not found
    df.columns = new_columns
    return df


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
    print(f"  âœ“ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Load schema and build mapping
    print(f"Loading schema: {resolved_schema}")
    schema_data = load_schema(
        str(resolved_schema),
        str(default_schema),
        alias_conflict_policy=args.alias_conflict_policy,
    )
    mapping = schema_data["mapping"]
    schema = schema_data["schema"]
    print(f"  âœ“ Loaded {len(mapping)} alias mappings")
    if schema_data.get("standard_mappings_applied"):
        print(
            "  âœ“ Combined selected schema with standard mapping "
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

