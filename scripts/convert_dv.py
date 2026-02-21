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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

# Import the new inference module
try:
    from dv_inference import batch_infer, get_measurement_from_schema
except ImportError:
    print("Warning: dv_inference module not found. Metadata inference will be disabled.")
    batch_infer = None


def load_schema(schema_path: str) -> Dict:
    """
    Load the DV mapping schema from YAML.

    Args:
        schema_path: Path to standard_dv_mapping.yaml

    Returns:
        Dictionary with alias_to_standard mapping and full schema
    """
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    alias_to_standard = {}

    # Handle new schema format (version 2.1+) with 'dvs' list
    if 'dvs' in schema:
        for dv in schema['dvs']:
            standard_name = dv.get('id')
            aliases = dv.get('aliases', [])
            for alias in aliases:
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

    return {"mapping": alias_to_standard, "schema": schema}


def load_input_file(input_path: str) -> pd.DataFrame:
    """Load tabular input data from CSV or Excel files."""
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
        return pd.read_csv(input_file)

    raise ValueError(
        f"Unsupported input format: '{suffix or 'no extension'}'. "
        "Supported formats are .csv, .xlsx, and .xls."
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
    """Save standardized data to CSV or Excel based on file extension."""
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

    raise ValueError(
        f"Unsupported output format: '{suffix or 'no extension'}'. "
        "Supported formats are .csv, .xlsx, and .xls."
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
        # If inference module not available, return basic metadata
        for col in standardized_df.columns:
            column_meta[col] = {
                "original_name": original_names.get(col, [col]),
                "inference_available": False
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
    schema_version: str = "2.1"
) -> None:
    """
    Export CSV with sidecar metadata JSON.

    Creates:
        - output_path: Standardized CSV
        - output_path_metadata.json: Column metadata
    """
    # Export standardized dataset with extension-aware serializer.
    save_output_file(df, output_path)
    print(f"✓ Standardized file saved to: {output_path}")

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
            "categories": {}
        }
    }

    # Count categories
    for col, meta in column_meta.items():
        cat = meta.get("category", "Unknown")
        metadata["summary"]["categories"][cat] = metadata["summary"]["categories"].get(cat, 0) + 1

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata JSON saved to: {meta_path}")

    # Warn if columns need review
    if metadata["summary"]["needs_review"] > 0:
        print(f"⚠ Warning: {metadata['summary']['needs_review']} column(s) flagged for manual review (low confidence)")


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
    print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Load schema and build mapping
    print(f"Loading schema: {resolved_schema}")
    schema_data = load_schema(str(resolved_schema))
    mapping = schema_data["mapping"]
    schema = schema_data["schema"]
    print(f"  ✓ Loaded {len(mapping)} alias mappings")

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
        export_with_metadata(df_standardized, column_meta, str(resolved_output), schema.get('version', '2.1'))
    else:
        print("Standardizing columns (metadata inference disabled)...")
        df_standardized = standardize_columns(df, mapping)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        save_output_file(df_standardized, str(resolved_output))
        print(f"✓ Standardized file saved to: {resolved_output}")

    print("=" * 60)
    print("✓ Standardization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
