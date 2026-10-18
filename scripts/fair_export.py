"""Self-describing batch outputs: Frictionless data packages and an RO-Crate.

After a batch run, every source directory under ``<output>/standardized``
gets a ``datapackage.json`` (Frictionless Data Package v2) that describes each
standardized table and, per column, what it is:

* ``opendv:canonicalDv``, ``title``, ``description``: the canonical DV, its
  label and cluster from the DV schema;
* ``opendv:originalNames``: the raw column names that were mapped to it;
* ``opendv:mappingSource``: how the mapping was made (schema alias,
  repository mapping, LLM deduction, ...), from ``meta_view.csv``;
* ``opendv:canonicalRange``, ``opendv:direction``, ``opendv:unit``;
* ``opendv:itemOf``: the composite a ``<dv>__item_N`` column belongs to.

The run directory gets an ``ro-crate-metadata.json`` (RO-Crate 1.3) linking
the manifest snapshot, the sources (with repository URL, commit and
publication DOI), the code version, the schema and rule versions, the LLM
model, and every output, as a ``CreateAction``.

Both are written with the reference libraries (``frictionless``,
``rocrate``). Run ``python -m scripts.fair_export <batch output dir>`` to add
them to an existing run.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.batch_reporting import categorize_mapping_source

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DV_SCHEMA = REPO_ROOT / "schemas" / "standard_dv_mapping.yaml"
TABULAR_FORMATS = {".csv": "csv", ".tsv": "tsv", ".xlsx": "xlsx", ".parquet": "parquet"}
STANDARDIZED_MARKER = "-standardized"
ITEM_SUFFIX = "__item_"
NAMESPACE = "opendv"


def _ns(key: str) -> str:
    return f"{NAMESPACE}:{key}"


# ── Inputs ───────────────────────────────────────────────────────────────────

def _load_dv_schema(path: Path) -> tuple[str, dict[str, dict]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return str(data.get("version", "")), {str(dv["id"]): dv for dv in data.get("dvs") or [] if isinstance(dv, dict)}


def frictionless_type(series: pd.Series) -> str:
    """Frictionless field type of a pandas column."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "any" if series.dropna().map(type).nunique() > 1 else "string"


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_excel(path)


def _dataset_prefixes(output_dir: Path, dataset_ids: Iterable[str], output_format: str = "native") -> dict[str, str]:
    """Artifact prefix -> dataset id, using the batch runner's own naming
    (fallback for runs whose quality JSON does not record the dataset id)."""
    from scripts.run_batch_standardization import _build_artifact_prefix, _resolve_output_suffix

    prefixes = {}
    for dataset_id in dataset_ids:
        suffix = _resolve_output_suffix(Path(dataset_id).suffix, output_format)
        length = max(len(f"{STANDARDIZED_MARKER}{suffix}"), len("-mapping-debug.json"), len("-quality.json"),
                     len("-error.log"))
        prefixes[_build_artifact_prefix(Path(dataset_id), output_dir=output_dir, artifact_suffix_length=length)] = (
            dataset_id
        )
    return prefixes


def _resource_name(prefix: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9._-]+", "-", prefix.lower()).strip("-") or "table"
    name, i = base, 2
    while name in used:
        name, i = f"{base}-{i}", i + 1
    used.add(name)
    return name


# ── Data packages ────────────────────────────────────────────────────────────

def _field_descriptor(
    column: str,
    field_type: str,
    dvs: Mapping[str, dict],
    mapping_rows: Mapping[str, list[dict]],
    recoding: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    field: dict[str, Any] = {"name": column, "type": field_type}
    base = column.split(ITEM_SUFFIX)[0] if ITEM_SUFFIX in column else None
    canonical = column if column in dvs else None
    if base and base in dvs:
        field[_ns("itemOf")] = base
    dv = dvs.get(canonical or "")
    if dv:
        measurement = dv.get("measurement") or {}
        field["title"] = str(dv.get("label", canonical))
        field[_ns("canonicalDv")] = canonical
        field[_ns("cluster")] = dv.get("cluster")
        if measurement.get("canonical_range") is not None:
            field[_ns("canonicalRange")] = measurement["canonical_range"]
        if measurement.get("direction"):
            field[_ns("direction")] = measurement["direction"]
        if measurement.get("primary_unit"):
            field[_ns("unit")] = measurement["primary_unit"]
    rows = mapping_rows.get(column, [])
    if rows:
        field[_ns("originalNames")] = sorted({str(r["original_alias"]) for r in rows})
        raw_sources = {str(r.get("mapping_source")) for r in rows}
        categories = sorted({categorize_mapping_source(r) for r in raw_sources})
        field[_ns("mappingSource")] = categories[0] if len(categories) == 1 else categories
        # Mapping files are recorded by name only: absolute local paths
        # would leak the machine layout into published metadata.
        files = sorted({Path(r).name for r in raw_sources if r.lower().endswith((".yaml", ".yml"))})
        if files:
            field[_ns("mappingFile")] = files[0] if len(files) == 1 else files
    # Recoding reports from the quality JSON are keyed by original column name.
    originals = [column, *field.get(_ns("originalNames"), [])]
    for key, report in recoding.items():
        hits = {o: report[o] for o in originals if o in report}
        if hits:
            field[_ns(key)] = hits
    return field


def build_data_package(
    source_dir: Path,
    source: Mapping[str, Any],
    meta_view: pd.DataFrame,
    dvs: Mapping[str, dict],
    schema_version: str,
    output_format: str = "native",
) -> dict[str, Any] | None:
    """Data Package descriptor for one source directory (``None`` if it has no tables)."""
    source_id = source_dir.name
    files = sorted(p for p in source_dir.iterdir() if STANDARDIZED_MARKER in p.stem and p.is_file())
    if not files:
        return None
    rows = meta_view[meta_view["source_id"].astype(str) == source_id]
    prefixes = _dataset_prefixes(source_dir, rows["dataset_id"].astype(str).unique(), output_format)
    by_dataset: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in rows.to_dict("records"):
        by_dataset[str(record["dataset_id"])][str(record["canonical_dv"])].append(record)

    resources, skipped, used = [], [], set()
    for path in files:
        fmt = TABULAR_FORMATS.get(path.suffix.lower())
        if fmt is None:
            skipped.append(path.name)
            continue
        prefix = path.stem[: path.stem.rindex(STANDARDIZED_MARKER)]
        quality_path = source_dir / f"{prefix}-quality.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.is_file() else {}
        dataset_id = quality.get("dataset_id") or prefixes.get(prefix)
        column_types = quality.get("standardized_column_types")
        if column_types and quality.get("standardized_file") == path.name:
            # Recorded by the batch runner when it wrote the table.
            n_rows = int(quality.get("standardized_rows", 0))
        else:
            df = _read_table(path)
            column_types = {str(c): frictionless_type(df[c]) for c in df.columns}
            n_rows = len(df)
        recoding = {
            "valueLabelsApplied": quality.get("value_labels_applied") or {},
            "negativeAnswerCodesRecoded": quality.get("negative_answer_codes_recoded") or {},
        }
        fields = [
            _field_descriptor(name, column_types[name], dvs, by_dataset.get(dataset_id or "", {}), recoding)
            for name in column_types
        ]
        resource: dict[str, Any] = {
            "name": _resource_name(prefix, used),
            "path": path.name,
            "format": fmt,
            "schema": {"fields": fields},
            _ns("rows"): n_rows,
        }
        if dataset_id:
            resource["sources"] = [{"title": dataset_id, "path": f"{source.get('location', '')}#{dataset_id}"}]
            resource[_ns("originalFile")] = dataset_id
        if quality.get("dataset_type"):
            resource[_ns("datasetType")] = quality["dataset_type"]
        if fmt == "tsv":
            resource["format"] = "csv"
            resource["dialect"] = {"csv": {"delimiter": "\t"}}
        resources.append(resource)
    if not resources:
        return None
    descriptor: dict[str, Any] = {
        "name": re.sub(r"[^a-z0-9._-]+", "-", source_id.lower()),
        "title": source_id,
        "resources": resources,
        _ns("schemaVersion"): schema_version,
        _ns("generatedAt"): datetime.now(UTC).isoformat(),
    }
    if source.get("location"):
        descriptor["homepage"] = source["location"]
        descriptor["sources"] = [{"title": source_id, "path": source["location"]}]
    if source.get("publication_doi"):
        descriptor[_ns("publicationDoi")] = source["publication_doi"]
    revisions = sorted({str(c) for c in rows.get("commit", pd.Series(dtype=str)).dropna().unique() if str(c)})
    if revisions:
        # Git commit for repository sources, the download URL/version otherwise.
        descriptor[_ns("sourceRevision")] = revisions[0] if len(revisions) == 1 else revisions
    if skipped:
        descriptor[_ns("nonTabularFiles")] = skipped
    return descriptor


def write_data_packages(
    output_dir: Path,
    schema_path: Path = DEFAULT_DV_SCHEMA,
    validate: bool = False,
) -> dict[str, Any]:
    """Write ``datapackage.json`` into every source directory of a batch run."""
    import frictionless

    standardized = output_dir / "standardized"
    meta_view = pd.read_csv(output_dir / "meta_view.csv", low_memory=False)
    current_version, dvs = _load_dv_schema(schema_path)
    # The version the run used, when it recorded one; else the schema now on disk.
    summary_path = output_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    schema_version = str(summary.get("schema_version") or current_version)
    sources = _manifest_sources(output_dir)
    written, invalid = [], {}
    for source_dir in sorted(p for p in standardized.iterdir() if p.is_dir()):
        descriptor = build_data_package(
            source_dir, sources.get(source_dir.name, {}), meta_view, dvs, schema_version,
            output_format=str(summary.get("output_format") or "native"),
        )
        if descriptor is None:
            continue
        package = frictionless.Package.from_descriptor(descriptor, basepath=str(source_dir))
        target = source_dir / "datapackage.json"
        package.to_json(str(target))
        written.append(target)
        if validate:
            # Blank rows are kept from the source files; they are not errors here.
            report = frictionless.validate(str(target), skip_errors=["blank-row"])
            if not report.valid:
                invalid[source_dir.name] = [e.message for task in report.tasks for e in task.errors][:5]
    return {"data_packages": [str(p.relative_to(output_dir)) for p in written], "invalid": invalid}


# ── RO-Crate ─────────────────────────────────────────────────────────────────

def _manifest_sources(output_dir: Path) -> dict[str, dict]:
    snapshot = output_dir / "manifest_snapshot.yaml"
    if not snapshot.is_file():
        return {}
    data = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
    return {str(s.get("source_id")): s for s in data.get("sources") or [] if isinstance(s, dict)}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("opendv-hci")
    except Exception:  # noqa: BLE001 - not installed as a package
        data = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', data, re.MULTILINE)
        return match.group(1) if match else "unknown"


RUN_FILES = {
    "run_summary.json": ("Run summary", "application/json"),
    "manifest_snapshot.yaml": ("Manifest snapshot", "application/yaml"),
    "meta_view.csv": ("Column-level mapping view", "text/csv"),
    "meta_view.json": ("Column-level mapping view", "application/json"),
    "llm_deductions.json": ("LLM-deduced aliases", "application/json"),
    "llm_rejected_suggestions.json": ("LLM suggestions rejected by the plausibility check", "application/json"),
    "unknown_alias_summary.json": ("Unmapped column names", "application/json"),
    "composites.json": ("Composed multi-item scales", "application/json"),
    "mapping_debug_summary.json": ("Mapping debug summary", "application/json"),
}


def write_ro_crate(output_dir: Path, llm_model: str | None = None) -> Path:
    """Write ``ro-crate-metadata.json`` describing a batch run in place."""
    from rocrate.model.contextentity import ContextEntity
    from rocrate.rocrate import ROCrate

    summary_path = output_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    sources = _manifest_sources(output_dir)

    crate = ROCrate()
    crate.name = f"OpenDV-HCI batch run {summary.get('run_timestamp', '')}".strip()
    crate.description = (
        "Dependent variables of open HCI datasets standardized onto the OpenDV-HCI canonical schema."
    )
    crate.license = "https://spdx.org/licenses/MIT"

    commit = _git_commit()
    software = crate.add(ContextEntity(crate, "#opendv-hci", properties={
        "@type": "SoftwareApplication",
        "name": "OpenDV-HCI",
        "version": _package_version(),
        "url": "https://github.com/M-Colley/dv-standardization-tool",
        **({"identifier": f"git:{commit}"} if commit else {}),
    }))
    instruments = [software]
    schema_file = Path(str(summary.get("schema", DEFAULT_DV_SCHEMA)))
    if schema_file.is_file():
        version = summary.get("schema_version") or _load_dv_schema(schema_file)[0]
        instruments.append(crate.add(ContextEntity(crate, "#dv-schema", properties={
            "@type": "CreativeWork", "name": "OpenDV-HCI DV schema", "version": version,
        })))
    if summary.get("llm_deduction_enabled"):
        model = llm_model or ", ".join(summary.get("llm_models_used") or []) or "unknown"
        instruments.append(crate.add(ContextEntity(crate, "#llm", properties={
            "@type": "SoftwareApplication",
            "name": model,
            "description": (
                f"Local LLM for unmapped columns; plausibility rules v{summary.get('llm_plausibility_rules_version')}, "
                f"{summary.get('llm_shortlist')} shortlist"
            ),
        })))

    inputs = []
    for source_id, source in sources.items():
        properties = {"@type": "Dataset", "name": source_id}
        if source.get("location", "").startswith("http"):
            properties["url"] = source["location"]
        if source.get("publication_doi"):
            properties["citation"] = {"@id": f"https://doi.org/{source['publication_doi']}"}
        inputs.append(crate.add(ContextEntity(crate, f"#source-{source_id}", properties=properties)))

    results = []
    for name, (title, media_type) in RUN_FILES.items():
        path = output_dir / name
        if path.is_file():
            properties = {"name": title, "encodingFormat": media_type}
            results.append(crate.add_file(path, dest_path=name, properties=properties))
    standardized = output_dir / "standardized"
    for package in sorted(standardized.glob("*/datapackage.json")) if standardized.is_dir() else []:
        rel = package.relative_to(output_dir).as_posix()
        results.append(crate.add_file(package, dest_path=rel, properties={
            "name": f"Data package of {package.parent.name}",
            "encodingFormat": "application/json",
            "conformsTo": {"@id": "https://datapackage.org/profiles/2.0/datapackage.json"},
        }))

    action = {
        "@type": "CreateAction",
        "name": "OpenDV-HCI batch standardization",
        "instrument": [{"@id": e.id} for e in instruments],
        "object": [{"@id": e.id} for e in inputs],
        "result": [{"@id": e.id} for e in results],
    }
    if summary.get("run_timestamp"):
        action["endTime"] = summary["run_timestamp"]
    crate.add(ContextEntity(crate, "#batch-run", properties=action))
    crate.metadata.write(str(output_dir))
    return output_dir / "ro-crate-metadata.json"


def export_run(
    output_dir: Path,
    schema_path: Path = DEFAULT_DV_SCHEMA,
    llm_model: str | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    """Data packages for every source, then the run's RO-Crate."""
    result = write_data_packages(output_dir, schema_path, validate=validate)
    result["ro_crate"] = str(write_ro_crate(output_dir, llm_model=llm_model).relative_to(output_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Add data packages and an RO-Crate to a batch run.")
    parser.add_argument("output_dir", type=Path, help="Batch output directory (contains run_summary.json)")
    parser.add_argument("--schema", type=Path, default=DEFAULT_DV_SCHEMA)
    parser.add_argument("--llm-model", default=None, help="LLM model id, for runs that did not record it")
    parser.add_argument("--validate", action="store_true", help="Validate every data package (reads all tables)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = export_run(args.output_dir, args.schema, llm_model=args.llm_model, validate=args.validate)
    print(json.dumps({k: (v if k != "data_packages" else len(v)) for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
