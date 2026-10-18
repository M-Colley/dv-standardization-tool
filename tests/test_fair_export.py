"""Frictionless data packages, RO-Crate, Parquet output and mapping validation."""

from __future__ import annotations

import json
from pathlib import Path

import frictionless
import pandas as pd
import pytest
import yaml

from scripts import fair_export
from scripts.data_loaders import save_output_file
from scripts.validate_schema import validate, validate_dataset_mapping

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def batch_dir(tmp_path):
    """A minimal batch output: one source, one standardized table."""
    out = tmp_path / "run"
    source = out / "standardized" / "demo_study"
    source.mkdir(parents=True)
    df = pd.DataFrame({
        "participant": ["P1", "P2", "P3"],
        "mental_demand": [5, 7, 9],
        "trust_rating__item_1": [3.0, 4.5, None],
        "trust_rating": [3.5, 4.0, 4.5],
    })
    df.to_csv(source / "data_csv-standardized.csv", index=False)
    (source / "data_csv-quality.json").write_text(json.dumps({
        "dataset_type": "questionnaire",
        "dataset_id": "raw/data.csv",
        "value_labels_applied": {"TLX_M": {"recoded": 3, "kept_numeric": 0, "unmatched": 0}},
    }), encoding="utf-8")
    pd.DataFrame([
        {"source_id": "demo_study", "dataset_id": "raw/data.csv", "canonical_dv": "mental_demand",
         "original_alias": "TLX_M", "mapping_source": r"C:\Users\someone\data\demo\demo_mapping.yaml",
         "commit": "abc123"},
        {"source_id": "demo_study", "dataset_id": "raw/data.csv", "canonical_dv": "trust_rating",
         "original_alias": "Trust", "mapping_source": "llm_deduction", "commit": "abc123"},
    ]).to_csv(out / "meta_view.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps({
        "run_timestamp": "2026-10-18T00:00:00+00:00", "schema_version": "9.9",
        "llm_deduction_enabled": True, "llm_models_used": ["org/model-x"],
        "llm_plausibility_rules_version": "1.1", "llm_shortlist": "definition",
    }), encoding="utf-8")
    (out / "manifest_snapshot.yaml").write_text(yaml.safe_dump({"sources": [{
        "source_id": "demo_study", "source_type": "github_repo",
        "location": "https://github.com/example/demo", "publication_doi": "10.1145/0000000.0000000",
    }]}), encoding="utf-8")
    return out


def test_data_package_is_valid_and_describes_columns(batch_dir):
    result = fair_export.export_run(batch_dir, validate=True)
    assert result["invalid"] == {}
    package_path = batch_dir / "standardized" / "demo_study" / "datapackage.json"
    assert frictionless.validate(str(package_path)).valid

    package = json.loads(package_path.read_text(encoding="utf-8"))
    assert package["opendv:schemaVersion"] == "9.9"  # the run's version, not today's schema
    assert package["opendv:publicationDoi"] == "10.1145/0000000.0000000"
    assert package["opendv:sourceRevision"] == "abc123"
    resource = package["resources"][0]
    assert resource["opendv:originalFile"] == "raw/data.csv"
    fields = {f["name"]: f for f in resource["schema"]["fields"]}

    md = fields["mental_demand"]
    assert md["type"] == "integer"
    assert md["opendv:canonicalDv"] == "mental_demand"
    assert md["opendv:canonicalRange"] == [0, 20]
    assert md["opendv:originalNames"] == ["TLX_M"]
    assert md["opendv:mappingSource"] == "repo_mapping"
    assert md["opendv:mappingFile"] == "demo_mapping.yaml"
    assert md["opendv:valueLabelsApplied"] == {"TLX_M": {"recoded": 3, "kept_numeric": 0, "unmatched": 0}}
    assert fields["trust_rating"]["opendv:mappingSource"] == "llm_deduced"
    assert fields["trust_rating__item_1"]["opendv:itemOf"] == "trust_rating"
    assert "opendv:canonicalDv" not in fields["participant"]
    # No local paths leak into the published metadata.
    assert "someone" not in package_path.read_text(encoding="utf-8")


def test_ro_crate_links_sources_software_and_outputs(batch_dir):
    fair_export.export_run(batch_dir)
    crate = json.loads((batch_dir / "ro-crate-metadata.json").read_text(encoding="utf-8"))
    graph = {entity["@id"]: entity for entity in crate["@graph"]}
    action = graph["#batch-run"]
    assert action["@type"] == "CreateAction"
    assert {"@id": "#source-demo_study"} in action["object"]
    assert {"@id": "standardized/demo_study/datapackage.json"} in action["result"]
    assert graph["#llm"]["name"] == "org/model-x"
    assert graph["#source-demo_study"]["citation"] == {"@id": "https://doi.org/10.1145/0000000.0000000"}
    assert graph["#dv-schema"]["version"] == "9.9"


def test_recorded_column_types_avoid_reading_the_table(batch_dir):
    source = batch_dir / "standardized" / "demo_study"
    quality = json.loads((source / "data_csv-quality.json").read_text(encoding="utf-8"))
    quality.update({
        "standardized_file": "data_csv-standardized.csv",
        "standardized_rows": 3,
        "standardized_column_types": {"participant": "string", "mental_demand": "integer",
                                      "trust_rating__item_1": "number", "trust_rating": "number"},
    })
    (source / "data_csv-quality.json").write_text(json.dumps(quality), encoding="utf-8")
    (source / "data_csv-standardized.csv").write_text("corrupt", encoding="utf-8")  # would break a re-read
    result = fair_export.write_data_packages(batch_dir)
    assert len(result["data_packages"]) == 1


def test_parquet_writer_handles_mixed_object_columns(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "mixed": pd.Series([1, "x"], dtype=object)})
    path = tmp_path / "t.parquet"
    save_output_file(df, str(path))
    back = pd.read_parquet(path)
    assert back["a"].tolist() == [1, 2]
    assert back["mixed"].tolist() == ["1", "x"]


# ── Dataset mapping validation ───────────────────────────────────────────────

def test_dataset_mapping_validation_flags_bad_blocks():
    errors, warnings = validate_dataset_mapping({
        "TLX1": "mental_demand",
        "Speed": "meta_condition_speed",
        "Broken": None,
        "scales": [{"canonical": "x", "aliases": ["a"], "reverse_aliases": ["b"], "scale_range": [7, 1]}],
        "value_labels": [{"columns": ["Q1"], "labels": {"agree": "high"}}],
    })
    assert any("Broken" in e for e in errors)
    assert any("reverse_aliases" in e for e in errors)
    assert any("scale_range" in e for e in errors)
    assert any("map to a number" in e for e in errors)
    assert warnings and "meta_condition_speed" in warnings[0]


@pytest.mark.parametrize("path", sorted((REPO / "schemas").glob("*_mapping.yaml")), ids=lambda p: p.name)
def test_repository_mappings_are_valid(path):
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate(schema, str(path)) == []


def test_citation_file_has_required_cff_keys():
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    assert {"message", "title", "authors"} <= set(cff)
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{cff["version"]}"' in pyproject
