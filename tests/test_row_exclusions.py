"""Tests for the ``exclude_rows`` mapping block (rows the authors flagged)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from opendv.convert_dv import _build_alias_mapping
from opendv.validate_schema import validate_dataset_mapping
from opendv.value_labels import apply_row_exclusions

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "schemas" / "standard_dv_mapping.yaml"
FLAG = {"column": "Outlier", "values": [True], "reason": "flagged by the authors"}


# ── Application ──────────────────────────────────────────────────────────────

def test_flagged_rows_are_dropped_and_counted():
    df = pd.DataFrame({"t": [0.5, 9.0, 0.6, 0.01], "Outlier": [False, True, False, True]})
    out, report = apply_row_exclusions(df, [FLAG])
    assert out["t"].tolist() == [0.5, 0.6]
    assert out.index.tolist() == [0, 1]                       # index reset
    assert report == {
        "rows_before": 4, "rows_dropped": 2,
        "rules": [{"column": "Outlier", "values": [True], "reason": "flagged by the authors", "rows": 2}],
    }


def test_true_matches_logical_text_but_not_numbers_or_gaps():
    # A logical column with a gap is read as text/objects; R writes TRUE.
    df = pd.read_csv(io.StringIO("t,Outlier\n1,TRUE\n2,False\n3,\n4,true\n5, True \n"))
    out, report = apply_row_exclusions(df, [FLAG])
    assert out["t"].tolist() == [2, 3]                        # a missing flag is kept
    assert report["rows_dropped"] == 3
    numbers = pd.DataFrame({"t": [1, 2], "Outlier": [1, 0]})
    assert apply_row_exclusions(numbers, [FLAG])[1]["rows_dropped"] == 0
    flags = pd.DataFrame({"t": [1, 2], "Outlier": [True, False]})
    assert apply_row_exclusions(flags, [{**FLAG, "values": [1]}])[1]["rows_dropped"] == 0


def test_numbers_and_text_values():
    df = pd.DataFrame({"block": ["practice", "main", " practice", "Main"], "exclude": [1, "0", "1", None]})
    out, _ = apply_row_exclusions(df, [{"column": "block", "values": ["practice"], "reason": "r"}])
    assert out["block"].tolist() == ["main", "Main"]          # trimmed, case-sensitive
    out, _ = apply_row_exclusions(df, [{"column": "exclude", "values": [1], "reason": "r"}])
    assert out["block"].tolist() == ["main", "Main"]          # 1 and "1"


def test_overlapping_blocks_count_each_and_drop_the_union():
    df = pd.DataFrame({"a": [1, 1, 0, 0], "b": [1, 0, 1, 0]})
    out, report = apply_row_exclusions(df, [
        {"column": "a", "values": [1], "reason": "a"},
        {"column": "b", "values": [1], "reason": "b"},
    ])
    assert len(out) == 1
    assert report["rows_dropped"] == 3
    assert [rule["rows"] for rule in report["rules"]] == [2, 2]


def test_absent_column_is_skipped_and_no_match_is_reported():
    df = pd.DataFrame({"Outlier": [False, False]})
    out, report = apply_row_exclusions(df, [{**FLAG, "column": "Excluded"}])
    assert out is df and report == {}
    out, report = apply_row_exclusions(df, [FLAG])
    assert out is df
    assert (report["rows_before"], report["rows_dropped"], report["rules"][0]["rows"]) == (2, 0, 0)


# ── Validation ───────────────────────────────────────────────────────────────

def test_validator_accepts_a_block_with_column_values_and_reason():
    assert validate_dataset_mapping({"exclude_rows": [FLAG]}) == ([], [])


@pytest.mark.parametrize(("block", "message"), [
    ({"values": [True], "reason": "r"}, "'column'"),
    ({"column": ["Outlier"], "values": [True], "reason": "r"}, "'column'"),
    ({"column": "Outlier", "values": [], "reason": "r"}, "'values'"),
    ({"column": "Outlier", "values": True, "reason": "r"}, "'values'"),
    ({"column": "Outlier", "values": [None], "reason": "r"}, "'values'"),
    ({"column": "Outlier", "values": [True]}, "'reason'"),
    ({"column": "Outlier", "values": [True], "reason": "  "}, "'reason'"),
    ("Outlier", "block"),
])
def test_validator_flags_bad_blocks(block, message):
    errors, _ = validate_dataset_mapping({"exclude_rows": [block]})
    assert len(errors) == 1 and errors[0].startswith("exclude_rows[0]") and message in errors[0]


def test_validator_needs_a_list_and_the_key_is_not_an_alias():
    errors, _ = validate_dataset_mapping({"exclude_rows": FLAG})
    assert errors == ["'exclude_rows' must be a list of {column, values, reason} blocks"]
    assert "exclude_rows" not in _build_alias_mapping({"Outlier": "meta_outlier_flag", "exclude_rows": [FLAG]})


# ── Batch, quality report, data package, relations ───────────────────────────

def _run(tmp_path: Path, table: str, mapping: dict, **kwargs):
    from opendv.run_batch_standardization import run_batch

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "trials.csv").write_text(table, encoding="utf-8")
    (tmp_path / "map.yaml").write_text(yaml.safe_dump(mapping), encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
        "source_id": "s", "source_type": "local_path", "location": str(tmp_path / "src"),
        "include_globs": ["*.csv"], "mapping_path": str(tmp_path / "map.yaml"),
    }]}), encoding="utf-8")
    run_batch(tmp_path / "manifest.yaml", tmp_path / "out", SCHEMA, llm_deduction_enabled=False,
              cache_root=tmp_path / "cache", **kwargs)
    return tmp_path / "out"


TRIALS = "pid,time,flag\n1,0.5,False\n1,9.5,True\n2,0.7,False\n2,0.6,False\n"
MAPPING = {
    "pid": "meta_participant_id", "time": "task_completion_time", "flag": "meta_outlier_flag",
    "exclude_rows": [{"column": "flag", "values": [True], "reason": "too slow"}],
}


def test_batch_drops_rows_before_summaries_and_reports_them(tmp_path):
    out = _run(tmp_path, TRIALS, MAPPING)
    table = pd.read_csv(out / "standardized" / "s" / "trials_csv-standardized.csv")
    assert table["task_completion_time"].tolist() == [0.5, 0.7, 0.6]
    row = pd.read_csv(out / "meta_view.csv").set_index("canonical_dv").loc["task_completion_time"]
    assert row["n"] == 3 and row["mean"] == pytest.approx(0.6)
    quality = json.loads((out / "standardized" / "s" / "trials_csv-quality.json").read_text(encoding="utf-8"))
    assert quality["standardized_rows"] == 3
    assert quality["rows_excluded"] == {
        "rows_before": 4, "rows_dropped": 1,
        "rules": [{"column": "flag", "values": [True], "reason": "too slow", "rows": 1}],
    }
    package = json.loads((out / "standardized" / "s" / "datapackage.json").read_text(encoding="utf-8"))
    resource = package["resources"][0]
    assert resource["opendv:rows"] == 3
    assert resource["opendv:rowsExcluded"]["rows_dropped"] == 1


def test_batch_without_the_block_keeps_every_row(tmp_path):
    mapping = {k: v for k, v in MAPPING.items() if k != "exclude_rows"}
    out = _run(tmp_path, TRIALS, mapping, fair_export=False)
    quality = json.loads((out / "standardized" / "s" / "trials_csv-quality.json").read_text(encoding="utf-8"))
    assert quality["standardized_rows"] == 4 and "rows_excluded" not in quality


def test_relations_count_every_column_of_a_filtered_file_as_curated(tmp_path):
    from opendv.analysis import construct_relations as cr

    folder = tmp_path / "standardized" / "s"
    folder.mkdir(parents=True)
    pd.DataFrame({"participant": [1, 2], "mental_demand": [1.0, 2.0]}).to_csv(
        folder / "t-standardized.csv", index=False
    )
    meta = pd.DataFrame([
        {"source_id": "s", "dataset_id": "t.csv", "canonical_dv": "participant", "original_alias": "participant",
         "mapping_method": "blocked", "mapping_source": "never_map_blocklist", "item_of": None},
        {"source_id": "s", "dataset_id": "t.csv", "canonical_dv": "mental_demand", "original_alias": "tlx1",
         "mapping_method": "mapping", "mapping_source": "schemas/standard_dv_mapping.yaml", "item_of": None},
    ])

    def level(rows_excluded):
        quality = {"dataset_id": "t.csv", "standardized_file": "t-standardized.csv", "rows_excluded": rows_excluded}
        (folder / "t-quality.json").write_text(json.dumps(quality), encoding="utf-8")
        return cr.column_provenance(tmp_path / "standardized", meta)[("s", "t-standardized.csv")]["mental_demand"].level

    assert level({"rows_before": 3, "rows_dropped": 0, "rules": []}) == cr.LEVEL_RANK["automatic"]
    assert level({"rows_before": 3, "rows_dropped": 1, "rules": []}) == cr.LEVEL_RANK["curated"]


def test_llm_benchmark_profiles_the_filtered_rows(tmp_path, monkeypatch):
    from opendv.analysis import evaluate_llm_mapping as ev
    from opendv.run_batch_standardization import SchemaFamilies

    out = _run(tmp_path, TRIALS, MAPPING, fair_export=False)
    monkeypatch.setattr(ev, "_source_context", lambda *_a, **_k: "")
    monkeypatch.setattr(ev, "_ask_llm_logged", lambda *_a, **_k: ([], 0.0, None, None))
    args = SimpleNamespace(
        manifest=str(tmp_path / "manifest.yaml"), batch_dir=str(out), max_per_source=10, seed=0,
        cache_dir=str(tmp_path / "cache"), models=[], shortlist="definition",
    )
    families = SchemaFamilies.load(SCHEMA)
    items = ev.run_e1(args, families, {"task_completion_time"}).set_index("alias")
    assert items.loc["time", "observed_max"] == pytest.approx(0.7)   # not the flagged 9.5 s
