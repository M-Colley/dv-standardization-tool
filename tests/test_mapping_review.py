"""Review queue, decisions and write-back of column mappings (scripts/mapping_review.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import mapping_review as mr

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"


def _meta_row(source, alias, method, canonical, n=10, mean=3.0, sd=1.0, dataset="data.csv", dtype="questionnaire"):
    return {"source_id": source, "dataset_id": dataset, "dataset_type": dtype, "canonical_dv": canonical,
            "original_alias": alias, "mapping_method": method, "n": n, "mean": mean, "sd": sd,
            "min": mean - 2, "max": mean + 2, "source_mapping": None}


@pytest.fixture()
def batch_dir(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    pd.DataFrame([
        _meta_row("s1", "MentalDemand", "llm", "mental_demand", n=20, mean=8.0),
        _meta_row("s1", "MentalDemand", "llm", "mental_demand", n=20, mean=10.0, dataset="b.csv"),
        _meta_row("s1", "Bin", "llm", "perceived_safety", mean=12.8),
        _meta_row("s1", "strangeScore", "unmapped", "strangeScore"),
        _meta_row("s1", "telemetry_x", "unmapped", "telemetry_x", dtype="sensor_stream"),
        _meta_row("s2", "TLX1", "mapping", "mental_demand"),
    ]).to_csv(out / "meta_view.csv", index=False)
    (out / "llm_rejected_suggestions.json").write_text(json.dumps([
        {"source_id": "s1", "alias": "age", "proposed_canonical": None, "stage": "pre_llm", "reason": "demographic",
         "min": 19, "max": 60, "datasets": ["data.csv"]},
        {"source_id": "s2", "alias": "Pulse", "proposed_canonical": "valence", "stage": "post_llm",
         "reason": "out_of_range", "min": 50, "max": 120, "datasets": ["x.csv"]},
    ]), encoding="utf-8")
    return out


def test_queue_orders_by_urgency_and_flags_impossible_values(batch_dir):
    queue = mr.build_review_queue(batch_dir, SCHEMA)
    assert queue["status"].tolist() == ["llm_accepted", "llm_accepted", "llm_rejected", "unmapped", "screened_out"]
    assert "telemetry_x" not in set(queue["raw_column"])  # sensor streams are not reviewed
    rows = queue.set_index("raw_column")
    assert rows.loc["MentalDemand", "n"] == 40
    assert rows.loc["MentalDemand", "mean"] == pytest.approx(9.0)
    assert rows.loc["MentalDemand", "datasets"] == "b.csv; data.csv"
    assert rows.loc["Bin", "outside_range"]  # mean 12.8 cannot lie on -3..3
    assert not rows.loc["MentalDemand", "outside_range"]
    assert rows.loc["Pulse", "suggestion"] == "valence" and rows.loc["Pulse", "min"] == 50
    assert rows.loc["MentalDemand", "candidates"].startswith("mental_demand (Mental Demand")


def _manifest(tmp_path, mapping_file: Path | None):
    s1 = {"source_id": "s1", "source_type": "local_path", "location": str(tmp_path)}
    if mapping_file is not None:
        s1["mapping_path"] = str(mapping_file)
    path = tmp_path / "manifest.yaml"
    text = "# my manifest comment\n" + yaml.safe_dump({"sources": [
        s1, {"source_id": "s2", "source_type": "local_path", "location": str(tmp_path)},
    ]}, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    return path


def test_apply_writes_mapping_with_comments_and_log(batch_dir, tmp_path):
    mapping = tmp_path / "s1_mapping.yaml"
    mapping.write_text("# curated by hand\nExisting: trust_rating  # keep me\n", encoding="utf-8")
    manifest = _manifest(tmp_path, mapping)
    queue = mr.build_review_queue(batch_dir, SCHEMA).set_index("raw_column")
    queue.loc["MentalDemand", "decision"] = "accept"
    queue.loc["Bin", "decision"] = "non_dv"
    queue.loc["strangeScore", "decision"] = "map"
    queue.loc["strangeScore", "target"] = "usability"
    queue.loc["Pulse", "decision"] = "study_specific"
    queue.loc["Pulse", "target"] = "heart rate"
    log = tmp_path / "log.csv"
    result = mr.apply_decisions(mr.decisions_from_queue(queue.reset_index()), manifest, "Ada",
                                schemas_dir=tmp_path, log_path=log, schema_path=SCHEMA)

    text = mapping.read_text(encoding="utf-8")
    assert "# curated by hand" in text and "# keep me" in text
    written = yaml.safe_load(text)
    assert written == {"Existing": "trust_rating", "MentalDemand": "mental_demand", "Bin": "meta_bin",
                       "strangeScore": "usability"}
    assert "reviewed" in text and "by Ada: non_dv (LLM had perceived_safety)" in text
    # s2 had no mapping file: one is created and registered in the manifest, comments kept.
    assert result["manifest_updated"]
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "# my manifest comment" in manifest_text
    s2 = yaml.safe_load(manifest_text)["sources"][1]
    assert Path(s2["mapping_path"]).name == "s2_mapping.yaml"
    assert yaml.safe_load((tmp_path / "s2_mapping.yaml").read_text(encoding="utf-8")) == {"Pulse": "heart_rate"}
    assert manifest.with_suffix(".yaml.bak").is_file()

    entries = pd.read_csv(log)
    assert len(entries) == 4 and set(entries["reviewer"]) == {"Ada"}
    summary = mr.review_summary(log)
    assert summary["llm_mappings_reviewed"] == 2
    assert summary["llm_precision_by_review"] == pytest.approx(0.5)


def test_invalid_decision_writes_nothing(batch_dir, tmp_path):
    mapping = tmp_path / "s1_mapping.yaml"
    mapping.write_text("{}\n", encoding="utf-8")
    manifest = _manifest(tmp_path, mapping)
    bad = [
        mr.Decision("s1", "MentalDemand", "accept", previous_suggestion="mental_demand"),
        mr.Decision("s1", "strangeScore", "map", target="not_a_dv"),
    ]
    with pytest.raises(ValueError, match="not a schema DV"):
        mr.apply_decisions(bad, manifest, "Ada", schemas_dir=tmp_path, log_path=tmp_path / "log.csv",
                           schema_path=SCHEMA)
    assert mapping.read_text(encoding="utf-8") == "{}\n"
    assert not (tmp_path / "log.csv").exists()
    with pytest.raises(ValueError, match="reviewer"):
        mr.apply_decisions([], manifest, " ", schemas_dir=tmp_path, schema_path=SCHEMA)


def test_reviewed_mappings_drive_the_next_run_and_the_benchmark(tmp_path):
    """A reviewed DV mapping becomes a curated mapping; a non_dv becomes a curated non-DV label."""
    from analyses.evaluate_llm_mapping import gold_items
    from scripts.run_batch_standardization import run_batch

    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"ParticipantNo": [1, 2, 3], "WorkloadQ": [4, 6, 5], "OddColumn": [9, 9, 8]}).to_csv(
        data / "survey.csv", index=False)
    mapping = tmp_path / "study_mapping.yaml"
    mapping.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"sources": [{
        "source_id": "study", "source_type": "local_path", "location": str(data), "mapping_path": str(mapping),
    }]}), encoding="utf-8")
    mr.apply_decisions([
        mr.Decision("study", "WorkloadQ", "map", target="mental_demand"),
        mr.Decision("study", "OddColumn", "non_dv"),
    ], manifest, "Ada", schemas_dir=tmp_path, log_path=tmp_path / "log.csv", schema_path=SCHEMA)

    run_batch(manifest, tmp_path / "out", SCHEMA, llm_deduction_enabled=False, fair_export=False)
    meta = pd.read_csv(tmp_path / "out" / "meta_view.csv")
    rows = meta.set_index("original_alias")
    assert rows.loc["WorkloadQ", "canonical_dv"] == "mental_demand"
    assert rows.loc["WorkloadQ", "mapping_method"] == "mapping"
    assert rows.loc["OddColumn", "canonical_dv"] == "meta_oddcolumn"
    gold = gold_items(meta, {"mental_demand"}, max_per_source=40, seed=1).set_index("original_alias")
    assert gold.loc["WorkloadQ", "label_kind"] == "dv"
    assert gold.loc["OddColumn", "label_kind"] == "curated_non_dv"



def test_composite_items_keep_their_own_mapping_and_values(tmp_path):
    """An LLM-mapped item composed with a schema-mapped one stays visible.

    The composite counts as LLM-derived for the provenance index (so analyses
    that exclude LLM mappings drop it), and the review queue lists the LLM
    item with its own values rather than the composite's.
    """
    from analyses.evaluate_llm_mapping import gold_items
    from analyses.multi_study_analysis import load_mapping_provenance
    from opendv.convert_dv import standardize_columns
    from opendv.item_composition import compose_item_groups
    from scripts.run_batch_standardization import _summarize_dataset

    original = pd.DataFrame({"TLX1": [1, 2, 3, 4, 5, 6, 7, 8], "Recognized": [2, 2, 3, 5, 5, 6, 8, 8],
                             "Seen": [1, 3, 3, 4, 6, 6, 7, 9]})
    mapping = dict.fromkeys(original.columns, "mental_demand")
    standardized, records = compose_item_groups(standardize_columns(original.copy(), mapping), "questionnaire")
    assert records[0]["decision"] == "composed"
    debug = [{"original_column": "TLX1", "mapping_method": "mapping",
              "mapping_source": "schemas/standard_dv_mapping.yaml"}] + [
        {"original_column": c, "mapping_method": "llm", "mapping_source": "llm_deduction"}
        for c in ("Recognized", "Seen")]
    rows, _ = _summarize_dataset("s", "d.csv", original, standardized, mapping, {"source_mapping": None}, debug,
                                 "questionnaire", composite_records=records)
    meta = pd.DataFrame(rows)
    composite = meta[meta["item_of"].isna()].iloc[0]
    items = meta[meta["item_of"].notna()].set_index("original_alias")
    assert composite["composite_items"] == "TLX1; Recognized; Seen"
    assert composite["mapping_method"] == "mapping"  # the composite row still describes its first item
    assert list(items.index) == ["TLX1", "Recognized", "Seen"]
    assert items.loc["Seen", "mean"] == pytest.approx(original["Seen"].mean())
    assert items.loc["Recognized", "mapping_method"] == "llm"

    batch = tmp_path / "run"
    batch.mkdir()
    meta.to_csv(batch / "meta_view.csv", index=False)
    assert load_mapping_provenance(batch / "meta_view.csv")[("s", "mental_demand")] == "llm_deduced"
    queue = mr.build_review_queue(batch, SCHEMA).set_index("raw_column")
    assert set(queue.index) == {"Recognized", "Seen"}
    assert queue.loc["Seen", "mean"] == pytest.approx(original["Seen"].mean())
    gold = gold_items(meta, {"mental_demand"}, max_per_source=40, seed=1)
    assert list(gold["original_alias"]) == ["TLX1"]  # one gold item per composite, as before items had rows


UI_SMOKE_SCRIPT = """
import sys
from streamlit.testing.v1 import AppTest

at = AppTest.from_string(
    "from opendv.ui.components.review_panel import render_review_tab\\nrender_review_tab()\\n", default_timeout=120
)
at.run()  # widgets exist only after the first run
at.text_input[0].set_value("does/not/exist")
at.run()
assert not at.exception, at.exception
assert any("meta_view.csv" in info.value for info in at.info)
print("UI_OK")
"""


def test_review_tab_renders_without_a_batch():
    """Run in a separate process: Streamlit's AppTest leaves process-wide state
    behind that makes later tests using a process pool hang."""
    pytest.importorskip("streamlit")  # UI extra; not installed in the core CI job
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", UI_SMOKE_SCRIPT], cwd=repo, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0 and "UI_OK" in result.stdout, result.stderr[-2000:]
