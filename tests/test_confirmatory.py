"""Confirmatory shortlist test: download checks, gold-label mappings and the pre-registered analysis."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
import yaml

from opendv.analysis import confirmatory as cf


def test_checksums_cover_size_sha256_md5_and_git_blob():
    content = b"PId,TCT\n1,512\n"
    blob = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()
    good = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "md5": hashlib.md5(content).hexdigest(), "git_blob": blob}
    assert cf.checksum_ok(content, good)
    for key in good:
        assert not cf.checksum_ok(content + b"x", {key: good[key]}), key


def test_mappings_from_gold_labels_leave_out_conflicting_names(tmp_path):
    labels = pd.DataFrame([
        ("s1", "a.csv", "TLX_1", "dv", "mental_demand"),
        ("s1", "b.csv", "TLX_1", "dv", "mental_demand"),     # same meaning in two tables: fine
        ("s1", "a.csv", "Q1", "dv", "SUS1"),
        ("s1", "b.csv", "Q1", "dv", "acceptance_rating"),    # same name, different meaning: left out
        ("s1", "a.csv", "PID", "non_dv", "meta_participant_id"),
        ("s1", "a.csv", "agency1", "unlabeled_dv", ""),
    ], columns=["source_id", "dataset_id", "column", "kind", "label"])
    labels.to_csv(tmp_path / "gold.csv", index=False)
    conflicts = cf.write_mappings(tmp_path / "gold.csv", tmp_path / "mappings")
    mapping = yaml.safe_load((tmp_path / "mappings" / "s1_mapping.yaml").read_text(encoding="utf-8"))
    assert mapping == {"TLX_1": "mental_demand", "PID": "meta_participant_id"}
    assert conflicts == {"s1": ["Q1"]}


def _run(tmp_path, name, answers, precision):
    rows = [{"experiment": "E1", "source_id": "s", "dataset_id": "d.csv", "alias": f"c{i}", "label_kind": "dv",
             "gold": "g", "gold_in_shortlist": True, "pre_llm_rejection": None,
             "final_unchecked": a, "final_checked": a} for i, a in enumerate(answers)]
    out = tmp_path / name
    out.mkdir()
    pd.DataFrame(rows).to_csv(out / "llm_mapping_evaluation_items.csv", index=False)
    summary = {"results": {"E1": {"checked": {"precision_of_accepted_dv_mappings": {"rate": precision},
                                              "abstained_on_dv_columns": {}}}}}
    (out / "llm_mapping_evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return out


def test_analysis_follows_the_preregistered_rule(tmp_path):
    base = ["g"] * 10 + ["x"] * 20
    better = ["g"] * 10 + ["g"] * 12 + ["x"] * 8          # 12 more correct, none lost
    runs = {"definition": _run(tmp_path, "definition", base, 0.5),
            "dense_h": _run(tmp_path, "dense_h", better, 0.7),
            "mixed_h": _run(tmp_path, "mixed_h", base, 0.5)}
    result = cf.analyze(runs)
    h1 = result["tests"]["dense_h_vs_definition"]
    assert (h1["new_only"], h1["baseline_only"]) == (12, 0)
    assert h1["p_two_sided"] == pytest.approx(2 * 0.5 ** 12)
    assert h1["p_holm"] == pytest.approx(2 * h1["p_two_sided"])
    assert result["shortlists"]["dense_h"]["correct"] == 22
    assert result["decision"]["switch_default_to_hierarchical"]

    runs["dense_h"] = _run(tmp_path, "dense_h_low_precision", better, 0.4)
    assert not cf.analyze(runs)["decision"]["switch_default_to_hierarchical"]
