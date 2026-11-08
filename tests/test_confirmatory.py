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
             "gold": "g", "gold_in_shortlist": True, "pre_llm_rejection": None, "post_llm_rejection": None,
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


def test_posthoc_compares_with_the_preregistered_run_and_adds_header_matches(tmp_path):
    names = ("definition", "dense_h", "mixed_h")
    before = {n: _run(tmp_path, f"before_{n}", ["g"] * 5 + ["x"] * 5, 0.5) for n in names}
    after = {n: _run(tmp_path, f"after_{n}", ["g"] * 7 + ["x"] * 3, 0.7) for n in names}
    # Header matching answers c8 correctly and c0 wrongly; it replaces the LLM's answer.
    header = {("s", "d.csv", "c8"): "g", ("s", "d.csv", "c0"): "y"}
    result = cf.posthoc(after, before, header_answers=header, dv_ids={"g", "x", "y"})
    stats = result["shortlists"]["definition"]
    assert stats["vs_preregistered_run"]["correct_before"] == 5
    assert stats["vs_preregistered_run"]["new_only"] == 2
    combined = stats["with_header_matching"]
    assert (combined["correct"], combined["header_matched_dv_columns"], combined["header_matched_correct"]) == (7, 2, 1)
    assert (combined["vs_llm_only_new_only"], combined["vs_llm_only_baseline_only"]) == (1, 1)
    # Every answer is a DV id, so all 10 count as accepted mappings, 7 of them correct.
    precision = combined["precision_of_accepted_dv_mappings"]
    assert (precision["k"], precision["n"]) == (7, 10)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_fetch_writes_only_the_pinned_members_of_a_whole_archive(tmp_path, monkeypatch):
    import zlib

    import httpx

    table, other = b"participant,mental_demand\n1,40\n", b"not needed"
    archive = _zip_bytes({"data/q.csv": table, "videos/v.csv": other})
    sources = {"sources": [
        {"source_id": "kept", "files": [{
            "path": "Supplementary material.zip", "size": len(archive),
            "md5": hashlib.md5(archive).hexdigest(), "url": "https://host/a.zip", "fetch": "whole",
            "members": [{"path": "data/q.csv", "size": len(table), "crc32": f"{zlib.crc32(table):08x}"}]}]},
        {"source_id": "gone", "dropped": "no table", "files": []},
    ]}
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(sources), encoding="utf-8")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=archive))
    original = httpx.Client
    monkeypatch.setattr(cf.httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    problems = cf.fetch(tmp_path / "raw", path)
    assert problems == {"kept": []}
    written = sorted(p.relative_to(tmp_path / "raw").as_posix() for p in (tmp_path / "raw").rglob("*") if p.is_file())
    assert written == ["kept/Supplementary material/data/q.csv"]


def test_archive_members_cannot_leave_the_source_folder(tmp_path):
    assert cf._member_target(tmp_path, "a.zip", "../../evil.csv") is None
    assert cf._member_target(tmp_path, "a.zip", "x/y.csv") == (tmp_path / "a" / "x" / "y.csv").resolve()
    assert cf.member_ok(b"abc", {"size": 3, "crc32": "352441c2"})
    assert not cf.member_ok(b"abd", {"size": 3, "crc32": "352441c2"})
