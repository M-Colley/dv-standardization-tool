"""Tests for the LLM mapping benchmark's summaries."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from opendv.analysis.evaluate_llm_mapping import load_subsets, summarize_subsets


def _item(source_id, label_kind, gold, answer):
    return {"experiment": "E1", "source_id": source_id, "label_kind": label_kind, "gold": gold,
            "gold_in_shortlist": gold is not None, "final_unchecked": answer, "final_checked": answer,
            "pre_llm_rejection": None, "post_llm_rejection": None}


def test_subsets_are_summarized_apart_from_the_other_sources(tmp_path):
    listed = tmp_path / "sources.yaml"
    listed.write_text(yaml.safe_dump({"sources": [{"source_id": "new_a"}, {"source_id": "new_b"}]}), encoding="utf-8")
    subsets = load_subsets([f"added={listed}"])
    assert subsets == {"added": {"new_a", "new_b"}}

    items = pd.DataFrame([
        _item("new_a", "dv", "g", "g"),
        _item("new_b", "dv", "g", "x"),
        _item("new_b", "curated_non_dv", None, "g"),
        _item("old", "dv", "g", "g"),
        {**_item("old", "dv", "g", "x"), "experiment": "E2"},  # E2 is never part of a subset
    ])
    result = summarize_subsets(items, {"g", "x"}, subsets)
    added, rest = result["added"], result["not_added"]
    assert (added["n_positive"], added["n_negative"]) == (2, 1)
    assert added["checked"]["accuracy_on_dv_columns"]["k"] == 1
    assert added["checked"]["curated_non_dv_mapped_to_a_dv"]["k"] == 1
    assert (rest["n_positive"], rest["checked"]["accuracy_on_dv_columns"]["k"]) == (1, 1)


def test_subset_needs_a_name_and_a_file():
    with pytest.raises(ValueError, match="NAME=SOURCES_YAML"):
        load_subsets(["sources.yaml"])
