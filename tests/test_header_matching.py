"""Question-text headers mapped by cleaned alias or item wording, and the instrument keying check."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from opendv import header_matching as hm
from opendv.instruments import keying_check, load_instruments

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"
SUS = {i.id: i.wording[0] for i in {inst.id: inst for inst in load_instruments()}["sus"].items.values()}
# Paraphrased SUS items, as a study might word them for its own interface.
HEADERS = {
    "SUS1": "I think that I would like to use the interaction technique frequently.",
    "SUS2": "I found the interaction technique unnecessarily complex.",
    "SUS3": "I found the interaction technique easy to use.",
    "SUS4": "I would need the help of a technically skilled person to use the technique.",
    "SUS5": "The various functions of the technique were well integrated.",
}


def fake_encoder(pairs: dict[str, str], dim: int = 256):
    """Each header shares its item wording's vector; everything else gets an unrelated random one."""
    rng = np.random.default_rng(0)
    vectors: dict[str, np.ndarray] = {}

    def vector(text: str) -> np.ndarray:
        text = pairs.get(text, text)
        if text not in vectors:
            v = rng.normal(size=dim)
            vectors[text] = v / np.linalg.norm(v)
        return vectors[text]

    return lambda texts: np.vstack([vector(t) for t in texts])


PAIRS = {hm.clean_header(h): SUS[item] for item, h in HEADERS.items()}


def test_clean_header_strips_numbers_anchors_and_html_but_keeps_subquestions():
    assert hm.clean_header("1. I found the system <strong>easy</strong> to use") == "I found the system easy to use"
    assert hm.clean_header("Mental Demand (0 = Low, 100 = High)") == "Mental Demand"
    assert hm.clean_header("TrustAutomation[SQ001]") == "TrustAutomation[SQ001]"


def test_cleaned_alias_maps_labelled_headers():
    columns = ["Mental Demand (0 = Low, 100 = High)", "Comment"]
    matches = hm.match_headers(columns, columns, {"mental_demand": "mental_demand"}, encoder=fake_encoder({}))
    assert {c: (m.dv, m.method) for c, m in matches.items()} == {
        "Mental Demand (0 = Low, 100 = High)": ("mental_demand", "cleaned_alias")}


def test_an_adjacent_block_of_item_wordings_is_matched_one_to_one():
    columns = ["UserID", "Condition", *HEADERS.values(), "Do you have any further comments on the technique?"]
    matches = hm.match_headers(columns, columns[2:], {}, encoder=fake_encoder(PAIRS))
    assert {m.dv: c for c, m in matches.items()} == {item: h for item, h in HEADERS.items()}
    assert {m.method for m in matches.values()} == {"item_wording"}


def test_too_few_or_scattered_lookalikes_are_not_matched():
    filler = [f"How much did you enjoy part {i} of the study overall?" for i in range(12)]
    three = list(HEADERS.values())[:3]
    assert hm.match_headers(three + filler, three + filler, {}, encoder=fake_encoder(PAIRS)) == {}
    filler = [f"How much did you enjoy part {i} of the study overall?" for i in range(48)]
    scattered = [*filler[:12], HEADERS["SUS1"], *filler[12:24], HEADERS["SUS2"], *filler[24:36], HEADERS["SUS3"],
                 *filler[36:], HEADERS["SUS4"]]
    assert hm.match_headers(scattered, scattered, {}, encoder=fake_encoder(PAIRS)) == {}


def test_batch_maps_question_headers_before_the_llm(tmp_path):
    from scripts.run_batch_standardization import run_batch

    rng = np.random.default_rng(1)
    data = tmp_path / "data"
    data.mkdir()
    frame = {"participant_code": [f"P{i}" for i in range(30)]}
    frame.update({h: rng.integers(1, 6, 30) for h in HEADERS.values()})
    frame["MysteryLikert"] = rng.integers(1, 8, 30)
    pd.DataFrame(frame).to_csv(data / "survey.csv", index=False)
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
        "source_id": "s", "source_type": "local_path", "location": str(data), "include_globs": ["*.csv"]}]}),
        encoding="utf-8")
    asked: list[str] = []

    def fake_llm(raw_column_name, canonical_candidates, **_kwargs):
        asked.append(raw_column_name)
        return None

    with mock.patch.object(hm, "_encoder", return_value=fake_encoder(PAIRS)), \
            mock.patch("scripts.llm_coordinator.deduce_standard_name_with_local_llm", side_effect=fake_llm), \
            mock.patch("scripts.run_batch_standardization.collect_repository_context", return_value=""), \
            mock.patch("scripts.llm_coordinator._select_llm_candidate_shortlist",
                       return_value=(["trust_rating"], 100.0)):
        summary = run_batch(tmp_path / "manifest.yaml", tmp_path / "out", SCHEMA, fair_export=False)
    meta = pd.read_csv(tmp_path / "out" / "meta_view.csv").set_index("original_alias")
    for item, header in HEADERS.items():
        assert meta.loc[header, "canonical_dv"] == item
        assert meta.loc[header, "mapping_method"] == "header_match"
        assert meta.loc[header, "mapping_source"].startswith("header_match:item_wording:sus:")
    assert asked == ["MysteryLikert"]
    assert summary["header_matching"] is True


def _tlx(n: int, reversed_performance: bool) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    load = rng.normal(size=n)
    cols = {name: load + rng.normal(scale=0.6, size=n) for name in
            ("mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration")}
    if reversed_performance:
        cols["performance"] = -cols["performance"]
    return pd.DataFrame(cols)


def test_keying_check_flags_a_success_keyed_tlx_performance_item():
    findings = keying_check(_tlx(60, reversed_performance=True))
    assert [(f["instrument"], f["item"]) for f in findings] == [("nasa_tlx_raw", "performance")]
    assert findings[0]["loading"] < -0.5 and findings[0]["n"] == 60
    assert keying_check(_tlx(60, reversed_performance=False)) == []
    assert keying_check(_tlx(10, reversed_performance=True)) == []  # too few rows to judge


def test_keying_check_accepts_raw_sus_and_flags_pre_reversed_items():
    rng = np.random.default_rng(3)
    usability = rng.normal(size=80)
    raw = pd.DataFrame({f"SUS{i}": np.clip(np.round(3 + (usability if i % 2 else -usability)
                                                     + rng.normal(scale=0.5, size=80)), 1, 5) for i in range(1, 11)})
    assert keying_check(raw) == []  # even items negatively worded, as the registry expects
    pre_reversed = raw.copy()
    for i in (2, 4, 6, 8, 10):
        pre_reversed[f"SUS{i}"] = 6 - raw[f"SUS{i}"]
    flagged = {f["item"] for f in keying_check(pre_reversed)}
    assert flagged == {"SUS2", "SUS4", "SUS6", "SUS8", "SUS10"}


def test_batch_reports_keying_warnings(tmp_path):
    from scripts.run_batch_standardization import run_batch

    data = tmp_path / "data"
    data.mkdir()
    frame = _tlx(40, reversed_performance=True).round(2)
    frame.rename(columns={"performance": "tlx4", "mental_demand": "tlx1"}).to_csv(data / "tlx.csv", index=False)
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump({"sources": [{
        "source_id": "s", "source_type": "local_path", "location": str(data), "include_globs": ["*.csv"]}]}),
        encoding="utf-8")
    summary = run_batch(tmp_path / "manifest.yaml", tmp_path / "out", SCHEMA, llm_deduction_enabled=False,
                        fair_export=False)
    [warning] = summary["instrument_keying_warnings"]
    assert (warning["item"], warning["original_column"], warning["source_id"]) == ("performance", "tlx4", "s")
    quality = json.loads(next((tmp_path / "out" / "standardized" / "s").glob("*-quality.json")).read_text())
    assert quality["instrument_keying_warnings"][0]["item"] == "performance"
    assert summary["header_matching"] is False


@pytest.mark.parametrize("raw, category", [
    ("header_match:item_wording:sus:0.951", "header_match"),
    ("header_match:cleaned_alias:Mental Demand", "header_match"),
    ("llm_deduction", "llm_deduced"),
])
def test_header_matches_have_their_own_provenance_category(raw, category):
    from opendv.batch_reporting import categorize_mapping_source

    assert categorize_mapping_source(raw) == category
