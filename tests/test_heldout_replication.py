"""Decision rules of the second held-out test (docs/preregistration_heldout_relations_2.md)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from opendv.analysis import heldout_replication as hr


def _finding(x, y, r, association="within_person", k=4, q=0.01, finding=True, pair_type="same_instrument"):
    a, b = sorted((x, y))
    return {"level": "curated", "association": association, "dv_x": a, "dv_y": b, "pair_type": pair_type,
            "k_studies": k, "pooled_r": r, "q_bh": q, "finding": finding}


def _study(x, y, study, r, association="within_person"):
    a, b = sorted((x, y))
    return {"level": "curated", "association": association, "dv_x": a, "dv_y": b, "study": study,
            "domain": "other", "n": 60, "n_participants": 20, "r": r, "p_value": 0.01,
            "ci95_low": r - 0.2, "ci95_high": r + 0.2}


def _pooled(x, y, r, p, association="within_person", k=2):
    a, b = sorted((x, y))
    return {"level": "curated", "association": association, "dv_x": a, "dv_y": b, "k_studies": k,
            "pair_type": "same_instrument", "pooled_r": r, "hksj_ci95_low": r - 0.1, "hksj_ci95_high": r + 0.1,
            "prediction_interval_low": r - 0.3, "prediction_interval_high": r + 0.3,
            "heterogeneity_i2_pct": 10.0, "tau": 0.1, "n_total": 40, "p_hksj": p, "q_bh": p, "finding": False}


def test_targets_are_the_union_of_findings_with_their_sign_and_family():
    development = pd.DataFrame([
        _finding("mental_demand", "perceived_safety", -0.44, pair_type="cross_construct"),
        _finding("effort", "frustration", 0.6),
        _finding("mental_demand", "usability", -0.42, association="between_person", pair_type="cross_construct"),
        _finding("effort", "performance", 0.1, association="between_person", finding=False),
        {**_finding("effort", "mental_demand", 0.7), "level": "llm"},  # not curated: ignored
    ])
    first_test = pd.DataFrame([
        _finding("effort", "performance", 0.42, association="between_person", k=7),
        _finding("mental_demand", "usability", -0.3, association="between_person", pair_type="cross_construct"),
    ])
    targets = hr.build_targets(development, first_test).set_index("target")
    assert list(targets.columns) == hr.TARGET_COLUMNS[1:]
    assert targets.loc["within:mental_demand~perceived_safety", "family"] == "H1"
    assert targets.loc["within:effort~frustration", "family"] == "H2"
    usability = targets.loc["between:mental_demand~usability"]
    assert usability["family"] == "H4" and usability["found_in"] == "both" and usability["predicted_sign"] == -1
    performance = targets.loc["between:effort~performance"]
    assert performance["found_in"] == "first test" and performance["predicted_sign"] == 1
    assert len(targets) == 4


def test_targets_refuse_findings_of_opposite_sign():
    development = pd.DataFrame([_finding("effort", "frustration", 0.6, association="between_person")])
    first_test = pd.DataFrame([_finding("effort", "frustration", -0.6, association="between_person")])
    with pytest.raises(ValueError, match="opposite signs"):
        hr.build_targets(development, first_test)


def test_the_committed_targets_match_the_committed_analyses():
    rebuilt = hr.build_targets(
        pd.read_csv(hr.REPO_ROOT / "analyses" / "output_relations" / "relations_pooled.csv"),
        pd.read_csv(hr.REPO_ROOT / "analyses" / "heldout" / "relations" / "relations_pooled.csv"),
    )
    committed = pd.read_csv(hr.OUT_DIR / "targets.csv")
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), committed, check_dtype=False)
    assert committed.groupby("family").size().to_dict() == {"H1": 1, "H2": 3, "H4": 33}


def _targets(*rows):
    return pd.DataFrame([{"target": f"{a.split('_')[0]}:{x}~{y}", "family": "H4", "association": a, "dv_x": x,
                          "dv_y": y, "pair_type": "same_instrument", "predicted_sign": s, "found_in": "development"}
                         for a, x, y, s in rows])


def test_h4_needs_two_studies_and_controls_the_false_discovery_rate():
    targets = _targets(("between_person", "effort", "frustration", 1),
                       ("between_person", "mental_demand", "usability", -1),
                       ("within_person", "effort", "performance", 1))
    by_study = pd.DataFrame(
        [_study("effort", "frustration", s, 0.5, "between_person") for s in ("a", "b", "c")]
        + [_study("mental_demand", "usability", s, -0.4, "between_person") for s in ("a", "b")]
        + [_study("effort", "performance", "a", 0.3)]
    )
    pooled = pd.DataFrame([
        _pooled("effort", "frustration", 0.5, 0.01, "between_person", k=3),
        _pooled("mental_demand", "usability", -0.4, 0.08, "between_person"),
    ])
    result = hr.h4(pooled, by_study, targets)
    assert result["targets"] == 3 and result["testable"] == 2
    tlx = result["results"]["between:effort~frustration"]
    sus = result["results"]["between:mental_demand~usability"]
    # one-sided p: 0.005 and 0.04; BH over two: 0.01 and 0.04
    assert tlx["p_one_sided"] == pytest.approx(0.005) and tlx["q_bh"] == pytest.approx(0.01)
    assert sus["q_bh"] == pytest.approx(0.04) and sus["replicated"] and sus["same_sign"]
    assert result["results"]["within:effort~performance"]["testable"] is False
    assert result["replicated"] == 2
    assert result["by_association"]["within_person"] == {"targets": 1, "testable": 0, "replicated": 0,
                                                        "same_sign": 0}


def test_h4_a_wrong_sign_does_not_replicate():
    targets = _targets(("between_person", "mental_demand", "usability", -1))
    by_study = pd.DataFrame([_study("mental_demand", "usability", s, 0.4, "between_person") for s in ("a", "b")])
    pooled = pd.DataFrame([_pooled("mental_demand", "usability", 0.4, 0.001, "between_person")])
    res = hr.h4(pooled, by_study, targets)["results"]["between:mental_demand~usability"]
    assert res["p_one_sided"] == pytest.approx(0.9995)
    assert not res["replicated"] and not res["same_sign"] and res["k_predicted_sign"] == 0


def test_decide_reuses_the_first_tests_rules_and_lists_new_findings(tmp_path):
    rows, pooled = [], []
    for x, y in hr.H2_PAIRS:
        rows += [_study(x, y, s, 0.6) for s in ("s1", "s2")]
        pooled.append(_pooled(x, y, 0.6, 0.02))
    pooled.append({**_pooled("arousal", "valence", -0.5, 0.001, "between_person", k=3), "finding": True})
    rows += [_study("arousal", "valence", s, -0.5, "between_person") for s in ("s1", "s2", "s3")]
    relations = tmp_path / "relations"
    relations.mkdir()
    pd.DataFrame(pooled).to_csv(relations / "relations_pooled.csv", index=False)
    pd.DataFrame(rows).to_csv(relations / "relations_by_study.csv", index=False)
    targets = pd.read_csv(hr.OUT_DIR / "targets.csv")
    targets_path = tmp_path / "targets.csv"
    targets.to_csv(targets_path, index=False)
    result = hr.decide(relations, targets_path)
    assert result["H1"]["decision"] == "not testable on this set"
    assert result["H2"]["decision"] == "supported"
    assert result["H4"]["testable"] == 0 and result["H4"]["decision"] == "not testable on this set"
    assert result["new_findings_exploratory"] == [
        {"association": "between_person", "pair": ["arousal", "valence"], "pair_type": "same_instrument",
         "k_studies": 3, "pooled_r": -0.5, "q_bh": 0.001}]
    json.dumps(result)  # serializable
