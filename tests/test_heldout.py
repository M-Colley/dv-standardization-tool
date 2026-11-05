"""Decision rules of the held-out test (docs/preregistration_heldout_relations.md)."""

from __future__ import annotations

import pandas as pd
import pytest

from opendv.analysis import heldout as ho


def _study(x, y, study, r, p=0.01, level="curated", association="within_person"):
    a, b = sorted((x, y))
    return {"level": level, "association": association, "dv_x": a, "dv_y": b, "study": study,
            "domain": "other", "n": 60, "n_participants": 20, "r": r, "p_value": p,
            "ci95_low": r - 0.2, "ci95_high": r + 0.2}


def _pooled(x, y, r, p, k=2):
    a, b = sorted((x, y))
    return {"level": "curated", "association": "within_person", "dv_x": a, "dv_y": b, "k_studies": k,
            "pooled_r": r, "hksj_ci95_low": r - 0.1, "hksj_ci95_high": r + 0.1,
            "prediction_interval_low": r - 0.3, "prediction_interval_high": r + 0.3,
            "heterogeneity_i2_pct": 10.0, "tau": 0.1, "n_total": 40, "p_hksj": p}


def test_one_sided_p_follows_the_predicted_sign():
    assert ho.one_sided_p(0.04, 0.5, +1) == pytest.approx(0.02)
    assert ho.one_sided_p(0.04, -0.5, +1) == pytest.approx(0.98)
    assert ho.one_sided_p(0.04, -0.5, -1) == pytest.approx(0.02)


def test_h1_needs_two_studies_and_is_one_sided():
    one = pd.DataFrame([_study("mental_demand", "perceived_safety", "s1", -0.4)])
    assert ho.h1(pd.DataFrame([_pooled("a", "b", 0.1, 0.5)]), one)["decision"] == "not testable on this set"
    two = pd.DataFrame([_study("mental_demand", "perceived_safety", s, -0.4) for s in ("s1", "s2")])
    supported = ho.h1(pd.DataFrame([_pooled("mental_demand", "perceived_safety", -0.4, 0.06)]), two)
    assert supported["p_one_sided"] == pytest.approx(0.03) and supported["decision"] == "supported"
    wrong_sign = ho.h1(pd.DataFrame([_pooled("mental_demand", "perceived_safety", 0.4, 0.06)]), two)
    assert wrong_sign["decision"] == "not supported"


def test_h2_holds_only_if_all_three_pairs_survive_holm():
    rows, pooled = [], []
    for x, y in ho.H2_PAIRS:
        rows += [_study(x, y, s, 0.6) for s in ("s1", "s2")]
        pooled.append(_pooled(x, y, 0.6, 0.02))
    result = ho.h2(pd.DataFrame(pooled), pd.DataFrame(rows))
    assert result["decision"] == "supported"
    assert all(pair["p_holm"] == pytest.approx(0.03) for pair in result["pairs"].values())
    pooled[0]["p_hksj"] = 0.5
    assert ho.h2(pd.DataFrame(pooled), pd.DataFrame(rows))["decision"] == "not supported"


def test_h2b_names_the_subscale_with_the_lowest_mean_r():
    pooled = []
    for i, x in enumerate(ho.TLX_SUBSCALES):
        for y in ho.TLX_SUBSCALES[i + 1:]:
            r = 0.2 if "performance" in (x, y) else 0.6
            pooled.append(_pooled(x, y, r, 0.01))
    result = ho.h2b(pd.DataFrame(pooled), pd.DataFrame([_study("effort", "frustration", "s1", 0.6)]))
    assert result["lowest"] == "performance" and result["decision"] == "met"
