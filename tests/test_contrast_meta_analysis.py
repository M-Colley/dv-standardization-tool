"""Condition-contrast effect sizes, checked against metafor 5.2.1.

Reference values in ``fixtures/metafor_contrast_reference.json`` come from
``escalc(measure = "SMD")``, ``escalc(measure = "SMCRPH")``,
``aggregate(..., struct = "CS")`` and ``rma(method = "REML", test = "knha")``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyses import contrast_meta_analysis as cma
from analyses import multi_study_analysis as msa

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "metafor_contrast_reference.json").read_text())


@pytest.mark.parametrize("i", range(len(FIXTURE["smd"]["input"])))
def test_smd_between_matches_metafor(i):
    row = FIXTURE["smd"]["input"][i]
    yi, vi = cma.smd_between(row["m1"], row["m2"], row["sd1"], row["sd2"], row["n1"], row["n2"])
    assert yi == pytest.approx(FIXTURE["smd"]["yi"][i], rel=1e-10)
    assert vi == pytest.approx(FIXTURE["smd"]["vi"][i], rel=1e-10)


@pytest.mark.parametrize("i", range(len(FIXTURE["smcrph"]["input"])))
def test_smcrph_within_matches_metafor(i):
    row = FIXTURE["smcrph"]["input"][i]
    yi, vi = cma.smcrph_within(row["m1"], row["m2"], row["sd1"], row["sd2"], row["r"], row["n"])
    assert yi == pytest.approx(FIXTURE["smcrph"]["yi"][i], rel=1e-10)
    assert vi == pytest.approx(FIXTURE["smcrph"]["vi"][i], rel=1e-10)


@pytest.mark.parametrize("rho", ["0.2", "0.5", "0.8"])
def test_aggregate_dependent_matches_metafor_cs(rho):
    data = pd.DataFrame(FIXTURE["aggregate"]["input"])
    expected = FIXTURE["aggregate"]["by_rho"][rho]
    for i, (_, group) in enumerate(data.groupby("study", sort=True)):
        yi, vi = cma.aggregate_dependent(group["yi"], group["vi"], float(rho))
        assert yi == pytest.approx(expected["yi"][i], rel=1e-10)
        assert vi == pytest.approx(expected["vi"][i], rel=1e-10)


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8])
def test_aggregate_dependent_mean_is_the_composite_of_borenstein(rho):
    # Mean of the effects, variance (sum of variances + covariances) / k²
    # (Borenstein et al. 2009, ch. 24; metafor aggregate(weighted = FALSE)).
    data = pd.DataFrame(FIXTURE["aggregate"]["input"])
    for _, group in data.groupby("study", sort=True):
        y, v = group["yi"].to_numpy(), group["vi"].to_numpy()
        k = len(y)
        cov = sum(v) + sum(rho * np.sqrt(v[i] * v[j]) for i in range(k) for j in range(k) if i != j)
        yi, vi = cma.aggregate_dependent(y, v, rho, weighted=False)
        assert yi == pytest.approx(y.mean(), rel=1e-12)
        assert vi == pytest.approx(cov / k**2, rel=1e-12)


def test_gls_leaves_the_range_of_unequally_precise_contrasts_the_mean_does_not():
    # Two contrasts against one reference (perceived risk at a 15 m and a 5 m
    # merging gap vs 25 m): with rho = 0.5 and a variance ratio above 4 the
    # GLS weight of the less precise one is negative.
    y, v = [0.751, 2.300], [0.029, 0.125]
    gls, _ = cma.aggregate_dependent(y, v, 0.5)
    mean, _ = cma.aggregate_dependent(y, v, 0.5, weighted=False)
    assert gls < min(y)
    assert min(y) < mean < max(y)


def test_pool_effects_matches_metafor_reml_knha():
    yi = np.array(FIXTURE["smcrph"]["yi"])
    vi = np.array(FIXTURE["smcrph"]["vi"])
    pooled = msa.pool_effects(yi, vi, "REML")
    ref = FIXTURE["pooled_knha"]
    # metafor stops REML iterations at a 1e-5 change in tau², our scoring at
    # 1e-8, so agreement is to about 1e-4 (as in test_meta_analysis_reference).
    assert pooled["random_effects_mean"] == pytest.approx(ref["estimate"], rel=1e-4)
    assert pooled["tau2"] == pytest.approx(ref["tau2"], rel=1e-4, abs=1e-8)
    assert pooled["hksj_ci95_low"] == pytest.approx(ref["ci_lb"], rel=1e-4)
    assert pooled["hksj_ci95_high"] == pytest.approx(ref["ci_ub"], rel=1e-4)


# ── Design parsing ───────────────────────────────────────────────────────────

DESIGN_YAML = """
version: "test"
canonical_factors:
  ehmi_presence:
    levels: [absent, present]
    reference: absent
studies:
  study_within:
    participant: [pid, participant_code]
    factors:
      - name: ehmi
        canonical: ehmi_presence
        design: within
        columns:
          ehmi: null
          code: {A: none, B: ring, C: band}
        levels: {none: absent, ring: present, band: present}
  study_between:
    participant: pid
    factors:
      - name: ehmi
        canonical: ehmi_presence
        design: between
        columns: {group: null}
        levels: {control: absent, display: present}
"""


@pytest.fixture()
def designs_path(tmp_path):
    path = tmp_path / "designs.yaml"
    path.write_text(DESIGN_YAML, encoding="utf-8")
    return path


def test_load_study_designs_resolves_canonical_reference(designs_path):
    registry = cma.load_study_designs(designs_path)
    factor = registry.studies["study_within"].factors[0]
    assert factor.reference == ("none",)
    assert registry.studies["study_within"].participant == ("pid", "participant_code")


def test_load_study_designs_rejects_unknown_canonical_level(tmp_path):
    bad = DESIGN_YAML.replace("display: present", "display: maybe")
    path = tmp_path / "bad.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="not in canonical levels"):
        cma.load_study_designs(path)


def test_repository_designs_file_is_valid():
    registry = cma.load_study_designs()
    assert "running_into_traffic_chi26" in registry.studies
    assert {"ehmi_presence", "display_dimensionality"} <= set(registry.canonical_factors)


def test_integer_codes_read_as_floats_match_integer_levels():
    assert cma._normalize_value(3.0) == "3"
    assert cma._normalize_value(np.float64(10.0)) == "10"
    assert cma._normalize_value(2.5) == "2.5"
    assert cma._normalize_value(" A ") == "A"
    assert pd.isna(cma._normalize_value(float("nan")))


def test_documented_condition_codes_are_declared():
    designs = cma.load_study_designs().studies
    roads = {f.name: f for f in designs["roads_chi25"].factors}
    frame = pd.DataFrame({"ConditionID": [1.0, 6.0, 11.0, None]})
    assert roads["interaction_concept"].level_series(frame).tolist()[:3] == ["Trajectory", "Waypoint", "Path planning"]
    assert roads["number_of_requests"].level_series(frame).tolist()[:3] == ["One", "Two", "Four"]
    psych = {f.name: f for f in designs["psych_conflict_handling"].factors}
    assert psych["strategy"].reference == ("Baseline",)
    assert psych["mechanism"].level_series(pd.DataFrame({"Condition": [12]})).tolist() == ["Politeness"]


def test_factor_levels_fall_back_to_code_column(designs_path):
    factor = cma.load_study_designs(designs_path).studies["study_within"].factors[0]
    df = pd.DataFrame({"ehmi": [" none", np.nan, "ring"], "code": [np.nan, "C", np.nan]})
    assert factor.level_series(df).tolist() == ["none", "band", "ring"]


def test_exclude_leaves_listed_rows_without_a_level(tmp_path):
    path = tmp_path / "designs.yaml"
    path.write_text(DESIGN_YAML.replace(
        "        columns: {group: null}\n",
        "        columns: {group: null}\n        exclude: {phase: [pre, train], block: 2}\n",
    ), encoding="utf-8")
    factor = cma.load_study_designs(path).studies["study_between"].factors[0]
    assert factor.exclude == {"phase": frozenset({"pre", "train"}), "block": frozenset({"2"})}
    df = pd.DataFrame({
        "group": ["control", "display", "display", "control", "display"],
        "phase": ["pre", "post", "train", np.nan, "post"],
        "block": [1.0, 1.0, 1.0, 1.0, 2.0],
    })
    levels = factor.level_series(df)
    # pre/train rows and block 2 drop out; a row without a phase keeps its level
    assert levels.isna().tolist() == [True, False, True, False, True]
    assert levels.dropna().tolist() == ["display", "control"]


def test_exclude_must_map_columns(tmp_path):
    path = tmp_path / "designs.yaml"
    path.write_text(DESIGN_YAML.replace(
        "        columns: {group: null}\n", "        columns: {group: null}\n        exclude: [pre]\n",
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="'exclude' must map columns"):
        cma.load_study_designs(path)


def test_declared_factors_of_the_2026_11_05_studies():
    designs = cma.load_study_designs().studies
    tac = designs["myocontrol_intent_2025"].factors[0]
    # the two files spell the TAC levels differently
    assert tac.level_series(pd.DataFrame({"meta_condition_tac_level": ["TAC 4", "TAC_4", "TAC_3"]})).tolist() == [
        "TAC 4", "TAC 4", "TAC 3"]
    user_type = {f.name: f for f in designs["nl2tap_filter_2026"].factors}["user_type"]
    nl2tap = pd.DataFrame({
        "participant_id": ["user_0001", "user_0067", "user_0101", "user_0205", "user_0101"],
        "meta_condition_complexity": [np.nan, "C3", "C3", np.nan, "C1"],
    })
    assert user_type.level_series(nl2tap).tolist()[:4] == ["non_expert", "non_expert", "expert", "expert"]
    assert pd.isna(user_type.level_series(nl2tap).iloc[4])  # scenario of another complexity class
    assert len(user_type.columns["participant_id"]) == 85
    gap = {f.name: f for f in designs["perceived_risk_merging_2022"].factors}["merging_gap"]
    risk = pd.DataFrame({
        "meta_condition_initial_merging_gap": [5.0, 25.0, 25.0],
        "meta_condition_event_type": ["Merging and hard brake", "Merging and hard brake", "Hard braking"],
    })
    assert gap.level_series(risk).tolist()[:2] == ["5 m", "25 m"]
    assert pd.isna(gap.level_series(risk).iloc[2])
    assert gap.canonical == "situation_criticality" and gap.reference == ("25 m",)
    path = {f.name: f for f in designs["tlx_formats_uas_2023"].factors}["path_visible"]
    assert path.level_series(pd.DataFrame({"meta_condition_path_visible": [True, False]})).tolist() == [
        "visible", "hidden"]
    canonical = {
        (study, f.name): f.canonical for study, d in designs.items() for f in d.factors if f.canonical
    }
    assert {k for k, v in canonical.items() if v == "task_demand"} == {
        ("tlx_formats_uas_2023", "play_speed"), ("tlx_formats_uas_2023", "cursor_noise"),
        ("tlx_formats_uas_2023", "view_radius"), ("tlx_formats_uas_2023", "path_visible"),
        ("concurrent_speech_2022", "presentation"), ("nl2tap_filter_2026", "scenario_complexity"),
        ("roads_chi25", "number_of_requests"),
    }


def test_within_contrast_without_variance_of_the_change_is_skipped():
    # Every participant has the same value at both levels: SMCRPH would give
    # vi = 0, which no pooling can weight.
    means = pd.DataFrame({"C1": [1.0, 1.0, 0.5, 1.0], "C3": [1.0, 1.0, 0.5, 1.0], "C2": [1.0, 0.5, 0.5, 1.0]})
    rows = cma._contrast_rows(means, ("C1",), "within", 3)
    assert [r["level"] for r in rows] == ["C2"]


# ── End to end on synthetic studies ──────────────────────────────────────────

def _within_study(rng, shift):
    rows = []
    for p in range(1, 21):
        base = rng.normal(10, 3)
        for level, delta in (("none", 0.0), ("ring", shift), ("band", shift)):
            for _ in range(2):  # two trials per cell, averaged per participant
                rows.append({"pid": p, "ehmi": level, "mental_demand": base + delta + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def _between_study(rng, shift):
    rows = []
    for p in range(1, 41):
        group = "control" if p <= 20 else "display"
        rows.append({
            "pid": p, "group": group,
            "mental_demand": rng.normal(10 + (shift if group == "display" else 0.0), 3),
        })
    return pd.DataFrame(rows)


def test_contrasts_end_to_end(designs_path, tmp_path):
    rng = np.random.default_rng(7)
    studies = {"study_within": _within_study(rng, -2.0), "study_between": _between_study(rng, -2.0)}
    registry = cma.load_study_designs(designs_path)
    effects = cma.compute_contrast_effects(studies, registry, {("study_between", "mental_demand"): "llm_deduced"})

    within = effects[effects["study"] == "study_within"]
    assert set(within["level"]) == {"ring", "band"}
    assert (within["measure"] == "SMCRPH").all() and (within["n"] == 20).all()
    assert (within["yi"] < 0).all()
    # lower mental demand is better, so a negative contrast favours the eHMI
    assert within["favours_level"].all()

    between = effects[effects["study"] == "study_between"]
    assert between["measure"].tolist() == ["SMD"]
    assert between[["n_level", "n_reference"]].iloc[0].tolist() == [20, 20]

    pooled = cma.pool_contrasts(effects)
    assert pooled["k_studies"].tolist() == [2]
    assert pooled["includes_llm_deduced"].tolist() == [True]
    assert pooled["random_effects_mean"].iloc[0] < 0
    assert {"estimate_rho_0.2", "estimate_rho_0.8"} <= set(pooled.columns)

    summary = cma.run_contrast_analysis(
        studies, tmp_path / "out", designs_path=designs_path,
        mapping_provenance={("study_between", "mental_demand"): "llm_deduced"},
    )
    assert summary["n_pooled"] == 1
    clean = pd.read_csv(tmp_path / "out" / "contrast_meta_analysis_llm_excluded.csv")
    assert clean.empty  # only one study is left once the LLM-mapped one is dropped


def test_declared_design_is_checked_against_data(designs_path, caplog):
    rng = np.random.default_rng(3)
    df = _within_study(rng, 1.0).rename(columns={"ehmi": "group"})
    df["group"] = df["group"].map({"none": "control", "ring": "display", "band": "display"})
    registry = cma.load_study_designs(designs_path)
    with caplog.at_level("WARNING"):
        effects = cma.compute_contrast_effects({"study_between": df}, registry)
    assert "declared between but the data look within" in caplog.text
    assert effects["design_observed"].unique().tolist() == ["within"]
    assert effects["measure"].unique().tolist() == ["SMCRPH"]
