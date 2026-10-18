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


def test_factor_levels_fall_back_to_code_column(designs_path):
    factor = cma.load_study_designs(designs_path).studies["study_within"].factors[0]
    df = pd.DataFrame({"ehmi": [" none", np.nan, "ring"], "code": [np.nan, "C", np.nan]})
    assert factor.level_series(df).tolist() == ["none", "band", "ring"]


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
