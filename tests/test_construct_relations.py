"""Tests for opendv.analysis.construct_relations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from opendv.analysis import construct_relations as cr
from opendv.analysis.contrast_meta_analysis import DesignRegistry, StudyDesign


def test_between_person_matches_scipy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=30)
    means = pd.DataFrame({"a": x, "b": 0.5 * x + rng.normal(size=30)})
    res = cr.between_person(means, "a", "b")
    ref = stats.pearsonr(means["a"], means["b"])
    assert res["r"] == pytest.approx(ref.statistic)
    assert res["p_value"] == pytest.approx(ref.pvalue)
    assert res["vi"] == pytest.approx(1 / 27)


def test_within_person_matches_the_ancova_definition():
    """rmcorr (Bakdash & Marusich 2017): r from the common slope of y on x
    with participant intercepts, sign of the slope, df = N - P - 1."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    rng = np.random.default_rng(2)
    rows = []
    for p in range(12):
        offset = rng.normal(0, 3)
        x = rng.normal(size=5)
        rows += [{"p": p, "x": xi + offset, "y": 0.6 * xi - offset + rng.normal(0, 0.8)} for xi in x]
    df = pd.DataFrame(rows)
    res = cr.within_person(df[["x", "y"]], df["p"], "x", "y")
    fit = smf.ols("y ~ C(p) + x", data=df).fit()
    table = anova_lm(fit, typ=3)
    ss_x, ss_res = table.loc["x", "sum_sq"], table.loc["Residual", "sum_sq"]
    r_ref = np.sign(fit.params["x"]) * np.sqrt(ss_x / (ss_x + ss_res))
    assert res["r"] == pytest.approx(r_ref)
    assert res["p_value"] == pytest.approx(table.loc["x", "PR(>F)"])
    assert res["n_participants"] == 12


def test_pair_types():
    assert cr.pair_type("TLX_SCORE", "mental_demand") == "shared_items"
    assert cr.pair_type("ssq_nausea", "ssq_oculomotor") == "shared_items"
    assert cr.pair_type("ueq_hedonic", "ueq_pragmatic") == "same_instrument"
    assert cr.pair_type("mental_demand", "trust_rating") == "cross_construct"


def _rows(pairs: dict[tuple[str, str], list[tuple[float, int]]], level="curated") -> pd.DataFrame:
    out = []
    for (x, y), studies in pairs.items():
        for i, (r, n) in enumerate(studies):
            z, vi = cr._fisher(r, n)
            out.append({
                "level": level, "association": "between_person", "dv_x": x, "dv_y": y,
                "study": f"s{i}", "domain": "automotive" if i % 2 == 0 else "other", "n": n,
                "n_participants": n, "r": r, "p_value": stats.pearsonr(np.arange(3), np.arange(3)).pvalue * 0
                + (2 * stats.t.sf(abs(r) * np.sqrt((n - 2) / (1 - r**2)), n - 2)),
                "ci95_low": np.nan, "ci95_high": np.nan, "z": z, "vi": vi,
                "max_level_x": 2, "max_level_y": 2,
            })
    rows = pd.DataFrame(out)
    rows["ci95_low"] = np.tanh(rows["z"] - 1.96 * np.sqrt(rows["vi"]))
    rows["ci95_high"] = np.tanh(rows["z"] + 1.96 * np.sqrt(rows["vi"]))
    return rows


def test_pooling_and_finding_rule():
    rows = _rows({
        ("mental_demand", "trust_rating"): [(-0.5, 40), (-0.45, 60), (-0.55, 50)],
        ("TLX_SCORE", "mental_demand"): [(0.8, 40), (0.85, 60), (0.82, 50)],
        ("effort", "trust_rating"): [(0.3, 40), (-0.3, 40)],
    })
    pooled = cr.pool_relations(rows).set_index(["dv_x", "dv_y"])
    hit = pooled.loc[("mental_demand", "trust_rating")]
    assert hit["k_studies"] == 3
    assert hit["pooled_r"] == pytest.approx(np.tanh(np.average(rows["z"][:3], weights=1 / rows["vi"][:3])), abs=0.01)
    assert bool(hit["finding"])
    assert hit["k_significant_alone"] == 3
    assert not bool(pooled.loc[("TLX_SCORE", "mental_demand")]["finding"])   # scored from shared items
    assert not bool(pooled.loc[("effort", "trust_rating")]["finding"])       # k = 2


def _write_study(root, study, files):
    """files: {name: (frame, [(alias, canonical, method, source)], quality extras)}"""
    meta_rows = []
    folder = root / "standardized" / study
    folder.mkdir(parents=True)
    for name, (frame, columns, extra) in files.items():
        frame.to_csv(folder / f"{name}-standardized.csv", index=False)
        (folder / f"{name}-quality.json").write_text(json.dumps(
            {"dataset_id": f"{name}.csv", "standardized_file": f"{name}-standardized.csv", **extra}))
        for alias, canonical, method, source in columns:
            meta_rows.append({
                "source_id": study, "dataset_id": f"{name}.csv", "canonical_dv": canonical,
                "original_alias": alias, "mapping_method": method, "mapping_source": source, "item_of": None,
            })
    return meta_rows


def test_levels_drop_columns_by_the_step_that_mapped_them(tmp_path):
    schema = "schemas/standard_dv_mapping.yaml"
    frame = pd.DataFrame({
        "participant": [1, 2, 3],
        "mental_demand": [1.0, 2.0, 3.0],     # raw "tlx1": a schema alias of mental_demand
        "trust_rating": [3.0, 2.0, 1.0],      # raw "Vertrauen": only the study mapping knows it
        "usability": [50.0, 60.0, 70.0],      # raw "UX": mapped by the LLM
    })
    meta = _write_study(tmp_path, "s", {"t": (frame, [
        ("participant", "participant", "blocked", "never_map_blocklist"),
        ("tlx1", "mental_demand", "mapping", schema),
        ("Vertrauen", "trust_rating", "mapping", "schemas/s_mapping.yaml"),
        ("UX", "usability", "llm", "llm_deduction"),
    ], {})})
    provenance = cr.column_provenance(tmp_path / "standardized", pd.DataFrame(meta))
    levels = {c: i.level for c, i in provenance[("s", "t-standardized.csv")].items()}
    assert levels == {"mental_demand": 1, "trust_rating": 2, "usability": 3}

    def dvs(level):
        frame = cr.level_frames(tmp_path / "standardized", provenance, level)["s"]
        return sorted(cr.dv_matrix(frame, level).columns)

    assert dvs("automatic") == ["mental_demand"]
    assert dvs("curated") == ["mental_demand", "trust_rating"]
    assert dvs("llm") == ["mental_demand", "trust_rating", "usability"]
    assert dvs("names") == ["name:tlx1@mental_demand", "name:vertrauen@trust_rating"]


def test_a_reverse_keyed_column_needs_the_study_mapping(tmp_path):
    frame = pd.DataFrame({"participant": [1, 2], "AOA1": [1.0, -1.0]})
    meta = _write_study(tmp_path, "s", {"t": (frame, [
        ("participant", "participant", "blocked", "never_map_blocklist"),
        ("aoa1", "AOA1", "mapping", "schemas/standard_dv_mapping.yaml"),
    ], {"reverse_keyed_items": {"aoa1": {"reversed": 2}}})})
    provenance = cr.column_provenance(tmp_path / "standardized", pd.DataFrame(meta))
    assert provenance[("s", "t-standardized.csv")]["AOA1"].level == 2


def test_end_to_end_relation_across_three_studies(tmp_path):
    rng = np.random.default_rng(4)
    meta = []
    designs = {}
    for k, study in enumerate(["s1", "s2", "s3"]):
        n = 40
        trust = rng.normal(size=n)
        frame = pd.DataFrame({
            "pid": np.repeat(np.arange(n), 2),
            "trust_rating": np.repeat(trust, 2) + rng.normal(0, 0.3, 2 * n),
            "perceived_safety": np.repeat(0.7 * trust, 2) + rng.normal(0, 0.5, 2 * n),
        })
        raw_trust = ["trust", "Vertrauen", "TrustScore"][k]   # one name per study
        meta += _write_study(tmp_path, study, {"t": (frame, [
            ("pid", "pid", "blocked", "never_map_blocklist"),
            (raw_trust, "trust_rating", "mapping", f"schemas/{study}_mapping.yaml"),
            ("PerceivedSafety", "perceived_safety", "mapping", "schemas/standard_dv_mapping.yaml"),
        ], {})})
        designs[study] = StudyDesign(study, ("pid",), domain="automotive")
    pd.DataFrame(meta).to_csv(tmp_path / "meta_view.csv", index=False)
    registry = DesignRegistry("t", {}, designs)

    import opendv.analysis.contrast_meta_analysis as cma
    original = cma.load_study_designs
    cma.load_study_designs = lambda path=None: registry
    try:
        cr.run(tmp_path / "standardized", tmp_path / "out", levels=("names", "automatic", "curated"))
    finally:
        cma.load_study_designs = original
    pooled = pd.read_csv(tmp_path / "out" / "relations_pooled.csv")
    hit = pooled[(pooled["level"] == "curated") & (pooled["association"] == "between_person")]
    assert len(hit) == 1 and hit.iloc[0]["k_studies"] == 3 and bool(hit.iloc[0]["finding"])
    assert pooled[(pooled["level"] == "names")].empty            # three different raw trust names
    attribution = pd.read_csv(tmp_path / "out" / "relations_attribution.csv")
    assert attribution.iloc[0]["first_finding_at"] == "curated"


def test_llm_replay_replaces_study_mappings_with_llm_answers(tmp_path):
    rng = np.random.default_rng(5)
    meta, designs = [], {}
    for k, study in enumerate(["s1", "s2", "s3"]):
        n = 40
        trust = rng.normal(size=n)
        frame = pd.DataFrame({
            "pid": np.arange(n),
            "trust_rating": trust,
            "perceived_safety": 0.7 * trust + rng.normal(0, 0.5, n),
            "meta_demographic_age": rng.integers(18, 60, n),
        })
        meta += _write_study(tmp_path, study, {"t": (frame, [
            ("pid", "pid", "blocked", "never_map_blocklist"),
            (f"Vertrauen{k}", "trust_rating", "mapping", f"schemas/{study}_mapping.yaml"),
            ("PerceivedSafety", "perceived_safety", "mapping", "schemas/standard_dv_mapping.yaml"),
            ("Alter", "meta_demographic_age", "mapping", f"schemas/{study}_mapping.yaml"),
        ], {})})
        designs[study] = StudyDesign(study, ("pid",), domain="automotive")
    meta_view = pd.DataFrame(meta)
    provenance = cr.column_provenance(tmp_path / "standardized", meta_view)
    e1 = pd.DataFrame([
        {"experiment": "E1", "source_id": "s1", "alias": "Vertrauen0", "final_checked": "trust_rating"},
        {"experiment": "E1", "source_id": "s2", "alias": "Vertrauen1", "final_checked": "jian_trust"},
        {"experiment": "E1", "source_id": "s3", "alias": "Vertrauen2", "final_checked": None},
        {"experiment": "E1", "source_id": "s3", "alias": "Alter", "final_checked": "retention_time"},
    ])
    replay, counts = cr.replay_map(tmp_path / "standardized", meta_view, provenance, e1)
    assert counts == {"curated_dv_columns": 3, "answered": 2, "unanswered_or_rejected": 1,
                      "same_dv_as_curated": 1, "curated_non_dv_mapped_by_llm": 1}
    frames = cr.level_frames(tmp_path / "standardized", provenance, cr.REPLAY_LEVEL, replay)
    assert sorted(cr.dv_matrix(frames["s1"], cr.REPLAY_LEVEL).columns) == ["perceived_safety", "trust_rating"]
    assert sorted(cr.dv_matrix(frames["s2"], cr.REPLAY_LEVEL).columns) == ["jian_trust", "perceived_safety"]
    assert sorted(cr.dv_matrix(frames["s3"], cr.REPLAY_LEVEL).columns) == ["perceived_safety", "retention_time"]

    registry = DesignRegistry("t", {}, designs)
    rows = pd.concat([
        cr.study_relations(cr.level_frames(tmp_path / "standardized", provenance, "curated"), registry, "curated")[0],
        cr.study_relations(frames, registry, cr.REPLAY_LEVEL)[0],
    ])
    pooled = cr.pool_relations(rows)
    table = cr.compare_conclusions(pooled, candidates=(cr.REPLAY_LEVEL,)).set_index("association")
    between = table.loc["between_person"]
    assert between["reference_findings"] == 1 and between["findings"] == 0


def test_llm_replay_reads_several_item_files(tmp_path):
    """Benchmark runs on disjoint sets of studies are concatenated for the replay."""
    rng = np.random.default_rng(6)
    meta, designs = [], {}
    for k, study in enumerate(["s1", "s2", "s3"]):
        n = 40
        trust = rng.normal(size=n)
        frame = pd.DataFrame({
            "pid": np.arange(n),
            "trust_rating": trust,
            "perceived_safety": 0.7 * trust + rng.normal(0, 0.5, n),
        })
        meta += _write_study(tmp_path, study, {"t": (frame, [
            ("pid", "pid", "blocked", "never_map_blocklist"),
            (f"Vertrauen{k}", "trust_rating", "mapping", f"schemas/{study}_mapping.yaml"),
            ("PerceivedSafety", "perceived_safety", "mapping", "schemas/standard_dv_mapping.yaml"),
        ], {})})
        designs[study] = StudyDesign(study, ("pid",), domain="automotive")
    pd.DataFrame(meta).to_csv(tmp_path / "meta_view.csv", index=False)
    items = pd.DataFrame([
        {"experiment": "E1", "source_id": f"s{k + 1}", "alias": f"Vertrauen{k}", "final_checked": "trust_rating"}
        for k in range(3)
    ])
    items.iloc[:1].to_csv(tmp_path / "e1_a.csv", index=False)
    items.iloc[1:].to_csv(tmp_path / "e1_b.csv", index=False)
    items.to_csv(tmp_path / "e1_all.csv", index=False)
    registry = DesignRegistry("t", {}, designs)

    import opendv.analysis.contrast_meta_analysis as cma
    original = cma.load_study_designs
    cma.load_study_designs = lambda path=None: registry
    try:
        split = cr.run(tmp_path / "standardized", tmp_path / "out_split", levels=("curated",),
                       llm_replay_items=[tmp_path / "e1_a.csv", tmp_path / "e1_b.csv"])
        joined = cr.run(tmp_path / "standardized", tmp_path / "out_joined", levels=("curated",),
                        llm_replay_items=tmp_path / "e1_all.csv")
    finally:
        cma.load_study_designs = original
    assert split["llm_replay"]["columns"] == joined["llm_replay"]["columns"]
    assert split["llm_replay"]["columns"]["same_dv_as_curated"] == 3
    assert split["llm_replay"]["e1_items"] == [str(tmp_path / "e1_a.csv"), str(tmp_path / "e1_b.csv")]
    assert joined["llm_replay"]["e1_items"] == str(tmp_path / "e1_all.csv")
    a = pd.read_csv(tmp_path / "out_split" / "relations_pooled.csv")
    b = pd.read_csv(tmp_path / "out_joined" / "relations_pooled.csv")
    pd.testing.assert_frame_equal(a, b)


# ── LLM level: the LLM fills gaps, never changes a DV mapped without it ─────

STUDY_MAPPING = "schemas/s_mapping.yaml"
PID = ("pid", "pid", "blocked", "never_map_blocklist")
TLX_SUBSCALES = ["mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration"]


def test_llm_level_sets_aside_llm_columns_of_a_curated_dv(tmp_path):
    """SenseCobot's case: EEG beta/alpha ratios LLM-mapped to mental demand,
    once beside the NASA-TLX rating (``__dup_1``) and once in other tables."""
    rng = np.random.default_rng(7)
    n = 30
    tlx = pd.DataFrame({
        "pid": np.arange(n),
        "mental_demand": rng.integers(0, 21, n).astype(float),   # curated: the NASA-TLX rating
        "mental_demand__dup_1": rng.normal(0.4, 0.05, n),       # LLM: an EEG ratio, same table
        "effort": rng.integers(0, 21, n).astype(float),
    })
    eeg = pd.DataFrame({
        "pid": np.repeat(np.arange(n), 3),
        "mental_demand": rng.normal(0.4, 0.05, 3 * n),          # LLM: an EEG ratio, other table
        "eda_gsr": rng.normal(5.0, 1.0, 3 * n),                  # LLM: no other column of this DV
    })
    eeg_only = pd.DataFrame({"pid": np.arange(n), "mental_demand": rng.normal(0.4, 0.05, n)})
    meta = _write_study(tmp_path, "s", {
        "tlx": (tlx, [PID, ("MD", "mental_demand", "mapping", STUDY_MAPPING),
                      ("beta_alpha", "mental_demand", "llm", "llm_deduction"),
                      ("EF", "effort", "mapping", STUDY_MAPPING)], {}),
        "eeg": (eeg, [PID, ("ratio", "mental_demand", "llm", "llm_deduction"),
                      ("gsr", "eda_gsr", "llm", "llm_deduction")], {}),
        "eeg_only": (eeg_only, [PID, ("ratio2", "mental_demand", "llm", "llm_deduction")], {}),
    })
    root = tmp_path / "standardized"
    provenance = cr.column_provenance(root, pd.DataFrame(meta))

    curated = cr.level_frames(root, provenance, "curated")
    llm = cr.level_frames(root, provenance, "llm")
    at_curated = cr.dv_matrix(curated["s"], "curated")
    at_llm = cr.dv_matrix(llm["s"], "llm")
    # The curated ratings are kept as they are: no EEG ratio averaged in or stacked below them.
    kept = at_llm["mental_demand"].dropna().reset_index(drop=True)
    pd.testing.assert_series_equal(kept, tlx["mental_demand"], check_names=False)
    pd.testing.assert_series_equal(kept, at_curated["mental_demand"].dropna().reset_index(drop=True))
    assert "eda_gsr" in at_llm and "eda_gsr" not in at_curated      # the LLM still fills a gap
    assert sorted(llm["s"]["_source_file"].unique()) == ["eeg.csv", "tlx.csv"]   # eeg_only: nothing left
    assert llm.column_levels["s"] == {"mental_demand": 2, "effort": 2, "eda_gsr": 3}
    aside = pd.DataFrame(llm.set_aside).sort_values(["dataset_id", "column"])
    assert aside[["dataset_id", "column", "raw_column", "dv", "kept_instead"]].to_dict("records") == [
        {"dataset_id": "eeg.csv", "column": "mental_demand", "raw_column": "ratio",
         "dv": "mental_demand", "kept_instead": "other table"},
        {"dataset_id": "eeg_only.csv", "column": "mental_demand", "raw_column": "ratio2",
         "dv": "mental_demand", "kept_instead": "other table"},
        {"dataset_id": "tlx.csv", "column": "mental_demand__dup_1", "raw_column": "beta_alpha",
         "dv": "mental_demand", "kept_instead": "same table"},
    ]
    assert curated.set_aside == []


def test_llm_level_keeps_a_score_from_curated_items_before_an_llm_column(tmp_path):
    rng = np.random.default_rng(8)
    n = 25
    frame = pd.DataFrame({"pid": np.arange(n), **{s: rng.integers(0, 21, n).astype(float) for s in TLX_SUBSCALES}})
    frame["TLX_SCORE"] = rng.normal(60, 10, n)               # LLM: some other workload index
    columns = [PID, *[(f"tlx_{s}", s, "mapping", STUDY_MAPPING) for s in TLX_SUBSCALES],
               ("Workload", "TLX_SCORE", "llm", "llm_deduction")]
    meta = _write_study(tmp_path, "s", {"t": (frame, columns, {})})
    root = tmp_path / "standardized"
    provenance = cr.column_provenance(root, pd.DataFrame(meta))
    llm = cr.level_frames(root, provenance, "llm")
    assert "TLX_SCORE" not in llm["s"] and "nasa_tlx_score" in llm["s"]
    matrix = cr.dv_matrix(llm["s"], "llm")
    pd.testing.assert_series_equal(matrix["TLX_SCORE"], llm["s"]["nasa_tlx_score"], check_names=False)
    assert llm.column_levels["s"]["nasa_tlx_score"] == 2
    (row,) = llm.set_aside
    assert (row["column"], row["dataset_id"], row["dv"], row["kept_instead"]) == \
        ("TLX_SCORE", "t.csv", "TLX_SCORE", "computed score")


def test_llm_level_keeps_a_curated_column_before_a_score_from_llm_items(tmp_path):
    rng = np.random.default_rng(9)
    n = 25
    frame = pd.DataFrame({"pid": np.arange(n), **{s: rng.integers(0, 21, n).astype(float) for s in TLX_SUBSCALES}})
    frame["TLX_SCORE"] = rng.normal(60, 10, n)               # curated overall workload
    columns = [PID, *[(f"q{i}", s, "llm", "llm_deduction") for i, s in enumerate(TLX_SUBSCALES)],
               ("RTLX", "TLX_SCORE", "mapping", STUDY_MAPPING)]
    meta = _write_study(tmp_path, "s", {"t": (frame, columns, {})})
    root = tmp_path / "standardized"
    provenance = cr.column_provenance(root, pd.DataFrame(meta))
    llm = cr.level_frames(root, provenance, "llm")
    matrix = cr.dv_matrix(llm["s"], "llm")
    pd.testing.assert_series_equal(matrix["TLX_SCORE"], frame["TLX_SCORE"], check_names=False)
    assert "mental_demand" in matrix                                # the LLM's subscales stay
    (row,) = llm.set_aside
    assert (row["column"], row["dataset_id"], row["kept_instead"]) == ("nasa_tlx_score", "", "mapped column")


def _relations_run(tmp_path, studies, levels, **kwargs):
    """Write studies ``{name: (frame, meta columns)}`` and run the relations on them."""
    meta, designs = [], {}
    for study, (frame, columns) in studies.items():
        meta += _write_study(tmp_path, study, {"t": (frame, columns, {})})
        designs[study] = StudyDesign(study, ("pid",), domain="automotive")
    pd.DataFrame(meta).to_csv(tmp_path / "meta_view.csv", index=False)
    registry = DesignRegistry("t", {}, designs)
    import opendv.analysis.contrast_meta_analysis as cma
    original = cma.load_study_designs
    cma.load_study_designs = lambda path=None: registry
    try:
        return cr.run(tmp_path / "standardized", tmp_path / "out", levels=levels, **kwargs)
    finally:
        cma.load_study_designs = original


def test_llm_level_relations_rest_on_the_curated_columns(tmp_path):
    """A study's LLM column of a curated DV no longer moves its correlation,
    and the pooled relation no longer counts as resting on LLM columns."""
    rng = np.random.default_rng(10)
    studies = {}
    for k, study in enumerate(["s1", "s2", "s3"]):
        n = 40
        trust = rng.normal(size=n)
        frame = pd.DataFrame({"pid": np.arange(n), "trust_rating": trust,
                              "perceived_safety": 0.7 * trust + rng.normal(0, 0.5, n)})
        columns = [PID, (f"Vertrauen{k}", "trust_rating", "mapping", f"schemas/{study}_mapping.yaml"),
                   ("PerceivedSafety", "perceived_safety", "mapping", "schemas/standard_dv_mapping.yaml")]
        if study == "s1":
            frame["trust_rating__dup_1"] = rng.normal(size=n)      # unrelated, LLM-mapped
            frame["acceptance_rating"] = rng.normal(size=n)       # LLM-only DV
            columns += [("Gefallen", "trust_rating", "llm", "llm_deduction"),
                        ("Akzeptanz", "acceptance_rating", "llm", "llm_deduction")]
        studies[study] = (frame, columns)
    _relations_run(tmp_path, studies, levels=("curated", "llm"))
    by_study = pd.read_csv(tmp_path / "out" / "relations_by_study.csv")
    pair = by_study[(by_study["dv_x"] == "perceived_safety") & (by_study["dv_y"] == "trust_rating")]
    r = pair.pivot(index="study", columns=["level", "association"], values="r")
    np.testing.assert_allclose(r[("llm", "between_person")], r[("curated", "between_person")])
    # Schema (1) and study mapping (2) at both levels. Before 2026-11-14 the
    # labels counted s1's LLM column (3) at the curated level, which never uses it.
    assert (pair["max_level_x"] == 1).all() and (pair["max_level_y"] == 2).all()
    accept = by_study[(by_study["level"] == "llm") & (by_study["dv_x"] == "acceptance_rating")]
    assert set(accept["max_level_x"]) == {3}                        # the LLM-only DV is still used
    pooled = pd.read_csv(tmp_path / "out" / "relations_pooled.csv")
    hit = pooled[(pooled["dv_x"] == "perceived_safety") & (pooled["dv_y"] == "trust_rating")]
    assert len(hit) == 2 and not hit["includes_llm_columns"].any()
    aside = pd.read_csv(tmp_path / "out" / "relations_llm_columns_set_aside.csv")
    assert aside[["level", "study", "column", "kept_instead"]].values.tolist() == [
        ["llm", "s1", "trust_rating__dup_1", "same table"]]
    summary = json.loads((tmp_path / "out" / "relations_summary.json").read_text())
    assert summary["llm_columns_set_aside"]["llm"] == {"columns": 1, "studies": 1}


def test_llm_replay_sets_aside_llm_answers_of_an_automatic_dv(tmp_path):
    """In the replay the study mappings are hidden; an LLM answer of a DV the
    study has a schema column of is set aside, as at the ``llm`` level."""
    rng = np.random.default_rng(11)
    n = 30
    frame = pd.DataFrame({"pid": np.arange(n), "perceived_safety": rng.normal(size=n),
                          "trust_rating": rng.normal(size=n), "effort": rng.normal(size=n)})
    meta = _write_study(tmp_path, "s", {"t": (frame, [
        PID, ("PerceivedSafety", "perceived_safety", "mapping", "schemas/standard_dv_mapping.yaml"),
        ("Sicherheit", "trust_rating", "mapping", STUDY_MAPPING),
        ("Anstrengung", "effort", "mapping", STUDY_MAPPING)], {})})
    meta_view = pd.DataFrame(meta)
    root = tmp_path / "standardized"
    provenance = cr.column_provenance(root, meta_view)
    e1 = pd.DataFrame([
        {"experiment": "E1", "source_id": "s", "alias": "Sicherheit", "final_checked": "perceived_safety"},
        {"experiment": "E1", "source_id": "s", "alias": "Anstrengung", "final_checked": "effort"},
    ])
    replay, _ = cr.replay_map(root, meta_view, provenance, e1)
    frames = cr.level_frames(root, provenance, cr.REPLAY_LEVEL, replay)
    matrix = cr.dv_matrix(frames["s"], cr.REPLAY_LEVEL)
    pd.testing.assert_series_equal(matrix["perceived_safety"], frame["perceived_safety"], check_names=False)
    pd.testing.assert_series_equal(matrix["effort"], frame["effort"], check_names=False)
    assert [(r["column"], r["raw_column"], r["kept_instead"]) for r in frames.set_aside] == [
        ("perceived_safety__dup_900", "Sicherheit", "same table")]
    assert frames.column_levels["s"] == {"perceived_safety": 1, "effort__dup_901": 3}
