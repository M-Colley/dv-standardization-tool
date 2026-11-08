"""Regression tests for the defects found in the 2026-11-04 code review.

Each test reproduces one reported failure and checks the corrected behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opendv.analysis import multi_study_analysis as msa

# ── Composition of columns that share a DV ──────────────────────────────────

def test_negatively_related_columns_are_not_flipped_or_averaged():
    from opendv.item_composition import compose_item_groups

    df = pd.DataFrame({
        "trust_rating": [5, 4, 5, 3, 4, 5],          # before a failure
        "trust_rating__dup_2": [1, 2, 1, 3, 2, 1],   # after it
    })
    out, records = compose_item_groups(df, "questionnaire")
    assert records[0]["decision"] == "skip_negative_item_correlation"
    assert records[0]["reverse_coded_items"] == []
    pd.testing.assert_frame_equal(out, df)


def test_llm_columns_are_never_composed_with_deterministic_ones():
    from opendv.item_composition import compose_item_groups

    df = pd.DataFrame({
        "perceived_safety": [1, 2, 3, 4, 5, 6],
        "perceived_safety__dup_2": [1, 2, 3, 4, 5, 7],
        "perceived_safety__dup_3": [6, 5, 4, 3, 2, 1],  # mapped by the LLM
    })
    out, records = compose_item_groups(df, "questionnaire", llm_columns={"perceived_safety__dup_3"})
    assert records[0]["decision"] == "skip_mixed_provenance"
    assert list(out["perceived_safety"]) == [1, 2, 3, 4, 5, 6]


def test_consistent_items_are_still_composed():
    from opendv.item_composition import compose_item_groups

    df = pd.DataFrame({"trust_rating": [1, 2, 3, 4, 5], "trust_rating__dup_2": [1, 2, 3, 5, 5]})
    out, records = compose_item_groups(df, "questionnaire")
    assert records[0]["decision"] == "composed"
    assert list(out["trust_rating"]) == [1.0, 2.0, 3.0, 4.5, 5.0]


def test_bookkeeping_columns_are_not_composed():
    from opendv.item_composition import compose_item_groups

    df = pd.DataFrame({"meta_fms_timepoint": [1, 2, 3, 4, 5], "meta_fms_timepoint__dup_2": [1, 2, 3, 4, 6]})
    out, records = compose_item_groups(df, "questionnaire")
    assert records == []
    pd.testing.assert_frame_equal(out, df)


# ── Scales and units declared in mapping files ──────────────────────────────

def test_scale_conversions_map_declared_columns_linearly():
    from opendv.value_labels import apply_scale_conversions

    df = pd.DataFrame({"TLX 1": [0, 5, 10, "round data missing"], "other": [1, 2, 3, 4]})
    out, report = apply_scale_conversions(df, [{"columns": ["TLX 1"], "from": [0, 10], "to": [0, 20]}])
    assert list(out["TLX 1"]) == [0, 10, 20, "round data missing"]
    assert list(out["other"]) == [1, 2, 3, 4]
    assert report == {"TLX 1": "[0, 10] -> [0, 20]"}


def test_scale_conversions_are_validated():
    from opendv.validate_schema import validate_dataset_mapping

    errors, _ = validate_dataset_mapping({
        "x": "mental_demand",
        "scale_conversions": [{"columns": ["x"], "from": [10, 0], "to": [0, 20]}],
    })
    assert any("scale_conversions[0]" in e for e in errors)


def test_drop_columns_table_option(tmp_path):
    from opendv.data_loaders import load_input_file

    path = tmp_path / "ObservationsPerEvaluation.csv"
    pd.DataFrame({"Trust": [4], "MentalLoad": [10]}).to_csv(path, index=False)
    df = load_input_file(str(path), {"drop_columns": ["MentalLoad"]})
    assert list(df.columns) == ["Trust"]
    with pytest.raises(ValueError, match="not found"):
        load_input_file(str(path), {"drop_columns": ["Nope"]})


@pytest.mark.parametrize("items, expected", [
    ([10, 8, 12, 14, 6, 4], 45.0),        # 0-20 marks
    ([11, 9, 13, 15, 7, 5], 45.0),        # 1-21 marks (one item at 21 picks this coding)
    ([50, 40, 60, 70, 30, 20], 45.0),     # 0-100
])
def test_raw_tlx_is_on_0_100_whatever_the_item_coding(items, expected):
    from opendv import instruments

    names = ["mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration"]
    df = pd.DataFrame({n: [v] for n, v in zip(names, items, strict=True)})
    if items[0] == 11:
        df = pd.concat([df, pd.DataFrame({n: [21 if n == "effort" else 1] for n in names})], ignore_index=True)
    out, _ = instruments.score_frame(df)
    assert out["nasa_tlx_score"].iloc[0] == pytest.approx(expected)


# ── Provenance ──────────────────────────────────────────────────────────────

def test_study_mapping_files_are_not_labelled_as_the_shared_schema():
    from opendv.batch_reporting import categorize_mapping_source

    assert categorize_mapping_source(r"C:\repo\schemas\pairsonic_cscw24_mapping.yaml") == "repo_mapping"
    assert categorize_mapping_source("schemas/standard_dv_mapping.yaml") == "schema"


def test_scores_inherit_the_provenance_of_their_items(tmp_path):
    rows = [{"source_id": "s", "canonical_dv": f"SUS{i}", "mapping_source": "schemas/standard_dv_mapping.yaml"}
            for i in range(1, 11)]
    rows[5]["mapping_source"] = "llm_deduction"
    path = tmp_path / "meta_view.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    provenance = msa.load_mapping_provenance(path)
    assert provenance[("s", "usability")] == "llm_deduced"


def test_composite_item_columns_are_not_read_as_scale_items():
    df = pd.DataFrame({"perceived_safety": [1.0, 2.0], "perceived_safety__item_2": [5.0, 6.0]})
    frame = msa._canonical_numeric_frame(df)
    assert list(frame.columns) == ["perceived_safety"]


def test_uncomposed_duplicates_are_averaged_not_dropped():
    df = pd.DataFrame({"fms_score": [0.0, 1.0], "fms_score__dup_2": [4.0, 5.0]})
    frame = msa._canonical_numeric_frame(df)
    assert list(frame["fms_score"]) == [2.0, 3.0]


def test_dwell_time_is_its_own_dv():
    assert msa._canonicalize_dv_name("dwell_time") == "dwell_time"


def test_reverse_keyed_items_are_lower_is_better():
    meta = msa._load_dv_measurement_metadata()
    assert meta["SUS2"]["direction"] == "lower_is_better"
    assert meta["AOA1"]["direction"] == "lower_is_better"
    assert meta["AOA3"]["direction"] == "higher_is_better"


# ── Participants ────────────────────────────────────────────────────────────

def test_participant_ids_can_come_from_file_names_and_be_file_scoped():
    from opendv.analysis.contrast_meta_analysis import StudyDesign

    df = pd.DataFrame({"User_ID": [10, None], "_source_file": ["u_10_obs.csv", "x_u_11_q.csv"]})
    design = StudyDesign("s", ("User_ID",), participant_from_file=r"_u_(\d+)_")
    assert design.participant_series(df).tolist() == ["10", "11"]
    scoped = StudyDesign("s", ("User_ID",), participant_scope="file")
    assert scoped.participant_series(df).tolist()[0] == "u_10_obs.csv::10"


def test_participant_means_use_the_declared_participant_column(tmp_path, monkeypatch):
    from opendv.analysis.contrast_meta_analysis import DesignRegistry, StudyDesign

    study = tmp_path / "st"
    study.mkdir()
    pd.DataFrame({"user_id": [1, 1, 2, 2], "pedestrian_crossing_time": [1.0, 2.0, 3.0, 4.0]}).to_csv(
        study / "log.csv", index=False)
    pd.DataFrame({"UserID": [1, 2], "mental_demand": [5.0, 7.0]}).to_csv(study / "survey.csv", index=False)
    registry = DesignRegistry("t", {}, {"st": StudyDesign("st", ("UserID", "user_id"))})
    monkeypatch.setattr(msa, "_study_designs", lambda: registry)
    out = msa.load_studies(tmp_path, repeated_measures_studies={"st"})["st"]
    assert len(out) == 2
    assert out["mental_demand"].tolist() == [5.0, 7.0]


# ── Synthesis ───────────────────────────────────────────────────────────────

def test_subgroups_count_each_study_once():
    rows = [{"study": "a", "dv": "usability", "hedges_g": 0.5, "var_g": 0.01}]
    rows += [{"study": "a", "dv": f"SUS{i}", "hedges_g": 0.5, "var_g": 0.01} for i in (1, 3, 5, 7, 9)]
    rows += [{"study": "b", "dv": "usability", "hedges_g": -0.5, "var_g": 0.01}]
    out = msa.subgroup_meta_analysis(pd.DataFrame(rows))
    ux = out[out["cluster"] == "ux_satisfaction"].iloc[0]
    assert ux["k_studies"] == 2
    assert ux["pooled_g"] == pytest.approx(0.0, abs=1e-12)


def test_subgroups_combine_a_studys_effects_by_their_mean():
    # A one-group g's variance grows with g², so a study's largest deviation is
    # its least precise one. With a variance ratio above 1 / rho² = 4, GLS gives
    # it a negative weight: study a comes out at 0.23, below both its effects.
    from opendv.analysis.contrast_meta_analysis import aggregate_dependent

    y_a, v_a = [0.4, 3.0], [0.01, 0.08]
    rows = [
        {"study": "a", "dv": "usability", "hedges_g": y_a[0], "var_g": v_a[0]},
        {"study": "a", "dv": "ueq_pragmatic", "hedges_g": y_a[1], "var_g": v_a[1]},
        {"study": "b", "dv": "usability", "hedges_g": -0.5, "var_g": 0.01},
    ]

    def pooled_with_b(study_a):
        g, v = study_a
        w = np.array([1.0 / v, 1.0 / 0.01])
        return float(w @ np.array([g, -0.5]) / w.sum())

    gls_a = aggregate_dependent(y_a, v_a, 0.5, weighted=True)
    assert gls_a[0] < min(y_a)
    ux = msa.subgroup_meta_analysis(pd.DataFrame(rows)).set_index("cluster").loc["ux_satisfaction"]
    assert ux["aggregation"] == "mean"
    assert ux["pooled_g"] == pytest.approx(pooled_with_b(aggregate_dependent(y_a, v_a, 0.5, weighted=False)))
    assert ux["pooled_g_gls"] == pytest.approx(pooled_with_b(gls_a))
    assert ux["pooled_g"] > 0 > ux["pooled_g_gls"]
    gls = msa.subgroup_meta_analysis(pd.DataFrame(rows), weighted=True).set_index("cluster").loc["ux_satisfaction"]
    assert gls["aggregation"] == "gls"
    assert gls["pooled_g"] == pytest.approx(ux["pooled_g_gls"])


def test_summary_counts_participants_and_caps_rows_without_an_id(monkeypatch):
    # Study "log" repeats measures: 3 participants, 300 rows, plus a file of 50
    # rows without ids (Road Bumps' accelerometer trace); "survey" has 1 row each.
    from opendv.analysis.contrast_meta_analysis import DesignRegistry, StudyDesign

    registry = DesignRegistry("t", {}, {"log": StudyDesign("log", ("PId",)),
                                        "survey": StudyDesign("survey", ("PId",))})
    monkeypatch.setattr(msa, "_study_designs", lambda: registry)
    rng = np.random.default_rng(3)
    log = pd.DataFrame({"PId": np.repeat([1.0, 2.0, 3.0], 100), "mental_demand": rng.normal(5, 1, 300)})
    log = pd.concat([log, pd.DataFrame({"pedestrian_crossing_time": rng.normal(2, 1, 50)})], ignore_index=True)
    survey = pd.DataFrame({"PId": np.arange(1.0, 9.0), "mental_demand": rng.normal(6, 1, 8)})
    summary = msa.harmonized_summary({"log": log, "survey": survey}).set_index(["study", "dv"])
    assert summary.loc[("log", "mental_demand"), ["n", "n_participants"]].tolist() == [300, 3]
    assert summary.loc[("log", "pedestrian_crossing_time"), ["n", "n_participants"]].tolist() == [50, 3]
    assert summary.loc[("survey", "mental_demand"), ["n", "n_participants"]].tolist() == [8, 8]
    assert msa._participant_count(None, pd.RangeIndex(5)) == 5


def test_study_vs_pool_deviations_use_participants_as_n():
    summary = pd.DataFrame([
        {"study": "a", "dv": "mental_demand", "n": 100_000, "n_participants": 16, "mean": 4.0, "sd": 1.0},
        {"study": "b", "dv": "mental_demand", "n": 30, "n_participants": 30, "mean": 6.0, "sd": 1.0},
    ])
    eff = msa.compute_standardized_effects(summary).set_index("study")
    grand = (16 * 4.0 + 30 * 6.0) / 46
    j = 1 - 3 / (4 * 15 - 1)
    g = (4.0 - grand) * j
    assert eff.loc["a", "hedges_g"] == pytest.approx(g)
    assert eff.loc["a", "var_g"] == pytest.approx(1 / 16 + g ** 2 / 32)
    assert eff.loc["a", ["n", "n_rows"]].tolist() == [16, 100_000]
    rows = msa.compute_standardized_effects(summary, unit="rows").set_index("study")
    assert rows.loc["a", "n"] == 100_000
    assert abs(rows.loc["a", "hedges_g"]) < 1e-3  # the row-weighted pool sits on study a
    without = msa.compute_standardized_effects(summary.drop(columns="n_participants"))
    pd.testing.assert_frame_equal(without, msa.compute_standardized_effects(summary, unit="rows"))


def test_subgroups_pool_studies_with_random_effects():
    # One very precise study and three imprecise ones that disagree with it:
    # the fixed-effect pool sits on the precise study, the random-effects pool does not.
    rows = [{"study": "big", "dv": "usability", "hedges_g": 0.0, "var_g": 1e-5}]
    rows += [{"study": s, "dv": "usability", "hedges_g": y, "var_g": 0.05}
             for s, y in (("b", 1.0), ("c", 0.6), ("d", 1.4))]
    ux = msa.subgroup_meta_analysis(pd.DataFrame(rows)).set_index("cluster").loc["ux_satisfaction"]
    y = np.array([0.0, 1.0, 0.6, 1.4])
    v = np.array([1e-5, 0.05, 0.05, 0.05])
    expected = msa.pool_effects(y, v, "DL")
    assert ux["pooling_method"] == "random_effects"
    assert ux["pooled_g"] == pytest.approx(expected["random_effects_mean"])
    assert ux["se"] == pytest.approx(expected["random_effects_se"])
    assert ux["hksj_ci95_high"] == pytest.approx(expected["hksj_ci95_high"])
    assert ux["tau2"] == pytest.approx(expected["tau2"]) and ux["tau2"] > 0
    assert ux["fixed_effect_g"] == pytest.approx(float(np.sum(y / v) / np.sum(1 / v)))
    assert ux["fixed_effect_largest_weight_pct"] > 99
    assert ux["largest_weight_study"] == "big" and ux["largest_weight_pct"] < 30
    assert ux["pooled_g"] > 0.5 > ux["fixed_effect_g"]
    reml = msa.subgroup_meta_analysis(pd.DataFrame(rows), estimator="REML").set_index("cluster")
    expected_reml = msa.pool_effects(y, v, "REML")["random_effects_mean"]
    assert reml.loc["ux_satisfaction", "pooled_g"] == pytest.approx(expected_reml)


def test_meta_regression_intercept_only_equals_dl_with_hksj():
    y = np.array([1.2, 0.8, 2.1, 1.5, 0.3, 1.9])
    v = np.array([0.04, 0.09, 0.05, 0.12, 0.06, 0.08])
    fit = msa.mixed_meta_regression(y, v, np.ones((6, 1)))
    pooled = msa.pool_effects(y, v, "DL")
    assert fit["coef"][0] == pytest.approx(pooled["random_effects_mean"])
    assert fit["se"][0] == pytest.approx(pooled["hksj_se"])
    assert fit["tau2"] == pytest.approx(pooled["tau2"])


def test_meta_regression_keeps_nominal_error_under_heterogeneity():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(600):
        x = rng.uniform(10, 100, 6)
        v = rng.uniform(0.02, 0.1, 6)
        y = rng.normal(1, 0.5, 6) + rng.normal(0, np.sqrt(v))
        hits += msa.mixed_meta_regression(y, v, np.column_stack([np.ones(6), x]))["p"][1] < 0.05
    assert hits / 600 < 0.1   # a fixed-effect slope test rejects about 60% here


def test_leave_one_out_reports_the_pooling_method():
    summary = pd.DataFrame({
        "study": ["a", "b", "c"], "dv": ["x"] * 3, "n": [20, 20, 20],
        "mean": [1.0, 2.0, 3.0], "sd": [1.0, 1.0, 1.0],
    })
    out = msa.leave_one_out_sensitivity(summary)
    assert set(out["pooling_method"]) == {"fixed_effects_k_lt_3"}
    assert set(out["k_remaining"]) == {2}


# ── LLM fallback and its benchmark ──────────────────────────────────────────

def test_prompt_does_not_show_the_queried_alias():
    from opendv.llm_utils import _format_candidate_reference

    shown = _format_candidate_reference(["task_completion_time"], query="time_spent")
    assert "time_spent" not in shown
    assert "time_spent" in _format_candidate_reference(["task_completion_time"])


def test_range_tolerance_applies_to_the_canonical_range_only():
    from opendv.llm_plausibility import range_is_compatible

    assert not range_is_compatible((-2.61, 1.84), (-2.0, 2.0))   # accelerations in g as AOA1
    assert range_is_compatible((1.0, 7.0), (-3.0, 3.0))          # a 1-7 recording of a -3..3 item


def test_timing_rule_needs_the_question_code():
    from opendv.llm_plausibility import name_rejection_reason

    assert name_rejection_reason("reactionTime", ["lastpage", "reactionCorrect"]) is None
    assert name_rejection_reason("waitTime", ["lastpage", "waitTimeTime"]) is None
    assert name_rejection_reason("SUSTime", ["lastpage", "SUS[SQ001]"]) == "survey_timing"


def test_header_with_a_unit_is_not_matched_by_its_cleaned_name():
    from opendv.header_matching import match_headers

    mapping = {"task_completion_time": "task_completion_time", "mental_demand": "mental_demand"}
    columns = ["Task Completion Time (ms)", "Mental Demand", "Mental Demand (0 = Low, 100 = High)"]
    matches = match_headers(columns, columns, mapping, instruments=(), encoder=lambda texts: None)
    assert "Task Completion Time (ms)" not in matches
    assert matches["Mental Demand"].dv == "mental_demand"
    assert matches["Mental Demand (0 = Low, 100 = High)"].dv == "mental_demand"


# ── Single-file conversion ──────────────────────────────────────────────────

def test_reshape_leaves_tables_whose_other_columns_vary_within_participants():
    from opendv.reshape_utils import auto_reshape_to_wide

    wide = pd.DataFrame({
        "participant": [1, 1, 2, 2, 3, 3, 4, 4],
        "condition": ["A", "B"] * 4,
        "sus": [70, 60, 75, 65, 80, 50, 72, 68],
        "tlx": [30, 40, 35, 45, 20, 50, 33, 41],
    })
    with pytest.warns(UserWarning):
        assert auto_reshape_to_wide(wide).equals(wide)
    long = pd.DataFrame({
        "participant": [1, 1, 1, 1, 2, 2, 2, 2],
        "condition": ["A", "A", "B", "B"] * 2,
        "measure": ["sus", "tlx"] * 4,
        "value": [70, 30, 60, 40, 75, 35, 65, 45],
    })
    out = auto_reshape_to_wide(long)
    # Either left as is, or pivoted on the measure with the condition kept apart;
    # never SUS and TLX averaged into one condition column.
    assert out.equals(long) or {"sus", "tlx"} <= set(out.columns)
