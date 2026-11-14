"""LLM-mapped columns are left out column by column, not study × DV by study × DV.

Until 2026-11-14 one LLM-mapped column marked its whole study × DV as
``llm_deduced`` (``load_mapping_provenance``), so the LLM-excluded analysis
dropped curated columns that shared a DV with it (e.g. SenseCobot's NASA-TLX
mental demand next to its EEG ratios), and an LLM column with more rows
displaced a curated column or a derived score when the columns of a DV were
folded into one (Curved Dragging's NASA-TLX total before 2026-11-14).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opendv.analysis import multi_study_analysis as msa
from opendv.analysis import multiverse_analysis

SCHEMA = "schemas/standard_dv_mapping.yaml"
STUDY_MAPPING = "schemas/s_mapping.yaml"
LLM = "llm_deduction"
TLX = ["mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration"]


def _write_table(study_dir: Path, name: str, dataset_id: str, df: pd.DataFrame, sources: dict[str, str],
                 meta_rows: list[dict], source_id: str, item_of: dict[str, str] | None = None) -> None:
    """Write a standardized table, its quality sidecar and its meta_view rows (in column order)."""
    study_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{name}-standardized.csv"
    df.to_csv(study_dir / file_name, index=False)
    (study_dir / f"{name}-quality.json").write_text(
        json.dumps({"standardized_file": file_name, "dataset_id": dataset_id}), encoding="utf-8"
    )
    item_of = item_of or {}
    for column in df.columns:
        base = msa._column_base(column)
        meta_rows.append({
            "source_id": source_id,
            "dataset_id": dataset_id,
            "canonical_dv": base,
            "original_alias": column,
            "mapping_source": sources.get(column, SCHEMA),
            "item_of": item_of.get(column, ""),
        })


def _tlx_frame(rng: np.random.Generator, n: int, pid_offset: int = 0, items: list[str] = TLX) -> pd.DataFrame:
    """NASA-TLX subscales (all six by default) on 0-20 for n participants."""
    data = {"participant_id": np.arange(pid_offset, pid_offset + n)}
    for name in items:
        data[name] = np.clip(np.round(rng.normal(10, 3, n)), 0, 20)
    return pd.DataFrame(data)


# ── Column provenance ───────────────────────────────────────────────────────

def test_meta_view_rows_give_each_column_its_own_category(tmp_path):
    meta_rows: list[dict] = []
    df = pd.DataFrame({
        "pid": [1, 2],
        "mental_demand": [5.0, 6.0],
        "mental_demand__dup_2": [0.1, 0.2],
        "trust_rating": [3.0, 4.0],
        "trust_rating__item_1": [3.0, 4.0],
        "trust_rating__item_2": [3.0, 4.0],
    })
    _write_table(tmp_path / "standardized" / "s", "a", "raw/a.csv", df,
                 {"mental_demand__dup_2": LLM, "trust_rating__item_2": LLM}, meta_rows, "s",
                 item_of={"trust_rating__item_1": "trust_rating", "trust_rating__item_2": "trust_rating"})
    pd.DataFrame(meta_rows).to_csv(tmp_path / "meta_view.csv", index=False)

    provenance = msa.load_column_provenance(tmp_path / "meta_view.csv")
    categories = msa._file_column_categories([*df.columns, msa.SOURCE_FILE_COLUMN], provenance[("s", "raw/a.csv")])
    assert categories["mental_demand"] == "schema"
    assert categories["mental_demand__dup_2"] == "llm_deduced"
    # A composite built from an LLM-mapped item is LLM-derived.
    assert categories["trust_rating"] == "llm_deduced"
    assert msa.SOURCE_FILE_COLUMN not in categories
    # The study × DV view merges them: one LLM column marks the whole DV.
    assert msa.load_mapping_provenance(tmp_path / "meta_view.csv")[("s", "mental_demand")] == "llm_deduced"


def test_rows_that_do_not_line_up_are_matched_by_canonical_name(caplog):
    entries = [("mental_demand", None, "schema"), ("mental_demand", None, "llm_deduced")]
    with caplog.at_level("WARNING"):
        categories = msa._file_column_categories(["mental_demand", "effort"], entries)
    assert "do not match" in caplog.text
    assert categories == {"mental_demand": "llm_deduced", "effort": "unknown"}


# ── Curated columns stay ────────────────────────────────────────────────────

SUBSCALES = ["mental_demand", "effort", "frustration"]


def _mixed_study_input(tmp_path: Path) -> Path:
    """Study ``s``: three curated NASA-TLX subscales in one table, EEG ratios
    LLM-mapped to mental demand in another, and an LLM-only trust column.
    Study ``t``: the same curated subscales and a curated trust column. (Not
    all six subscales, so the causal step does not replace them by the total.)"""
    rng = np.random.default_rng(20261114)
    root = tmp_path / "standardized"
    meta_rows: list[dict] = []

    s_tlx = _tlx_frame(rng, 60, items=SUBSCALES)
    s_tlx["group"] = np.where(s_tlx["participant_id"] % 2 == 0, "control", "display")
    s_tlx.loc[s_tlx["group"] == "display", "mental_demand"] -= 3
    s_tlx["trust_rating"] = rng.normal(4, 1, 60)
    _write_table(root / "s", "labels", "S/Labels.csv", s_tlx,
                 {**{c: STUDY_MAPPING for c in SUBSCALES}, "trust_rating": LLM}, meta_rows, "s")
    eeg = pd.DataFrame({"participant_id": np.arange(60).repeat(2), "mental_demand": rng.normal(0.4, 0.1, 120)})
    _write_table(root / "s", "fusion", "S/Fusion.csv", eeg, {"mental_demand": LLM}, meta_rows, "s")

    t_tlx = _tlx_frame(rng, 60, pid_offset=100, items=SUBSCALES)
    t_tlx["group"] = np.where(t_tlx["participant_id"] % 2 == 0, "control", "display")
    t_tlx.loc[t_tlx["group"] == "display", "mental_demand"] -= 3
    t_tlx["trust_rating"] = rng.normal(4, 1, 60)
    _write_table(root / "t", "survey", "T/survey.csv", t_tlx, {}, meta_rows, "t")

    pd.DataFrame(meta_rows).to_csv(tmp_path / "meta_view.csv", index=False)
    (tmp_path / "designs.yaml").write_text(
        """
version: "test"
canonical_factors:
  ehmi_presence:
    levels: [absent, present]
    reference: absent
studies:
  s:
    participant: participant_id
    factors:
      - {name: display, canonical: ehmi_presence, design: between, columns: {group: null},
         levels: {control: absent, display: present}}
  t:
    participant: participant_id
    factors:
      - {name: display, canonical: ehmi_presence, design: between, columns: {group: null},
         levels: {control: absent, display: present}}
""",
        encoding="utf-8",
    )
    return root


def test_llm_excluded_analysis_keeps_a_curated_column_that_shares_its_dv(tmp_path):
    root = _mixed_study_input(tmp_path)
    out = tmp_path / "out"
    msa.main([
        "--input-dir", str(root), "--output-dir", str(out), "--exclude-llm-deduced",
        "--no-measurement-invariance", "--study-designs", str(tmp_path / "designs.yaml"),
        "--causal-min-rows", "50",
    ])
    curated = pd.read_csv(root / "s" / "labels-standardized.csv")["mental_demand"]

    clean = pd.read_csv(out / "harmonized_dv_summary_llm_excluded.csv").set_index(["study", "dv"])
    row = clean.loc[("s", "mental_demand")]
    assert row["n"] == 60
    assert row["mean"] == pytest.approx(curated.mean())
    assert row["mapping_source"] == "repo_mapping"
    # The LLM-only DV is gone; its curated twin in t is not.
    assert ("s", "trust_rating") not in clean.index
    assert ("t", "trust_rating") in clean.index
    meta = pd.read_csv(out / "meta_analysis_summary_llm_excluded.csv").set_index("dv")
    assert meta.loc["mental_demand", "k_studies"] == 2
    assert not meta["includes_llm_deduced"].any()

    # With all mappings the curated column is preferred and the EEG ratios set aside.
    everything = pd.read_csv(out / "harmonized_dv_summary.csv").set_index(["study", "dv"])
    assert everything.loc[("s", "mental_demand"), "mean"] == pytest.approx(curated.mean())
    assert everything.loc[("s", "mental_demand"), "mapping_source"] == "repo_mapping"
    assert everything.loc[("s", "trust_rating"), "mapping_source"] == "llm_deduced"
    set_aside = pd.read_csv(out / "llm_columns_set_aside.csv")
    assert set_aside[["study", "dataset_id", "column", "dv"]].values.tolist() == [
        ["s", "S/Fusion.csv", "mental_demand", "mental_demand"]
    ]

    # Contrasts: s's curated mental demand joins the LLM-excluded pool.
    pooled = pd.read_csv(out / "contrast_meta_analysis_llm_excluded.csv").set_index("dv")
    assert pooled.loc["mental_demand", "k_studies"] == 2
    assert "trust_rating" not in pooled.index
    effects = pd.read_csv(out / "contrast_effects.csv")
    assert set(effects.loc[effects["dv"] == "mental_demand", "mapping_source"]) == {"repo_mapping", "schema"}

    # Causal discovery reads the studies without LLM-mapped columns: s's
    # curated mental demand is shared with t, its LLM-only trust column is not.
    causal = json.loads((out / "analysis_results.json").read_text(encoding="utf-8"))["causal_discovery"]
    assert causal is not None
    assert "mental_demand" in causal["dvs_used"]
    assert "trust_rating" not in causal["dvs_used"]
    assert sorted(causal["studies_used"]) == ["s", "t"]


def test_multiverse_without_llm_keeps_curated_columns(tmp_path):
    root = _mixed_study_input(tmp_path)
    specs = multiverse_analysis.run(root, tmp_path / "meta_view.csv")
    no_llm = specs[(specs["mappings"] == "no_llm") & (specs["dv"] == "mental_demand")]
    assert set(no_llm["k_studies"]) == {2}
    assert not no_llm["includes_llm_deduced"].any()
    trust = specs[specs["dv"] == "trust_rating"]
    assert set(trust["mappings"]) == {"all"}


def test_llm_duplicate_in_the_same_table_is_not_averaged_in(tmp_path):
    meta_rows: list[dict] = []
    df = pd.DataFrame({"pid": [1, 2, 3], "trust_rating": [3.0, 4.0, 5.0], "trust_rating__dup_2": [90.0, 95.0, 99.0]})
    _write_table(tmp_path / "standardized" / "s", "a", "raw/a.csv", df,
                 {"trust_rating__dup_2": LLM}, meta_rows, "s")
    pd.DataFrame(meta_rows).to_csv(tmp_path / "meta_view.csv", index=False)
    provenance = msa.load_column_provenance(tmp_path / "meta_view.csv")
    root = tmp_path / "standardized"

    for mode in ("prefer_curated", "exclude"):
        studies = msa.load_studies(root, column_provenance=provenance, llm_columns=mode)
        canonical = msa._canonicalize_studies(studies)
        assert canonical["s"]["trust_rating"].tolist() == [3.0, 4.0, 5.0]
        assert canonical.mapping_sources[("s", "trust_rating")] == "schema"
    # Without provenance the two are averaged, as before.
    assert msa._canonicalize_studies(msa.load_studies(root))["s"]["trust_rating"].tolist() == [46.5, 49.5, 52.0]


# ── Derived scores ──────────────────────────────────────────────────────────

def _score_vs_llm_column_input(tmp_path: Path) -> tuple[Path, msa.ColumnProvenance]:
    """Curated subscales (40 rows) in one table and an LLM column read as the
    NASA-TLX total, with more rows (120), in another."""
    rng = np.random.default_rng(7)
    meta_rows: list[dict] = []
    root = tmp_path / "standardized"
    _write_table(root / "s", "tlx", "exp1/tlx.csv", _tlx_frame(rng, 40), {}, meta_rows, "s")
    width = pd.DataFrame({"pid": np.arange(120), "TLX_SCORE": rng.normal(5, 1, 120)})
    _write_table(root / "s", "exp2", "exp2/targets.csv", width, {"TLX_SCORE": LLM}, meta_rows, "s")
    pd.DataFrame(meta_rows).to_csv(tmp_path / "meta_view.csv", index=False)
    return root, msa.load_column_provenance(tmp_path / "meta_view.csv")



def test_derived_score_is_not_displaced_by_an_llm_column_with_more_rows(tmp_path):
    root, provenance = _score_vs_llm_column_input(tmp_path)
    for mode in ("prefer_curated", "exclude"):
        studies = msa.load_studies(root, column_provenance=provenance, llm_columns=mode)
        canonical = msa._canonicalize_studies(studies)
        score = canonical["s"]["TLX_SCORE"].dropna()
        assert len(score) == 40
        assert score.tolist() == pytest.approx(studies["s"]["nasa_tlx_score"].dropna().tolist())
        assert canonical.mapping_sources[("s", "TLX_SCORE")] == "schema"
        summary = msa.harmonized_summary(studies, canonical_studies=canonical).set_index(["study", "dv"])
        assert summary.loc[("s", "TLX_SCORE"), "n"] == 40
        assert summary.loc[("s", "TLX_SCORE"), "mapping_source"] == "schema"
    # The LLM column is kept with all mappings (no curated column of that
    # name), but the score computed from curated items stands for the DV.
    assert "TLX_SCORE" in msa.load_studies(root, column_provenance=provenance)["s"].columns
    # Without provenance the column with more rows wins, as before.
    assert msa._canonicalize_studies(msa.load_studies(root))["s"]["TLX_SCORE"].notna().sum() == 120


def test_a_score_with_an_llm_item_is_llm_derived_and_left_out_without_llm(tmp_path):
    rng = np.random.default_rng(3)
    meta_rows: list[dict] = []
    root = tmp_path / "standardized"
    _write_table(root / "s", "tlx", "tlx.csv", _tlx_frame(rng, 30), {"frustration": LLM}, meta_rows, "s")
    pd.DataFrame(meta_rows).to_csv(tmp_path / "meta_view.csv", index=False)
    provenance = msa.load_column_provenance(tmp_path / "meta_view.csv")

    everything = msa.load_studies(root, column_provenance=provenance)
    assert everything.column_provenance["s"]["nasa_tlx_score"] == "llm_deduced"
    assert msa._canonicalize_studies(everything).mapping_sources[("s", "TLX_SCORE")] == "llm_deduced"
    clean = msa._canonicalize_studies(msa.load_studies(root, column_provenance=provenance, llm_columns="exclude"))
    assert "TLX_SCORE" not in clean["s"].columns
    assert "frustration" not in clean["s"].columns
    assert "mental_demand" in clean["s"].columns


def test_design_and_metadata_columns_are_never_left_out(tmp_path):
    meta_rows: list[dict] = []
    df = pd.DataFrame({"pid": [1, 2], "meta_condition_speed": [10, 20], "mental_demand": [5.0, 6.0]})
    _write_table(tmp_path / "standardized" / "s", "a", "a.csv", df,
                 {"meta_condition_speed": LLM, "mental_demand": LLM}, meta_rows, "s")
    pd.DataFrame(meta_rows).to_csv(tmp_path / "meta_view.csv", index=False)
    provenance = msa.load_column_provenance(tmp_path / "meta_view.csv")
    frame = msa.load_studies(tmp_path / "standardized", column_provenance=provenance, llm_columns="exclude")["s"]
    assert "meta_condition_speed" in frame.columns
    assert "mental_demand" not in frame.columns


def test_variants_read_the_same_rows(tmp_path):
    root = _mixed_study_input(tmp_path)
    provenance = msa.load_column_provenance(tmp_path / "meta_view.csv")
    loaded = msa.load_study_variants(
        root,
        {"all": {}, "clean": {"llm_columns": "exclude"}, "means": {"repeated_measures_studies": True}},
        column_provenance=provenance,
    )
    assert loaded["all"]["s"].index.equals(loaded["clean"]["s"].index)
    assert len(loaded["means"]["s"]) == 60
    with pytest.raises(ValueError, match="llm_columns"):
        msa.load_study_variants(root, {"x": {"llm_columns": "drop"}})
