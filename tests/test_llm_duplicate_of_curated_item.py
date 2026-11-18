"""An LLM-mapped column of a questionnaire item that a table already has a
curated column of is set aside before scoring, and the curated column is scored.

The batch names a table's columns of one DV in raw column order: the first
gets the bare name, the others ``__dup_N``. Longitudinal US/DE's web files list
``com_eval_sq004`` (a 1-7 rating of the eHMI's clarity, LLM-mapped to
``perceived_safety_item_04``) before their curated ``perceived_safety_sq004``,
which became ``perceived_safety_item_04__dup_2``. Registry scores read items by
their bare name only, so once the LLM column was set aside (curated level of the
relations, ``llm_columns="prefer_curated"`` or ``"exclude"``) the files had no
item 4 and no perceived-safety score: 29 of the study's 55 participants were
missing from the curated level until 2026-11-16.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opendv.analysis import construct_relations as cr
from opendv.analysis import multi_study_analysis as msa

STUDY_MAPPING = "schemas/s_mapping.yaml"
LLM = "llm_deduction"
ITEMS = [f"perceived_safety_item_0{i}" for i in range(1, 5)]


def test_freed_bare_name_goes_to_the_first_remaining_column_of_its_dv():
    columns = ["pid", "perceived_safety_item_01", "perceived_safety_item_04__dup_2",
               "perceived_safety_item_04__dup_3", "trust_rating__dup_2", "trust_rating__item_1"]
    renames = msa.reclaim_freed_names(columns, ["perceived_safety_item_04"])
    assert renames == {"perceived_safety_item_04__dup_2": "perceived_safety_item_04"}
    # Nothing freed: a removed __dup_N column, or a bare name still present.
    assert msa.reclaim_freed_names(columns, ["perceived_safety_item_04__dup_9"]) == {}
    assert msa.reclaim_freed_names(["x", "x__dup_2"], ["x"]) == {}
    # Composite inputs (__item_N) are never renamed.
    assert msa.reclaim_freed_names(["trust_rating__item_1"], ["trust_rating"]) == {}


def _web_and_vr_tables(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The VR table: the four curated items. The web table: an LLM column of
    item 4 (1-7) listed before the curated items (-3..+3), as in Longitudinal."""
    vr = pd.DataFrame({"pid": np.arange(20), **{i: rng.integers(-3, 4, 20).astype(float) for i in ITEMS}})
    web = pd.DataFrame({"pid": np.arange(100, 130)})
    web["perceived_safety_item_04"] = rng.integers(1, 8, 30).astype(float)         # LLM: com_eval_sq004
    for item in ITEMS[:3]:
        web[item] = rng.integers(-3, 4, 30).astype(float)
    web["perceived_safety_item_04__dup_2"] = rng.integers(-3, 4, 30).astype(float)  # curated item 4
    return vr, web


def _expected_score(vr: pd.DataFrame, web: pd.DataFrame) -> list[float]:
    curated_web = web.drop(columns="perceived_safety_item_04").rename(
        columns={"perceived_safety_item_04__dup_2": "perceived_safety_item_04"})
    return pd.concat([vr[ITEMS], curated_web[ITEMS]]).mean(axis=1).tolist()


def _write_table(study_dir: Path, name: str, df: pd.DataFrame, llm_columns: set[str], meta_rows: list[dict]) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(study_dir / f"{name}-standardized.csv", index=False)
    (study_dir / f"{name}-quality.json").write_text(
        json.dumps({"standardized_file": f"{name}-standardized.csv", "dataset_id": f"{name}.xlsx"}),
        encoding="utf-8",
    )
    for column in df.columns:
        meta_rows.append({
            "source_id": "s", "dataset_id": f"{name}.xlsx", "canonical_dv": msa._column_base(column),
            "original_alias": column, "item_of": "",
            "mapping_method": "llm" if column in llm_columns else "mapping",
            "mapping_source": LLM if column in llm_columns else STUDY_MAPPING,
        })


@pytest.fixture
def longitudinal_like(tmp_path):
    vr, web = _web_and_vr_tables(np.random.default_rng(20261116))
    meta_rows: list[dict] = []
    root = tmp_path / "standardized"
    _write_table(root / "s", "data_main", vr, set(), meta_rows)
    _write_table(root / "s", "web", web, {"perceived_safety_item_04"}, meta_rows)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(tmp_path / "meta_view.csv", index=False)
    return root, meta, vr, web


def test_pooled_means_score_the_curated_item_after_the_llm_column_is_set_aside(longitudinal_like, tmp_path):
    root, _meta, vr, web = longitudinal_like
    provenance = msa.load_column_provenance(tmp_path / "meta_view.csv")
    for mode in ("prefer_curated", "exclude"):
        studies = msa.load_studies(root, column_provenance=provenance, llm_columns=mode)
        frame = studies["s"]
        assert frame["perceived_safety_score"].tolist() == pytest.approx(_expected_score(vr, web))
        assert studies.column_provenance["s"]["perceived_safety_score"] == "repo_mapping"
        assert "perceived_safety_item_04__dup_2" not in frame.columns
        assert [(row["dataset_id"], row["column"]) for row in studies.set_aside] == [
            ("web.xlsx", "perceived_safety_item_04")]


@pytest.mark.parametrize("level", ["curated", "llm"])
def test_relations_levels_score_the_curated_item_of_every_table(longitudinal_like, level):
    root, meta, vr, web = longitudinal_like
    provenance = cr.column_provenance(root, meta)
    frames = cr.level_frames(root, provenance, level)
    frame = frames["s"]
    assert frame["perceived_safety_score"].tolist() == pytest.approx(_expected_score(vr, web))
    assert frames.column_levels["s"]["perceived_safety_score"] == cr.LEVEL_RANK["curated"]
    assert frames.column_levels["s"]["perceived_safety_item_04"] == cr.LEVEL_RANK["curated"]
    matrix = cr.dv_matrix(frame, level)
    assert matrix["perceived_safety"].notna().sum() == len(vr) + len(web)
    if level == "llm":
        (row,) = frames.set_aside
        assert (row["dataset_id"], row["column"], row["raw_column"], row["kept_instead"]) == (
            "web.xlsx", "perceived_safety_item_04", "perceived_safety_item_04", "same and other tables")
    else:
        assert frames.set_aside == []
