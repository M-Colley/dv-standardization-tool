"""The OpenDV corpus: its manifest, study table and the value measures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from analyses.opendv_corpus import build_manifest, studies, value

HERE = Path(build_manifest.__file__).resolve().parent


def test_manifest_unites_the_three_sets_and_is_up_to_date():
    sources = build_manifest.build()
    assert len(sources) == 88
    assert len({s["source_id"] for s in sources}) == 88
    counts = pd.Series([s["study_set"] for s in sources]).value_counts().to_dict()
    assert counts == {"development": 39, "heldout_2": 30, "heldout_1": 19}
    committed = yaml.safe_load((HERE / "sources_manifest.yaml").read_text(encoding="utf-8"))["sources"]
    assert committed == sources, "run python -m analyses.opendv_corpus.build_manifest"


def test_study_table_matches_the_manifest():
    table = pd.read_csv(HERE / "corpus_studies.csv")
    assert set(table["source_id"]) == {s["source_id"] for s in build_manifest.build()}
    assert table.equals(studies.build()), "run python -m analyses.opendv_corpus.studies"
    assert (table["venue_group"] == "sigchi").sum() == 21


def test_venue_groups():
    assert studies.venue_group("CHI 2024", "10.1145/3613904.3642298") == "sigchi"
    assert studies.venue_group("Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems 2023",
                               "10.1145/3544548.3580778") == "sigchi"
    assert studies.venue_group("2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI) 2025",
                               "10.1109/hri61500.2025.10974182") == "sigchi"
    assert studies.venue_group("ACM Transactions on Human-Robot Interaction 2025", "10.1145/3727989") == "acm_other"
    assert studies.venue_group("Mensch und Computer 2023 2023", "10.1145/3603555.3608534") == "acm_other"
    assert studies.venue_group("IEEE Transactions on Visualization and Computer Graphics 2025",
                               "10.1109/TVCG.2024.3470214") == "ieee"
    assert studies.venue_group("Applied Ergonomics 2021", "10.1016/j.apergo.2021.103428") == "journal_other"
    assert studies.venue_group(None, None) == "no_paper"


def _meta_view() -> pd.DataFrame:
    rows = [
        # study, raw name, DV, matched by the shared schema?
        ("s1", "Mental Demand", "mental_demand", True),
        ("s1", "A1", "AOA1", False),
        ("s2", "mental_demand", "mental_demand", True),
        ("s2", "A1", "efficiency", False),
        ("s3", "TLX1", "mental_demand", False),
        ("s3", "SUS", "usability", True),
    ]
    return pd.DataFrame({
        "source_id": [r[0] for r in rows], "original_alias": [r[1] for r in rows],
        "canonical_dv": [r[2] for r in rows], "mapping_domain": "dv",
        "mapping_source": ["x/schemas/standard_dv_mapping.yaml" if r[3] else "x/schemas/s_mapping.yaml"
                           for r in rows],
    })


def test_poolability_counts_shared_measures_per_identifier():
    dv = value.dv_columns(_meta_view())
    table = pd.DataFrame({"source_id": ["s1", "s2", "s3"], "venue_group": ["sigchi", "sigchi", "ieee"]})
    pool = value.poolability(dv, table).set_index(["subset", "strategy"])
    # raw names: 'A1' is shared (s1, s2) but stands for two DVs; nothing reaches 3 studies
    assert pool.loc[("all", "raw_label"), "measures_in_2plus_studies"] == 1
    assert pool.loc[("all", "normalized_label"), "measures_in_2plus_studies"] == 2  # a1, mentaldemand
    # OpenDV: mental_demand is in all three studies
    assert pool.loc[("all", "opendv"), "measures_in_3plus_studies"] == 1
    assert pool.loc[("all", "opendv"), "share_of_pairs_sharing"] == 1.0
    assert pool.loc[("sigchi", "opendv"), "studies"] == 2


def test_label_ambiguity_finds_names_with_two_meanings():
    amb = value.label_ambiguity(value.dv_columns(_meta_view())).set_index("normalized_label")
    assert amb.loc["a1", "n_dvs"] == 2 and amb.loc["a1", "dvs"] == "AOA1; efficiency"
    assert amb.loc["mentaldemand", "n_dvs"] == 1


def test_harmonization_counts_raw_names_per_dv():
    harm = value.harmonization_by_dv(value.dv_columns(_meta_view())).set_index("dv")
    assert harm.loc["mental_demand", "studies"] == 3
    assert harm.loc["mental_demand", "distinct_raw_labels"] == 2  # mentaldemand, tlx1
    assert harm.loc["mental_demand", "share_matched_by_study_mappings"] == round(1 / 3, 3)


def test_corrected_mappings_are_up_to_date_and_keep_the_frozen_file():
    from analyses.opendv_corpus import corrections

    repo = HERE.parents[1]
    for sid, spec in corrections.CORRECTIONS.items():
        copy = corrections.corrected_path(sid).read_text(encoding="utf-8")
        assert copy == corrections.corrected_text(sid), "run python -m analyses.opendv_corpus.corrections"
        frozen = (repo / spec["frozen"]).read_text(encoding="utf-8").splitlines()
        relocked = set(spec.get("relock") or {})
        kept = set(copy.splitlines())
        for line in frozen:
            if line.split(":", 1)[0].strip() in relocked and not line.lstrip().startswith("#"):
                continue
            assert line in kept, (sid, line)


def test_corpus_manifest_points_at_corrected_copies():
    from analyses.opendv_corpus import corrections

    by_id = {s["source_id"]: s for s in build_manifest.build()}
    for sid, spec in corrections.CORRECTIONS.items():
        assert by_id[sid]["mapping_path"].startswith("analyses/opendv_corpus/mappings/")
        assert by_id[sid]["mapping_corrected_from"] == spec["frozen"]
