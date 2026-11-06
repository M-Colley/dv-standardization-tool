"""Results must not depend on where the batch wrote its output.

The batch replaces part of a standardized file's name with a hash when the
full output path would pass ``WINDOWS_PATH_SOFT_LIMIT`` characters. Until
2026-11-06 participant ids declared with ``participant_from_file`` were read
from that name, so a checkout in a long folder (e.g. ``.claude/worktrees/<name>``)
lost eHMI Optimization's ids and changed its participant-level results
without a warning. They are now read from the table's path in its source,
and files are read in the order of their unshortened names.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from opendv import run_batch_standardization as rbs
from opendv._paths import SCHEMAS_DIR
from opendv.analysis import contrast_meta_analysis as cma
from opendv.analysis import multi_study_analysis as msa

SCHEMA = SCHEMAS_DIR / "standard_dv_mapping.yaml"
EHMI = "ehmi_optimization_chi25"
OLD_FILE_NAME_PATTERN = r"_u_(\d+)_"   # the pattern written against standardized file names


def _shipped_pattern() -> str:
    pattern = cma.load_study_designs().studies[EHMI].participant_from_file
    assert pattern is not None
    return pattern


def _write_source(root):
    """Per-participant folders laid out like eHMI Optimization's repository:
    a log that carries the id and a questionnaire that only its folder names."""
    rng = np.random.default_rng(7)
    for pid in (10, 11, 12):
        folder = root / "participantdata_round_one" / f"u_{pid}"
        (folder / f"QuestionnaireData_{pid}").mkdir(parents=True)
        pd.DataFrame({"User_ID": [pid] * 4, "task_completion_time": rng.normal(10, 1, 4).round(3)}).to_csv(
            folder / "ObservationsPerEvaluation.csv", index=False)
        pd.DataFrame({"mental_demand": rng.integers(1, 21, 4)}).to_csv(
            folder / f"QuestionnaireData_{pid}" / "QuestionnaireResponses.csv", index=False)


def _run(tmp_path, name, monkeypatch, shorten: bool):
    data = tmp_path / "data"
    if not data.exists():
        _write_source(data)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"sources": [{
        "source_id": "st", "source_type": "local_path", "location": str(data), "include_globs": ["**/*.csv"],
    }]}), encoding="utf-8")
    with monkeypatch.context() as patch:
        if shorten:
            # Leaves only MIN_ARTIFACT_PREFIX_LENGTH characters for every name,
            # as an output folder of about 170 characters does.
            patch.setattr(rbs, "WINDOWS_PATH_SOFT_LIMIT", 0)
        rbs.run_batch(manifest, tmp_path / name, SCHEMA, llm_deduction_enabled=False, fair_export=False)
    return tmp_path / name / "standardized"


@pytest.fixture
def two_runs(tmp_path, monkeypatch):
    short = _run(tmp_path, "short", monkeypatch, shorten=False)
    long = _run(tmp_path, "long", monkeypatch, shorten=True)
    return short, long


def test_shortened_names_lose_the_id_but_participants_resolve(two_runs, monkeypatch):
    short, long = two_runs
    # The precondition of the defect: the shortened questionnaire names no
    # longer contain the participant's folder, the short ones do.
    short_names = {p.name for p in (short / "st").glob("*-standardized.csv")}
    long_names = {p.name for p in (long / "st").glob("*-standardized.csv")}
    assert len(short_names) == len(long_names) == 6
    assert all(re.search(OLD_FILE_NAME_PATTERN, n) for n in short_names)
    assert not any(re.search(OLD_FILE_NAME_PATTERN, n) for n in long_names)
    # ... and sort in another order.
    def by_name(root, names):
        ids = msa.dataset_ids_by_file(root / "st")
        return [ids[n] for n in sorted(names)]
    assert by_name(short, short_names) != by_name(long, long_names)

    # The same rows in the same order, so even the last digits of sums agree.
    frames = {label: msa.load_studies(root)["st"] for label, root in (("short", short), ("long", long))}
    pd.testing.assert_frame_equal(frames["short"], frames["long"])

    pattern = _shipped_pattern()
    designs = {
        "study": cma.StudyDesign("st", ("User_ID",), participant_from_file=pattern),
        "file": cma.StudyDesign("st", ("User_ID",), participant_from_file=pattern, participant_scope="file"),
    }
    ids = {scope: design.participant_series(frames["long"]) for scope, design in designs.items()}
    assert ids["study"].notna().all() and ids["file"].notna().all()
    assert sorted(set(ids["study"])) == ["10", "11", "12"]
    # File-scoped ids are qualified by the source path, the same for both runs.
    assert "participantdata_round_one/u_10/QuestionnaireData_10/QuestionnaireResponses.csv::10" in set(ids["file"])

    # Participant-level unit (``--repeated-measures``): same participants, same means.
    registry = cma.DesignRegistry("t", {}, {"st": designs["study"]})
    monkeypatch.setattr(msa, "_study_designs", lambda: registry)
    means = {label: msa.load_studies(root, repeated_measures_studies={"st"})["st"]
             for label, root in (("short", short), ("long", long))}
    assert len(means["short"]) == len(means["long"]) == 3
    pd.testing.assert_frame_equal(means["short"], means["long"])
    assert means["long"]["mental_demand"].notna().all()


def test_source_file_is_the_dataset_id_recorded_by_the_batch(two_runs):
    _, long = two_runs
    frame = msa.load_studies(long)["st"]
    meta = pd.read_csv(long.parent / "meta_view.csv")
    assert set(frame[msa.SOURCE_FILE_COLUMN]) == set(meta["dataset_id"])


def test_relations_frames_do_not_depend_on_the_output_path(two_runs):
    from opendv.analysis import construct_relations as cr

    frames = {}
    for root in two_runs:
        provenance = cr.column_provenance(root, pd.read_csv(root.parent / "meta_view.csv"))
        frames[root] = cr.level_frames(root, provenance, "automatic")["st"]
    short, long = (frames[root] for root in two_runs)
    pd.testing.assert_frame_equal(short, long)
    design = cma.StudyDesign("st", ("User_ID",), participant_from_file=_shipped_pattern())
    assert sorted(set(design.participant_series(long))) == ["10", "11", "12"]


def test_unshortened_name_is_the_name_of_a_short_run(two_runs):
    short, long = two_runs
    short_ids = {v: k for k, v in msa.dataset_ids_by_file(short / "st").items()}
    for name, dataset_id in msa.dataset_ids_by_file(long / "st").items():
        assert msa.unshortened_file_name(name, dataset_id) == short_ids[dataset_id]
    assert msa.unshortened_file_name("notes.csv", None) == "notes.csv"


def test_shipped_ehmi_pattern_reads_the_source_folder():
    design = cma.load_study_designs().studies[EHMI]
    frame = pd.DataFrame({
        "User_ID": [np.nan, 12.0],
        msa.SOURCE_FILE_COLUMN: [
            "eHMI-bo-participantdata/u_10/QuestionnaireData_10/QuestionnaireResponses.csv",
            "eHMI-bo-participantdata/u_12/ObservationsPerEvaluation.csv",
        ],
    })
    assert design.participant_series(frame).tolist() == ["10", "12"]


def test_a_file_without_a_sidecar_keeps_its_name(tmp_path):
    study = tmp_path / "st"
    study.mkdir()
    pd.DataFrame({"mental_demand": [1.0, 2.0]}).to_csv(study / "x_u_7_answers.csv", index=False)
    frame = msa.load_studies(tmp_path)["st"]
    assert frame[msa.SOURCE_FILE_COLUMN].tolist() == ["x_u_7_answers.csv"] * 2


def test_a_pattern_that_matches_no_file_is_reported(caplog):
    frame = pd.DataFrame({"User_ID": [np.nan, 3.0], msa.SOURCE_FILE_COLUMN: ["a/answers.csv", "a/log.csv"]})
    design = cma.StudyDesign("unmatched_pattern_study", ("User_ID",), participant_from_file=r"(?:^|/)u_(\d+)/")
    with caplog.at_level(logging.WARNING, logger=cma.logger.name):
        series = design.participant_series(frame)
    assert series.tolist()[1] == "3" and pd.isna(series.tolist()[0])
    assert "matches the path of none" in caplog.text
