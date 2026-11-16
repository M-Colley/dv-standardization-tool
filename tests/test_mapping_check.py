"""opendv check-mapping: what it flags in a study mapping before the batch runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from opendv.mapping_check import check_mapping, load_vocabulary, main


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "study_mapping.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _messages(report, level):
    return {f["alias"]: f["message"] for f in report[level]}


def test_valid_targets_raise_nothing(tmp_path, vocab):
    report = check_mapping(_write(tmp_path, (
        "TLX_md: mental_demand\n"
        "q_sus1: SUS1\n"
        "ueqs1: ueq_s_1\n"
        "pid: meta_participant_id\n"
        "reverse_items: [{columns: [q2], scale_range: [1, 5], reason: negatively worded}]\n"
    )), vocabulary=vocab)
    assert report["errors"] == [] and report["warnings"] == []
    assert report["to_dvs"] == 2 and report["to_instrument_items"] == 1 and report["to_meta"] == 1
    assert report["aliases"] == 4  # reverse_items is an option, not a column


def test_unknown_target_is_a_warning_with_the_closest_dv(tmp_path, vocab):
    report = check_mapping(_write(tmp_path, "md: mental_demnd\n"), vocabulary=vocab)
    msg = _messages(report, "warnings")["md"]
    assert "not in the schema" in msg and "mental_demand" in msg
    assert report["to_study_specific"] == 1


def test_non_string_target_is_an_error_and_fails_the_cli(tmp_path, vocab, capsys):
    path = _write(tmp_path, "md: 3\n")
    assert check_mapping(path, vocabulary=vocab)["errors"]
    assert main([str(path)]) == 1
    assert main([str(_write(tmp_path, "md: mental_demand\n"))]) == 0
    capsys.readouterr()


def test_shared_alias_with_another_target_wins_unless_overridden(tmp_path, vocab):
    path = _write(tmp_path, "sus_score: meta_derived_authors_sus\n")
    msg = _messages(check_mapping(path, vocabulary=vocab), "warnings")["sus_score"]
    assert "usability" in msg and "override_standard_aliases" in msg
    assert check_mapping(path, override_standard_aliases=True, vocabulary=vocab)["warnings"] == []


def test_sensor_alias_can_capture_a_dv_column(tmp_path, vocab):
    # osf_cwd6h's perceived-risk column has the name of a sensor channel
    path = _write(tmp_path, "ArduinoData1: perceived_risk\n")
    msg = _messages(check_mapping(path, vocabulary=vocab), "warnings")["ArduinoData1"]
    assert "sensor_arduino_data_1" in msg and "override_standard_aliases" in msg
    # The batch merges metadata and sensor aliases into the shared mapping that the flag overrides
    assert check_mapping(path, override_standard_aliases=True, vocabulary=vocab)["warnings"] == []


def test_never_map_names_and_case_duplicates(tmp_path, vocab):
    report = check_mapping(_write(tmp_path, (
        "condition: mental_demand\n"
        "ID: meta_participant_id\n"
        "id: meta_response_id\n"
    )), vocabulary=vocab)
    msgs = _messages(report, "warnings")
    assert "never-map" in msgs["condition"]
    assert "case-insensitively" in msgs["id"]


def test_instrument_scores_report_missing_items(tmp_path, vocab):
    full = "\n".join(f"s{i}: SUS{i}" for i in range(1, 11)) + "\n"
    scores = check_mapping(_write(tmp_path, full), vocabulary=vocab)["instrument_scores"]
    sus = next(s for s in scores if s["instrument"] == "sus")
    assert sus["computed"] and sus["dv"] == "usability"

    partial = "\n".join(f"s{i}: SUS{i}" for i in range(1, 10)) + "\n"
    sus = next(s for s in check_mapping(_write(tmp_path, partial), vocabulary=vocab)["instrument_scores"]
               if s["instrument"] == "sus")
    assert not sus["computed"] and sus["missing"] == ["SUS10"]


def test_every_corpus_mapping_has_no_errors(vocab):
    repo = Path(__file__).resolve().parents[1]
    paths = sorted(repo.glob("schemas/*_mapping.yaml")) + sorted(repo.glob("analyses/heldout*/mappings/*.yaml"))
    paths = [p for p in paths if p.name != "standard_dv_mapping.yaml" and not p.name.startswith("standard_")]
    assert len(paths) > 70
    for path in paths:
        assert check_mapping(path, vocabulary=vocab)["errors"] == [], path
