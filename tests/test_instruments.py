"""Instrument registry: loading, validation and scoring against worked examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import instruments

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"
DV_SCHEMA = {dv["id"]: dv for dv in yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))["dvs"]}
REGISTRY = {inst.id: inst for inst in instruments.load_instruments()}


def _score(columns: dict[str, list[float]]) -> pd.DataFrame:
    out, _ = instruments.score_frame(pd.DataFrame(columns), REGISTRY.values())
    return out


def test_every_score_maps_to_a_schema_dv():
    assert instruments.schema_errors(REGISTRY.values(), DV_SCHEMA) == []


def test_every_score_column_is_recognised_as_its_dv():
    """A score written under a name the analysis does not map would be lost."""
    from analyses import multi_study_analysis as msa

    unmapped = {
        s.column: msa._canonicalize_dv_name(s.column)
        for inst in REGISTRY.values() for s in inst.scores
        if msa._canonicalize_dv_name(s.column) != s.dv
    }
    assert unmapped == {}


@pytest.mark.parametrize(
    "inst_id, score_name",
    [(i.id, s.name) for i in REGISTRY.values() for s in i.scores],
)
def test_score_range_matches_dv_canonical_range(inst_id, score_name):
    """The lowest and highest possible score equal the DV's canonical range."""
    inst = REGISTRY[inst_id]
    score = next(s for s in inst.scores if s.name == score_name)
    lo, hi = inst.codings[0]
    canonical = DV_SCHEMA[score.dv]["measurement"].get("canonical_range")
    if canonical is None:
        pytest.skip(f"{score.dv} has no canonical_range")
    best = {i: [hi if inst.items[i].key == 1 else lo] for i in dict.fromkeys(score.items)}
    worst = {i: [lo if inst.items[i].key == 1 else hi] for i in dict.fromkeys(score.items)}
    top = instruments.score_frame(pd.DataFrame(best), [inst])[0][score.column].iloc[0]
    bottom = instruments.score_frame(pd.DataFrame(worst), [inst])[0][score.column].iloc[0]
    assert [bottom, top] == pytest.approx([float(v) for v in canonical])


# ── Worked examples ──────────────────────────────────────────────────────────

def test_sus_worked_example():
    # Odd items 4, even items 2: each item contributes 3, 10 x 3 x 2.5 = 75.
    out = _score({f"SUS{i}": [4.0 if i % 2 else 2.0] for i in range(1, 11)})
    assert out["sus_score"].iloc[0] == pytest.approx(75.0)


def test_sus_missing_item_gives_missing_score():
    cols = {f"sus_{i}": [3.0, 3.0] for i in range(1, 11)}
    cols["sus_4"] = [3.0, np.nan]
    out = _score(cols)
    assert out["sus_score"].iloc[0] == pytest.approx(50.0)
    assert np.isnan(out["sus_score"].iloc[1])


def test_ssq_weights_follow_kennedy_1993():
    # Every symptom rated 1 ("slight"): raw sums N = O = D = 7.
    out = _score({f"ssq{i}": [1.0] for i in range(1, 17)})
    assert out["ssq_nausea_score"].iloc[0] == pytest.approx(7 * 9.54)
    assert out["ssq_oculomotor_score"].iloc[0] == pytest.approx(7 * 7.58)
    assert out["ssq_disorientation_score"].iloc[0] == pytest.approx(7 * 13.92)
    assert out["ssq_total_score"].iloc[0] == pytest.approx(21 * 3.74)


def test_ueq_s_midpoint_scoring():
    cols = {f"ueqs{i}": [7.0 if i <= 4 else 4.0] for i in range(1, 9)}
    out = _score(cols)
    assert out["ueq_pragmatic"].iloc[0] == pytest.approx(3.0)
    assert out["ueq_hedonic"].iloc[0] == pytest.approx(0.0)


def test_tia_reverse_keyed_items():
    # Agreeing fully (5) with every item: reverse-keyed items 10 and 15 count as 1.
    out = _score({f"tia_koerber_{i:02d}": [5.0] for i in range(1, 20)})
    assert out["tia_reliability_competence"].iloc[0] == pytest.approx((5 * 4 + 1 * 2) / 6)
    assert out["tia_trust_in_automation"].iloc[0] == pytest.approx(5.0)
    assert out["tia_propensity_to_trust"].iloc[0] == pytest.approx((1 + 5 + 5) / 3)


@pytest.mark.parametrize("coding, positive, negative", [((1, 5), 1.0, 5.0), ((-2, 2), -2.0, 2.0)])
def test_van_der_laan_is_coding_invariant(coding, positive, negative):
    """Positive pole first except items 3, 6 and 8; both codings give +2."""
    mirrored = {3, 6, 8}
    cols = {f"AOA{i}": [negative if i in mirrored else positive] for i in range(1, 10)}
    out = _score(cols)
    assert out["aoa_usefulness"].iloc[0] == pytest.approx(2.0)
    assert out["aoa_satisfying"].iloc[0] == pytest.approx(2.0)


def test_ambiguous_coding_is_reported(caplog):
    cols = {f"AOA{i}": [0.0, 2.0] for i in range(1, 10)}
    with caplog.at_level("WARNING"):
        _score(cols)
    assert "fits codings" in caplog.text


def test_out_of_range_answers_are_not_scored():
    cols = {f"SUS{i}": [7.0] for i in range(1, 11)}
    out, records = instruments.score_frame(pd.DataFrame(cols), [REGISTRY["sus"]])
    assert "sus_score" not in out.columns
    assert records[0]["decision"] == "skipped_out_of_range"


# ── Registry validation ─────────────────────────────────────────────────────

BASE = {
    "label": "x",
    "citation": "x",
    "response_codings": [[1, 5]],
    "items": [{"id": "a", "key": 1}, {"id": "b", "key": -1}],
    "scores": {"s": {"dv": "usability", "items": ["a", "b"], "centering": "min", "aggregate": "sum"}},
}


def _write(tmp_path, instrument):
    path = tmp_path / "instruments.yaml"
    path.write_text(yaml.safe_dump({"instruments": {"x": instrument}}), encoding="utf-8")
    return path


def test_valid_minimal_registry(tmp_path):
    (inst,) = instruments.load_instruments(_write(tmp_path, BASE))
    assert inst.items["b"].key == -1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"scores": {"s": {"dv": "usability", "items": ["a", "zz"]}}}, "unknown items"),
        ({"scores": {"s": {"dv": "usability", "items": ["a", "b"], "centering": "none"}}}, "reverse-keyed"),
        ({"scores": {"s": {"dv": "usability", "items": ["a"], "centering": "odd"}}}, "centering"),
        ({"response_codings": [[5, 1]]}, "min < max"),
        ({"citation": ""}, "citation"),
    ],
)
def test_invalid_registry_is_rejected(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        instruments.load_instruments(_write(tmp_path, {**BASE, **change}))


def test_unknown_score_dv_is_reported(tmp_path):
    bad = {**BASE, "scores": {"s": {"dv": "not_a_dv", "items": ["a"]}}}
    registry = instruments.load_instruments(_write(tmp_path, bad))
    assert instruments.schema_errors(registry, DV_SCHEMA) == ["x/s: dv 'not_a_dv' is not in the DV schema"]
