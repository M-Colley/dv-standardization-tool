"""Answers are scored and pooled only on a plausible scale (fixes of 2026-11-15)."""

from __future__ import annotations

import pandas as pd

from opendv.analysis.multi_study_analysis import harmonized_summary
from opendv.instruments import choose_coding

TLX = ((0.0, 20.0), (1.0, 21.0), (0.0, 100.0))
UEQ_S = ((-3.0, 3.0), (1.0, 7.0))


def _codes(*values):
    return choose_coding([pd.Series(values, dtype=float)], TLX)


def test_tlx_answers_one_point_off_a_21_point_coding_are_not_read_as_0_100():
    # Running into Traffic: subscales 0..21 were scored on 0-100, about five times too low;
    # they are read on 0-20 widened by the one point instead
    assert _codes(0, 4, 18, 21) == (0.0, 21.0)
    assert _codes(0, 4, 18) == (0.0, 20.0)


def test_tlx_codings_that_fit_are_kept():
    assert _codes(0, 7, 20) == (0.0, 20.0)
    assert _codes(1, 12, 21) == (1.0, 21.0)
    assert _codes(0, 35, 85) == (0.0, 100.0)
    assert _codes(25, 60, 95) == (0.0, 100.0)


def test_narrow_scales_are_not_affected():
    assert choose_coding([pd.Series([0.0, 1.0])], UEQ_S) == (-3.0, 3.0)
    assert choose_coding([pd.Series([1.0, 7.0])], UEQ_S) == (1.0, 7.0)


def test_values_on_no_known_scale_are_not_pooled():
    # weighted NASA-TLX sums (Music Intensity) used to be min-max rescaled to 0-100
    df = pd.DataFrame({"TLX_SCORE": [193.3, 600.0, 1493.3, 900.0]})
    summary = harmonized_summary({"study_a": df}, harmonize_scales=True, canonical_studies={"study_a": df})
    row = summary[(summary["study"] == "study_a") & (summary["dv"] == "TLX_SCORE")].iloc[0]
    assert row["n"] == 0
    assert row["scale_note"].startswith("not pooled")
