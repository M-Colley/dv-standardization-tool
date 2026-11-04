"""Declarative recoding of text-labelled responses into numbers.

Survey exports often store Likert answers as their labels ("Rather agree",
"Stimme eher zu") instead of numeric codes. Such columns are invisible to every
numeric analysis — a dataset can be "standardized" yet contribute nothing. A
repository mapping YAML can declare the codes once::

    value_labels:
      - columns: [C1, C2, C3, C4, C5]
        labels:
          Do not agree at all: 1
          Rather disagree: 2
          Neither: 3
          Rather agree: 4
          Fully agree: 5

Matching ignores case and surrounding whitespace. Numeric cells are kept;
non-empty cells whose label is not declared become missing and are counted in
the report, so a typo in the YAML or an unexpected label is visible in the
dataset's ``-quality.json`` rather than silently guessed.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd


def _normalize_label(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def apply_value_labels(
    df: pd.DataFrame,
    value_labels: Iterable[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Recode declared columns of *df*; return the frame and a per-column report.

    The report maps column -> {"recoded": n, "kept_numeric": n, "unmatched": n}.
    Columns not present in *df* are ignored.
    """
    report: dict[str, dict[str, int]] = {}
    out = df
    for spec in value_labels or []:
        if not isinstance(spec, dict):
            continue
        labels = {_normalize_label(k): float(v) for k, v in (spec.get("labels") or {}).items()}
        if not labels:
            continue
        for column in spec.get("columns") or []:
            if column not in out.columns:
                continue
            if out is df:
                out = df.copy()
            series = out[column]
            numeric = pd.to_numeric(series, errors="coerce")
            text = series.where(numeric.isna() & series.notna())
            mapped = text.map(lambda v: labels.get(_normalize_label(v)) if pd.notna(v) else None)
            result = numeric.where(numeric.notna(), pd.to_numeric(mapped, errors="coerce"))
            report[str(column)] = {
                "recoded": int(mapped.notna().sum()),
                "kept_numeric": int(numeric.notna().sum()),
                "unmatched": int((text.notna() & mapped.isna()).sum()),
            }
            out[column] = result
    return out, report


# LimeSurvey answer codes cannot contain "-", so negative scale points are
# exported as "neg1", "neg2", ... or, in some surveys, "1neg", "2neg", ...
# (e.g. a -2..+2 acceptance item, a -3..+3 safety item). Coercing such a column
# with pd.to_numeric silently drops every negative answer.
_NEGATIVE_ANSWER_CODE = re.compile(r"^(?:neg(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)neg)$", re.IGNORECASE)
MIN_NUMERIC_SHARE_AFTER_RECODING = 0.9


def recode_negative_answer_codes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Turn LimeSurvey ``negN`` / ``Nneg`` answer codes into ``-N``.

    Only text columns that contain at least one such code and become at least
    90% numeric after recoding are converted, so free-text columns are left
    alone. Returns the frame and ``{column: n_codes_recoded}``.
    """
    report: dict[str, int] = {}
    out = df
    for position, column in enumerate(df.columns):
        series = df.iloc[:, position]
        if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
            continue
        text = series.dropna().astype(str).str.strip()
        groups = text.str.extract(_NEGATIVE_ANSWER_CODE)
        codes = groups[0].fillna(groups[1])  # prefix or suffix form
        n_codes = int(codes.notna().sum())
        if not n_codes:
            continue
        recoded_text = text.where(codes.isna(), "-" + codes)
        numeric = pd.to_numeric(recoded_text, errors="coerce")
        if numeric.notna().mean() < MIN_NUMERIC_SHARE_AFTER_RECODING:
            continue
        if out is df:
            out = df.copy()
        values = pd.Series(float("nan"), index=series.index)
        values.loc[numeric.index] = numeric
        out.isetitem(position, values)
        report[str(column)] = n_codes
    return out, report


# LimeSurvey exports the answers of array (matrix) questions as answer codes
# ("A1".."A7") unless "answer codes as numbers" is chosen. Array sub-questions
# carry a bracketed or _SQnnn suffix (``TrustAutomation[SQ001]``); single
# list questions such as gender also use A-codes but are nominal, so only
# array sub-questions are recoded.
_ARRAY_SUBQUESTION = re.compile(r"(\[[^\]]+\]|_sq\d+)$", re.IGNORECASE)
_ANSWER_CODE = re.compile(r"^a0*(\d+)$", re.IGNORECASE)
_YES_NO = {"y": 1, "yes": 1, "ja": 1, "j": 1, "n": 0, "no": 0, "nein": 0}
# pandas reads True/False cells as booleans only when the column has no gaps;
# with a missing cell the column stays text (object) and looks non-numeric.
_TRUE_FALSE = {"true": 1, "false": 0}


def recode_survey_answer_codes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Turn survey answer codes into numbers: ``A1``.. -> 1.. and yes/no -> 1/0.

    * LimeSurvey answer codes (``A1``..``An``) in array sub-question columns
      become their number (the code order is the order of the answer scale).
    * Columns holding only yes/no answers (``Y``/``N``, ``Yes``/``No``,
      ``Ja``/``Nein``) become 1/0.
    * Columns holding only ``True``/``False`` (as text or as booleans in a
      column with gaps) become 1/0.

    A column is recoded only when every non-empty cell is such a code, so
    mixed or free-text columns are left alone. Returns the frame and
    ``{column: "answer_code" | "yes_no" | "true_false"}``.
    """
    report: dict[str, str] = {}
    out = df
    for position, column in enumerate(df.columns):
        series = df.iloc[:, position]
        if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
            continue
        text = series.dropna().astype(str).str.strip()
        text = text[text != ""]
        if text.empty:
            continue
        lowered = text.str.lower()
        if lowered.isin(_YES_NO.keys()).all():
            values, kind = lowered.map(_YES_NO), "yes_no"
        elif lowered.isin(_TRUE_FALSE.keys()).all():
            values, kind = lowered.map(_TRUE_FALSE), "true_false"
        elif _ARRAY_SUBQUESTION.search(str(column)) and text.str.fullmatch(_ANSWER_CODE).all():
            values, kind = text.str.extract(_ANSWER_CODE)[0].astype(float), "answer_code"
        else:
            continue
        if out is df:
            out = df.copy()
        recoded = pd.Series(float("nan"), index=series.index)
        recoded.loc[values.index] = values.astype(float)
        out.isetitem(position, recoded)
        report[str(column)] = kind
    return out, report


def apply_missing_values(
    df: pd.DataFrame,
    specs: Iterable[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set declared missing-value codes of one dataset to missing.

    Survey exports often store "no answer" or "not applicable" as numbers
    (98, 99, -99), which numeric analyses would otherwise read as answers::

        missing_values:
          - columns: [sus6, sus7]     # or "*" for every column
            values: [98, 99]
            reason: 99 = no answer (README)

    Applied first, before ``value_labels``. Codes are compared as numbers
    when the cell is numeric and as trimmed text otherwise. Returns the frame
    and ``{column: n_cells_set_missing}`` for the columns that had any.
    """
    out = df
    report: dict[str, int] = {}
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        codes = list(spec.get("values") or [])
        numeric_codes = [float(c) for c in codes if isinstance(c, (int, float)) and not isinstance(c, bool)]
        text_codes = {str(c).strip() for c in codes}
        columns = list(out.columns) if spec.get("columns") == "*" else list(spec.get("columns") or [])
        for column in columns:
            if column not in out.columns:
                continue
            series = out[column]
            hit = pd.to_numeric(series, errors="coerce").isin(numeric_codes)
            hit |= series.astype(str).str.strip().isin(text_codes) & series.notna()
            if not hit.any():
                continue
            if out is df:
                out = df.copy()
            out[column] = series.mask(hit)
            report[str(column)] = report.get(str(column), 0) + int(hit.sum())
    return out, report


__all__ = [
    "apply_missing_values",
    "apply_value_labels",
    "recode_negative_answer_codes",
    "recode_survey_answer_codes",
]


def apply_reverse_keying(
    df: pd.DataFrame,
    specs: Iterable[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Reverse-score declared items of one dataset: ``x -> min + max - x``.

    A repository mapping YAML declares items whose positive pole is on the
    other side in *this* survey (or that the source analysis reverse-scores)::

        reverse_items:
          - columns: [C2]
            scale_range: [1, 5]
            reason: reverse-scored in the source analysis (C2_inv)

    Applied after ``value_labels`` and before mapping, so every downstream
    score sees consistently keyed items. Values outside ``scale_range`` are
    left unchanged and counted, so a wrong range is visible in the report.
    """
    out = df
    report: dict[str, dict[str, Any]] = {}
    for spec in specs:
        lo, hi = (float(v) for v in spec["scale_range"])
        for column in spec.get("columns") or []:
            if column not in out.columns:
                continue
            if out is df:
                out = df.copy()
            values = pd.to_numeric(out[column], errors="coerce")
            inside = values.between(lo, hi)
            out[column] = values.where(~inside, lo + hi - values)
            report[str(column)] = {
                "reversed": int(inside.sum()),
                "out_of_range": int((values.notna() & ~inside).sum()),
                "scale_range": [lo, hi],
                "reason": str(spec.get("reason", "")),
            }
    return out, report
