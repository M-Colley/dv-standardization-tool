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

import numpy as np
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


_BOOLEAN_TEXT = {"true": True, "false": False}


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _cells_matching(series: pd.Series, values: Iterable[Any]) -> pd.Series:
    """Rows of *series* whose cell equals one of *values*; missing cells never match.

    Booleans match boolean cells and the text ``true`` / ``false`` in any case
    (a logical column with gaps is read as text; R writes ``TRUE``). Numbers
    match numerically equal non-boolean cells. Text matches the trimmed cell.
    """
    values = list(values)
    bool_values = {v for v in values if isinstance(v, bool)}
    numeric_values = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    text_values = {str(v).strip() for v in values if not isinstance(v, bool)}
    present = series.notna()
    hit = pd.Series(False, index=series.index)
    if pd.api.types.is_bool_dtype(series.dtype):
        if bool_values:
            hit |= series.isin(list(bool_values)) & present
        return hit
    if bool_values:
        as_bool = series.map(
            lambda v: bool(v) if _is_bool(v) else _BOOLEAN_TEXT.get(str(v).strip().casefold())
        )
        hit |= as_bool.isin(list(bool_values)) & present
    if numeric_values:
        is_bool_cell = series.map(_is_bool)
        hit |= pd.to_numeric(series.where(~is_bool_cell), errors="coerce").isin(numeric_values)
    if text_values:
        hit |= series.astype(str).str.strip().isin(text_values) & present
    return hit


def apply_row_exclusions(
    df: pd.DataFrame,
    specs: Iterable[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop the rows a dataset's authors flagged for exclusion.

    Trial-level exports often keep the trials their authors left out of the
    analysis (too fast, too slow, practice) and mark them in a column. A
    mapping YAML declares which values of that column leave a row out::

        exclude_rows:
          - column: Outlier
            values: [true]
            reason: trials the authors flagged as too fast or too slow (man/HandSize.Rd)

    Applied first, to the values as read, so nothing downstream (recodings,
    summaries, the LLM's profiles) sees those rows. A row is dropped when any
    block matches it; cells are compared as in ``_cells_matching``. Blocks
    whose column the dataset lacks are skipped. Returns the frame (index
    reset) and, if any block applied, ``{"rows_before", "rows_dropped",
    "rules": [{column, values, reason, rows}]}`` with each block's own match
    count, so blocks that overlap are visible.
    """
    drop = pd.Series(False, index=df.index)
    rules: list[dict[str, Any]] = []
    for spec in specs or []:
        column = spec.get("column") if isinstance(spec, dict) else None
        if not isinstance(column, str) or column not in df.columns:
            continue
        values = list(spec.get("values") or [])
        hit = _cells_matching(df[column], values)
        drop |= hit
        rules.append({
            "column": column,
            "values": values,
            "reason": str(spec.get("reason", "")),
            "rows": int(hit.sum()),
        })
    if not rules:
        return df, {}
    report = {"rows_before": int(len(df)), "rows_dropped": int(drop.sum()), "rules": rules}
    if not drop.any():
        return df, report
    return df.loc[~drop].reset_index(drop=True), report


def unit_factor(source_unit: str, target_unit: str) -> float:
    """Multiplier from one time unit to another (``ms`` -> ``s`` is 0.001).

    Units are the ones ``pandas.Timedelta`` understands (``ms``, ``s``,
    ``min``, ``h``, ...); anything else raises ``ValueError``.
    """
    return pd.Timedelta(1, unit=str(source_unit)) / pd.Timedelta(1, unit=str(target_unit))


def apply_unit_conversions(
    df: pd.DataFrame,
    specs: Iterable[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert the declared columns of one dataset to the unit of their DV.

    The schema states each time DV's unit (``crossing_initiation_time`` in
    seconds); a dataset that recorded milliseconds would otherwise be pooled
    with seconds::

        unit_conversions:
          - columns: [CIT_1_overall, CIT_2_overall]
            from: ms
            to: s
            reason: crossing initiation time in ms (README)

    Numeric cells are multiplied, other cells are kept. Returns the frame and
    ``{column: "ms -> s"}`` for the columns that were converted.
    """
    out = df
    report: dict[str, str] = {}
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        factor = unit_factor(spec["from"], spec["to"])
        for column in spec.get("columns") or []:
            if column not in out.columns:
                continue
            if out is df:
                out = df.copy()
            numbers = pd.to_numeric(out[column], errors="coerce")
            out[column] = (numbers * factor).where(numbers.notna(), out[column])
            report[str(column)] = f"{spec['from']} -> {spec['to']}"
    return out, report


def _linear_range(value: Any) -> tuple[float, float]:
    lo, hi = (float(v) for v in value)
    if not lo < hi:
        raise ValueError(f"range must be [min, max] with min < max, got {value!r}")
    return lo, hi


def apply_scale_conversions(
    df: pd.DataFrame,
    specs: Iterable[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map declared columns linearly from the scale they were recorded on to
    the DV's scale, e.g. NASA-TLX subscales stored on 0-10 to the schema's 0-20::

        scale_conversions:
          - columns: ["TLX 1", "TLX 2"]
            from: [0, 10]
            to: [0, 20]
            reason: subscales stored 0-10 (codebook)

    The analysis rescales a DV only when its values fall outside the DV's
    range, so a narrower recording scale that fits inside it (0-10 within
    0-20, a 0-1 proportion within 0-100 %) has to be declared. Numeric cells
    are mapped, other cells are kept. Returns the frame and
    ``{column: "[0, 10] -> [0, 20]"}``.
    """
    out = df
    report: dict[str, str] = {}
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        (a, b), (c, d) = _linear_range(spec["from"]), _linear_range(spec["to"])
        for column in spec.get("columns") or []:
            if column not in out.columns:
                continue
            if out is df:
                out = df.copy()
            numbers = pd.to_numeric(out[column], errors="coerce")
            out[column] = (c + (numbers - a) * (d - c) / (b - a)).where(numbers.notna(), out[column])
            report[str(column)] = f"[{a:g}, {b:g}] -> [{c:g}, {d:g}]"
    return out, report


__all__ = [
    "apply_missing_values",
    "apply_row_exclusions",
    "apply_scale_conversions",
    "apply_unit_conversions",
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
