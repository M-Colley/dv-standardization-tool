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
# exported as "neg1", "neg2", ... (e.g. a -2..+2 acceptance item). Coercing such
# a column with pd.to_numeric silently drops every negative answer.
_NEGATIVE_ANSWER_CODE = re.compile(r"^neg(\d+(?:\.\d+)?)$", re.IGNORECASE)
MIN_NUMERIC_SHARE_AFTER_RECODING = 0.9


def recode_negative_answer_codes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Turn LimeSurvey ``negN`` answer codes into ``-N``.

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
        codes = text.str.extract(_NEGATIVE_ANSWER_CODE, expand=False)
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


__all__ = ["apply_value_labels", "recode_negative_answer_codes"]
