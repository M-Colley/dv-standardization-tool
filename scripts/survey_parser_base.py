"""Base class + low-level helpers for survey-export parsers.

Extracted from ``scripts.survey_parsers`` so the abstract ``SurveyParser``
contract and its supporting primitives (encoding/delimiter detection,
column slugification, duplicate-column resolution, file head reader) can
be unit-tested without loading the three concrete parsers
(Qualtrics/LimeSurvey/REDCap), each of which carries hundreds of lines of
platform-specific logic.

``scripts.survey_parsers`` re-exports every public name from this module
so existing imports keep working:

    from scripts.survey_parsers import (
        SurveyParser,
        _detect_encoding,
        _detect_delimiter,
        _slugify,
        _ensure_unique_columns,
        _read_file_bytes,
    )

continues to resolve.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level file + text helpers
# ---------------------------------------------------------------------------

def _read_file_bytes(file_path: Path, max_bytes: int = 65536) -> bytes:
    """Read up to *max_bytes* from the beginning of a file for sniffing."""
    with open(file_path, "rb") as fh:
        return fh.read(max_bytes)


def _detect_encoding(file_path: Path) -> str:
    """Best-effort encoding detection for survey-export files.

    Delegates to the shared ``encoding_utils.detect_text_encoding`` helper so
    survey exports are classified identically to the convert_dv loader. BOMs
    are honoured (utf-8-sig / utf-16-le / utf-16-be) and plain ASCII is
    upgraded to utf-8.
    """
    from scripts.encoding_utils import detect_text_encoding

    raw = _read_file_bytes(file_path, max_bytes=32768)
    return detect_text_encoding(raw)


def _detect_delimiter(sample: str, prefer: str | None = None) -> str:
    """Pick the most consistent delimiter from the first lines of *sample*.

    Counts each candidate (``,``, ``;``, ``\\t``, ``|``) per row and selects
    the one with the lowest coefficient of variation across the first 15
    rows — i.e. the most consistent count is the most likely delimiter.
    Falls back to ``prefer`` (or ``,``) when no rows can be sampled.
    """
    candidates = [",", ";", "\t", "|"]
    if prefer:
        candidates = [prefer] + [c for c in candidates if c != prefer]

    rows = [r for r in sample.splitlines() if r.strip()][:15]
    if not rows:
        return prefer or ","

    best_delim = prefer or ","
    best_score = float("inf")

    for delim in candidates:
        counts = [row.count(delim) for row in rows]
        if max(counts) == 0:
            continue
        mean = sum(counts) / len(counts)
        if mean == 0:
            continue
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        cv = (variance ** 0.5) / mean
        if cv < best_score:
            best_score = cv
            best_delim = delim

    return best_delim


# ---------------------------------------------------------------------------
# Column-name normalization
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 80) -> str:
    """Turn arbitrary text into a short, filesystem/column-safe slug.

    Collapses any run of non-alphanumeric characters to a single underscore,
    trims leading/trailing underscores, truncates to ``max_len``, and falls
    back to ``"unnamed"`` if the result is empty.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len] if slug else "unnamed"


def _ensure_unique_columns(columns: Sequence[str]) -> list[str]:
    """Deduplicate column names by appending ``_2``, ``_3``, etc.

    Order-preserving: the first occurrence keeps its original name, every
    subsequent collision gets a ``_<n>`` suffix where ``n`` is the running
    occurrence count.
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            result.append(col)
    return result


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SurveyParser(ABC):
    """Base class for survey platform parsers.

    Subclasses must implement :meth:`parse` which reads a file exported by
    a survey platform and returns a clean :class:`pandas.DataFrame` in
    wide format (one row per participant, one column per variable).
    """

    @abstractmethod
    def parse(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Parse a survey export file.

        Parameters
        ----------
        file_path:
            Path to the exported file.
        **kwargs:
            Parser-specific options.

        Returns
        -------
        pd.DataFrame
            Wide-format DataFrame with one row per participant and
            descriptive column names.
        """
        ...


__all__ = [
    "_read_file_bytes",
    "_detect_encoding",
    "_detect_delimiter",
    "_slugify",
    "_ensure_unique_columns",
    "SurveyParser",
]
