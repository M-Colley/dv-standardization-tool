"""Map question-text column headers to DVs before the LLM is asked.

Survey exports often use the question itself, or a label with its anchors, as
the column header ("I found the system unnecessarily complex.", "Wie hoch waren
die geistigen Anforderungen der Aufgaben?", "Mental Demand (0 = Low, 100 =
High)"). Aliases do not match such headers, and the LLM maps them poorly (2 of
35 in the VR study of the confirmatory test, docs/confirmatory_shortlist_results.md).
Two deterministic steps run on the columns no alias maps:

1. **Cleaned alias.** HTML tags, a leading item number and parenthesized
   anchors are removed and the rest is looked up among the aliases again
   ("Mental Demand (0 = Low, 100 = High)" -> ``mental_demand``). Square brackets
   are kept: in LimeSurvey exports they name the sub-question.
2. **Item wording.** For every registered instrument whose items carry a
   ``wording`` (``schemas/instruments.yaml``), the table's question-like headers
   and the item wordings are embedded with a multilingual sentence model and
   assigned one to one (Hungarian assignment, :func:`scipy.optimize.linear_sum_assignment`).
   A header is the *seed* of an item when it is that item's most similar header.
   The instrument is accepted only if at least :data:`MIN_BLOCK_ITEMS` seeds lie
   in a run of columns no wider than the instrument (plus two): questionnaire
   items are adjacent, while scattered questions that happen to resemble some
   items are not. Within that run, every assigned pair above
   :data:`MIN_SIMILARITY` is a match.

Matches carry their own provenance (``mapping_method`` ``header_match``), so
they are neither curated gold labels nor LLM deductions.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from opendv.instruments import Instrument, load_instruments

logger = logging.getLogger(__name__)

HEADER_MATCH_MODEL = "intfloat/multilingual-e5-small"
MIN_BLOCK_ITEMS = 4
MIN_SIMILARITY = 0.80
BLOCK_SLACK_COLUMNS = 2
MIN_QUESTION_WORDS = 3

_HTML_TAG = re.compile(r"<[^>]+>")
_ITEM_NUMBER = re.compile(r"^\s*\d{1,3}\s*[.):]\s*")
_PARENTHESIZED = re.compile(r"\([^)]*\)")
# A unit in parentheses ("Completion time (ms)") changes what the number means,
# so such a header is not matched by its cleaned name (the DV may use seconds).
_UNIT_IN_PARENTHESES = re.compile(
    r"\(\s*(?:in\s+)?(?:ms|msec|millisec\w*|s|sec|secs|seconds?|min|mins|minutes?|h|hr|hrs|hours?|"
    r"%|percent|px|pixels?|mm|cm|m|km|deg|degrees?|hz|bpm)\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HeaderMatch:
    column: str
    dv: str
    method: str  # "cleaned_alias" | "item_wording"
    detail: str  # the alias, or "<instrument>:<similarity>"


def clean_header(header: str) -> str:
    """Header without HTML tags, a leading item number and parenthesized anchors."""
    text = _HTML_TAG.sub("", str(header))
    text = _ITEM_NUMBER.sub("", text)
    text = _PARENTHESIZED.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _alias_key(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.strip()).lower()


@lru_cache(maxsize=1)
def _encoder():
    from opendv.dv_retrieval import sentence_transformer_encoder

    encode = sentence_transformer_encoder(HEADER_MATCH_MODEL, device="cpu")
    return lambda texts: encode(["query: " + t for t in texts])


def _wording_matches(
    headers: Sequence[str], positions: Sequence[int], instrument: Instrument, encode
) -> list[tuple[int, str, float]]:
    """(header index, item id, similarity) of the instrument's accepted block, if any."""
    items = [item for item in instrument.items.values() if item.wording]
    if len(items) < MIN_BLOCK_ITEMS or len(headers) < MIN_BLOCK_ITEMS:
        return []
    from scipy.optimize import linear_sum_assignment

    header_vectors = encode([clean_header(h) for h in headers])
    similarity = np.full((len(headers), len(items)), -1.0)
    for j, item in enumerate(items):
        wording_vectors = encode(list(item.wording))
        similarity[:, j] = (header_vectors @ wording_vectors.T).max(axis=1)
    rows, cols = linear_sum_assignment(-similarity)
    assigned = [(r, c) for r, c in zip(rows, cols, strict=True) if similarity[r, c] >= MIN_SIMILARITY]
    seeds = sorted(positions[r] for r, c in assigned if similarity[:, c].argmax() == r)
    width = len(items) + BLOCK_SLACK_COLUMNS
    best: list[int] = []
    for start in seeds:
        window = [p for p in seeds if start <= p < start + width]
        if len(window) > len(best):
            best = window
    if len(best) < MIN_BLOCK_ITEMS:
        return []
    slack = len(items) - len(best)
    lo, hi = best[0] - slack, best[-1] + slack
    return [
        (r, items[c].id, float(similarity[r, c]))
        for r, c in assigned
        if lo <= positions[r] <= hi
    ]


def match_headers(
    columns: Sequence[str],
    candidates: Iterable[str],
    alias_mapping: Mapping[str, str],
    instruments: Iterable[Instrument] | None = None,
    encoder=None,
) -> dict[str, HeaderMatch]:
    """Matches for the ``candidates`` (unmapped columns) of one table.

    ``columns`` is the table's full column order (blocks are found by position),
    ``alias_mapping`` the alias -> DV mapping in force for the table.
    """
    position = {str(col): i for i, col in enumerate(columns)}
    lowered = {str(alias).lower(): dv for alias, dv in alias_mapping.items()}
    matches: dict[str, HeaderMatch] = {}
    remaining: list[str] = []
    for column in candidates:
        column = str(column)
        cleaned = clean_header(column)
        dv = None
        # Plain headers ("Mental Demand") are tried too: the alias mapping only
        # matched them case-insensitively, not with spaces read as underscores.
        if cleaned and not _UNIT_IN_PARENTHESES.search(column):
            dv = lowered.get(cleaned.lower()) or lowered.get(_alias_key(cleaned))
        if dv:
            matches[column] = HeaderMatch(column, dv, "cleaned_alias", cleaned)
        elif len(clean_header(column).split()) >= MIN_QUESTION_WORDS and column in position:
            remaining.append(column)
    if len(remaining) < MIN_BLOCK_ITEMS:
        return matches
    instruments = load_instruments() if instruments is None else instruments
    encode = encoder or _encoder()
    taken: set[str] = set()
    for instrument in instruments:
        free = [c for c in remaining if c not in taken]
        for index, item_id, sim in _wording_matches(free, [position[c] for c in free], instrument, encode):
            column = free[index]
            taken.add(column)
            matches[column] = HeaderMatch(column, item_id, "item_wording", f"{instrument.id}:{sim:.3f}")
    return matches


__all__ = ["HeaderMatch", "clean_header", "match_headers"]
