"""Instrument registry: declarative scoring of standard questionnaires.

``schemas/instruments.yaml`` records, per instrument, its items (with keying
and the raw column names they appear under), the candidate response codings
and the scoring rule of each score, with a citation. This module loads and
validates that registry and scores a data frame: every score whose items are
all present is added as a column.

Scoring an item depends on the score's ``centering`` and the item's ``key``
(+1: a higher answer means a higher score; -1: reverse-keyed):

=========  ======================  =======================
centering  key +1                  key -1
=========  ======================  =======================
none       x                       (not allowed)
reflect    x                       min + max - x
min        x - min                 max - x
midpoint   x - (min + max) / 2     (min + max) / 2 - x
=========  ======================  =======================

Item scores are then summed or averaged (missing items make the score
missing) and multiplied by ``multiplier``. ``min``/``max`` come from the
first response coding of the instrument that contains every observed answer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_INSTRUMENTS_PATH = Path(__file__).resolve().parents[1] / "schemas" / "instruments.yaml"
CENTERINGS = ("none", "reflect", "min", "midpoint")
AGGREGATES = ("mean", "sum")


def normalize_column_name(name: str) -> str:
    """Case- and punctuation-insensitive form used to match column names."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


@dataclass(frozen=True)
class Item:
    id: str
    aliases: tuple[str, ...]
    key: int

    @property
    def candidates(self) -> tuple[str, ...]:
        """Column names tried in order: the aliases, then the item id."""
        return (*self.aliases, self.id)


@dataclass(frozen=True)
class Score:
    name: str
    dv: str
    items: tuple[str, ...]
    centering: str
    aggregate: str
    multiplier: float = 1.0

    @property
    def column(self) -> str:
        return self.name


@dataclass(frozen=True)
class Instrument:
    id: str
    label: str
    citation: str
    codings: tuple[tuple[float, float], ...]
    items: Mapping[str, Item]
    scores: tuple[Score, ...]
    doi: str = ""


def _parse_instrument(iid: str, raw: Mapping[str, Any]) -> Instrument:
    codings = tuple((float(lo), float(hi)) for lo, hi in raw.get("response_codings") or [])
    if not codings or any(lo >= hi for lo, hi in codings):
        raise ValueError(f"{iid}: response_codings must be a non-empty list of [min, max] with min < max")
    items: dict[str, Item] = {}
    for entry in raw.get("items") or []:
        key = int(entry.get("key", 1))
        if key not in (1, -1):
            raise ValueError(f"{iid}/{entry.get('id')}: key must be +1 or -1")
        item = Item(id=str(entry["id"]), aliases=tuple(str(a) for a in entry.get("aliases") or []), key=key)
        if item.id in items:
            raise ValueError(f"{iid}: duplicate item {item.id}")
        items[item.id] = item
    scores = []
    for name, spec in (raw.get("scores") or {}).items():
        score = Score(
            name=str(name),
            dv=str(spec["dv"]),
            items=tuple(str(i) for i in spec["items"]),
            centering=str(spec.get("centering", "none")),
            aggregate=str(spec.get("aggregate", "mean")),
            multiplier=float(spec.get("multiplier", 1.0)),
        )
        if score.centering not in CENTERINGS:
            raise ValueError(f"{iid}/{name}: centering must be one of {CENTERINGS}")
        if score.aggregate not in AGGREGATES:
            raise ValueError(f"{iid}/{name}: aggregate must be one of {AGGREGATES}")
        unknown = sorted(set(score.items) - set(items))
        if unknown:
            raise ValueError(f"{iid}/{name}: unknown items {unknown}")
        if score.centering == "none" and any(items[i].key == -1 for i in score.items):
            raise ValueError(f"{iid}/{name}: reverse-keyed items need a centering other than 'none'")
        scores.append(score)
    if not scores:
        raise ValueError(f"{iid}: no scores defined")
    return Instrument(
        id=iid,
        label=str(raw.get("label", iid)),
        citation=str(raw.get("citation", "")).strip(),
        codings=codings,
        items=items,
        scores=tuple(scores),
        doi=str(raw.get("doi", "") or ""),
    )


@lru_cache(maxsize=4)
def _load_cached(path_str: str, mtime_ns: int) -> tuple[Instrument, ...]:
    with open(path_str, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    instruments = tuple(_parse_instrument(str(iid), raw) for iid, raw in (data.get("instruments") or {}).items())
    columns = [s.column for inst in instruments for s in inst.scores]
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"score columns defined twice: {duplicates}")
    for inst in instruments:
        if not inst.citation:
            raise ValueError(f"{inst.id}: a citation is required")
    return instruments


def load_instruments(path: Path = DEFAULT_INSTRUMENTS_PATH) -> tuple[Instrument, ...]:
    """Parse and validate the registry (cached per file modification time)."""
    path = Path(path)
    return _load_cached(str(path), path.stat().st_mtime_ns)


def schema_errors(instruments: Iterable[Instrument], dv_ids: Iterable[str]) -> list[str]:
    """Problems linking the registry to the DV schema (each score must map to a DV)."""
    known = set(dv_ids)
    return [
        f"{inst.id}/{score.name}: dv {score.dv!r} is not in the DV schema"
        for inst in instruments
        for score in inst.scores
        if score.dv not in known
    ]


def item_column_names(instruments: Iterable[Instrument]) -> frozenset[str]:
    """Normalized raw column names of every registered item."""
    return frozenset(
        normalize_column_name(candidate)
        for inst in instruments
        for item in inst.items.values()
        for candidate in item.candidates
    )


def construct_subitems(instruments: Iterable[Instrument]) -> dict[str, list[str]]:
    """Score DV -> the item ids it is computed from (for excluding tautological edges)."""
    mapping: dict[str, list[str]] = {}
    for inst in instruments:
        for score in inst.scores:
            merged = mapping.setdefault(score.dv, [])
            merged.extend(i for i in dict.fromkeys(score.items) if i not in merged)
    return mapping


def _resolve_items(df: pd.DataFrame, instrument: Instrument) -> dict[str, pd.Series]:
    lookup = {normalize_column_name(col): col for col in df.columns}
    resolved: dict[str, pd.Series] = {}
    for item in instrument.items.values():
        for candidate in item.candidates:
            col = lookup.get(normalize_column_name(candidate))
            if col is not None:
                resolved[item.id] = pd.to_numeric(df[col], errors="coerce")
                break
    return resolved


def choose_coding(
    values: Iterable[pd.Series], codings: tuple[tuple[float, float], ...], label: str = ""
) -> tuple[float, float] | None:
    """First candidate coding that contains every observed answer."""
    observed = pd.concat(list(values), ignore_index=True).dropna()
    if observed.empty:
        return codings[0]
    lo, hi = float(observed.min()), float(observed.max())
    fitting = [c for c in codings if c[0] <= lo and hi <= c[1]]
    if not fitting:
        logger.warning("%s: answers span %g..%g, outside every coding %s; not scored", label, lo, hi, codings)
        return None
    if len(fitting) > 1:
        logger.warning(
            "%s: answers span %g..%g, which fits codings %s; assuming %s", label, lo, hi, fitting, fitting[0]
        )
    return fitting[0]


def _item_score(x: pd.Series, key: int, centering: str, lo: float, hi: float) -> pd.Series:
    if centering == "none":
        return x
    if centering == "reflect":
        return x if key == 1 else (lo + hi) - x
    if centering == "min":
        return x - lo if key == 1 else hi - x
    mid = (lo + hi) / 2.0
    return x - mid if key == 1 else mid - x


def score_frame(
    df: pd.DataFrame, instruments: Iterable[Instrument] | None = None
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Add every score whose items are all present; return the frame and a record per score."""
    if instruments is None:
        instruments = load_instruments()
    out = df.copy()
    records: list[dict[str, Any]] = []
    for inst in instruments:
        resolved = _resolve_items(out, inst)
        ready = [s for s in inst.scores if all(i in resolved for i in s.items)]
        if not ready:
            continue
        coding = choose_coding(resolved.values(), inst.codings, label=inst.id)
        for score in ready:
            record = {"instrument": inst.id, "score": score.column, "dv": score.dv}
            if coding is None:
                records.append({**record, "decision": "skipped_out_of_range"})
                continue
            lo, hi = coding
            parts = [_item_score(resolved[i], inst.items[i].key, score.centering, lo, hi) for i in score.items]
            stacked = pd.concat(parts, axis=1)
            if score.aggregate == "sum":
                value = stacked.sum(axis=1, skipna=False)
            else:
                value = stacked.mean(axis=1, skipna=False)
            out[score.column] = value * score.multiplier
            records.append({
                **record,
                "decision": "scored",
                "coding": [lo, hi],
                "n_scored": int(out[score.column].notna().sum()),
            })
    return out, records


__all__ = [
    "DEFAULT_INSTRUMENTS_PATH",
    "Instrument",
    "Item",
    "Score",
    "choose_coding",
    "construct_subitems",
    "item_column_names",
    "load_instruments",
    "normalize_column_name",
    "schema_errors",
    "score_frame",
]
