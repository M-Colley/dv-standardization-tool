"""Plausibility screening for LLM-suggested DV mappings.

The local LLM (``scripts.llm_utils``) is the last-resort mapper for columns no
explicit schema or repository mapping covers. Left unchecked it maps columns
that cannot be dependent variables — participant age to ``retention_time``,
a survey's random seed to ``trust_rating``, LimeSurvey per-question timings to
``mental_demand`` — and those mappings then flow straight into pooled
estimates. This module screens both sides of the LLM call:

* :func:`screen_llm_candidates` — before the call, drop columns whose name
  marks them as demographics, identifiers, timestamps, conditions, ...
  (``schemas/llm_plausibility_rules.yaml``) or whose values are not numeric;
* :func:`check_llm_suggestion` — after the call, reject a suggestion when the
  observed values fit neither the canonical DV's range nor a standard
  alternative response scale.

Every rejection is returned as a record so the batch runner can log it
(``llm_rejected_suggestions.json``). Explicit mappings never pass through here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yaml
from opendv._paths import SCHEMAS_DIR

RULES_PATH = SCHEMAS_DIR / "llm_plausibility_rules.yaml"

STAGE_PRE_LLM = "pre_llm"
STAGE_POST_LLM = "post_llm"

_UPPER = "A-ZÀ-ÖØ-Þ"
_LOWER = "a-zß-öø-ÿ"
_SPLIT = re.compile(rf"[^0-9{_UPPER}{_LOWER}]+")
_CAMEL = re.compile(rf"(?<=[{_LOWER}])(?=[{_UPPER}])")
_ACRONYM = re.compile(rf"(?<=[{_UPPER}])(?=[{_UPPER}][{_LOWER}])")
_DIGIT = re.compile(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)")


def name_tokens(name: Any) -> list[str]:
    """Split a column name into lower-cased tokens.

    ``"probandenIDTime"`` -> ``["probanden", "id", "time"]``;
    ``"attention_check001"`` -> ``["attention", "check", "001"]``.
    """
    tokens: list[str] = []
    for part in _SPLIT.split(str(name)):
        if not part:
            continue
        part = _DIGIT.sub(" ", _ACRONYM.sub(" ", _CAMEL.sub(" ", part)))
        tokens.extend(token.lower() for token in part.split())
    return tokens


def normalize_name(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


@dataclass(frozen=True)
class PlausibilityRules:
    version: str
    name_rules: tuple[dict[str, Any], ...]
    min_numeric_fraction: float
    range_tolerance: float
    alternative_ranges: tuple[tuple[float, float], ...]
    limesurvey_markers: frozenset[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlausibilityRules:
        return cls(
            version=str(data.get("version", "unversioned")),
            name_rules=tuple(dict(rule) for rule in data.get("name_rules") or []),
            min_numeric_fraction=float(data.get("min_numeric_fraction", 0.9)),
            range_tolerance=float(data.get("range_tolerance", 0.15)),
            alternative_ranges=tuple(
                (float(lo), float(hi)) for lo, hi in (data.get("alternative_ranges") or [])
            ),
            limesurvey_markers=frozenset(normalize_name(m) for m in data.get("limesurvey_markers") or []),
        )


@lru_cache(maxsize=4)
def load_rules(path: str = str(RULES_PATH)) -> PlausibilityRules:
    with open(path, encoding="utf-8") as f:
        return PlausibilityRules.from_dict(yaml.safe_load(f) or {})


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.lower()}
    return {str(v).lower() for v in value}


def _rule_matches(rule: dict[str, Any], tokens: list[str], normalized: str) -> bool:
    keys = set(rule) - {"reason"}
    if not keys:
        return False
    token_set = set(tokens)
    if "min_tokens" in rule and len(tokens) < int(rule["min_tokens"]):
        return False
    if "any_token" in rule and not token_set & _as_set(rule["any_token"]):
        return False
    if "all_tokens" in rule and not _as_set(rule["all_tokens"]) <= token_set:
        return False
    if "first_token" in rule and (not tokens or tokens[0] not in _as_set(rule["first_token"])):
        return False
    if "second_token" in rule and (len(tokens) < 2 or tokens[1] not in _as_set(rule["second_token"])):
        return False
    if "last_token" in rule and (not tokens or tokens[-1] not in _as_set(rule["last_token"])):
        return False
    if "exact" in rule and normalized not in {normalize_name(v) for v in rule["exact"]}:
        return False
    if "contains" in rule and not any(normalize_name(v) in normalized for v in rule["contains"]):
        return False
    return True


def name_rejection_reason(
    column: str,
    other_columns: Iterable[str] = (),
    rules: PlausibilityRules | None = None,
) -> str | None:
    """Return why *column*'s name rules it out as a DV, or None."""
    rules = rules or load_rules()
    tokens = name_tokens(column)
    normalized = normalize_name(column)
    for rule in rules.name_rules:
        if _rule_matches(rule, tokens, normalized):
            return str(rule.get("reason", "name_rule"))

    others = [normalize_name(c) for c in other_columns if str(c) != str(column)]
    if tokens and tokens[-1] == "time" and normalized.endswith("time") and rules.limesurvey_markers & set(others):
        prefix = normalized[: -len("time")]
        if prefix and any(other.startswith(prefix) for other in others):
            return "survey_timing"
    return None


@dataclass(frozen=True)
class ColumnProfile:
    """Numeric summary of one column, enough to screen LLM suggestions."""

    n_values: int
    n_numeric: int
    minimum: float | None
    maximum: float | None

    @property
    def numeric_fraction(self) -> float:
        return self.n_numeric / self.n_values if self.n_values else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_values": self.n_values,
            "n_numeric": self.n_numeric,
            "numeric_fraction": round(self.numeric_fraction, 4),
            "min": self.minimum,
            "max": self.maximum,
        }


def profile_column(series: pd.Series) -> ColumnProfile:
    non_null = series.dropna()
    if pd.api.types.is_bool_dtype(non_null.dtype):
        numeric = non_null.astype(float)
    elif pd.api.types.is_numeric_dtype(non_null.dtype):
        numeric = non_null
    else:
        text = non_null.astype(str).str.strip()
        text = text[text != ""]
        numeric = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce").dropna()
        non_null = text
    numeric = numeric[np.isfinite(numeric.astype(float))] if len(numeric) else numeric
    if not len(numeric):
        return ColumnProfile(int(len(non_null)), 0, None, None)
    return ColumnProfile(
        int(len(non_null)), int(len(numeric)), float(numeric.min()), float(numeric.max())
    )


def profile_columns(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, ColumnProfile]:
    """Profile the named columns of *df* (first occurrence for duplicated names)."""
    positions: dict[str, int] = {}
    for position, column in enumerate(df.columns):
        positions.setdefault(str(column), position)
    return {
        str(column): profile_column(df.iloc[:, positions[str(column)]])
        for column in columns
        if str(column) in positions
    }


def data_rejection_reason(profile: ColumnProfile, rules: PlausibilityRules | None = None) -> str | None:
    rules = rules or load_rules()
    if profile.n_values == 0:
        return "no_values"
    if profile.numeric_fraction < rules.min_numeric_fraction:
        return "non_numeric"
    return None


def screen_llm_candidates(
    unknown_columns: list[str],
    all_columns: list[str],
    profiles: dict[str, ColumnProfile],
    rules: PlausibilityRules | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Split *unknown_columns* into those worth sending to the LLM and rejections."""
    rules = rules or load_rules()
    eligible: list[str] = []
    rejected: list[dict[str, Any]] = []
    for column in unknown_columns:
        profile = profiles.get(str(column))
        reason = name_rejection_reason(column, all_columns, rules)
        if reason is None and profile is not None:
            reason = data_rejection_reason(profile, rules)
        if reason is None:
            eligible.append(column)
            continue
        rejected.append({
            "alias": str(column),
            "proposed_canonical": None,
            "stage": STAGE_PRE_LLM,
            "reason": reason,
            **(profile.as_dict() if profile is not None else {}),
        })
    return eligible, rejected


def range_is_compatible(
    observed: tuple[float, float],
    canonical_range: tuple[float, float],
    rules: PlausibilityRules | None = None,
) -> bool:
    """True when *observed* fits the canonical range or a standard alternative scale."""
    rules = rules or load_rules()
    lo_obs, hi_obs = observed
    for lo, hi in (canonical_range, *rules.alternative_ranges):
        tolerance = (hi - lo) * rules.range_tolerance
        if lo_obs >= lo - tolerance and hi_obs <= hi + tolerance:
            return True
    return False


def check_llm_suggestion(
    alias: str,
    canonical: str,
    profile: ColumnProfile | None,
    canonical_ranges: dict[str, tuple[float, float]],
    rules: PlausibilityRules | None = None,
) -> dict[str, Any] | None:
    """Return a rejection record for an implausible suggestion, or None to accept it."""
    canonical_range = canonical_ranges.get(str(canonical))
    if (
        canonical_range is None
        or profile is None
        or profile.minimum is None
        or profile.maximum is None
        or range_is_compatible((profile.minimum, profile.maximum), canonical_range, rules)
    ):
        return None
    return {
        "alias": str(alias),
        "proposed_canonical": str(canonical),
        "stage": STAGE_POST_LLM,
        "reason": "out_of_range",
        "canonical_range": list(canonical_range),
        **profile.as_dict(),
    }


_SCALE_RANGE_PATTERNS = [
    # "21-point (0-20)", "7-point (-3-3)", "21-point (0–20)"
    re.compile(r"\d+-point\s*\((-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\)"),
    # "5-point (-2 to +2)"
    re.compile(r"\d+-point\s*\((-?\+?\d+(?:\.\d+)?)\s+to\s+(\+?-?\d+(?:\.\d+)?)\)"),
]
_KNOWN_UNIT_RANGES = {
    "proportion": (0.0, 1.0),
    "percentage": (0.0, 100.0),
    "5-point": (1.0, 5.0),
    "7-point": (1.0, 7.0),
    "9-point (SAM)": (1.0, 9.0),
}


def canonical_ranges_from_schema(schema: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Canonical value range per DV id (``measurement.canonical_range`` or the
    range written in ``primary_unit``), as used by the meta-analysis."""
    ranges: dict[str, tuple[float, float]] = {}
    for entry in schema.get("dvs") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        measurement = entry.get("measurement") or {}
        explicit = measurement.get("canonical_range")
        if isinstance(explicit, (list, tuple)) and len(explicit) == 2:
            ranges[str(entry["id"])] = (float(explicit[0]), float(explicit[1]))
            continue
        unit = str(measurement.get("primary_unit", ""))
        if unit in _KNOWN_UNIT_RANGES:
            ranges[str(entry["id"])] = _KNOWN_UNIT_RANGES[unit]
            continue
        for pattern in _SCALE_RANGE_PATTERNS:
            match = pattern.search(unit)
            if match:
                ranges[str(entry["id"])] = (
                    float(match.group(1).replace("+", "")),
                    float(match.group(2).replace("+", "")),
                )
                break
    return ranges


__all__ = [
    "ColumnProfile",
    "PlausibilityRules",
    "STAGE_POST_LLM",
    "STAGE_PRE_LLM",
    "canonical_ranges_from_schema",
    "check_llm_suggestion",
    "data_rejection_reason",
    "load_rules",
    "name_rejection_reason",
    "name_tokens",
    "profile_column",
    "profile_columns",
    "range_is_compatible",
    "screen_llm_candidates",
]
