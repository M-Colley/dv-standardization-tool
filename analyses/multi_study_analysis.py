#!/usr/bin/env python3
"""Cross-study analysis for standardized DV datasets.

This script demonstrates analyses that remain informative even when independent
variables are unknown or inconsistent across studies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import lingam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from sklearn.decomposition import PCA

# libyaml's C parser is ~7x faster on the 2k-line DV schema and yields
# identical objects; fall back to the pure-Python loader when unavailable.
_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


@lru_cache(maxsize=16)
def _read_yaml_cached(path_str: str, mtime_ns: int) -> Any:
    with open(path_str, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_YAML_SAFE_LOADER)


def _load_yaml(path: Path) -> Any:
    """Parse a schema YAML once per (path, mtime). Callers must not mutate the result."""
    return _read_yaml_cached(str(path), path.stat().st_mtime_ns)

ID_LIKE_COLUMNS = {
    "id",
    "participant_id",
    "userid",
    "user_id",
    "subject_id",
    "session_id",
    "submitdate",
    "seed",
    "lastpage",
}
DEFAULT_DV_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "standard_dv_mapping.yaml"
HARMONIZED_SUMMARY_COLUMNS = [
    "study", "dv", "n", "mean", "sd", "mean_z_vs_global", "scale_note",
    "mapping_source",
]
META_ANALYSIS_COLUMNS = [
    "dv",
    "k_studies",
    "study_coverage_pct",
    "pooling_method",
    "random_effects_mean",
    "random_effects_se",
    "ci95_low",
    "ci95_high",
    "hksj_se",
    "hksj_ci95_low",
    "hksj_ci95_high",
    "prediction_interval_low",
    "prediction_interval_high",
    "heterogeneity_q",
    "q_pvalue",
    "heterogeneity_i2_pct",
    "heterogeneity_warning",
    "tau2",
    "tau",
    "h2",
    "estimator",
    "mapping_source_categories",
    "k_llm_deduced",
    "includes_llm_deduced",
]
# DerSimonian-Laird tau² is unstable at k=2 — fall back to a fixed-effects
# estimate so the pooled mean is at least well-defined. The result is still
# emitted (so the DV doesn't silently vanish), just flagged.
MIN_K_FOR_RANDOM_EFFECTS = 3
HIGH_HETEROGENEITY_I2_THRESHOLD = 75.0
STANDARDIZED_EFFECTS_COLUMNS = [
    "study", "dv", "cohens_d", "hedges_g", "var_g", "se_g", "mapping_source",
]

# ── Mapping-source provenance ───────────────────────────────────────────────
# Categories assigned to each (study, canonical_dv) pair based on how the DV
# alias was resolved during batch standardization.  Used to distinguish
# explicit-schema alignments from uncertain LLM-deduced ones so the meta-
# analysis can quarantine or down-weight the latter.
_MAPPING_SOURCE_PRIORITY = {
    "llm_deduced": 3,
    "repo_mapping": 2,
    "schema": 1,
    "blocked": 0,
    "unknown": -1,
}


def _categorize_mapping_source(raw: object) -> str:
    """Map a raw mapping_source string from meta_view.csv to a coarse category."""
    if raw is None:
        return "unknown"
    try:
        if pd.isna(raw):
            return "unknown"
    except (TypeError, ValueError):
        pass
    text = str(raw).strip().lower()
    if not text or text == "n/a":
        return "unknown"
    if text == "llm_deduction":
        return "llm_deduced"
    if text in {"never_map_blocklist", "blocked"}:
        return "blocked"
    if text == "in_memory_mapping":
        return "schema"
    schema_markers = (
        "standard_dv_mapping",
        "standard_dv_metadata",
        "standard_detection",
        "standard_sensor_mapping",
        "schemas/",
        "schemas\\",
    )
    if any(marker in text for marker in schema_markers):
        return "schema"
    return "repo_mapping"


def load_mapping_provenance(meta_view_path: Path) -> Dict[tuple, str]:
    """Build a ``{(source_id, canonical_dv): category}`` index from meta_view.csv.

    ``meta_view.csv`` is emitted by ``scripts/run_batch_standardization.py``
    and records the raw ``mapping_source`` per dataset/column.  When a pair
    has multiple rows with conflicting categories, the most cautious (highest
    priority) category wins, so any LLM-deduced appearance flags the DV.

    Returns an empty dict when the file is absent or malformed so callers can
    safely fall back to ``mapping_source == "unknown"``.
    """
    if not meta_view_path.is_file():
        return {}
    try:
        df = pd.read_csv(meta_view_path)
    except Exception:
        return {}
    if "source_id" not in df.columns or "canonical_dv" not in df.columns:
        return {}
    frame = pd.DataFrame({
        "source_id": df["source_id"].astype(str),
        "canonical_dv": df["canonical_dv"].astype(str),
        "mapping_source": df["mapping_source"] if "mapping_source" in df.columns else None,
    })
    frame = frame[~frame["canonical_dv"].isin({"", "nan", "None"})]
    if frame.empty:
        return {}
    # meta_view has one row per (dataset, column) — tens of thousands for
    # telemetry-heavy runs — but only a handful of distinct mapping sources.
    categories = {
        raw: _categorize_mapping_source(raw)
        for raw in frame["mapping_source"].drop_duplicates()
    }
    frame["category"] = frame["mapping_source"].map(categories).fillna("unknown")
    frame["priority"] = frame["category"].map(_MAPPING_SOURCE_PRIORITY).fillna(-1)
    # idxmax keeps the first row on ties, i.e. the earliest-seen category wins
    # unless a later one is strictly more cautious.
    winners = frame.loc[frame.groupby(["source_id", "canonical_dv"], sort=False)["priority"].idxmax()]
    return {
        (source_id, canonical_dv): category
        for source_id, canonical_dv, category in zip(
            winners["source_id"], winners["canonical_dv"], winners["category"], strict=True
        )
    }


OVERLAP_DETAIL_COLUMNS = [
    "study_a",
    "study_b",
    "shared_dv_count",
    "union_dv_count",
    "jaccard_overlap",
    "shared_dvs",
    "study_a_only_dvs",
    "study_b_only_dvs",
]

# ── Scale range parsing ──────────────────────────────────────────────────────

_SCALE_RANGE_PATTERNS = [
    # "21-point (0-20)", "21-point (0–20)"
    re.compile(r"\d+-point\s*\((\-?\d+(?:\.\d+)?)\s*[-–]\s*(\-?\d+(?:\.\d+)?)\)"),
    # "5-point (-2 to +2)"
    re.compile(r"\d+-point\s*\((\-?\+?\d+(?:\.\d+)?)\s+to\s+(\+?\-?\d+(?:\.\d+)?)\)"),
]

_KNOWN_UNIT_RANGES: Dict[str, tuple[float, float]] = {
    "proportion": (0.0, 1.0),
    "percentage": (0.0, 100.0),
    "5-point": (1.0, 5.0),
    "7-point": (1.0, 7.0),
    "9-point (SAM)": (1.0, 9.0),
}


def _parse_scale_range(primary_unit: str) -> tuple[float, float] | None:
    if primary_unit in _KNOWN_UNIT_RANGES:
        return _KNOWN_UNIT_RANGES[primary_unit]
    for pat in _SCALE_RANGE_PATTERNS:
        m = pat.search(primary_unit)
        if m:
            lo = float(m.group(1).replace("+", ""))
            hi = float(m.group(2).replace("+", ""))
            return (lo, hi)
    return None


# ── Schema helpers ───────────────────────────────────────────────────────────

def _normalize_colname(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {_normalize_colname(col): col for col in df.columns}


@lru_cache(maxsize=4)
def _load_standard_dv_lookup(schema_path: str = str(DEFAULT_DV_SCHEMA_PATH)) -> Dict[str, str]:
    path = Path(schema_path)
    if not path.is_file():
        return {}

    try:
        data = _load_yaml(path) or {}
    except OSError:
        return {}

    lookup: Dict[str, str] = {}
    for entry in data.get("dvs", []):
        if not isinstance(entry, dict):
            continue
        canonical = str(entry.get("id", "")).strip()
        if not canonical:
            continue

        candidates = [canonical, str(entry.get("label", "")).strip(), *(entry.get("aliases") or [])]
        for candidate in candidates:
            normalized = _normalize_colname(str(candidate))
            if normalized and normalized not in lookup:
                lookup[normalized] = canonical

    return lookup


@lru_cache(maxsize=4)
def _load_dv_measurement_metadata(
    schema_path: str = str(DEFAULT_DV_SCHEMA_PATH),
) -> Dict[str, dict]:
    """Return measurement metadata per canonical DV id.

    Each value dict has keys: primary_unit, scale_type, direction, cluster,
    and optionally canonical_range (parsed from primary_unit).
    """
    path = Path(schema_path)
    if not path.is_file():
        return {}

    try:
        data = _load_yaml(path) or {}
    except OSError:
        return {}

    meta: Dict[str, dict] = {}
    for entry in data.get("dvs", []):
        if not isinstance(entry, dict):
            continue
        dv_id = str(entry.get("id", "")).strip()
        if not dv_id:
            continue
        meas = entry.get("measurement") or {}
        primary_unit = str(meas.get("primary_unit", ""))
        info: dict = {
            "primary_unit": primary_unit,
            "scale_type": str(meas.get("scale_type", "")),
            "direction": str(meas.get("direction", "neutral")),
            "cluster": str(entry.get("cluster", "")),
        }
        cr = meas.get("canonical_range")
        if cr and isinstance(cr, (list, tuple)) and len(cr) == 2:
            info["canonical_range"] = (float(cr[0]), float(cr[1]))
        elif rng := _parse_scale_range(primary_unit):
            info["canonical_range"] = rng
        meta[dv_id] = info

    return meta


def validate_schema_clusters(
    dv_schema_path: Path = DEFAULT_DV_SCHEMA_PATH,
    cluster_schema_path: Path | None = None,
) -> None:
    """Warn about any DV cluster references not defined in thematic_clusters.yaml."""
    if cluster_schema_path is None:
        cluster_schema_path = dv_schema_path.parent / "thematic_clusters.yaml"

    # Load DV schema
    if not dv_schema_path.is_file():
        return
    try:
        dv_data = _load_yaml(dv_schema_path) or {}
    except OSError:
        return

    # Collect all cluster IDs referenced by DVs
    referenced_clusters: set[str] = set()
    for entry in dv_data.get("dvs", []):
        if not isinstance(entry, dict):
            continue
        cluster = str(entry.get("cluster", "")).strip()
        if cluster:
            referenced_clusters.add(cluster)

    if not referenced_clusters:
        return

    # If thematic_clusters.yaml doesn't exist, skip gracefully
    if not cluster_schema_path.is_file():
        return

    try:
        cluster_data = _load_yaml(cluster_schema_path) or {}
    except OSError:
        return

    # Collect defined cluster IDs
    defined_clusters: set[str] = set()
    clusters_list = cluster_data.get("clusters", [])
    if isinstance(clusters_list, list):
        for entry in clusters_list:
            if isinstance(entry, dict):
                cid = str(entry.get("id", "")).strip()
                if cid:
                    defined_clusters.add(cid)
            elif isinstance(entry, str):
                defined_clusters.add(entry.strip())
    elif isinstance(clusters_list, dict):
        defined_clusters.update(clusters_list.keys())

    _logger = logging.getLogger(__name__)
    for cluster_id in sorted(referenced_clusters - defined_clusters):
        _logger.warning(
            "Schema cluster '%s' is referenced by DVs but not defined in %s",
            cluster_id,
            cluster_schema_path.name,
        )


def _canonicalize_dv_name(name: str) -> str | None:
    return _load_standard_dv_lookup().get(_normalize_colname(str(name)))


# ── Scale harmonization ─────────────────────────────────────────────────────

def _detect_scale_range(series: pd.Series, canonical_range: tuple[float, float]) -> tuple[float, float]:
    """Heuristically detect the actual scale range of a data series.

    If the observed data clearly exceeds the canonical range, infer an
    alternative scale (e.g. a 0-20 Likert when canonical is 1-5).
    More specific ranges are tried before broad catch-alls like 0-100.
    """
    smin, smax = float(series.min()), float(series.max())
    cmin, cmax = canonical_range
    cspan = cmax - cmin
    if cspan <= 0:
        return (smin, smax)

    # If data fits within canonical range (with small tolerance), keep it.
    # PREFER canonical range — it's the authoritative definition.
    tol = cspan * 0.15
    if smin >= cmin - tol and smax <= cmax + tol:
        return canonical_range

    # Try common alternative ranges — more specific first so a 0-20 Likert
    # is not swallowed by the broad 0-100 VAS catch-all.
    # Only consider alternatives that are WIDER than the canonical range.
    for alt_lo, alt_hi in [
        (0, 20), (1, 20),   # 21-point scales (common in psychology)
        (0, 21), (1, 21),   # explicit 21-point
        (0, 10), (1, 10),   # 11-point
        (0, 9), (1, 9),     # 9-point (SAM)
        (1, 7), (0, 7),     # 7-point
        (1, 5), (0, 5),     # 5-point
        (0, 100), (1, 100), # VAS / percentage (broad catch-all last)
    ]:
        alt_tol = (alt_hi - alt_lo) * 0.15
        if smin >= alt_lo - alt_tol and smax <= alt_hi + alt_tol:
            return (float(alt_lo), float(alt_hi))

    return (smin, smax)


def _rescale_to_canonical(
    series: pd.Series,
    detected_range: tuple[float, float],
    canonical_range: tuple[float, float],
) -> pd.Series:
    d_lo, d_hi = detected_range
    c_lo, c_hi = canonical_range
    if d_hi == d_lo:
        return series
    return c_lo + (series - d_lo) * (c_hi - c_lo) / (d_hi - d_lo)


# ── Derived scale scores ────────────────────────────────────────────────────

def _resolve_series(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    lookup = _column_lookup(df)
    for cand in candidates:
        col = lookup.get(_normalize_colname(cand))
        if col is not None:
            return pd.to_numeric(df[col], errors="coerce")
    return None


# ── Derived-scale registry ─────────────────────────────────────────────────
# Each entry maps a higher-order construct to its sub-item candidate lists.
# `add_derived_scale_scores` and `CONSTRUCT_SUBITEMS` are BOTH derived from
# this single registry so they never go out of sync.  To add a new derived
# scale, just add an entry here and a scoring function below.

_DERIVED_SCALES: dict[str, dict] = {
    # NASA-TLX overall score
    "nasa_tlx_score": {
        "item_candidates": [
            ["tlx1", "nasa_tlx1", "mental_demand", "tlx_mental_demand", "nasa_tlx_mental"],
            ["tlx2", "nasa_tlx2", "physical_demand"],
            ["tlx3", "nasa_tlx3", "temporal_demand"],
            ["tlx4", "nasa_tlx4", "performance"],
            ["tlx5", "nasa_tlx5", "effort"],
            ["tlx6", "nasa_tlx6", "frustration"],
        ],
        # Canonical DV ids of the sub-items (used for DAG exclusion)
        "canonical_subitems": [
            "mental_demand", "physical_demand", "temporal_demand",
            "performance", "effort", "frustration",
        ],
        "formula": "mean",
    },
    # SUS total score  (canonical id: usability)
    "sus_score": {
        "item_candidates": [
            ["sus1", "sus_1"], ["sus2", "sus_2"], ["sus3", "sus_3"],
            ["sus4", "sus_4"], ["sus5", "sus_5"], ["sus6", "sus_6"],
            ["sus7", "sus_7"], ["sus8", "sus_8"], ["sus9", "sus_9"],
            ["sus10", "sus_10"],
        ],
        "canonical_subitems": [f"SUS{i}" for i in range(1, 11)],
        "formula": "custom",
    },
    # AOA usefulness subscale
    "aoa_usefulness": {
        "item_candidates": [
            ["aoa1", "aoa_1"], ["aoa2", "aoa_2"], ["aoa3", "aoa_3"],
            ["aoa4", "aoa_4"], ["aoa5", "aoa_5"], ["aoa6", "aoa_6"],
            ["aoa7", "aoa_7"], ["aoa8", "aoa_8"], ["aoa9", "aoa_9"],
        ],
        "canonical_subitems": [f"AOA{i}" for i in range(1, 10)],
        "formula": "custom",
    },
    # AOA satisfying subscale  (shares same items as usefulness)
    "aoa_satisfying": {
        "item_candidates": [
            ["aoa1", "aoa_1"], ["aoa2", "aoa_2"], ["aoa3", "aoa_3"],
            ["aoa4", "aoa_4"], ["aoa5", "aoa_5"], ["aoa6", "aoa_6"],
            ["aoa7", "aoa_7"], ["aoa8", "aoa_8"], ["aoa9", "aoa_9"],
        ],
        "canonical_subitems": [f"AOA{i}" for i in range(1, 10)],
        "formula": "custom",
    },
}


def _build_construct_subitems() -> dict[str, list[str]]:
    """Build construct → sub-item mapping from the derived-scale registry.

    The keys are the *canonical DV ids* that the derived score maps to
    (looked up via ``_canonicalize_dv_name``), so the mapping stays correct
    even when the schema renames things.
    """
    lookup = _load_standard_dv_lookup()
    mapping: dict[str, list[str]] = {}
    for col_name, spec in _DERIVED_SCALES.items():
        # Resolve the column name to its canonical DV id
        canonical = lookup.get(_normalize_colname(col_name), col_name)
        subitems = spec.get("canonical_subitems", [])
        if subitems:
            existing = mapping.get(canonical, [])
            merged = list(dict.fromkeys(existing + subitems))  # dedupe, preserve order
            mapping[canonical] = merged
    return mapping


def _aoa_scale_midpoint(items: list[pd.Series]) -> float:
    """Midpoint of the 5-point coding used for AOA items: 0 for -2..+2, 3 for 1..5."""
    values = pd.concat(items, ignore_index=True).dropna()
    if values.empty:
        return 3.0
    lo, hi = float(values.min()), float(values.max())
    if lo < 0:
        return 0.0
    if lo >= 1:
        return 3.0
    if hi <= 2:
        # Only 0..2 observed: cannot tell -2..+2 (no negative answers) from
        # 0..4; the -2..+2 export is the one seen in practice.
        logging.getLogger(__name__).warning(
            "AOA items span %.0f..%.0f; assuming -2..+2 coding (midpoint 0)", lo, hi
        )
        return 0.0
    return 2.0  # 0..4 coding


def add_derived_scale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add TLX/SUS/AOA derived scores when all required mapped item columns are present.

    Scale definitions come from ``_DERIVED_SCALES`` — add new entries there
    so the DAG sub-item filter picks them up automatically.

    Entries with ``formula: "mean"`` are computed automatically as the mean of all
    item series.  Entries with ``formula: "custom"`` use bespoke Python code below.
    Adding a new scale with ``formula: "mean"`` requires no additional Python code.
    """
    out = df.copy()

    # --- Auto-compute "mean" formula scales ---
    for scale_name, spec in _DERIVED_SCALES.items():
        if spec.get("formula") != "mean":
            continue
        items = [_resolve_series(out, cands) for cands in spec["item_candidates"]]
        if all(item is not None for item in items):
            out[scale_name] = pd.concat(items, axis=1).mean(axis=1, skipna=False)

    # --- Custom formula scales ---

    # SUS: alternating forward/reverse scoring, scaled to 0-100
    spec = _DERIVED_SCALES["sus_score"]
    sus_items = [_resolve_series(out, cands) for cands in spec["item_candidates"]]
    if all(item is not None for item in sus_items):
        out["sus_score"] = (
            (sus_items[0] - 1)
            + (sus_items[2] - 1)
            + (sus_items[4] - 1)
            + (sus_items[6] - 1)
            + (sus_items[8] - 1)
            + (5 - sus_items[1])
            + (5 - sus_items[3])
            + (5 - sus_items[5])
            + (5 - sus_items[7])
            + (5 - sus_items[9])
        ) * 2.5

    # AOA (Van der Laan acceptance): usefulness and satisfying subscales.
    # Items are stored as the answer *position* on a 5-point semantic
    # differential; items 3, 6 and 8 have the positive pole on the other side.
    # Each item is centred on the scale midpoint (mid - x, or x - mid for the
    # mirrored items), so scores are -2..+2 whether a study codes positions
    # 1..5 (midpoint 3) or -2..+2 (midpoint 0, e.g. LimeSurvey "neg2".."2").
    spec = _DERIVED_SCALES["aoa_usefulness"]
    aoa_items = [_resolve_series(out, cands) for cands in spec["item_candidates"]]
    if all(item is not None for item in aoa_items):
        mid = _aoa_scale_midpoint(aoa_items)
        out["aoa_usefulness"] = (
            (mid - aoa_items[0])
            + (aoa_items[2] - mid)
            + (mid - aoa_items[4])
            + (mid - aoa_items[6])
            + (mid - aoa_items[8])
        ) / 5.0
        out["aoa_satisfying"] = (
            (mid - aoa_items[1])
            + (mid - aoa_items[3])
            + (aoa_items[5] - mid)
            + (aoa_items[7] - mid)
        ) / 4.0

    return out


# ── Study loading ────────────────────────────────────────────────────────────

_PICKLE_SUFFIXES = {".pkl", ".pickle"}


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in _PICKLE_SUFFIXES:
        return pd.read_pickle(path)
    return pd.read_excel(path)


def _read_header(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return [str(c) for c in pd.read_csv(path, nrows=0).columns]
    return [str(c) for c in pd.read_excel(path, nrows=0).columns]


def _derived_scale_item_names() -> frozenset[str]:
    return frozenset(
        _normalize_colname(candidate)
        for spec in _DERIVED_SCALES.values()
        for candidates in spec["item_candidates"]
        for candidate in candidates
    )


def _is_analysis_column(name: str, derived_items: frozenset[str] | None = None) -> bool:
    """True when a column can feed any analysis: a canonical DV or a derived-scale item."""
    if derived_items is None:
        derived_items = _derived_scale_item_names()
    return bool(_canonicalize_dv_name(name)) or _normalize_colname(name) in derived_items


def load_studies(
    input_dir: Path,
    repeated_measures_studies: set[str] | None = None,
    skip_files_without_dvs: bool = False,
    max_workers: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load studies, combining all files that share the same subdirectory.

    Directory layout
    ----------------
    input_dir/
        study_a/
            part1.csv       ← combined into study "study_a"
            part2.csv
        study_b/
            results.xlsx    ← single-file study "study_b"
        study_c.csv         ← files at root are treated as individual studies

    Each subdirectory becomes one study key (with row-wise concatenation).
    Files directly in input_dir are treated as separate studies by file stem.
    Derived scale scores are computed after grouping.

    Parameters
    ----------
    repeated_measures_studies:
        When a study key is in this set, rows are aggregated to participant-level
        means (detected by ID_LIKE_COLUMNS) before storing in the output dict.
    skip_files_without_dvs:
        Read each CSV/XLSX header first and skip files with no canonical DV
        (or derived-scale item) column — typically per-trial telemetry logs.
        Such files only contribute rows in which every DV is missing, which no
        analysis uses, so results are unchanged while loading gets much
        faster. A study whose files are all skipped is kept as an empty frame
        so study counts and coverage percentages stay the same.
    max_workers:
        Threads used to read files (default: up to 8).
    """
    _load_logger = logging.getLogger(__name__)
    files = sorted(
        list(input_dir.rglob("*.csv"))
        + list(input_dir.rglob("*.xlsx"))
        + list(input_dir.rglob("*.pkl"))
        + list(input_dir.rglob("*.pickle"))
    )

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    if not files:
        raise FileNotFoundError(f"No CSV/XLSX/PKL files found in {input_dir}")

    # Group files by their immediate parent directory relative to input_dir.
    # Files sitting directly in input_dir are treated as separate studies.
    groups: Dict[str, List[Path]] = {}
    for path in files:
        rel_parent = path.parent.relative_to(input_dir)
        key = path.stem if str(rel_parent) == "." else str(rel_parent)
        groups.setdefault(key, []).append(path)

    rm_keys = repeated_measures_studies or set()

    # Computed here (not cached globally) so edits to _DERIVED_SCALES apply;
    # also warms the schema lookup cache before threads share it.
    derived_items = _derived_scale_item_names()
    _load_standard_dv_lookup()

    def _load(path: Path) -> pd.DataFrame | None:
        if skip_files_without_dvs and path.suffix.lower() not in _PICKLE_SUFFIXES:
            if not any(_is_analysis_column(col, derived_items) for col in _read_header(path)):
                return None
        return _read_file(path)
    n_threads = max_workers or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        loaded = dict(zip(files, pool.map(_load, files), strict=True))
    n_skipped = sum(1 for df in loaded.values() if df is None)
    if n_skipped:
        print(f"Skipped {n_skipped} file(s) without canonical DV columns.")

    studies: Dict[str, pd.DataFrame] = {}
    for project_key, paths in sorted(groups.items()):
        frames = []
        for path in paths:
            df = loaded[path]
            if df is None:
                continue
            df["_source_file"] = path.name  # traceability column
            frames.append(df)
            print(f"  [{project_key}] loaded {path.name} ({len(df)} rows)")

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        print(
            f"  -> project '{project_key}': {len(combined)} total rows "
            f"from {len(frames)} of {len(paths)} file(s)"
        )
        combined = add_derived_scale_scores(combined)

        # Repeated-measures aggregation: collapse to participant-level means
        if project_key in rm_keys:
            id_col = next(
                (col for col in combined.columns if col.lower() in ID_LIKE_COLUMNS),
                None,
            )
            if id_col is not None:
                n_raw = len(combined)
                numeric_cols = [
                    c for c in combined.select_dtypes(include="number").columns
                    if c != id_col
                ]
                agg_df = (
                    combined.groupby(id_col)[numeric_cols]
                    .mean()
                    .reset_index()
                )
                n_agg = len(agg_df)
                _load_logger.info(
                    "Study '%s': aggregated %d rows -> %d participant means (repeated measures)",
                    project_key, n_raw, n_agg,
                )
                combined = agg_df
            else:
                _load_logger.warning(
                    "Study '%s': marked as repeated-measures but no participant ID column found "
                    "(checked %s); skipping aggregation",
                    project_key, sorted(ID_LIKE_COLUMNS),
                )

        studies[project_key] = combined

    return studies


# ── Canonical DV extraction ──────────────────────────────────────────────────

def _canonical_series_priority(series: pd.Series, raw_name: str, canonical_name: str) -> tuple[int, int, int]:
    non_null = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    exact_match = int(_normalize_colname(raw_name) == _normalize_colname(canonical_name))
    return (non_null, exact_match, unique)


def _canonical_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    selected: dict[str, pd.Series] = {}
    priorities: dict[str, tuple[int, int, int]] = {}

    for col in df.columns:
        lowered = col.lower()
        if lowered in ID_LIKE_COLUMNS or col == "_source_file":
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        canonical = _canonicalize_dv_name(col)
        if not canonical:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue

        priority = _canonical_series_priority(series, col, canonical)
        if canonical not in selected or priority > priorities[canonical]:
            selected[canonical] = series
            priorities[canonical] = priority

    if not selected:
        return pd.DataFrame(index=df.index)

    ordered = {canonical: selected[canonical] for canonical in sorted(selected)}
    return pd.DataFrame(ordered, index=df.index)


def _canonicalize_studies(studies: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {name: _canonical_numeric_frame(df) for name, df in studies.items()}


def _resolve_canonical(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None,
) -> Dict[str, pd.DataFrame]:
    """Return precomputed canonical frames, or canonicalize on demand.

    Canonicalization scans every column of every study; orchestrators
    (``main``, ``run_catalog_meta_analysis``) compute it once and pass it to
    every analysis function instead of paying the scan ~12 times per run.
    """
    if canonical_studies is not None:
        return canonical_studies
    return _canonicalize_studies(studies)


def numeric_dvs(df: pd.DataFrame) -> List[str]:
    return list(_canonical_numeric_frame(df).columns)


def _study_numeric_dv_sets_from_canonical(studies: Dict[str, pd.DataFrame]) -> Dict[str, set[str]]:
    return {name: set(df.columns) for name, df in studies.items()}


def study_numeric_dv_sets(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, set[str]]:
    return _study_numeric_dv_sets_from_canonical(_resolve_canonical(studies, canonical_studies))


# ── Overlap analysis ─────────────────────────────────────────────────────────

def compute_overlap(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    dv_sets = _study_numeric_dv_sets_from_canonical(_resolve_canonical(studies, canonical_studies))
    index = list(studies.keys())
    overlap = pd.DataFrame(index=index, columns=index, dtype=float)
    for a in index:
        for b in index:
            union = dv_sets[a] | dv_sets[b]
            overlap.loc[a, b] = len(dv_sets[a] & dv_sets[b]) / len(union) if union else np.nan
    return overlap


def compute_dv_presence_matrix(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    dv_sets = _study_numeric_dv_sets_from_canonical(_resolve_canonical(studies, canonical_studies))
    all_dvs = sorted({dv for dvs in dv_sets.values() for dv in dvs})
    presence = pd.DataFrame(0, index=sorted(studies.keys()), columns=all_dvs, dtype=int)
    for study, dvs in dv_sets.items():
        for dv in dvs:
            presence.loc[study, dv] = 1
    return presence


def compute_overlap_details(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    dv_sets = _study_numeric_dv_sets_from_canonical(_resolve_canonical(studies, canonical_studies))
    rows = []
    for study_a, study_b in combinations(sorted(dv_sets.keys()), 2):
        shared = sorted(dv_sets[study_a] & dv_sets[study_b])
        union = sorted(dv_sets[study_a] | dv_sets[study_b])
        rows.append(
            {
                "study_a": study_a,
                "study_b": study_b,
                "shared_dv_count": len(shared),
                "union_dv_count": len(union),
                "jaccard_overlap": (len(shared) / len(union)) if union else np.nan,
                "shared_dvs": "; ".join(shared),
                "study_a_only_dvs": "; ".join(sorted(dv_sets[study_a] - dv_sets[study_b])),
                "study_b_only_dvs": "; ".join(sorted(dv_sets[study_b] - dv_sets[study_a])),
            }
        )
    return pd.DataFrame(rows, columns=OVERLAP_DETAIL_COLUMNS)


# ── Harmonized summary (with optional scale harmonization) ───────────────────

def harmonized_summary(
    studies: Dict[str, pd.DataFrame],
    harmonize_scales: bool = False,
    mapping_provenance: Dict[tuple, str] | None = None,
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    metadata = _load_dv_measurement_metadata() if harmonize_scales else {}
    rows = []
    for study, df in _resolve_canonical(studies, canonical_studies).items():
        for dv in df.columns:
            s = df[dv].dropna()
            scale_note = ""

            if harmonize_scales and dv in metadata:
                info = metadata[dv]
                canonical_range = info.get("canonical_range")
                if canonical_range is not None and len(s) > 0:
                    detected = _detect_scale_range(s, canonical_range)
                    if detected != canonical_range:
                        s = _rescale_to_canonical(s, detected, canonical_range)
                        scale_note = f"rescaled {detected[0]}-{detected[1]} -> {canonical_range[0]}-{canonical_range[1]}"

            mapping_source = (
                mapping_provenance.get((study, dv), "unknown")
                if mapping_provenance
                else "unknown"
            )
            rows.append(
                {
                    "study": study,
                    "dv": dv,
                    "n": s.shape[0],
                    "mean": s.mean(),
                    "sd": s.std(ddof=1),
                    "scale_note": scale_note,
                    "mapping_source": mapping_source,
                }
            )
    summary = pd.DataFrame(rows, columns=HARMONIZED_SUMMARY_COLUMNS)
    if summary.empty:
        return pd.DataFrame(columns=HARMONIZED_SUMMARY_COLUMNS)
    global_mean = summary.groupby("dv")["mean"].transform("mean")
    pooled_sd = summary.groupby("dv")["sd"].transform(
        lambda x: (
            np.sqrt(np.nanmean(np.square(x.dropna())))
            if not x.dropna().empty
            else np.nan
        )
    )
    summary["mean_z_vs_global"] = (summary["mean"] - global_mean) / pooled_sd.replace(0, np.nan)
    return summary.sort_values(["dv", "study"])


# ── Standardized effect sizes ────────────────────────────────────────────────

def compute_standardized_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute the study-vs-pool standardized deviation (Hedges' g) per DV.

    IMPORTANT: this is a *descriptive* one-group effect size, **not** a
    contrast-based effect (there is no IV, no control group, no paired
    condition).  Each row expresses how far one study's mean sits from the
    cross-study grand mean, in pooled-SD units, with Hedges' small-sample
    correction.  Do not interpret these as evidence of an intervention
    effect — treat them as a normalized view of between-study location shifts.
    The values are also written to ``study_vs_pool_standardized_deviation.csv``
    under that clearer name.
    """
    rows = []
    for dv, sub in summary.groupby("dv"):
        if len(sub) < 2:
            continue
        sub = sub[(sub["n"] > 1) & (sub["sd"] > 0)].copy()
        if len(sub) < 2:
            continue

        grand_mean = np.average(sub["mean"], weights=sub["n"])
        pooled_sd = np.sqrt(np.average(sub["sd"] ** 2, weights=sub["n"]))
        if pooled_sd <= 0:
            continue

        for _, row in sub.iterrows():
            n = row["n"]
            d = (row["mean"] - grand_mean) / pooled_sd
            # Hedges' correction factor (exact for n >= 4, good approx otherwise)
            j = 1 - 3 / (4 * (n - 1) - 1) if n > 1 else 1.0
            g = d * j
            var_g = (1 / n) + (g ** 2 / (2 * n))
            rows.append({
                "study": row["study"],
                "dv": dv,
                "cohens_d": d,
                "hedges_g": g,
                "var_g": var_g,
                "se_g": np.sqrt(var_g),
                "mapping_source": row.get("mapping_source", "unknown"),
            })
    return pd.DataFrame(rows, columns=STANDARDIZED_EFFECTS_COLUMNS)


# Clearer public alias — preferred name for new callers.
compute_study_vs_pool_standardized_deviation = compute_standardized_effects


# ── Tau-squared estimators ───────────────────────────────────────────────────

def _estimate_tau2_dl(effects: np.ndarray, variances: np.ndarray) -> float:
    """DerSimonian-Laird estimator for tau-squared."""
    w_fixed = 1.0 / variances
    fixed_mean = np.sum(w_fixed * effects) / np.sum(w_fixed)
    q = np.sum(w_fixed * np.square(effects - fixed_mean))
    df_q = len(effects) - 1
    c = np.sum(w_fixed) - (np.sum(np.square(w_fixed)) / np.sum(w_fixed))
    return max(0.0, (q - df_q) / c) if c > 0 else 0.0


def _estimate_tau2_reml(
    effects: np.ndarray,
    variances: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Restricted Maximum Likelihood (REML) estimator for tau-squared.

    Uses Fisher scoring iteration. Falls back to DL if convergence fails.
    """
    k = len(effects)
    if k < 2:
        return 0.0

    tau2 = _estimate_tau2_dl(effects, variances)  # warm start

    for _ in range(max_iter):
        w = 1.0 / (variances + tau2)
        mu = np.sum(w * effects) / np.sum(w)
        residuals_sq = np.square(effects - mu)

        # REML score and expected (Fisher) information for tau², with
        # P = W - w w' / sum(w):  score = (y'PPy - tr P) / 2,  I = tr(PP) / 2,
        # tr(PP) = Σw² - 2Σw³/Σw + (Σw²/Σw)²  (Viechtbauer, 2005).
        sum_w = np.sum(w)
        sum_w2 = np.sum(w ** 2)
        gradient = -0.5 * sum_w + 0.5 * np.sum(w ** 2 * residuals_sq) + 0.5 * (sum_w2 / sum_w)
        fisher_info = 0.5 * (sum_w2 - 2.0 * np.sum(w ** 3) / sum_w + (sum_w2 / sum_w) ** 2)

        if fisher_info <= 0:
            break

        tau2_new = tau2 + gradient / fisher_info
        tau2_new = max(0.0, tau2_new)

        if abs(tau2_new - tau2) < tol:
            return tau2_new
        tau2 = tau2_new

    return max(0.0, tau2)


# ── Meta-analysis ────────────────────────────────────────────────────────────

def _run_meta_for_dv(
    sub: pd.DataFrame,
    total_studies: int | None,
    estimator: str,
) -> dict | None:
    """Core random-effects meta-analysis for a single DV group."""
    sub = sub[(sub["n"] > 1) & (sub["sd"] > 0)].copy()
    if len(sub) < 2:
        return None
    dv = sub["dv"].iloc[0]
    k = len(sub)
    means = sub["mean"].values
    variances = (sub["sd"].values ** 2) / sub["n"].values
    if (variances <= 0).any():
        return None

    # Fixed-effects quantities (needed for Q regardless of estimator)
    w_fixed = 1.0 / variances
    fixed_mean = np.sum(w_fixed * means) / np.sum(w_fixed)
    q = float(np.sum(w_fixed * np.square(means - fixed_mean)))
    df_q = k - 1

    # Tau-squared and pooling. At k < MIN_K_FOR_RANDOM_EFFECTS, DL/REML are
    # unstable, so we report a fixed-effects pooled estimate and flag the
    # method explicitly. The DV is still kept so downstream readers see the
    # coverage; they can filter on pooling_method if they want only RE rows.
    if k < MIN_K_FOR_RANDOM_EFFECTS:
        tau2 = 0.0
        pooling_method = f"fixed_effects_k_lt_{MIN_K_FOR_RANDOM_EFFECTS}"
        random_mean = float(fixed_mean)
        random_se = float(np.sqrt(1.0 / np.sum(w_fixed)))
    else:
        if estimator == "REML":
            tau2 = _estimate_tau2_reml(means, variances)
        else:
            c = np.sum(w_fixed) - (np.sum(np.square(w_fixed)) / np.sum(w_fixed))
            tau2 = max(0.0, (q - df_q) / c) if c > 0 else 0.0
        pooling_method = "random_effects"
        w_random = 1.0 / (variances + tau2)
        random_mean = float(np.sum(w_random * means) / np.sum(w_random))
        random_se = float(np.sqrt(1.0 / np.sum(w_random)))

    # Hartung-Knapp-Sidik-Jonkman small-sample interval (metafor test="knha"):
    # the SE is rescaled by the weighted residual variance of the pooled model
    # and the interval uses t(k-1). With the k = 2-3 studies typical here the
    # normal-theory interval above is far too narrow (IntHout et al., 2014).
    w_pooled = w_fixed if k < MIN_K_FOR_RANDOM_EFFECTS else w_random
    q_hk = float(np.sum(w_pooled * np.square(means - random_mean))) / df_q
    hksj_se = float(np.sqrt(q_hk / np.sum(w_pooled)))
    hksj_half = float(stats.t.ppf(0.975, df_q)) * hksj_se

    # Heterogeneity
    i2 = max(0.0, (q - df_q) / q) * 100 if q > 0 else 0.0
    q_pvalue = float(stats.chi2.sf(q, df_q)) if df_q > 0 else np.nan
    h2 = q / df_q if df_q > 0 else np.nan
    tau = np.sqrt(tau2)
    heterogeneity_warning = (
        f"i2_exceeds_{HIGH_HETEROGENEITY_I2_THRESHOLD:g}pct"
        if i2 > HIGH_HETEROGENEITY_I2_THRESHOLD
        else ""
    )

    # Prediction interval (requires k >= 3 and random-effects pooling)
    if k >= 3 and pooling_method == "random_effects":
        t_crit = float(stats.t.ppf(0.975, k - 2))
        pi_half = t_crit * np.sqrt(tau2 + random_se ** 2)
        pi_low = random_mean - pi_half
        pi_high = random_mean + pi_half
    else:
        pi_low = np.nan
        pi_high = np.nan

    # Aggregate mapping-source provenance across contributing studies so
    # readers can see whether a pooled estimate rests on uncertain
    # LLM-deduced alignments or schema-backed ones.
    if "mapping_source" in sub.columns:
        source_values = [str(v) for v in sub["mapping_source"].tolist()]
    else:
        source_values = ["unknown"] * k
    k_llm_deduced = int(sum(1 for s in source_values if s == "llm_deduced"))
    categories_joined = "; ".join(sorted(set(source_values))) if source_values else "unknown"

    return {
        "dv": dv,
        "k_studies": k,
        "study_coverage_pct": (
            (k / total_studies) * 100.0
            if total_studies and total_studies > 0
            else np.nan
        ),
        "pooling_method": pooling_method,
        "random_effects_mean": random_mean,
        "random_effects_se": random_se,
        "ci95_low": random_mean - 1.96 * random_se,
        "ci95_high": random_mean + 1.96 * random_se,
        "hksj_se": hksj_se,
        "hksj_ci95_low": random_mean - hksj_half,
        "hksj_ci95_high": random_mean + hksj_half,
        "prediction_interval_low": pi_low,
        "prediction_interval_high": pi_high,
        "heterogeneity_q": q,
        "q_pvalue": q_pvalue,
        "heterogeneity_i2_pct": i2,
        "heterogeneity_warning": heterogeneity_warning,
        "tau2": tau2,
        "tau": tau,
        "h2": h2,
        "estimator": estimator,
        "mapping_source_categories": categories_joined,
        "k_llm_deduced": k_llm_deduced,
        "includes_llm_deduced": k_llm_deduced > 0,
    }


def meta_analysis_summary(
    summary: pd.DataFrame,
    total_studies: int | None = None,
    estimator: str = "DL",
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=META_ANALYSIS_COLUMNS)

    rows = []
    for dv, sub in summary.groupby("dv"):
        result = _run_meta_for_dv(sub, total_studies, estimator)
        if result is not None:
            rows.append(result)

    meta = pd.DataFrame(rows, columns=META_ANALYSIS_COLUMNS)
    if meta.empty:
        return meta
    return meta.sort_values("dv").reset_index(drop=True)


# ── Egger's regression test ──────────────────────────────────────────────────

def eggers_test(summary: pd.DataFrame, dv: str) -> dict | None:
    """Egger's regression test for funnel plot asymmetry.

    Regresses standardized effect (z = mean / se) on precision (1 / se).
    Requires k >= 3. Returns None if insufficient studies.
    """
    sub = summary[(summary["dv"] == dv) & (summary["n"] > 1) & (summary["sd"] > 0)].copy()
    if len(sub) < 3:
        return None

    se = sub["sd"].values / np.sqrt(sub["n"].values)
    precision = 1.0 / se
    z = sub["mean"].values / se

    slope, intercept, r_value, p_value, std_err = stats.linregress(precision, z)

    # The intercept test: t-statistic for intercept being zero
    k = len(sub)
    # Standard error of intercept from linear regression
    x = precision
    x_bar = np.mean(x)
    ss_x = np.sum((x - x_bar) ** 2)
    z_pred = intercept + slope * x
    residual_se = np.sqrt(np.sum((z - z_pred) ** 2) / (k - 2)) if k > 2 else np.nan
    se_intercept = residual_se * np.sqrt(1.0 / k + x_bar ** 2 / ss_x) if ss_x > 0 else np.nan

    t_stat = intercept / se_intercept if se_intercept and se_intercept > 0 else np.nan
    p_intercept = float(2 * stats.t.sf(abs(t_stat), k - 2)) if np.isfinite(t_stat) else np.nan

    return {
        "dv": dv,
        "k_studies": k,
        "intercept": intercept,
        "se_intercept": se_intercept,
        "t_stat": t_stat,
        "p_value": p_intercept,
        "significant_at_10pct": p_intercept < 0.10 if np.isfinite(p_intercept) else False,
    }


# ── Trim-and-fill (Duval & Tweedie) ─────────────────────────────────────────

def _estimate_tau2(effects: np.ndarray, variances: np.ndarray, method: str) -> float:
    """Between-study variance for ``method`` in {"FE", "DL", "REML"}."""
    if method == "FE":
        return 0.0
    if method == "REML":
        return _estimate_tau2_reml(effects, variances)
    if method == "DL":
        return _estimate_tau2_dl(effects, variances)
    raise ValueError(f"Unknown pooling method '{method}' (expected FE, DL or REML).")


def _pool_inverse_variance(
    effects: np.ndarray,
    variances: np.ndarray,
    method: str,
) -> tuple[float, float]:
    """Inverse-variance pooled estimate and its SE (fixed or random effects)."""
    tau2 = _estimate_tau2(effects, variances, method)
    w = 1.0 / (variances + tau2)
    return float(np.sum(w * effects) / np.sum(w)), float(np.sqrt(1.0 / np.sum(w)))


def _small_study_slope(effects: np.ndarray, variances: np.ndarray, method: str) -> float:
    """Slope of a meta-regression of the effects on their standard errors.

    Used to pick the side on which trim-and-fill imputes studies: a negative
    slope (small studies show smaller effects) means studies are missing on
    the right. Mirrors ``metafor::trimfill``, which fits
    ``rma(yi, vi, mods = sqrt(vi), method = <model method>)``; for the
    random-effects methods the residual tau² uses the DerSimonian-Laird
    meta-regression estimator (only the sign of the slope is used). Returns
    NaN when all studies have the same SE (slope undefined); callers then
    fall back to ``side="left"`` as metafor does when this model fails.
    """
    sei = np.sqrt(variances)
    if np.ptp(sei) <= 1e-12 * max(float(np.max(sei)), 1.0):
        return float("nan")
    X = np.column_stack([np.ones_like(effects), sei])

    def _wls(weights: np.ndarray) -> np.ndarray:
        xtw = X.T * weights
        return np.linalg.solve(xtw @ X, xtw @ effects)

    w = 1.0 / variances
    if method != "FE":
        k, p = X.shape
        beta = _wls(w)
        q_resid = float(np.sum(w * (effects - X @ beta) ** 2))
        xtwx_inv = np.linalg.inv((X.T * w) @ X)
        trace_p = float(np.sum(w) - np.trace(xtwx_inv @ ((X.T * w**2) @ X)))
        tau2 = max(0.0, (q_resid - (k - p)) / trace_p) if trace_p > 0 else 0.0
        w = 1.0 / (variances + tau2)
    return float(_wls(w)[1])


def trim_and_fill(
    summary: pd.DataFrame,
    dv: str,
    estimator: str = "DL",
    side: str | None = None,
    max_iter: int = 100,
    k0_estimator: str = "L0",
) -> dict | None:
    """Duval & Tweedie (2000) trim-and-fill funnel-asymmetry adjustment.

    Follows ``metafor::trimfill`` (Viechtbauer, 2010) and is validated against
    it in ``tests/test_meta_analysis_reference.py``:

    1. Flip the effects if studies are to be imputed on the right, then sort.
    2. Iterate: pool the ``k - k0`` least extreme studies, rank the absolute
       deviations of *all* studies from that estimate, and re-estimate the
       number of suppressed studies ``k0`` with the L0 (default) or R0
       estimator, until ``k0`` stops changing.
    3. Mirror the ``k0`` most extreme studies around the final truncated
       estimate and pool the original plus imputed studies.

    ``estimator`` is the pooling model used throughout ("FE", "DL" or
    "REML"), so the unadjusted estimate matches the main meta-analysis.
    ``side`` is the side on which studies are imputed ("left"/"right");
    ``None`` infers it from the sign of the small-study slope. Requires
    k >= 3. Trim-and-fill assumes funnel asymmetry stems from publication
    bias; under high heterogeneity it can impute spurious studies, so treat
    the adjusted estimate as a sensitivity analysis, not a correction.
    """
    if k0_estimator not in {"L0", "R0"}:
        raise ValueError(f"k0_estimator must be 'L0' or 'R0', got '{k0_estimator}'")
    sub = summary[(summary["dv"] == dv) & (summary["n"] > 1) & (summary["sd"] > 0)]
    if len(sub) < 3:
        return None

    effects = sub["mean"].to_numpy(dtype=float)
    variances = (sub["sd"].to_numpy(dtype=float) ** 2) / sub["n"].to_numpy(dtype=float)
    k = len(effects)
    pooled_orig, se_orig = _pool_inverse_variance(effects, variances, estimator)

    if side is None:
        side = "right" if _small_study_slope(effects, variances, estimator) < 0 else "left"
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left', 'right' or None, got '{side}'")

    # Work in a frame where suppressed studies are always on the left, i.e.
    # the excess (to be trimmed) sits at the top of the sorted effects.
    sign = -1.0 if side == "right" else 1.0
    order = np.argsort(sign * effects, kind="stable")
    y = sign * effects[order]
    v = variances[order]

    k0 = 0
    k0_previous = -1
    truncated_mean = float("nan")
    converged = False
    for _ in range(max_iter):
        truncated_mean, _ = _pool_inverse_variance(y[: k - k0], v[: k - k0], estimator)
        centered = y - truncated_mean
        signed_ranks = np.sign(centered) * stats.rankdata(np.abs(centered), method="ordinal")
        if k0_estimator == "L0":
            s_rank = float(np.sum(signed_ranks[signed_ranks > 0]))
            k0_raw = (4.0 * s_rank - k * (k + 1)) / (2.0 * k - 1.0)
        else:  # R0: length of the rightmost run of positive ranks, minus one
            negative = -signed_ranks[signed_ranks < 0]
            k0_raw = (k - float(np.max(negative))) - 1.0 if negative.size else k - 1.0
        # np.round rounds half to even, like R's round().
        k0 = int(max(0.0, np.round(k0_raw)))
        if k0 == k0_previous:
            converged = True
            break
        k0_previous = k0
    if not converged:
        logging.getLogger(__name__).warning(
            "trim_and_fill(%s): k0 did not converge in %d iterations", dv, max_iter
        )

    pooled_adj, se_adj = pooled_orig, se_orig
    if k0 > 0:
        filled = sign * (2.0 * truncated_mean - y[k - k0:])
        pooled_adj, se_adj = _pool_inverse_variance(
            np.concatenate([effects, filled]),
            np.concatenate([variances, v[k - k0:]]),
            estimator,
        )

    return {
        "dv": dv,
        "k_original": k,
        "k_imputed": k0,
        "k_total": k + k0,
        "side": side,
        "k0_estimator": k0_estimator,
        "pooling": estimator,
        "original_mean": pooled_orig,
        "original_se": se_orig,
        "adjusted_mean": pooled_adj,
        "adjusted_se": se_adj,
        "adjusted_ci95_low": pooled_adj - 1.96 * se_adj,
        "adjusted_ci95_high": pooled_adj + 1.96 * se_adj,
    }


# ── Meta-regression ─────────────────────────────────────────────────────────

def meta_regression(
    summary: pd.DataFrame,
    dv: str,
    moderator_col: str = "n",
) -> dict | None:
    """Weighted least squares meta-regression of study-level means on a moderator.

    Default moderator is sample size (n). Returns slope, intercept, R², and
    test of residual heterogeneity (QE).
    """
    sub = summary[(summary["dv"] == dv) & (summary["n"] > 1) & (summary["sd"] > 0)].copy()
    if len(sub) < 3 or moderator_col not in sub.columns:
        return None

    means = sub["mean"].values
    variances = (sub["sd"].values ** 2) / sub["n"].values
    moderator = sub[moderator_col].values.astype(float)

    if variances.min() <= 0 or np.any(np.isnan(moderator)):
        return None

    w = 1.0 / variances
    k = len(means)

    # Weighted least squares: Y = a + b*X
    xw = moderator * w
    yw = means * w
    sw = np.sum(w)
    sx = np.sum(xw)
    sy = np.sum(yw)
    sxx = np.sum(moderator ** 2 * w)
    sxy = np.sum(moderator * means * w)

    denom = sw * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return None

    b = (sw * sxy - sx * sy) / denom
    a = (sy - b * sx) / sw

    # Predicted and residuals
    predicted = a + b * moderator
    residuals = means - predicted
    q_residual = float(np.sum(w * residuals ** 2))
    df_residual = k - 2
    q_p = float(stats.chi2.sf(q_residual, df_residual)) if df_residual > 0 else np.nan

    # R² analog: proportion of heterogeneity explained
    q_total = float(np.sum(w * (means - np.sum(w * means) / sw) ** 2))
    r2 = max(0.0, 1.0 - q_residual / q_total) if q_total > 0 else 0.0

    # Standard error of slope
    se_b = np.sqrt(sw / denom) if denom > 0 else np.nan
    t_b = b / se_b if se_b > 0 else np.nan
    p_b = float(2 * stats.t.sf(abs(t_b), df_residual)) if np.isfinite(t_b) and df_residual > 0 else np.nan

    return {
        "dv": dv,
        "moderator": moderator_col,
        "k": k,
        "intercept": a,
        "slope": b,
        "se_slope": se_b,
        "t_slope": t_b,
        "p_slope": p_b,
        "r2_analog": r2,
        "q_residual": q_residual,
        "q_residual_p": q_p,
    }


# ── Cumulative meta-analysis ────────────────────────────────────────────────

def cumulative_meta_analysis(
    summary: pd.DataFrame,
    dv: str,
    sort_by: str = "n",
    estimator: str = "DL",
) -> pd.DataFrame:
    """Add studies one at a time (sorted by *sort_by*) and recompute the pooled estimate.

    Returns a DataFrame showing how the pooled estimate evolves.
    """
    sub = summary[(summary["dv"] == dv) & (summary["n"] > 1) & (summary["sd"] > 0)].copy()
    if len(sub) < 2:
        return pd.DataFrame()

    sub = sub.sort_values(sort_by, ascending=False).reset_index(drop=True)
    rows = []
    for i in range(1, len(sub)):
        cumul = sub.iloc[: i + 1]
        result = _run_meta_for_dv(cumul, total_studies=None, estimator=estimator)
        if result is not None:
            result["added_study"] = sub.iloc[i]["study"]
            result["cumulative_k"] = i + 1
            rows.append(result)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Effect size conversion utilities ────────────────────────────────────────

def convert_r_to_d(r: float) -> float:
    """Convert Pearson's r to Cohen's d."""
    return 2 * r / np.sqrt(1 - r ** 2) if abs(r) < 1 else np.nan


def convert_d_to_r(d: float) -> float:
    """Convert Cohen's d to Pearson's r (approximation)."""
    return d / np.sqrt(d ** 2 + 4)


def convert_or_to_d(odds_ratio: float) -> float:
    """Convert odds ratio to Cohen's d (Hasselblad & Hedges 1995)."""
    return np.log(odds_ratio) * np.sqrt(3) / np.pi if odds_ratio > 0 else np.nan


def convert_eta2_to_d(eta2: float, n1: int = 50, n2: int = 50) -> float:
    """Convert η² to Cohen's d (assuming two groups of size n1, n2)."""
    if eta2 < 0 or eta2 >= 1:
        return np.nan
    f2 = eta2 / (1 - eta2)
    return np.sqrt(f2 * (n1 + n2) / (n1 * n2) * (n1 + n2))


# ── Power analysis for meta-analysis ───────────────────────────────────────

def meta_analysis_power(
    expected_effect: float,
    average_n: int,
    k_studies: int,
    tau2: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Approximate power for a random-effects meta-analysis.

    Uses the formula from Valentine et al. (2010).
    """
    within_var = 1.0 / average_n + expected_effect ** 2 / (2 * average_n)
    total_var = within_var + tau2
    se_pooled = np.sqrt(total_var / k_studies)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_power = abs(expected_effect) / se_pooled - z_crit
    return float(stats.norm.cdf(z_power))


def studies_needed_for_power(
    expected_effect: float,
    average_n: int,
    tau2: float = 0.0,
    target_power: float = 0.80,
    alpha: float = 0.05,
    max_k: int = 500,
) -> int | None:
    """Return minimum number of studies needed to achieve *target_power*."""
    for k in range(2, max_k + 1):
        if meta_analysis_power(expected_effect, average_n, k, tau2, alpha) >= target_power:
            return k
    return None


# ── Data quality / outlier flagging ─────────────────────────────────────────

def flag_data_quality(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Produce a data quality report for each study × DV.

    Flags: out-of-range values (vs canonical_range), extreme kurtosis,
    suspiciously low variance (all identical), and small sample sizes.
    """
    metadata = _load_dv_measurement_metadata()
    canonical_studies = _resolve_canonical(studies, canonical_studies)
    rows = []
    for study, df in canonical_studies.items():
        for dv in df.columns:
            s = df[dv].dropna()
            if len(s) == 0:
                continue
            flags = []
            info = metadata.get(dv, {})
            cr = info.get("canonical_range")

            # Range check
            if cr is not None:
                n_below = int((s < cr[0]).sum())
                n_above = int((s > cr[1]).sum())
                if n_below > 0:
                    flags.append(f"{n_below} values below {cr[0]}")
                if n_above > 0:
                    flags.append(f"{n_above} values above {cr[1]}")

            # Zero/near-zero variance
            if s.std() < 1e-10:
                flags.append("zero variance (all identical)")

            # Small n
            if len(s) < 10:
                flags.append(f"small n ({len(s)})")

            # Extreme kurtosis (|kurt| > 7 suggests heavy tails or outliers)
            if len(s) >= 4:
                kurt = float(s.kurtosis())
                if abs(kurt) > 7:
                    flags.append(f"extreme kurtosis ({kurt:.1f})")

            rows.append({
                "study": study,
                "dv": dv,
                "n": len(s),
                "mean": float(s.mean()),
                "sd": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "n_flags": len(flags),
                "flags": "; ".join(flags) if flags else "",
            })

    return pd.DataFrame(rows).sort_values(["study", "dv"]).reset_index(drop=True)


# ── Non-Gaussianity testing ────────────────────────────────────────────────

def test_non_gaussianity(
    studies: Dict[str, pd.DataFrame],
    min_n: int = 8,
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Shapiro-Wilk normality test per study × DV.

    Used to validate the non-Gaussianity assumption required by LiNGAM.
    """
    canonical_studies = _resolve_canonical(studies, canonical_studies)
    rows = []
    for study, df in canonical_studies.items():
        for dv in df.columns:
            s = df[dv].dropna()
            if len(s) < min_n or len(s) > 5000:
                continue
            try:
                stat, p = stats.shapiro(s)
            except Exception:
                continue
            rows.append({
                "study": study,
                "dv": dv,
                "n": len(s),
                "shapiro_w": float(stat),
                "shapiro_p": float(p),
                "is_normal_p05": p > 0.05,
                "is_non_gaussian_p05": p <= 0.05,
            })

    return pd.DataFrame(rows).sort_values(["study", "dv"]).reset_index(drop=True)


# ── Reverse-coding detection ───────────────────────────────────────────────

def detect_potential_reverse_coding(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Flag DVs that might be reverse-coded based on inter-DV correlations.

    Within each study, if a DV with direction=higher_is_better is strongly
    negatively correlated (r < -0.3) with other same-cluster DVs, it may
    need reverse-coding.
    """
    metadata = _load_dv_measurement_metadata()
    canonical_studies = _resolve_canonical(studies, canonical_studies)
    flags = []

    for study, df in canonical_studies.items():
        if len(df.columns) < 2:
            continue
        corr = df.corr()
        for dv in df.columns:
            info = metadata.get(dv, {})
            direction = info.get("direction", "neutral")
            cluster = info.get("cluster", "")
            if direction == "neutral" or not cluster:
                continue

            # Check correlations with same-cluster DVs
            same_cluster = [
                c for c in df.columns
                if c != dv and metadata.get(c, {}).get("cluster") == cluster
            ]
            for other in same_cluster:
                r = corr.loc[dv, other] if dv in corr.index and other in corr.columns else np.nan
                if np.isfinite(r) and r < -0.3:
                    flags.append({
                        "study": study,
                        "dv": dv,
                        "direction": direction,
                        "cluster": cluster,
                        "correlated_with": other,
                        "r": float(r),
                        "warning": "possible reverse coding",
                    })

    return pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=["study", "dv", "direction", "cluster", "correlated_with", "r", "warning"]
    )


# ── Overlap statistics ──────────────────────────────────────────────────────

def compute_extended_overlap_stats(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Compute Jaccard, Sørensen-Dice, overlap coefficient, and DV frequency table."""
    dv_sets = _study_numeric_dv_sets_from_canonical(_resolve_canonical(studies, canonical_studies))
    all_dvs = sorted({dv for dvs in dv_sets.values() for dv in dvs})

    # DV frequency table
    freq = {}
    for dv in all_dvs:
        freq[dv] = sum(1 for dvs in dv_sets.values() if dv in dvs)
    freq_df = pd.DataFrame([
        {"dv": dv, "n_studies": cnt, "coverage_pct": cnt / len(studies) * 100}
        for dv, cnt in sorted(freq.items(), key=lambda x: -x[1])
    ])

    # Pairwise overlap statistics
    pair_rows = []
    for a, b in combinations(sorted(dv_sets.keys()), 2):
        intersection = len(dv_sets[a] & dv_sets[b])
        union = len(dv_sets[a] | dv_sets[b])
        min_size = min(len(dv_sets[a]), len(dv_sets[b]))
        pair_rows.append({
            "study_a": a,
            "study_b": b,
            "jaccard": intersection / union if union else np.nan,
            "sorensen_dice": 2 * intersection / (len(dv_sets[a]) + len(dv_sets[b]))
                if (len(dv_sets[a]) + len(dv_sets[b])) > 0 else np.nan,
            "overlap_coefficient": intersection / min_size if min_size > 0 else np.nan,
        })

    return {
        "dv_frequency": freq_df,
        "pairwise_overlap": pd.DataFrame(pair_rows),
    }


# ── Bron-Kerbosch maximum clique ────────────────────────────────────────────

def _bron_kerbosch_max_clique(adjacency: dict[str, set[str]]) -> list[str]:
    """Exact maximum clique via Bron-Kerbosch with pivoting.

    *adjacency* maps each node to its set of neighbors.
    Returns the largest clique found.
    """
    best: list[str] = []

    def _bk(R: set, P: set, X: set) -> None:
        nonlocal best
        if not P and not X:
            if len(R) > len(best):
                best = sorted(R)
            return
        # Pivot: choose u in P∪X with max neighbors in P
        pivot = max(P | X, key=lambda u: len(adjacency[u] & P))
        for v in list(P - adjacency[pivot]):
            _bk(R | {v}, P & adjacency[v], X & adjacency[v])
            P = P - {v}
            X = X | {v}

    all_nodes = set(adjacency.keys())
    _bk(set(), all_nodes, set())
    return best


# ── Leave-one-out sensitivity analysis ───────────────────────────────────────

def leave_one_out_sensitivity(
    summary: pd.DataFrame,
    total_studies: int | None = None,
    estimator: str = "DL",
) -> pd.DataFrame:
    """For each DV, iteratively remove each study and recompute the pooled estimate."""
    rows = []
    for dv, sub in summary.groupby("dv"):
        sub = sub[(sub["n"] > 1) & (sub["sd"] > 0)].copy()
        if len(sub) < 3:
            continue
        for idx in sub.index:
            remaining = sub.drop(idx)
            result = _run_meta_for_dv(remaining, total_studies, estimator)
            if result is None:
                continue
            result["omitted_study"] = sub.loc[idx, "study"]
            rows.append(result)

    if not rows:
        return pd.DataFrame(columns=["dv", "omitted_study", "random_effects_mean",
                                      "random_effects_se", "ci95_low", "ci95_high",
                                      "tau2", "heterogeneity_i2_pct"])

    df = pd.DataFrame(rows)
    return df[["dv", "omitted_study", "random_effects_mean", "random_effects_se",
               "ci95_low", "ci95_high", "tau2", "heterogeneity_i2_pct"]].sort_values(
        ["dv", "omitted_study"]
    ).reset_index(drop=True)


# ── Subgroup analysis by schema cluster ──────────────────────────────────────

def subgroup_meta_analysis(
    effects_df: pd.DataFrame,
    schema_path: str = str(DEFAULT_DV_SCHEMA_PATH),
) -> pd.DataFrame:
    """Pool standardized effects (Hedges' g) by schema cluster.

    Returns between-cluster heterogeneity test when multiple clusters exist.
    """
    if effects_df.empty:
        return pd.DataFrame(columns=[
            "cluster", "k_dvs", "k_effects", "pooled_g", "se", "ci95_low", "ci95_high",
        ])

    metadata = _load_dv_measurement_metadata(schema_path)

    # Map each DV to its cluster
    effects = effects_df.copy()
    effects["cluster"] = effects["dv"].map(
        lambda d: metadata.get(d, {}).get("cluster", "unknown")
    )

    # Flip sign for lower_is_better DVs so positive = better
    effects["direction"] = effects["dv"].map(
        lambda d: metadata.get(d, {}).get("direction", "neutral")
    )
    mask = effects["direction"] == "lower_is_better"
    effects.loc[mask, "hedges_g"] = -effects.loc[mask, "hedges_g"]

    rows = []
    for cluster, grp in effects.groupby("cluster"):
        grp = grp[grp["var_g"] > 0].copy()
        if len(grp) < 2:
            continue
        w = 1.0 / grp["var_g"].values
        g = grp["hedges_g"].values
        pooled = float(np.sum(w * g) / np.sum(w))
        se = float(np.sqrt(1.0 / np.sum(w)))
        rows.append({
            "cluster": cluster,
            "k_dvs": grp["dv"].nunique(),
            "k_effects": len(grp),
            "pooled_g": pooled,
            "se": se,
            "ci95_low": pooled - 1.96 * se,
            "ci95_high": pooled + 1.96 * se,
        })

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# ── PCA composite index ─────────────────────────────────────────────────────

def _prepare_composite_matrix(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    canonical_studies = _resolve_canonical(studies, canonical_studies)
    counts = {}
    for df in canonical_studies.values():
        for dv in set(df.columns):
            counts[dv] = counts.get(dv, 0) + 1
    common_cols = sorted([dv for dv, k in counts.items() if k >= 2])
    if len(common_cols) < 2:
        raise ValueError("Need at least two DVs shared by at least two studies for PCA composite index.")

    stacked = []
    for study, df in canonical_studies.items():
        block = df.reindex(columns=common_cols).copy()
        block["study"] = study
        stacked.append(block)
    long = pd.concat(stacked, ignore_index=True)

    # Standardize within each study to remove study-specific scaling artifacts.
    standardized = []
    for study, sub in long.groupby("study", sort=False):
        x = sub[common_cols].copy()
        x = x.apply(pd.to_numeric, errors="coerce")
        # Rows that observed none of the shared DVs (e.g. telemetry rows of a
        # study whose questionnaire lives in another file) carry no
        # information. Median-imputing them would turn each into an all-zero
        # z-row, inflating the study's n and shrinking its composite SD.
        x = x.dropna(how="all")
        if x.empty:
            continue
        x = x.fillna(x.median())
        x = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
        x = x.reindex(columns=common_cols)
        x = x.fillna(0.0)
        x["study"] = study
        standardized.append(x)
    if not standardized:
        raise ValueError("No study observed any shared DV; cannot build a PCA composite index.")
    z = pd.concat(standardized, ignore_index=True)

    feature_matrix = z[common_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    usable_cols = [col for col in common_cols if feature_matrix[col].nunique(dropna=True) > 1]
    if len(usable_cols) < 2:
        raise ValueError("Need at least two shared DVs with non-zero variance for PCA composite index.")
    return z, usable_cols


def compute_composite_scores(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, List[str], float]:
    """Per-row PC1 scores on within-study z-scored shared DVs.

    Returns ``(scores, usable_cols, explained_variance_ratio)`` where
    ``scores`` has a ``study`` and a ``cross_study_composite`` column. Compute
    it once and pass it to :func:`build_composite_index` and
    :func:`save_composite_plot`. Raises ``ValueError`` when fewer than two
    shared DVs are usable.
    """
    z, usable_cols = _prepare_composite_matrix(studies, canonical_studies)
    pca = PCA(n_components=1)
    z["cross_study_composite"] = pca.fit_transform(z[usable_cols])
    return z, usable_cols, float(pca.explained_variance_ratio_[0])


def build_composite_index(
    studies: Dict[str, pd.DataFrame],
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
    scores: tuple[pd.DataFrame, List[str], float] | None = None,
) -> pd.DataFrame:
    z, usable_cols, explained_variance_ratio = scores or compute_composite_scores(studies, canonical_studies)

    summary = (
        z.groupby("study")["cross_study_composite"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n", "std": "sd"})
        .reset_index()
    )
    summary["explained_variance_ratio"] = explained_variance_ratio
    summary["n_shared_dvs_used"] = len(usable_cols)
    return summary


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_plots(overlap: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(overlap, annot=True, cmap="Blues", vmin=0, vmax=1)
    plt.title("DV overlap across standardized studies (Jaccard)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "dv_overlap_heatmap.png", dpi=150)
    plt.close()

    if not summary.empty:
        shared = summary.groupby("dv")["study"].nunique()
        shared = shared[shared >= 2].index
        if len(shared):
            plot_df = summary[summary["dv"].isin(shared)]
            plt.figure(figsize=(9, 5))
            sns.barplot(data=plot_df, x="dv", y="mean_z_vs_global", hue="study")
            plt.axhline(0, color="black", linewidth=1)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Study mean shift (z vs global DV mean)")
            plt.title("Comparable DV-level differences without using IVs")
            plt.tight_layout()
            plt.savefig(output_dir / "dv_mean_shift.png", dpi=150)
            plt.close()

        dv_coverage = summary.groupby("study")["dv"].nunique().reset_index(name="n_numeric_dvs")
        plt.figure(figsize=(7, 4.5))
        sns.barplot(data=dv_coverage, x="study", y="n_numeric_dvs", hue="study", dodge=False, legend=False)
        for idx, row in dv_coverage.iterrows():
            plt.text(idx, row["n_numeric_dvs"] + 0.05, f"{int(row['n_numeric_dvs'])}", ha="center", va="bottom")
        plt.ylabel("Count of canonical numeric DVs")
        plt.xlabel("Study")
        plt.title("Canonical DV coverage by study")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "dv_coverage_by_study.png", dpi=150)
        plt.close()


def save_composite_plot(
    studies: Dict[str, pd.DataFrame],
    output_dir: Path,
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
    scores: tuple[pd.DataFrame, List[str], float] | None = None,
) -> None:
    if scores is None:
        try:
            scores = compute_composite_scores(studies, canonical_studies)
        except ValueError:
            return
    z = scores[0]

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=z, x="study", y="cross_study_composite", inner="quartile", hue="study", legend=False)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Study")
    plt.ylabel("Cross-study composite score (PCA PC1)")
    plt.title("Distribution of cross-study composite scores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "cross_study_composite_distribution.png", dpi=150)
    plt.close()


def save_forest_plot(
    summary: pd.DataFrame,
    meta_row: pd.Series,
    dv: str,
    output_dir: Path,
) -> None:
    """Generate a Cochrane-style forest plot with study-level annotation table."""
    sub = summary[(summary["dv"] == dv) & (summary["n"] > 1) & (summary["sd"] > 0)].copy()
    if len(sub) < 2:
        return

    sub = sub.sort_values("study").reset_index(drop=True)
    k = len(sub)
    means = sub["mean"].values
    sds = sub["sd"].values
    ns = sub["n"].values.astype(int)
    ses = sds / np.sqrt(ns)
    ci_lo = means - 1.96 * ses
    ci_hi = means + 1.96 * ses

    # Weights proportional to inverse variance
    variances = ses ** 2
    w = 1.0 / variances
    w_pct = 100 * w / np.sum(w)

    pooled_mean = meta_row["random_effects_mean"]
    pooled_se = meta_row["random_effects_se"]
    pooled_lo = meta_row["ci95_low"]
    pooled_hi = meta_row["ci95_high"]

    # --- Layout: forest on the left (ax_forest), annotation table on the right (ax_table) ---
    fig_height = max(4, 1.8 + 0.55 * k)
    fig, (ax_forest, ax_table) = plt.subplots(
        1, 2,
        figsize=(14, fig_height),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.02},
    )

    y_positions = list(range(k, 0, -1))
    mono_font = {"family": "monospace", "size": 8}

    # ---- Forest panel (left) ----
    for i, y in enumerate(y_positions):
        marker_size = max(4, min(15, w_pct[i] * 0.5))
        ax_forest.plot(means[i], y, "s", color="#2166ac", markersize=marker_size, zorder=3)
        ax_forest.hlines(y, ci_lo[i], ci_hi[i], color="#2166ac", linewidth=1.5, zorder=2)

    # Pooled diamond
    diamond_y = 0.3
    diamond_half_h = 0.25
    diamond_x = [pooled_lo, pooled_mean, pooled_hi, pooled_mean]
    diamond_yy = [diamond_y, diamond_y + diamond_half_h, diamond_y, diamond_y - diamond_half_h]
    ax_forest.fill(diamond_x, diamond_yy, color="#b2182b", alpha=0.7, zorder=3)

    # Prediction interval (if available)
    pi_lo = meta_row.get("prediction_interval_low")
    pi_hi = meta_row.get("prediction_interval_high")
    if pd.notna(pi_lo) and pd.notna(pi_hi):
        ax_forest.hlines(diamond_y, pi_lo, pi_hi, color="#b2182b", linewidth=1, linestyle="--", alpha=0.5, zorder=2)

    # Reference line at pooled mean
    ax_forest.axvline(pooled_mean, color="#b2182b", linewidth=0.8, linestyle="--", alpha=0.5)

    # Y-axis labels (study names on left)
    ax_forest.set_yticks(y_positions + [0])
    study_labels = list(sub["study"].values) + ["Pooled RE"]
    ax_forest.set_yticklabels(study_labels, fontsize=9)
    ax_forest.set_xlabel(f"{dv} (mean)")
    ax_forest.set_title(f"Forest plot: {dv} (k={k})")
    ax_forest.spines["right"].set_visible(False)
    ax_forest.spines["top"].set_visible(False)
    ax_forest.set_ylim(-0.5, k + 1.0)

    # ---- Annotation table panel (right) ----
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(-0.5, k + 1.0)
    ax_table.axis("off")

    # Column x-positions (normalised 0-1 within ax_table)
    col_x = {"n": 0.02, "mean": 0.14, "sd": 0.30, "weight": 0.46, "ci": 0.62}

    # Column headers
    header_y = k + 0.6
    header_font = {"family": "monospace", "size": 8, "weight": "bold"}
    ax_table.text(col_x["n"], header_y, "N", fontdict=header_font, va="center")
    ax_table.text(col_x["mean"], header_y, "Mean", fontdict=header_font, va="center")
    ax_table.text(col_x["sd"], header_y, "SD", fontdict=header_font, va="center")
    ax_table.text(col_x["weight"], header_y, "Weight", fontdict=header_font, va="center")
    ax_table.text(col_x["ci"], header_y, "Mean [95% CI]", fontdict=header_font, va="center")

    # Separator line below header
    ax_table.axhline(y=k + 0.35, color="black", linewidth=0.6, xmin=0.0, xmax=1.0)

    # Study rows
    for i, y in enumerate(y_positions):
        ci_text = f"{means[i]:>6.2f} [{ci_lo[i]:.2f}, {ci_hi[i]:.2f}]"
        ax_table.text(col_x["n"], y, f"{ns[i]:>5d}", fontdict=mono_font, va="center")
        ax_table.text(col_x["mean"], y, f"{means[i]:>7.2f}", fontdict=mono_font, va="center")
        ax_table.text(col_x["sd"], y, f"{sds[i]:>7.2f}", fontdict=mono_font, va="center")
        ax_table.text(col_x["weight"], y, f"{w_pct[i]:>5.1f}%", fontdict=mono_font, va="center")
        ax_table.text(col_x["ci"], y, ci_text, fontdict=mono_font, va="center")

    # Separator line above pooled row
    ax_table.axhline(y=0.7, color="black", linewidth=0.6, xmin=0.0, xmax=1.0)

    # Pooled estimate row
    pooled_ci_text = f"{pooled_mean:>6.2f} [{pooled_lo:.2f}, {pooled_hi:.2f}]"
    total_n = int(ns.sum())
    pooled_font = {"family": "monospace", "size": 8, "weight": "bold"}
    ax_table.text(col_x["n"], 0, f"{total_n:>5d}", fontdict=pooled_font, va="center")
    ax_table.text(col_x["weight"], 0, "100.0%", fontdict=pooled_font, va="center")
    ax_table.text(col_x["ci"], 0, pooled_ci_text, fontdict=pooled_font, va="center")

    # ---- Heterogeneity summary below the plot ----
    i2_val = meta_row["heterogeneity_i2_pct"]
    tau2_val = meta_row["tau2"]
    q_val = meta_row["heterogeneity_q"]
    q_p = meta_row["q_pvalue"]

    if q_p < 0.001:
        p_str = "p < 0.001"
    else:
        p_str = f"p = {q_p:.3f}"

    het_text = (
        f"Heterogeneity: I\u00B2 = {i2_val:.1f}%, "
        f"\u03C4\u00B2 = {tau2_val:.4f}, "
        f"Q({k - 1}) = {q_val:.2f}, {p_str}"
    )
    if pd.notna(meta_row.get("hksj_ci95_low")):
        het_text += (
            f"\nHartung-Knapp-Sidik-Jonkman 95% CI: "
            f"[{meta_row['hksj_ci95_low']:.2f}, {meta_row['hksj_ci95_high']:.2f}]"
        )
    fig.text(
        0.02, -0.01, het_text,
        fontsize=8, fontfamily="monospace", va="top", ha="left",
    )

    # bbox_inches="tight" handles cropping; tight_layout is incompatible with
    # the text-only annotation axis and only emitted warnings.
    plt.savefig(output_dir / f"forest_{dv}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_funnel_plot(
    summary: pd.DataFrame,
    meta_row: pd.Series,
    dv: str,
    output_dir: Path,
) -> None:
    """Generate a funnel plot for a single DV."""
    sub = summary[(summary["dv"] == dv) & (summary["n"] > 1) & (summary["sd"] > 0)].copy()
    if len(sub) < 3:
        return

    means = sub["mean"].values
    ses = sub["sd"].values / np.sqrt(sub["n"].values)
    pooled = meta_row["random_effects_mean"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(means, ses, s=50, color="#2166ac", zorder=3)

    # Pseudo-95% CI bounds
    se_range = np.linspace(0, max(ses) * 1.2, 100)
    ax.plot(pooled + 1.96 * se_range, se_range, "--", color="gray", linewidth=0.8)
    ax.plot(pooled - 1.96 * se_range, se_range, "--", color="gray", linewidth=0.8)
    ax.axvline(pooled, color="#b2182b", linewidth=1, linestyle="-", alpha=0.7)

    ax.set_xlabel(f"{dv} (study mean)")
    ax.set_ylabel("Standard error")
    ax.invert_yaxis()
    ax.set_title(f"Funnel plot: {dv} (k={len(sub)})")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"funnel_{dv}.png", dpi=150)
    plt.close()


def save_sensitivity_plot(
    sensitivity_df: pd.DataFrame,
    dv: str,
    original_meta_row: pd.Series,
    output_dir: Path,
) -> None:
    """Leave-one-out forest plot for a single DV."""
    sub = sensitivity_df[sensitivity_df["dv"] == dv].copy()
    if sub.empty:
        return

    sub = sub.sort_values("omitted_study").reset_index(drop=True)
    k = len(sub)

    fig, ax = plt.subplots(figsize=(9, max(3, 1 + 0.5 * k)))

    y_positions = list(range(k, 0, -1))
    for i, y in enumerate(y_positions):
        row = sub.iloc[i]
        ax.plot(row["random_effects_mean"], y, "o", color="#2166ac", markersize=6, zorder=3)
        ax.hlines(y, row["ci95_low"], row["ci95_high"], color="#2166ac", linewidth=1.5, zorder=2)

    # Full-data reference
    ax.axvline(original_meta_row["random_effects_mean"], color="#b2182b",
               linewidth=1.5, linestyle="--", alpha=0.7, label="Full-data estimate")
    ax.axvspan(original_meta_row["ci95_low"], original_meta_row["ci95_high"],
               alpha=0.1, color="#b2182b")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Omit: {s}" for s in sub["omitted_study"].values], fontsize=9)
    ax.set_xlabel(f"{dv} (pooled mean)")
    ax.set_title(f"Leave-one-out sensitivity: {dv}")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"sensitivity_{dv}.png", dpi=150)
    plt.close()


# ── Construct → sub-item hierarchy ───────────────────────────────────────────
# When a higher-order construct is present in the shared DVs, its constituent
# sub-items should be excluded from the LiNGAM DAG to avoid tautological
# causal edges (e.g., SUS items → usability).
#
# Built automatically from ``_DERIVED_SCALES`` — adding a new derived scale
# there will auto-exclude its sub-items from the DAG — and from the DV schema:
# an entry declares its parent with ``subitem_of: <construct id>``, or is
# recognised by the ``<construct id>_item_NN`` naming convention when that
# construct id exists in the schema (e.g. perceived_safety_item_01).

_SUBITEM_ID_PATTERN = re.compile(r"^(?P<construct>.+)_item_\d+$")


def _schema_construct_subitems(schema_path: Path = DEFAULT_DV_SCHEMA_PATH) -> dict[str, list[str]]:
    """Construct → item-level DV ids declared by (or implied in) the DV schema."""
    if not schema_path.is_file():
        return {}
    entries = [entry for entry in (_load_yaml(schema_path) or {}).get("dvs", []) if isinstance(entry, dict)]
    known_ids = {str(entry.get("id", "")).strip() for entry in entries}
    mapping: dict[str, list[str]] = {}
    for entry in entries:
        dv_id = str(entry.get("id", "")).strip()
        parent = str(entry.get("subitem_of") or "").strip()
        if not parent:
            match = _SUBITEM_ID_PATTERN.match(dv_id)
            if match and match.group("construct") in known_ids:
                parent = match.group("construct")
        if dv_id and parent and parent != dv_id:
            mapping.setdefault(parent, []).append(dv_id)
    return mapping


def _merge_subitem_maps(*maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for mapping in maps:
        for construct, items in mapping.items():
            merged[construct] = list(dict.fromkeys(merged.get(construct, []) + list(items)))
    return merged


CONSTRUCT_SUBITEMS: dict[str, list[str]] = _merge_subitem_maps(
    _build_construct_subitems(), _schema_construct_subitems()
)


def _filter_subitems_when_construct_present(dvs: list[str]) -> list[str]:
    """Remove sub-items from *dvs* if their parent construct is also present."""
    dv_set = set(dvs)
    exclude: set[str] = set()
    for construct, subitems in CONSTRUCT_SUBITEMS.items():
        if construct in dv_set:
            exclude.update(item for item in subitems if item in dv_set)
    if exclude:
        logging.getLogger(__name__).info(
            "LiNGAM: excluding %d sub-items in favour of higher-order "
            "constructs: %s", len(exclude), sorted(exclude),
        )
    return [dv for dv in dvs if dv not in exclude]


# ── LiNGAM causal discovery ─────────────────────────────────────────────────

_lingam_logger = logging.getLogger(__name__ + ".lingam")


def _pool_standardized_shared_dvs(
    canonical_studies: Dict[str, pd.DataFrame],
    min_shared_studies: int,
    min_rows: int,
) -> tuple[pd.DataFrame, list[str]] | None:
    """Pool within-study z-scored rows of the DVs shared across studies.

    Shared DVs are those present in at least *min_shared_studies* studies,
    minus sub-items whose parent construct is also shared. Each study is
    z-scored on the shared DVs it actually has (rows complete on those DVs,
    at least *min_rows* of them); blocks are then stacked, so DVs a study
    lacks are NaN. Returns ``(pooled, usable_dvs)`` where every usable DV has
    at least *min_rows* pooled observations, or None when fewer than two DVs
    qualify. Shared by the LiNGAM, bootstrap and PC passes so they all see
    the same data.
    """
    dv_counts: Dict[str, int] = {}
    for df in canonical_studies.values():
        for dv in df.columns:
            dv_counts[dv] = dv_counts.get(dv, 0) + 1
    shared_dvs = sorted(dv for dv, cnt in dv_counts.items() if cnt >= min_shared_studies)

    # Remove sub-items when their higher-order construct is also shared
    shared_dvs = _filter_subitems_when_construct_present(shared_dvs)
    _lingam_logger.info("Shared DVs after subitem filtering (%d): %s",
                        len(shared_dvs), shared_dvs)
    if len(shared_dvs) < 2:
        return None

    pooled_blocks = []
    for study_name, df in canonical_studies.items():
        available = [dv for dv in shared_dvs if dv in df.columns]
        if len(available) < 2:
            continue
        sub = df[available].dropna()          # drop rows with NaN *within available DVs only*
        if len(sub) < min_rows:
            continue
        # Within-study standardization to remove scale differences
        z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
        z = z.dropna(axis=1, how="all").dropna()
        if z.empty:
            continue
        _lingam_logger.info("  Study '%s': %d rows × %d DVs (%s)",
                            study_name, len(z), len(z.columns), list(z.columns))
        # Provenance only: lets callers report which studies a fit rests on.
        z[STUDY_COLUMN] = study_name
        pooled_blocks.append(z)

    if not pooled_blocks:
        return None

    pooled = pd.concat(pooled_blocks, ignore_index=True)

    # Keep DVs that appear in enough pooled rows (may have NaN from
    # studies that lack a given DV).
    usable_dvs = [c for c in shared_dvs
                  if c in pooled.columns and pooled[c].notna().sum() >= min_rows]
    _lingam_logger.info("Usable DVs after min-row filter (%d): %s",
                        len(usable_dvs), usable_dvs)
    if len(usable_dvs) < 2:
        return None
    return pooled, usable_dvs


STUDY_COLUMN = "_study"


def discover_causal_structure(
    studies: Dict[str, pd.DataFrame],
    min_shared_studies: int = 2,
    min_rows: int = 200,
    refuse_synthetic: bool = False,
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> dict | None:
    """Discover causal ordering among shared DVs using DirectLiNGAM.

    Pools standardized (within-study z-scored) data across studies for DVs
    shared by at least *min_shared_studies* studies. Applies DirectLiNGAM to
    the pooled matrix and returns the adjacency matrix + causal order.

    The result dict includes ``used_method`` (``complete_case`` |
    ``clique_complete_case`` | ``synthetic_cholesky``), ``synthetic_fallback``
    (bool), and ``n_complete_rows`` (complete-case row count before any
    fallback).  When ``refuse_synthetic=True``, the function returns None
    instead of running LiNGAM on Cholesky-synthesized rows — recommended when
    you do not want fabricated observations driving causal claims.

    Returns None if fewer than 2 shared DVs have sufficient data.
    """
    canonical_studies = _resolve_canonical(studies, canonical_studies)
    pooled_result = _pool_standardized_shared_dvs(canonical_studies, min_shared_studies, min_rows)
    if pooled_result is None:
        return None
    pooled, usable_dvs = pooled_result

    # Build the data matrix for LiNGAM.
    # With partial DV overlap across studies, not all DVs co-occur.
    # Strategy: find the largest subset of DVs that have pairwise
    # complete observations (≥ min_rows) for ALL pairs, then use
    # pairwise-complete correlation → Cholesky → synthetic data if
    # complete-case rows are insufficient.
    sub_pooled = pooled[usable_dvs]

    # Try the simple path first: rows complete on ALL usable DVs
    X_complete = sub_pooled.dropna()
    n_complete_rows = int(len(X_complete))
    _lingam_logger.info("Complete-case rows: %d (need %d)", n_complete_rows, min_rows)
    used_method = "complete_case"
    if len(X_complete) >= min_rows:
        X = X_complete
        _lingam_logger.info("Using complete-case path with %d rows × %d DVs",
                            len(X), len(usable_dvs))
    else:
        # Find the largest clique of DVs with full pairwise coverage
        min_periods = max(min_rows, 10)
        corr_full = sub_pooled.corr(min_periods=min_periods)

        # Build adjacency: DVs i,j are connected if their pairwise corr is not NaN
        pairwise_ok = ~corr_full.isna()
        _lingam_logger.info("Pairwise coverage matrix (NaN = no co-occurrence):\n%s",
                            corr_full.round(2).to_string())

        # Exact maximum clique via Bron-Kerbosch with pivoting.
        dv_list = list(corr_full.index)
        adjacency = {
            dv: {other for other in dv_list if other != dv and pairwise_ok.loc[dv, other]}
            for dv in dv_list
        }
        best_clique = _bron_kerbosch_max_clique(adjacency)

        if len(best_clique) < 2:
            _lingam_logger.warning("No clique of ≥2 DVs with full pairwise coverage")
            return None

        usable_dvs = sorted(best_clique)
        _lingam_logger.info("Largest pairwise-complete clique (%d DVs): %s",
                            len(usable_dvs), usable_dvs)

        # Try complete-case on the reduced DV set
        X_clique = sub_pooled[usable_dvs].dropna()
        n_complete_rows = int(len(X_clique))
        if len(X_clique) >= min_rows:
            X = X_clique
            used_method = "clique_complete_case"
            _lingam_logger.info("Using complete-case on clique: %d rows × %d DVs",
                                len(X), len(usable_dvs))
        else:
            if refuse_synthetic:
                _lingam_logger.warning(
                    "Refusing synthetic-Cholesky fallback (refuse_synthetic=True); "
                    "clique complete rows %d < min_rows %d",
                    len(X_clique), min_rows,
                )
                return None
            # Pairwise correlation → synthetic observations via Cholesky
            corr = sub_pooled[usable_dvs].corr(min_periods=min_periods)
            if corr.isna().any().any():
                _lingam_logger.warning("Clique still has NaN correlations — giving up")
                return None
            n_obs_pairwise = int(sub_pooled[usable_dvs].notna().sum().min())
            # Ensure positive-definiteness via eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(corr.values)
            eigvals = np.clip(eigvals, 1e-6, None)
            corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            np.fill_diagonal(corr_pd, 1.0)
            try:
                L = np.linalg.cholesky(corr_pd)
            except np.linalg.LinAlgError:
                _lingam_logger.warning("Cholesky decomposition failed")
                return None
            rng = np.random.default_rng(42)
            n_synth = max(n_obs_pairwise, min_rows * 2, 200)
            Z = rng.standard_normal((n_synth, len(usable_dvs)))
            X = pd.DataFrame(Z @ L.T, columns=usable_dvs)
            used_method = "synthetic_cholesky"
            _lingam_logger.warning(
                "LiNGAM: using Cholesky-synthesized rows (%d synthetic × %d DVs) — "
                "edges are read off a correlation matrix, NOT observed data; "
                "results marked synthetic_fallback=True",
                n_synth, len(usable_dvs),
            )

    if len(X) < min_rows:
        return None

    model = lingam.DirectLiNGAM()
    model.fit(X.values)

    adjacency = pd.DataFrame(
        model.adjacency_matrix_,
        index=usable_dvs,
        columns=usable_dvs,
    )
    causal_order = [usable_dvs[i] for i in model.causal_order_]

    # Extract significant edges (|coefficient| > threshold)
    threshold = 0.1
    edges = []
    adj = model.adjacency_matrix_
    for i in range(len(usable_dvs)):
        for j in range(len(usable_dvs)):
            if abs(adj[i, j]) > threshold:
                edges.append({
                    "from": usable_dvs[j],
                    "to": usable_dvs[i],
                    "coefficient": adj[i, j],
                })

    # Which studies the fitted rows come from: a "cross-study" DAG fit on a
    # single study's rows is a within-study result and must be reported so.
    if used_method == "synthetic_cholesky":
        source_rows = sub_pooled[usable_dvs].notna().any(axis=1)
        studies_used = sorted(pooled.loc[source_rows[source_rows].index, STUDY_COLUMN].unique())
    else:
        studies_used = sorted(pooled.loc[X.index, STUDY_COLUMN].unique())

    return {
        "adjacency_matrix": adjacency,
        "causal_order": causal_order,
        "edges": pd.DataFrame(edges) if edges else pd.DataFrame(columns=["from", "to", "coefficient"]),
        "studies_used": [str(s) for s in studies_used],
        "n_observations": len(X),
        "n_dvs": len(usable_dvs),
        "dvs_used": usable_dvs,
        "used_method": used_method,
        "synthetic_fallback": used_method == "synthetic_cholesky",
        "n_complete_rows": n_complete_rows,
        # The exact matrix DirectLiNGAM was fit on, so stability checks
        # (bootstrap, PC) can be run on the same observations.
        "data": X.reset_index(drop=True),
    }


def save_causal_dag_plot(
    causal_result: dict,
    output_dir: Path,
) -> None:
    """Visualize the LiNGAM causal DAG as a heatmap of the adjacency matrix."""
    adj = causal_result["adjacency_matrix"]
    order = causal_result["causal_order"]

    # Reorder by causal order
    adj_ordered = adj.reindex(index=order, columns=order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Adjacency heatmap
    ax = axes[0]
    mask = np.abs(adj_ordered.values) < 0.1
    sns.heatmap(
        adj_ordered.astype(float),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        mask=mask,
        ax=ax,
        cbar_kws={"label": "Causal coefficient"},
    )
    ax.set_title("LiNGAM causal adjacency matrix\n(rows <- columns, causal order)")
    ax.set_xlabel("Cause")
    ax.set_ylabel("Effect")

    # Causal order as a simple flow diagram
    ax2 = axes[1]
    n = len(order)
    for i, dv in enumerate(order):
        y = n - i
        ax2.text(0.5, y, f"{i+1}. {dv}", ha="center", va="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4e6f1", edgecolor="#2c3e50"))
        if i < n - 1:
            ax2.annotate("", xy=(0.5, y - 0.35), xytext=(0.5, y - 0.65),
                         arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.2, n + 0.8)
    ax2.axis("off")
    ax2.set_title("Estimated causal order\n(LiNGAM DirectLiNGAM)")

    plt.tight_layout()
    plt.savefig(output_dir / "causal_dag_lingam.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── IRT linking / ordinal rescaling warnings ─────────────────────────────────

def check_ordinal_rescaling_warnings(
    studies: Dict[str, pd.DataFrame],
    harmonize_scales: bool = True,
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Emit warnings when linear rescaling is applied to ordinal Likert data.

    Linear rescaling between two ordinal Likert scales (e.g., 1-5 to 1-7) treats
    category boundaries as equidistant, which is a strong assumption.  For robust
    cross-scale comparisons, Item Response Theory (IRT) linking or equipercentile
    equating should be used instead.

    Returns a DataFrame of warnings with columns:
        study, dv, detected_range, canonical_range, scale_type, warning
    """
    cols = ["study", "dv", "detected_range", "canonical_range", "scale_type", "warning"]
    if not harmonize_scales:
        return pd.DataFrame(columns=cols)

    metadata = _load_dv_measurement_metadata()
    canonical_studies = _resolve_canonical(studies, canonical_studies)
    warnings_rows: list[dict] = []
    _logger = logging.getLogger(__name__)

    for study_name, df in canonical_studies.items():
        for dv in df.columns:
            if dv not in metadata:
                continue
            info = metadata[dv]
            canonical_range = info.get("canonical_range")
            scale_type = info.get("scale_type", "")
            if canonical_range is None:
                continue

            s = df[dv].dropna()
            if len(s) == 0:
                continue

            detected = _detect_scale_range(s, canonical_range)
            if detected == canonical_range:
                continue

            # Check if this is ordinal data being linearly rescaled
            is_ordinal = scale_type.lower() in ("ordinal", "likert", "")
            d_lo, d_hi = detected
            c_lo, c_hi = canonical_range
            is_likert_to_likert = (
                d_hi - d_lo <= 20 and c_hi - c_lo <= 20
                and d_hi - d_lo != c_hi - c_lo
            )

            if is_ordinal or is_likert_to_likert:
                warning_msg = (
                    f"Linear rescaling from [{d_lo}, {d_hi}] to [{c_lo}, {c_hi}] "
                    f"assumes equidistant category boundaries. Consider IRT linking "
                    f"or equipercentile equating for more accurate cross-scale comparison."
                )
                warnings_rows.append({
                    "study": study_name,
                    "dv": dv,
                    "detected_range": f"[{d_lo}, {d_hi}]",
                    "canonical_range": f"[{c_lo}, {c_hi}]",
                    "scale_type": scale_type or "unknown (assumed ordinal)",
                    "warning": warning_msg,
                })
                _logger.info("IRT warning: %s/%s — %s", study_name, dv, warning_msg)

    return pd.DataFrame(warnings_rows, columns=cols)


# ── Bootstrap causal edge stability ──────────────────────────────────────────

def bootstrap_causal_stability(
    studies: Dict[str, pd.DataFrame],
    n_bootstrap: int = 100,
    min_shared_studies: int = 2,
    min_rows: int = 30,
    edge_threshold: float = 0.1,
    random_seed: int = 42,
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assess stability of LiNGAM causal edges via bootstrap resampling.

    Uses ``lingam.DirectLiNGAM.bootstrap`` (rows resampled with replacement,
    model refit per sample) and records, per directed edge, how often
    ``|coefficient| > edge_threshold`` appeared.

    Pass ``data`` — the matrix the main LiNGAM run was fit on
    (``discover_causal_structure(...)["data"]``) — so the stability estimates
    describe the reported DAG. Without it, rows complete on every usable
    shared DV are pooled from *studies*; with partial DV overlap across
    studies that is often empty.

    Returns a DataFrame with columns:
        from, to, appearance_rate, mean_coefficient, sd_coefficient
    where appearance_rate in [0, 1] indicates how frequently the edge appeared.
    Edges with appearance_rate > 0.5 are considered stable.
    """
    result_cols = ["from", "to", "appearance_rate", "mean_coefficient", "sd_coefficient"]
    if data is not None:
        X_full = data.dropna()
        usable_dvs = [str(c) for c in X_full.columns]
        if len(usable_dvs) < 2 or len(X_full) < 2:
            return pd.DataFrame(columns=result_cols)
    else:
        pooled_result = _pool_standardized_shared_dvs(
            _resolve_canonical(studies, canonical_studies), min_shared_studies, min_rows
        )
        if pooled_result is None:
            return pd.DataFrame(columns=result_cols)
        pooled, usable_dvs = pooled_result
        X_full = pooled[usable_dvs].dropna()
        if len(X_full) < min_rows:
            return pd.DataFrame(columns=result_cols)

    # lingam resamples via sklearn.utils.resample with the global NumPy RNG;
    # seed it for reproducibility and restore the caller's state afterwards.
    rng_state = np.random.get_state()
    np.random.seed(random_seed)
    try:
        bootstrap = lingam.DirectLiNGAM().bootstrap(X_full.to_numpy(dtype=float), n_sampling=n_bootstrap)
    finally:
        np.random.set_state(rng_state)

    # adjacency_matrices_[b, i, j] is the effect of variable j on variable i.
    matrices = np.asarray(bootstrap.adjacency_matrices_)
    present = np.abs(matrices) > edge_threshold
    rows = []
    for i, target in enumerate(usable_dvs):
        for j, source in enumerate(usable_dvs):
            hits = matrices[present[:, i, j], i, j]
            if hits.size == 0:
                continue
            rows.append({
                "from": source,
                "to": target,
                "appearance_rate": hits.size / n_bootstrap,
                "mean_coefficient": float(np.mean(hits)),
                "sd_coefficient": float(np.std(hits)),
            })
    if not rows:
        return pd.DataFrame(columns=result_cols)

    result = pd.DataFrame(rows, columns=result_cols).sort_values(["from", "to"])
    return result.sort_values("appearance_rate", ascending=False, kind="stable").reset_index(drop=True)


# ── PC algorithm for Gaussian data ──────────────────────────────────────────

def discover_causal_structure_pc(
    studies: Dict[str, pd.DataFrame],
    alpha: float = 0.05,
    min_shared_studies: int = 2,
    min_rows: int = 30,
    canonical_studies: Dict[str, pd.DataFrame] | None = None,
    data: pd.DataFrame | None = None,
) -> dict | None:
    """Constraint-based causal discovery using the PC algorithm skeleton.

    The PC algorithm is appropriate when data may be Gaussian (where LiNGAM's
    non-Gaussianity assumption fails).  This implements the skeleton phase:
    edges are removed when conditional independence is established via
    partial correlation tests with Fisher z-transformation.

    Pass ``data`` (``discover_causal_structure(...)["data"]``) to run on the
    same observations as the LiNGAM DAG.

    Returns dict with keys:
        skeleton: pd.DataFrame (adjacency matrix, 1 = edge present)
        n_observations: int
        n_dvs: int
        dvs_used: list[str]
        removed_edges: list[dict] with from, to, conditioning_set, p_value
    """
    if data is not None:
        # Same observations the main LiNGAM run used (see discover_causal_structure).
        X = data.dropna()
        usable_dvs = [str(c) for c in X.columns]
        if len(usable_dvs) < 2 or len(X) < 4:
            return None
    else:
        pooled_result = _pool_standardized_shared_dvs(
            _resolve_canonical(studies, canonical_studies), min_shared_studies, min_rows
        )
        if pooled_result is None:
            return None
        pooled, usable_dvs = pooled_result
        X = pooled[usable_dvs].dropna()
        if len(X) < min_rows:
            return None

    p = len(usable_dvs)
    n = len(X)

    # Initialize fully connected undirected skeleton
    skeleton = np.ones((p, p), dtype=int)
    np.fill_diagonal(skeleton, 0)
    removed_edges: list[dict] = []

    def _partial_corr_pvalue(i: int, j: int, cond_set: list[int]) -> float:
        """Fisher z-test for conditional independence via partial correlation."""
        if not cond_set:
            r = np.corrcoef(X.iloc[:, i], X.iloc[:, j])[0, 1]
        else:
            Z_cond = X.iloc[:, cond_set].values
            beta_i, _, _, _ = np.linalg.lstsq(Z_cond, X.iloc[:, i].values, rcond=None)
            res_i = X.iloc[:, i].values - Z_cond @ beta_i
            beta_j, _, _, _ = np.linalg.lstsq(Z_cond, X.iloc[:, j].values, rcond=None)
            res_j = X.iloc[:, j].values - Z_cond @ beta_j
            r = np.corrcoef(res_i, res_j)[0, 1]

        r = np.clip(r, -0.9999, 0.9999)
        z_stat = 0.5 * np.log((1 + r) / (1 - r))
        dof = n - len(cond_set) - 3
        if dof < 1:
            return 1.0
        se = 1.0 / np.sqrt(max(dof, 1))
        return float(2 * (1 - stats.norm.cdf(abs(z_stat / se))))

    # PC skeleton: iterate over conditioning set sizes
    max_cond_size = min(p - 2, 4)

    for cond_size in range(max_cond_size + 1):
        for i in range(p):
            neighbors_i = [j for j in range(p) if skeleton[i, j] == 1 and j != i]
            for j in neighbors_i:
                if skeleton[i, j] == 0:
                    continue

                candidates = [k for k in range(p)
                              if skeleton[i, k] == 1 and k != i and k != j]
                if len(candidates) < cond_size:
                    continue

                for cond_set in combinations(candidates, cond_size):
                    pval = _partial_corr_pvalue(i, j, list(cond_set))
                    if pval > alpha:
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                        removed_edges.append({
                            "from": usable_dvs[i],
                            "to": usable_dvs[j],
                            "conditioning_set": [usable_dvs[k] for k in cond_set],
                            "p_value": pval,
                        })
                        break

    skeleton_df = pd.DataFrame(skeleton, index=usable_dvs, columns=usable_dvs)
    return {
        "skeleton": skeleton_df,
        "n_observations": n,
        "n_dvs": p,
        "dvs_used": usable_dvs,
        "removed_edges": removed_edges,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multiple standardized studies.")
    parser.add_argument(
        "--input-dir",
        default="data/processed/multi_study_examples",
        help="Directory with standardized study files (.csv/.xlsx).",
    )
    parser.add_argument("--output-dir", default="analyses/output_python", help="Output directory.")
    parser.add_argument(
        "--estimator",
        choices=["DL", "REML"],
        default="DL",
        help="Tau-squared estimator: DerSimonian-Laird (DL) or REML.",
    )
    parser.add_argument(
        "--harmonize-scales",
        action="store_true",
        default=True,
        help="Rescale DVs to canonical ranges before pooling (default: on).",
    )
    parser.add_argument(
        "--no-harmonize-scales",
        dest="harmonize_scales",
        action="store_false",
        help="Disable scale harmonization.",
    )
    parser.add_argument(
        "--repeated-measures",
        default="",
        help=(
            "Comma-separated list of study keys to aggregate to participant-level means "
            "before analysis (e.g., ehmi_for_all_chi26,roads_chi25)."
        ),
    )
    parser.add_argument(
        "--exclude-llm-deduced",
        action="store_true",
        help=(
            "Run a sensitivity meta-analysis that drops study×DV rows whose alias "
            "was resolved via LLM deduction. Writes *_llm_excluded.csv companion "
            "files so you can compare against the full pooling."
        ),
    )
    parser.add_argument(
        "--meta-view",
        default=None,
        help=(
            "Optional path to meta_view.csv from run_batch_standardization. "
            "Defaults to '<input-dir>/../meta_view.csv' when present."
        ),
    )
    parser.add_argument(
        "--refuse-synthetic-causal",
        action="store_true",
        help=(
            "Refuse the Cholesky-synthesized fallback in LiNGAM causal "
            "discovery. Edges are only emitted when complete-case rows are "
            "sufficient; otherwise the pass is skipped."
        ),
    )
    parser.add_argument(
        "--causal-min-rows",
        type=int,
        default=200,
        help=(
            "Minimum complete-case rows required to run LiNGAM. Defaults to "
            "200; lower values produce noisier edges and should be justified."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    repeated_measures: set[str] = (
        {k.strip() for k in args.repeated_measures.split(",") if k.strip()}
        if args.repeated_measures
        else set()
    )

    # Locate the batch runner's meta_view.csv (or user-supplied path) to build
    # per-(study, DV) mapping provenance.  When absent, every row is stamped
    # "unknown" and the exclude-llm-deduced flag becomes a no-op.
    if args.meta_view:
        meta_view_path = Path(args.meta_view)
    else:
        meta_view_path = input_dir.parent / "meta_view.csv"
    mapping_provenance = load_mapping_provenance(meta_view_path)
    if mapping_provenance:
        print(f"Loaded mapping provenance for {len(mapping_provenance)} (study, DV) pairs "
              f"from {meta_view_path}")
    elif args.exclude_llm_deduced:
        print(
            f"[WARNING] --exclude-llm-deduced set but no mapping_source provenance was "
            f"found at {meta_view_path}. Nothing will be excluded."
        )

    validate_schema_clusters()

    studies = load_studies(
        input_dir,
        repeated_measures_studies=repeated_measures or None,
        skip_files_without_dvs=True,
    )
    # Canonicalize once; every analysis below reuses the same frames instead
    # of re-scanning all study columns per function call.
    canonical = _canonicalize_studies(studies)
    overlap = compute_overlap(studies, canonical_studies=canonical)
    presence = compute_dv_presence_matrix(studies, canonical_studies=canonical)
    overlap_details = compute_overlap_details(studies, canonical_studies=canonical)
    summary = harmonized_summary(
        studies,
        harmonize_scales=args.harmonize_scales,
        mapping_provenance=mapping_provenance,
        canonical_studies=canonical,
    )
    meta_summary_df = meta_analysis_summary(summary, total_studies=len(studies), estimator=args.estimator)
    effects = compute_standardized_effects(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(output_dir / "dv_overlap_matrix.csv")
    presence.to_csv(output_dir / "dv_presence_matrix.csv")
    overlap_details.to_csv(output_dir / "dv_overlap_details.csv", index=False)
    summary.to_csv(output_dir / "harmonized_dv_summary.csv", index=False)
    meta_summary_df.to_csv(output_dir / "meta_analysis_summary.csv", index=False)
    save_plots(overlap, summary, output_dir)

    # Sensitivity pass: rerun the pooling after dropping any study×DV whose
    # alias was resolved by the local LLM rather than an explicit schema.
    # Always safe to run — a no-op when no provenance is present.
    if args.exclude_llm_deduced:
        summary_clean = summary[summary["mapping_source"] != "llm_deduced"].copy()
        dropped_rows = int(len(summary) - len(summary_clean))
        clean_meta = meta_analysis_summary(
            summary_clean,
            total_studies=len(studies),
            estimator=args.estimator,
        )
        clean_effects = compute_standardized_effects(summary_clean)
        summary_clean.to_csv(output_dir / "harmonized_dv_summary_llm_excluded.csv", index=False)
        clean_meta.to_csv(output_dir / "meta_analysis_summary_llm_excluded.csv", index=False)
        if not clean_effects.empty:
            clean_effects.to_csv(
                output_dir / "study_vs_pool_standardized_deviation_llm_excluded.csv",
                index=False,
            )
        print(
            f"\nSensitivity (LLM-deduced rows excluded): dropped {dropped_rows} rows; "
            f"{len(clean_meta)} DV(s) remain in meta_analysis_summary_llm_excluded.csv"
        )

    # Build JSON results dict incrementally
    results_json: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "n_studies": len(studies),
        "studies": sorted(studies.keys()),
        "meta_analysis": [],
        "dv_overlap": {},
        "causal_discovery": None,
        "harmonized_summary": [],
    }

    # Populate meta_analysis section (with egger_p merged in later)
    egger_p_by_dv: dict[str, float] = {}

    print("Loaded studies:", ", ".join(studies.keys()))
    print("\nDV overlap matrix:\n", overlap.round(2).to_string())
    print("\nTop harmonized summaries:\n", summary.head(12).round(3).to_string(index=False))
    if not meta_summary_df.empty:
        print("\nMeta-analysis summaries:\n", meta_summary_df.round(3).to_string(index=False))

    # Standardized effects — this is a DESCRIPTIVE one-group effect (study
    # mean vs cross-study grand mean in pooled-SD units), not a contrast-based
    # intervention effect.  The preferred filename reflects that; a legacy
    # alias is retained for existing downstream consumers.
    if not effects.empty:
        effects.to_csv(output_dir / "study_vs_pool_standardized_deviation.csv", index=False)
        effects.to_csv(output_dir / "standardized_effects.csv", index=False)
        print(
            "\nStudy-vs-pool standardized deviation (one-group Hedges' g, descriptive):\n",
            effects.round(3).to_string(index=False),
        )

    # Subgroup analysis
    subgroup = subgroup_meta_analysis(effects)
    if not subgroup.empty:
        subgroup.to_csv(output_dir / "subgroup_meta_analysis.csv", index=False)
        print("\nSubgroup analysis by cluster:\n", subgroup.round(3).to_string(index=False))

    # Leave-one-out sensitivity
    sensitivity = leave_one_out_sensitivity(summary, total_studies=len(studies), estimator=args.estimator)
    if not sensitivity.empty:
        sensitivity.to_csv(output_dir / "leave_one_out_sensitivity.csv", index=False)
        print("\nLeave-one-out sensitivity:\n", sensitivity.round(3).to_string(index=False))

    # Egger's test for publication bias
    egger_rows = []
    for _, mrow in meta_summary_df.iterrows():
        result = eggers_test(summary, mrow["dv"])
        if result is not None:
            egger_rows.append(result)
            egger_p_by_dv[mrow["dv"]] = result.get("p_value")
    if egger_rows:
        egger_df = pd.DataFrame(egger_rows)
        egger_df.to_csv(output_dir / "eggers_test.csv", index=False)
        print("\nEgger's test for funnel asymmetry:\n", egger_df.round(3).to_string(index=False))

    # Trim-and-fill for publication bias. Every evaluated DV is written —
    # k_imputed == 0 is itself a result (no funnel asymmetry detected).
    tf_rows = []
    for _, mrow in meta_summary_df.iterrows():
        tf = trim_and_fill(summary, mrow["dv"], estimator=args.estimator)
        if tf is not None:
            tf_rows.append(tf)
    if tf_rows:
        tf_df = pd.DataFrame(tf_rows)
        tf_df.to_csv(output_dir / "trim_and_fill.csv", index=False)
        imputed = tf_df[tf_df["k_imputed"] > 0]
        print(f"\nTrim-and-fill (L0, {args.estimator}): "
              f"{len(imputed)} of {len(tf_df)} DVs with imputed studies")
        if not imputed.empty:
            print(imputed.round(3).to_string(index=False))

    # Meta-regression (sample size as default moderator)
    mr_rows = []
    for _, mrow in meta_summary_df.iterrows():
        if mrow["k_studies"] >= 3:
            mr = meta_regression(summary, mrow["dv"], moderator_col="n")
            if mr is not None:
                mr_rows.append(mr)
    if mr_rows:
        mr_df = pd.DataFrame(mr_rows)
        mr_df.to_csv(output_dir / "meta_regression.csv", index=False)
        print("\nMeta-regression (moderator=n):\n", mr_df.round(3).to_string(index=False))

    # Data quality report
    quality_df = flag_data_quality(studies, canonical_studies=canonical)
    flagged = quality_df[quality_df["n_flags"] > 0]
    if not flagged.empty:
        quality_df.to_csv(output_dir / "data_quality_report.csv", index=False)
        print(f"\nData quality: {len(flagged)} study×DV combinations flagged")

    # Reverse-coding detection
    rev_df = detect_potential_reverse_coding(studies, canonical_studies=canonical)
    if not rev_df.empty:
        rev_df.to_csv(output_dir / "reverse_coding_warnings.csv", index=False)
        print(f"\nReverse-coding warnings: {len(rev_df)} potential issues detected")

    # Extended overlap statistics
    ext_overlap = compute_extended_overlap_stats(studies, canonical_studies=canonical)
    ext_overlap["dv_frequency"].to_csv(output_dir / "dv_frequency.csv", index=False)
    ext_overlap["pairwise_overlap"].to_csv(output_dir / "pairwise_overlap_extended.csv", index=False)
    print(f"\nDV frequency table saved ({len(ext_overlap['dv_frequency'])} DVs)")

    # Build meta_analysis JSON entries (after egger results are available)
    for _, mrow in meta_summary_df.iterrows():
        dv = mrow["dv"]
        results_json["meta_analysis"].append({
            "dv": dv,
            "k": int(mrow["k_studies"]),
            "pooled_mean": mrow["random_effects_mean"],
            "pooled_se": mrow["random_effects_se"],
            "ci_lower": mrow["ci95_low"],
            "ci_upper": mrow["ci95_high"],
            "heterogeneity_i2": mrow["heterogeneity_i2_pct"],
            "heterogeneity_q": mrow["heterogeneity_q"],
            "heterogeneity_p": mrow["q_pvalue"],
            "egger_p": egger_p_by_dv.get(dv),
        })

    # Populate dv_overlap as nested dict
    results_json["dv_overlap"] = {
        str(idx): {str(col): v for col, v in row.items()}
        for idx, row in overlap.iterrows()
    }

    # Populate harmonized_summary
    results_json["harmonized_summary"] = summary.to_dict(orient="records")

    # Forest plots, funnel plots, sensitivity plots
    for _, mrow in meta_summary_df.iterrows():
        dv = mrow["dv"]
        save_forest_plot(summary, mrow, dv, output_dir)
        save_funnel_plot(summary, mrow, dv, output_dir)
        save_sensitivity_plot(sensitivity, dv, mrow, output_dir)

    # Composite index
    try:
        composite_scores = compute_composite_scores(studies, canonical_studies=canonical)
        composite = build_composite_index(studies, scores=composite_scores)
        composite.to_csv(output_dir / "cross_study_composite_summary.csv", index=False)
        save_composite_plot(studies, output_dir, scores=composite_scores)
        print("\nComposite index by study:\n", composite.round(3).to_string(index=False))
    except ValueError as e:
        print(f"\n[WARNING] Skipping composite index: {e}")

    # LiNGAM causal discovery
    causal_result = None
    try:
        causal_result = discover_causal_structure(
            studies,
            min_rows=args.causal_min_rows,
            refuse_synthetic=args.refuse_synthetic_causal,
            canonical_studies=canonical,
        )
        if causal_result is not None:
            causal_result["adjacency_matrix"].to_csv(output_dir / "causal_adjacency_matrix.csv")
            causal_result["edges"].to_csv(output_dir / "causal_edges.csv", index=False)
            save_causal_dag_plot(causal_result, output_dir)
            method = causal_result.get("used_method", "complete_case")
            synthetic = bool(causal_result.get("synthetic_fallback", False))
            n_complete = int(causal_result.get("n_complete_rows", 0))
            print(
                f"\nLiNGAM causal discovery ({causal_result['n_observations']} obs, "
                f"{causal_result['n_dvs']} DVs, method={method}, complete_case_rows={n_complete}):"
            )
            if synthetic:
                print(
                    "  [WARNING] synthetic_fallback=True — edges were fit on "
                    "Cholesky-synthesized rows, not observed data. Do not treat "
                    "these as causal claims. Re-run with --refuse-synthetic-causal "
                    "to suppress this fallback."
                )
            print("  Causal order:", " -> ".join(causal_result["causal_order"]))
            if not causal_result["edges"].empty:
                print("  Significant edges:")
                for _, edge in causal_result["edges"].iterrows():
                    print(f"    {edge['from']} -> {edge['to']}  (b={edge['coefficient']:.3f})")
            else:
                print("  No significant causal edges detected (all |b| < 0.1)")
        else:
            print("\n[WARNING] Skipping LiNGAM: insufficient shared DVs or complete-case rows "
                  "(synthetic fallback may be refused).")
    except Exception as e:
        print(f"\n[WARNING] Skipping LiNGAM causal discovery: {e}")

    # Populate causal_discovery section of JSON
    if causal_result is not None:
        edges_list = (
            causal_result["edges"].to_dict(orient="records")
            if not causal_result["edges"].empty
            else []
        )
        results_json["causal_discovery"] = {
            "dvs_used": causal_result["dvs_used"],
            "n_observations": causal_result["n_observations"],
            "causal_order": causal_result["causal_order"],
            "edges": edges_list,
            "used_method": causal_result.get("used_method"),
            "synthetic_fallback": bool(causal_result.get("synthetic_fallback", False)),
            "n_complete_rows": int(causal_result.get("n_complete_rows", 0)),
            "studies_used": causal_result.get("studies_used", []),
        }

    # Stability checks run on the exact observations the DAG was fit on.
    # Cholesky-synthesized rows are excluded: resampling fabricated data says
    # nothing about the stability of real edges.
    causal_data = None
    if causal_result is not None and not causal_result.get("synthetic_fallback", False):
        causal_data = causal_result.get("data")

    # Bootstrap causal edge stability
    try:
        boot_df = (
            bootstrap_causal_stability(studies, n_bootstrap=50, data=causal_data)
            if causal_data is not None
            else bootstrap_causal_stability(studies, n_bootstrap=50, canonical_studies=canonical)
        )
        if not boot_df.empty:
            boot_df.to_csv(output_dir / "causal_bootstrap_stability.csv", index=False)
            stable = boot_df[boot_df["appearance_rate"] >= 0.5]
            print(f"\nBootstrap causal stability ({len(boot_df)} edges, "
                  f"{len(stable)} stable at >= 50%):")
            if not stable.empty:
                for _, row in stable.iterrows():
                    print(f"  {row['from']} -> {row['to']}  "
                          f"(rate={row['appearance_rate']:.2f}, "
                          f"b={row['mean_coefficient']:.3f} +/- {row['sd_coefficient']:.3f})")
    except Exception as e:
        print(f"\n[WARNING] Skipping bootstrap stability: {e}")

    # PC algorithm (alternative for Gaussian data)
    try:
        pc_result = discover_causal_structure_pc(studies, canonical_studies=canonical, data=causal_data)
        if pc_result is not None:
            pc_result["skeleton"].to_csv(output_dir / "pc_skeleton_matrix.csv")
            if pc_result["removed_edges"]:
                pd.DataFrame(pc_result["removed_edges"]).to_csv(
                    output_dir / "pc_removed_edges.csv", index=False
                )
            n_edges = int(pc_result["skeleton"].values.sum()) // 2
            print(f"\nPC algorithm skeleton ({pc_result['n_dvs']} DVs, "
                  f"{n_edges} undirected edges retained)")
    except Exception as e:
        print(f"\n[WARNING] Skipping PC algorithm: {e}")

    # IRT linking / ordinal rescaling warnings
    irt_warnings = check_ordinal_rescaling_warnings(
        studies, harmonize_scales=args.harmonize_scales, canonical_studies=canonical
    )
    if not irt_warnings.empty:
        irt_warnings.to_csv(output_dir / "irt_rescaling_warnings.csv", index=False)
        print(f"\nIRT/ordinal rescaling warnings: {len(irt_warnings)} DVs flagged")
        print("  (See irt_rescaling_warnings.csv for details)")

    # Write comprehensive JSON output
    json_path = output_dir / "analysis_results.json"
    json_path.write_text(
        json.dumps(results_json, default=str, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote analysis results to {json_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    main()
