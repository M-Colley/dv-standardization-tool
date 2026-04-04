#!/usr/bin/env python3
"""Cross-study analysis for standardized DV datasets.

This script demonstrates analyses that remain informative even when independent
variables are unknown or inconsistent across studies.
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import lingam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from sklearn.decomposition import PCA

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
]
META_ANALYSIS_COLUMNS = [
    "dv",
    "k_studies",
    "study_coverage_pct",
    "random_effects_mean",
    "random_effects_se",
    "ci95_low",
    "ci95_high",
    "prediction_interval_low",
    "prediction_interval_high",
    "heterogeneity_q",
    "q_pvalue",
    "heterogeneity_i2_pct",
    "tau2",
    "tau",
    "h2",
    "estimator",
]
STANDARDIZED_EFFECTS_COLUMNS = [
    "study", "dv", "cohens_d", "hedges_g", "var_g", "se_g",
]
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
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
        rng = _parse_scale_range(primary_unit)
        if rng is not None:
            info["canonical_range"] = rng
        meta[dv_id] = info

    return meta


def _canonicalize_dv_name(name: str) -> str | None:
    return _load_standard_dv_lookup().get(_normalize_colname(str(name)))


# ── Scale harmonization ─────────────────────────────────────────────────────

def _detect_scale_range(series: pd.Series, canonical_range: tuple[float, float]) -> tuple[float, float]:
    """Heuristically detect the actual scale range of a data series.

    If the observed data clearly exceeds the canonical range, infer a wider
    scale (e.g. 0-100 VAS vs the declared 0-20 Likert).
    """
    smin, smax = float(series.min()), float(series.max())
    cmin, cmax = canonical_range
    cspan = cmax - cmin
    if cspan <= 0:
        return (smin, smax)

    # If data fits within canonical range (with small tolerance), keep it.
    tol = cspan * 0.15
    if smin >= cmin - tol and smax <= cmax + tol:
        return canonical_range

    # Try common alternative ranges.
    for alt_lo, alt_hi in [(0, 100), (1, 100), (0, 10), (1, 10), (1, 7), (1, 5)]:
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


def add_derived_scale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add TLX/SUS/AOA derived scores when all required mapped item columns are present."""
    out = df.copy()

    tlx_item_candidates = [
        ["tlx1", "nasa_tlx1", "mental_demand", "tlx_mental_demand", "nasa_tlx_mental"],
        ["tlx2", "nasa_tlx2", "physical_demand"],
        ["tlx3", "nasa_tlx3", "temporal_demand"],
        ["tlx4", "nasa_tlx4", "performance"],
        ["tlx5", "nasa_tlx5", "effort"],
        ["tlx6", "nasa_tlx6", "frustration"],
    ]
    tlx_items = [_resolve_series(out, cands) for cands in tlx_item_candidates]
    if all(item is not None for item in tlx_items):
        out["nasa_tlx_score"] = pd.concat(tlx_items, axis=1).mean(axis=1, skipna=False)

    sus_item_candidates = [
        ["sus1", "sus_1"],
        ["sus2", "sus_2"],
        ["sus3", "sus_3"],
        ["sus4", "sus_4"],
        ["sus5", "sus_5"],
        ["sus6", "sus_6"],
        ["sus7", "sus_7"],
        ["sus8", "sus_8"],
        ["sus9", "sus_9"],
        ["sus10", "sus_10"],
    ]
    sus_items = [_resolve_series(out, cands) for cands in sus_item_candidates]
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

    aoa_item_candidates = [
        ["aoa1", "aoa_1"],
        ["aoa2", "aoa_2"],
        ["aoa3", "aoa_3"],
        ["aoa4", "aoa_4"],
        ["aoa5", "aoa_5"],
        ["aoa6", "aoa_6"],
        ["aoa7", "aoa_7"],
        ["aoa8", "aoa_8"],
        ["aoa9", "aoa_9"],
    ]
    aoa_items = [_resolve_series(out, cands) for cands in aoa_item_candidates]
    if all(item is not None for item in aoa_items):
        out["aoa_usefulness"] = (
            (3 - aoa_items[0])
            + (-3 + aoa_items[2])
            + (3 - aoa_items[4])
            + (3 - aoa_items[6])
            + (3 - aoa_items[8])
        ) / 5.0
        out["aoa_satisfying"] = (
            (3 - aoa_items[1])
            + (3 - aoa_items[3])
            + (-3 + aoa_items[5])
            + (-3 + aoa_items[7])
        ) / 4.0

    return out


# ── Study loading ────────────────────────────────────────────────────────────

def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_excel(path)


def load_studies(input_dir: Path) -> Dict[str, pd.DataFrame]:
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
    """
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

    studies: Dict[str, pd.DataFrame] = {}
    for project_key, paths in sorted(groups.items()):
        frames = []
        for path in paths:
            df = _read_file(path)
            df["_source_file"] = path.name  # traceability column
            frames.append(df)
            print(f"  [{project_key}] loaded {path.name} ({len(df)} rows)")

        combined = pd.concat(frames, ignore_index=True)
        print(
            f"  -> project '{project_key}': {len(combined)} total rows "
            f"from {len(paths)} file(s)"
        )
        studies[project_key] = add_derived_scale_scores(combined)

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


def numeric_dvs(df: pd.DataFrame) -> List[str]:
    return list(_canonical_numeric_frame(df).columns)


def _study_numeric_dv_sets_from_canonical(studies: Dict[str, pd.DataFrame]) -> Dict[str, set[str]]:
    return {name: set(df.columns) for name, df in studies.items()}


def study_numeric_dv_sets(studies: Dict[str, pd.DataFrame]) -> Dict[str, set[str]]:
    return _study_numeric_dv_sets_from_canonical(_canonicalize_studies(studies))


# ── Overlap analysis ─────────────────────────────────────────────────────────

def compute_overlap(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    dv_sets = _study_numeric_dv_sets_from_canonical(_canonicalize_studies(studies))
    index = list(studies.keys())
    overlap = pd.DataFrame(index=index, columns=index, dtype=float)
    for a in index:
        for b in index:
            union = dv_sets[a] | dv_sets[b]
            overlap.loc[a, b] = len(dv_sets[a] & dv_sets[b]) / len(union) if union else np.nan
    return overlap


def compute_dv_presence_matrix(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    dv_sets = _study_numeric_dv_sets_from_canonical(_canonicalize_studies(studies))
    all_dvs = sorted({dv for dvs in dv_sets.values() for dv in dvs})
    presence = pd.DataFrame(0, index=sorted(studies.keys()), columns=all_dvs, dtype=int)
    for study, dvs in dv_sets.items():
        for dv in dvs:
            presence.loc[study, dv] = 1
    return presence


def compute_overlap_details(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    dv_sets = _study_numeric_dv_sets_from_canonical(_canonicalize_studies(studies))
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
) -> pd.DataFrame:
    metadata = _load_dv_measurement_metadata() if harmonize_scales else {}
    rows = []
    for study, df in _canonicalize_studies(studies).items():
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

            rows.append(
                {
                    "study": study,
                    "dv": dv,
                    "n": s.shape[0],
                    "mean": s.mean(),
                    "sd": s.std(ddof=1),
                    "scale_note": scale_note,
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
    """Compute Hedges' g for each study relative to the grand mean per DV.

    For one-group descriptive meta-analysis (no shared control group), we
    standardize each study's mean relative to the pooled cross-study mean.
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
            })
    return pd.DataFrame(rows, columns=STANDARDIZED_EFFECTS_COLUMNS)


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

        # REML log-likelihood gradient and Hessian (Fisher information)
        gradient = -0.5 * np.sum(w) + 0.5 * np.sum(w ** 2 * residuals_sq) + 0.5 * (np.sum(w ** 2) / np.sum(w))
        fisher_info = 0.5 * (np.sum(w ** 2) - np.sum(w ** 3) / np.sum(w))

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

    # Tau-squared
    if estimator == "REML":
        tau2 = _estimate_tau2_reml(means, variances)
    else:
        c = np.sum(w_fixed) - (np.sum(np.square(w_fixed)) / np.sum(w_fixed))
        tau2 = max(0.0, (q - df_q) / c) if c > 0 else 0.0

    # Random-effects pooling
    w_random = 1.0 / (variances + tau2)
    random_mean = float(np.sum(w_random * means) / np.sum(w_random))
    random_se = float(np.sqrt(1.0 / np.sum(w_random)))

    # Heterogeneity
    i2 = max(0.0, (q - df_q) / q) * 100 if q > 0 else 0.0
    q_pvalue = float(stats.chi2.sf(q, df_q)) if df_q > 0 else np.nan
    h2 = q / df_q if df_q > 0 else np.nan
    tau = np.sqrt(tau2)

    # Prediction interval (requires k >= 3)
    if k >= 3:
        t_crit = float(stats.t.ppf(0.975, k - 2))
        pi_half = t_crit * np.sqrt(tau2 + random_se ** 2)
        pi_low = random_mean - pi_half
        pi_high = random_mean + pi_half
    else:
        pi_low = np.nan
        pi_high = np.nan

    return {
        "dv": dv,
        "k_studies": k,
        "study_coverage_pct": (
            (k / total_studies) * 100.0
            if total_studies and total_studies > 0
            else np.nan
        ),
        "random_effects_mean": random_mean,
        "random_effects_se": random_se,
        "ci95_low": random_mean - 1.96 * random_se,
        "ci95_high": random_mean + 1.96 * random_se,
        "prediction_interval_low": pi_low,
        "prediction_interval_high": pi_high,
        "heterogeneity_q": q,
        "q_pvalue": q_pvalue,
        "heterogeneity_i2_pct": i2,
        "tau2": tau2,
        "tau": tau,
        "h2": h2,
        "estimator": estimator,
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

def _prepare_composite_matrix(studies: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, List[str]]:
    canonical_studies = _canonicalize_studies(studies)
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
        x = x.fillna(x.median())
        x = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
        x = x.reindex(columns=common_cols)
        x = x.fillna(0.0)
        x["study"] = study
        standardized.append(x)
    z = pd.concat(standardized, ignore_index=True)

    feature_matrix = z[common_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    usable_cols = [col for col in common_cols if feature_matrix[col].nunique(dropna=True) > 1]
    if len(usable_cols) < 2:
        raise ValueError("Need at least two shared DVs with non-zero variance for PCA composite index.")
    return z, usable_cols


def build_composite_index(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    z, usable_cols = _prepare_composite_matrix(studies)

    pca = PCA(n_components=1)
    z["cross_study_composite"] = pca.fit_transform(z[usable_cols])

    summary = (
        z.groupby("study")["cross_study_composite"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n", "std": "sd"})
        .reset_index()
    )
    summary["explained_variance_ratio"] = pca.explained_variance_ratio_[0]
    summary["n_shared_dvs_used"] = len(usable_cols)
    return summary


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_plots(overlap: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(overlap, annot=True, cmap="Blues", vmin=0, vmax=1)
    plt.title("DV overlap across standardized studies (Jaccard)")
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
            plt.xticks(rotation=25, ha="right")
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
        plt.tight_layout()
        plt.savefig(output_dir / "dv_coverage_by_study.png", dpi=150)
        plt.close()


def save_composite_plot(studies: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    try:
        z, usable_cols = _prepare_composite_matrix(studies)
    except ValueError:
        return

    pca = PCA(n_components=1)
    z["cross_study_composite"] = pca.fit_transform(z[usable_cols])

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=z, x="study", y="cross_study_composite", inner="quartile", hue="study", legend=False)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Study")
    plt.ylabel("Cross-study composite score (PCA PC1)")
    plt.title("Distribution of cross-study composite scores")
    plt.tight_layout()
    plt.savefig(output_dir / "cross_study_composite_distribution.png", dpi=150)
    plt.close()


def save_forest_plot(
    summary: pd.DataFrame,
    meta_row: pd.Series,
    dv: str,
    output_dir: Path,
) -> None:
    """Generate a forest plot for a single DV."""
    sub = summary[(summary["dv"] == dv) & (summary["n"] > 1) & (summary["sd"] > 0)].copy()
    if len(sub) < 2:
        return

    sub = sub.sort_values("study").reset_index(drop=True)
    k = len(sub)
    means = sub["mean"].values
    ses = sub["sd"].values / np.sqrt(sub["n"].values)
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

    fig, ax = plt.subplots(figsize=(10, max(3, 1 + 0.6 * k)))

    y_positions = list(range(k, 0, -1))

    # Study-level estimates
    for i, y in enumerate(y_positions):
        # Square proportional to weight
        marker_size = max(4, min(15, w_pct[i] * 0.5))
        ax.plot(means[i], y, "s", color="#2166ac", markersize=marker_size, zorder=3)
        ax.hlines(y, ci_lo[i], ci_hi[i], color="#2166ac", linewidth=1.5, zorder=2)

        # Annotation
        label = f"{means[i]:.2f} [{ci_lo[i]:.2f}, {ci_hi[i]:.2f}]  w={w_pct[i]:.1f}%"
        ax.text(
            ax.get_xlim()[1] if ax.get_xlim()[1] != 1.0 else ci_hi[i] + ses[i],
            y, f"  {label}", va="center", fontsize=8, color="gray",
        )

    # Pooled diamond
    diamond_y = 0.3
    diamond_half_h = 0.25
    diamond_x = [pooled_lo, pooled_mean, pooled_hi, pooled_mean]
    diamond_yy = [diamond_y, diamond_y + diamond_half_h, diamond_y, diamond_y - diamond_half_h]
    ax.fill(diamond_x, diamond_yy, color="#b2182b", alpha=0.7, zorder=3)

    # Prediction interval (if available)
    pi_lo = meta_row.get("prediction_interval_low")
    pi_hi = meta_row.get("prediction_interval_high")
    if pd.notna(pi_lo) and pd.notna(pi_hi):
        ax.hlines(diamond_y, pi_lo, pi_hi, color="#b2182b", linewidth=1, linestyle="--", alpha=0.5, zorder=2)

    # Reference lines
    ax.axvline(pooled_mean, color="#b2182b", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_yticks(y_positions + [0])
    ax.set_yticklabels(list(sub["study"].values) + ["Pooled RE"], fontsize=9)
    ax.set_xlabel(f"{dv} (mean)")
    ax.set_title(f"Forest plot: {dv} (k={k}, I\u00B2={meta_row['heterogeneity_i2_pct']:.1f}%)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Re-draw annotations with correct xlim
    fig.tight_layout()
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


# ── LiNGAM causal discovery ─────────────────────────────────────────────────

def discover_causal_structure(
    studies: Dict[str, pd.DataFrame],
    min_shared_studies: int = 2,
    min_rows: int = 30,
) -> dict | None:
    """Discover causal ordering among shared DVs using DirectLiNGAM.

    Pools standardized (within-study z-scored) data across studies for DVs
    shared by at least *min_shared_studies* studies. Applies DirectLiNGAM to
    the pooled matrix and returns the adjacency matrix + causal order.

    Returns None if fewer than 2 shared DVs have sufficient data.
    """
    canonical_studies = _canonicalize_studies(studies)

    # Find DVs shared across enough studies
    dv_counts: Dict[str, int] = {}
    for df in canonical_studies.values():
        for dv in df.columns:
            dv_counts[dv] = dv_counts.get(dv, 0) + 1
    shared_dvs = sorted(dv for dv, cnt in dv_counts.items() if cnt >= min_shared_studies)

    if len(shared_dvs) < 2:
        return None

    # Pool data: z-score within each study, then concatenate
    pooled_blocks = []
    for study_name, df in canonical_studies.items():
        sub = df.reindex(columns=shared_dvs).dropna()
        if len(sub) < min_rows:
            continue
        # Within-study standardization to remove scale differences
        z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
        z = z.dropna(axis=1, how="all").dropna()
        pooled_blocks.append(z)

    if not pooled_blocks:
        return None

    pooled = pd.concat(pooled_blocks, ignore_index=True)

    # Only keep columns present in all blocks
    usable_dvs = [c for c in shared_dvs if c in pooled.columns and pooled[c].notna().sum() >= min_rows]
    if len(usable_dvs) < 2:
        return None

    X = pooled[usable_dvs].dropna()
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

    return {
        "adjacency_matrix": adjacency,
        "causal_order": causal_order,
        "edges": pd.DataFrame(edges) if edges else pd.DataFrame(columns=["from", "to", "coefficient"]),
        "n_observations": len(X),
        "n_dvs": len(usable_dvs),
        "dvs_used": usable_dvs,
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
        default=False,
        help="Rescale DVs to canonical ranges before pooling.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    studies = load_studies(input_dir)
    overlap = compute_overlap(studies)
    presence = compute_dv_presence_matrix(studies)
    overlap_details = compute_overlap_details(studies)
    summary = harmonized_summary(studies, harmonize_scales=args.harmonize_scales)
    meta_summary_df = meta_analysis_summary(summary, total_studies=len(studies), estimator=args.estimator)
    effects = compute_standardized_effects(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(output_dir / "dv_overlap_matrix.csv")
    presence.to_csv(output_dir / "dv_presence_matrix.csv")
    overlap_details.to_csv(output_dir / "dv_overlap_details.csv", index=False)
    summary.to_csv(output_dir / "harmonized_dv_summary.csv", index=False)
    meta_summary_df.to_csv(output_dir / "meta_analysis_summary.csv", index=False)
    save_plots(overlap, summary, output_dir)

    print("Loaded studies:", ", ".join(studies.keys()))
    print("\nDV overlap matrix:\n", overlap.round(2).to_string())
    print("\nTop harmonized summaries:\n", summary.head(12).round(3).to_string(index=False))
    if not meta_summary_df.empty:
        print("\nMeta-analysis summaries:\n", meta_summary_df.round(3).to_string(index=False))

    # Standardized effects
    if not effects.empty:
        effects.to_csv(output_dir / "standardized_effects.csv", index=False)
        print("\nStandardized effects (Hedges' g):\n", effects.round(3).to_string(index=False))

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
    if egger_rows:
        egger_df = pd.DataFrame(egger_rows)
        egger_df.to_csv(output_dir / "eggers_test.csv", index=False)
        print("\nEgger's test for funnel asymmetry:\n", egger_df.round(3).to_string(index=False))

    # Forest plots, funnel plots, sensitivity plots
    for _, mrow in meta_summary_df.iterrows():
        dv = mrow["dv"]
        save_forest_plot(summary, mrow, dv, output_dir)
        save_funnel_plot(summary, mrow, dv, output_dir)
        save_sensitivity_plot(sensitivity, dv, mrow, output_dir)

    # Composite index
    try:
        composite = build_composite_index(studies)
        composite.to_csv(output_dir / "cross_study_composite_summary.csv", index=False)
        save_composite_plot(studies, output_dir)
        print("\nComposite index by study:\n", composite.round(3).to_string(index=False))
    except ValueError as e:
        print(f"\n[WARNING] Skipping composite index: {e}")

    # LiNGAM causal discovery
    try:
        causal_result = discover_causal_structure(studies)
        if causal_result is not None:
            causal_result["adjacency_matrix"].to_csv(output_dir / "causal_adjacency_matrix.csv")
            causal_result["edges"].to_csv(output_dir / "causal_edges.csv", index=False)
            save_causal_dag_plot(causal_result, output_dir)
            print(f"\nLiNGAM causal discovery ({causal_result['n_observations']} obs, "
                  f"{causal_result['n_dvs']} DVs):")
            print("  Causal order:", " -> ".join(causal_result["causal_order"]))
            if not causal_result["edges"].empty:
                print("  Significant edges:")
                for _, edge in causal_result["edges"].iterrows():
                    print(f"    {edge['from']} -> {edge['to']}  (b={edge['coefficient']:.3f})")
            else:
                print("  No significant causal edges detected (all |b| < 0.1)")
        else:
            print("\n[WARNING] Skipping LiNGAM: fewer than 2 DVs shared across 2+ studies")
    except Exception as e:
        print(f"\n[WARNING] Skipping LiNGAM causal discovery: {e}")


if __name__ == "__main__":
    main()
