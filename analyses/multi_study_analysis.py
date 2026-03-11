#!/usr/bin/env python3
"""Cross-study analysis for standardized DV datasets.

This script demonstrates analyses that remain informative even when independent
variables are unknown or inconsistent across studies.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
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
HARMONIZED_SUMMARY_COLUMNS = ["study", "dv", "n", "mean", "sd", "mean_z_vs_global"]
META_ANALYSIS_COLUMNS = [
    "dv",
    "k_studies",
    "study_coverage_pct",
    "random_effects_mean",
    "random_effects_se",
    "ci95_low",
    "ci95_high",
    "heterogeneity_q",
    "heterogeneity_i2_pct",
    "tau2",
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


def _normalize_colname(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {_normalize_colname(col): col for col in df.columns}


@lru_cache(maxsize=1)
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

        candidates = [canonical, str(entry.get("label", "")).strip(), *entry.get("aliases", [])]
        for candidate in candidates:
            normalized = _normalize_colname(str(candidate))
            if normalized and normalized not in lookup:
                lookup[normalized] = canonical

    return lookup


def _canonicalize_dv_name(name: str) -> str | None:
    return _load_standard_dv_lookup().get(_normalize_colname(str(name)))


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
            part1.csv       â† combined into study "study_a"
            part2.csv
        study_b/
            results.xlsx    â† single-file study "study_b"
        study_c.csv         â† files at root are treated as individual studies

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


def harmonized_summary(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for study, df in _canonicalize_studies(studies).items():
        for dv in df.columns:
            s = df[dv].dropna()
            rows.append(
                {
                    "study": study,
                    "dv": dv,
                    "n": s.shape[0],
                    "mean": s.mean(),
                    "sd": s.std(ddof=1),
                }
            )
    summary = pd.DataFrame(rows, columns=HARMONIZED_SUMMARY_COLUMNS[:-1])
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


def meta_analysis_summary(summary: pd.DataFrame, total_studies: int | None = None) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=META_ANALYSIS_COLUMNS)

    rows = []
    for dv, sub in summary.groupby("dv"):
        sub = sub[(sub["n"] > 1) & (sub["sd"] > 0)].copy()
        if len(sub) < 2:
            continue
        var = (sub["sd"] ** 2) / sub["n"]
        if (var <= 0).any():
            continue
        w_fixed = 1.0 / var
        fixed_mean = np.sum(w_fixed * sub["mean"]) / np.sum(w_fixed)
        q = np.sum(w_fixed * np.square(sub["mean"] - fixed_mean))
        df_q = len(sub) - 1
        c = np.sum(w_fixed) - (np.sum(np.square(w_fixed)) / np.sum(w_fixed))
        tau2 = max(0.0, (q - df_q) / c) if c > 0 else 0.0
        w_random = 1.0 / (var + tau2)
        random_mean = np.sum(w_random * sub["mean"]) / np.sum(w_random)
        random_se = np.sqrt(1.0 / np.sum(w_random))
        rows.append(
            {
                "dv": dv,
                "k_studies": len(sub),
                "study_coverage_pct": (
                    (len(sub) / total_studies) * 100.0
                    if total_studies and total_studies > 0
                    else np.nan
                ),
                "random_effects_mean": random_mean,
                "random_effects_se": random_se,
                "ci95_low": random_mean - 1.96 * random_se,
                "ci95_high": random_mean + 1.96 * random_se,
                "heterogeneity_q": q,
                "heterogeneity_i2_pct": max(0.0, (q - df_q) / q) * 100 if q > 0 else 0.0,
                "tau2": tau2,
            }
        )
    meta = pd.DataFrame(rows, columns=META_ANALYSIS_COLUMNS)
    if meta.empty:
        return meta
    return meta.sort_values("dv").reset_index(drop=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multiple standardized studies.")
    parser.add_argument(
        "--input-dir",
        default="data/processed/multi_study_examples",
        help="Directory with standardized study files (.csv/.xlsx).",
    )
    parser.add_argument("--output-dir", default="analyses/output_python", help="Output directory.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    studies = load_studies(input_dir)
    overlap = compute_overlap(studies)
    presence = compute_dv_presence_matrix(studies)
    overlap_details = compute_overlap_details(studies)
    summary = harmonized_summary(studies)
    meta_summary = meta_analysis_summary(summary, total_studies=len(studies))

    output_dir.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(output_dir / "dv_overlap_matrix.csv")
    presence.to_csv(output_dir / "dv_presence_matrix.csv")
    overlap_details.to_csv(output_dir / "dv_overlap_details.csv", index=False)
    summary.to_csv(output_dir / "harmonized_dv_summary.csv", index=False)
    meta_summary.to_csv(output_dir / "meta_analysis_summary.csv", index=False)
    save_plots(overlap, summary, output_dir)

    print("Loaded studies:", ", ".join(studies.keys()))
    print("\nDV overlap matrix:\n", overlap.round(2).to_string())
    print("\nTop harmonized summaries:\n", summary.head(12).round(3).to_string(index=False))
    if not meta_summary.empty:
        print("\nMeta-analysis summaries:\n", meta_summary.round(3).to_string(index=False))

    try:
        composite = build_composite_index(studies)
        composite.to_csv(output_dir / "cross_study_composite_summary.csv", index=False)
        save_composite_plot(studies, output_dir)
        print("\nComposite index by study:\n", composite.round(3).to_string(index=False))
    except ValueError as e:
        print(f"\n[WARNING] Skipping composite index: {e}")


if __name__ == "__main__":
    main()
