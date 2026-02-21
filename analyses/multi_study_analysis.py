#!/usr/bin/env python3
"""Cross-study analysis for standardized DV datasets.

This script demonstrates analyses that remain informative even when independent
variables are unknown or inconsistent across studies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def _normalize_colname(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {_normalize_colname(col): col for col in df.columns}


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


def load_studies(input_dir: Path) -> Dict[str, pd.DataFrame]:
    files = sorted(list(input_dir.glob("*.csv")) + list(input_dir.glob("*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {input_dir}")

    studies: Dict[str, pd.DataFrame] = {}
    for path in files:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        studies[path.stem] = add_derived_scale_scores(df)
    return studies


def numeric_dvs(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col.lower() in ID_LIKE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def compute_overlap(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    dv_sets = {name: set(numeric_dvs(df)) for name, df in studies.items()}
    index = list(studies.keys())
    overlap = pd.DataFrame(index=index, columns=index, dtype=float)
    for a in index:
        for b in index:
            union = dv_sets[a] | dv_sets[b]
            overlap.loc[a, b] = len(dv_sets[a] & dv_sets[b]) / len(union) if union else np.nan
    return overlap


def harmonized_summary(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for study, df in studies.items():
        for dv in numeric_dvs(df):
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
    summary = pd.DataFrame(rows)
    global_mean = summary.groupby("dv")["mean"].transform("mean")
    pooled_sd = summary.groupby("dv")["sd"].transform(lambda x: np.sqrt(np.nanmean(np.square(x))))
    summary["mean_z_vs_global"] = (summary["mean"] - global_mean) / pooled_sd.replace(0, np.nan)
    return summary.sort_values(["dv", "study"])


def build_composite_index(studies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    counts = {}
    for df in studies.values():
        for dv in set(numeric_dvs(df)):
            counts[dv] = counts.get(dv, 0) + 1
    common_cols = sorted([dv for dv, k in counts.items() if k >= 2])
    if len(common_cols) < 2:
        raise ValueError("Need at least two DVs shared by at least two studies for PCA composite index.")

    stacked = []
    for study, df in studies.items():
        block = df.reindex(columns=common_cols).copy()
        block["study"] = study
        stacked.append(block)
    long = pd.concat(stacked, ignore_index=True)

    # Standardize within each study to remove study-specific scaling artifacts.
    standardized = []
    for study, sub in long.groupby("study", sort=False):
        x = sub[common_cols].copy()
        x = x.fillna(x.median())
        x = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
        x = x.fillna(0.0)
        x["study"] = study
        standardized.append(x)
    z = pd.concat(standardized, ignore_index=True)

    pca = PCA(n_components=1)
    z["cross_study_composite"] = pca.fit_transform(z[common_cols])

    summary = (
        z.groupby("study")["cross_study_composite"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n", "std": "sd"})
        .reset_index()
    )
    summary["explained_variance_ratio"] = pca.explained_variance_ratio_[0]
    return summary


def save_plots(overlap: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(overlap, annot=True, cmap="Blues", vmin=0, vmax=1)
    plt.title("DV overlap across standardized studies (Jaccard)")
    plt.tight_layout()
    plt.savefig(output_dir / "dv_overlap_heatmap.png", dpi=150)
    plt.close()

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
    plt.ylabel("Count of numeric standardized DVs")
    plt.xlabel("Study")
    plt.title("Numeric DV coverage by study")
    plt.tight_layout()
    plt.savefig(output_dir / "dv_coverage_by_study.png", dpi=150)
    plt.close()


def save_composite_plot(studies: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    counts = {}
    for df in studies.values():
        for dv in set(numeric_dvs(df)):
            counts[dv] = counts.get(dv, 0) + 1
    common_cols = sorted([dv for dv, k in counts.items() if k >= 2])
    if len(common_cols) < 2:
        return

    stacked = []
    for study, df in studies.items():
        block = df.reindex(columns=common_cols).copy()
        block["study"] = study
        stacked.append(block)
    long = pd.concat(stacked, ignore_index=True)

    standardized = []
    for study, sub in long.groupby("study", sort=False):
        x = sub[common_cols].copy()
        x = x.fillna(x.median())
        x = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
        x = x.fillna(0.0)
        x["study"] = study
        standardized.append(x)
    z = pd.concat(standardized, ignore_index=True)

    pca = PCA(n_components=1)
    z["cross_study_composite"] = pca.fit_transform(z[common_cols])

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
    summary = harmonized_summary(studies)
    composite = build_composite_index(studies)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(output_dir / "dv_overlap_matrix.csv")
    summary.to_csv(output_dir / "harmonized_dv_summary.csv", index=False)
    composite.to_csv(output_dir / "cross_study_composite_summary.csv", index=False)
    save_plots(overlap, summary, output_dir)
    save_composite_plot(studies, output_dir)

    print("Loaded studies:", ", ".join(studies.keys()))
    print("\nDV overlap matrix:\n", overlap.round(2).to_string())
    print("\nTop harmonized summaries:\n", summary.head(12).round(3).to_string(index=False))
    print("\nComposite index by study:\n", composite.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
