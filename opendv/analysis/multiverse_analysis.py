#!/usr/bin/env python3
"""Harmonization multiverse: how much do pooled DV estimates depend on choices?

Cross-study pooling of standardized DVs involves several defensible but
arbitrary decisions. This script crosses them (Steegen et al., 2016;
Simonsohn et al., 2020) and re-runs the meta-analysis for every combination:

==================  =========================================================
choice              options
==================  =========================================================
mappings            all  |  without LLM-deduced (study, DV) rows
scales              harmonized to the canonical range  |  as recorded
unit                observation rows  |  participant means (where an ID exists)
tau2                DerSimonian-Laird  |  REML
interval            Wald (normal)  |  Hartung-Knapp-Sidik-Jonkman (t, k-1 df)
==================  =========================================================

For each DV pooled under at least one specification it writes every
specification's estimate and interval (``multiverse_specifications.csv``), a
per-DV robustness summary (``multiverse_summary.csv``) and a specification
curve per DV pooled across >= 3 studies (``spec_curve_<dv>.png``).

Example::

    python analyses/multiverse_analysis.py \\
        --input-dir data/processed/catalog_meta_analysis/standardized \\
        --output-dir analyses/output_multiverse
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


from opendv.analysis import multi_study_analysis as msa  # noqa: E402

CHOICES = {
    "mappings": ["all", "no_llm"],
    "scales": ["harmonized", "raw"],
    "unit": ["observations", "participants"],
    "tau2": ["DL", "REML"],
    "interval": ["wald", "hksj"],
}


def _summaries(input_dir: Path, meta_view: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Harmonized per-(study, DV) summaries for every (unit, scales) choice."""
    provenance = msa.load_mapping_provenance(meta_view)
    loaded = {}
    rows_studies = msa.load_studies(input_dir, skip_files_without_dvs=True)
    participant_studies = msa.load_studies(
        input_dir, repeated_measures_studies=set(rows_studies), skip_files_without_dvs=True
    )
    for unit, studies in (("observations", rows_studies), ("participants", participant_studies)):
        canonical = msa._canonicalize_studies(studies)
        for scales in CHOICES["scales"]:
            loaded[(unit, scales)] = msa.harmonized_summary(
                studies,
                harmonize_scales=scales == "harmonized",
                mapping_provenance=provenance,
                canonical_studies=canonical,
            )
    return loaded


def run(input_dir: Path, meta_view: Path) -> pd.DataFrame:
    summaries = _summaries(input_dir, meta_view)
    records = []
    for spec in itertools.product(*CHOICES.values()):
        choice = dict(zip(CHOICES, spec, strict=True))
        summary = summaries[(choice["unit"], choice["scales"])]
        if choice["mappings"] == "no_llm":
            summary = summary[summary["mapping_source"] != "llm_deduced"]
        meta = msa.meta_analysis_summary(summary, estimator=choice["tau2"])
        for row in meta.itertuples(index=False):
            low, high = (
                (row.hksj_ci95_low, row.hksj_ci95_high)
                if choice["interval"] == "hksj"
                else (row.ci95_low, row.ci95_high)
            )
            records.append({
                **choice,
                "dv": row.dv,
                "k_studies": row.k_studies,
                "estimate": row.random_effects_mean,
                "ci95_low": low,
                "ci95_high": high,
                "pooling_method": row.pooling_method,
                "includes_llm_deduced": row.includes_llm_deduced,
            })
    return pd.DataFrame(records)


def summarize(specs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_specs = int(np.prod([len(v) for v in CHOICES.values()]))
    for dv, sub in specs.groupby("dv", sort=True):
        reference = sub[
            (sub["mappings"] == "all") & (sub["scales"] == "harmonized") & (sub["unit"] == "observations")
            & (sub["tau2"] == "DL") & (sub["interval"] == "wald")
        ]
        widths = sub["ci95_high"] - sub["ci95_low"]
        ref_row = reference.iloc[0] if len(reference) else None
        rows.append({
            "dv": dv,
            "n_specifications": len(sub),
            "n_specifications_possible": n_specs,
            "k_min": int(sub["k_studies"].min()),
            "k_max": int(sub["k_studies"].max()),
            "estimate_default": ref_row["estimate"] if ref_row is not None else np.nan,
            "estimate_min": sub["estimate"].min(),
            "estimate_max": sub["estimate"].max(),
            "estimate_range": sub["estimate"].max() - sub["estimate"].min(),
            "ci_width_min": widths.min(),
            "ci_width_max": widths.max(),
            "ci_width_ratio_max_min": (widths.max() / widths.min()) if widths.min() > 0 else np.nan,
            # Share of specifications whose interval contains the default estimate.
            "share_covering_default": (
                float(((sub["ci95_low"] <= ref_row["estimate"]) & (sub["ci95_high"] >= ref_row["estimate"])).mean())
                if ref_row is not None else np.nan
            ),
            "vanishes_without_llm": not (sub["mappings"] == "no_llm").any(),
        })
    return pd.DataFrame(rows)


def plot_curve(specs: pd.DataFrame, dv: str, output_dir: Path) -> None:
    sub = specs[specs["dv"] == dv].sort_values("estimate").reset_index(drop=True)
    x = np.arange(len(sub))
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(7.0, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.6], "hspace": 0.05}
    )
    colors = np.where(sub["interval"] == "hksj", "#b2182b", "#2166ac")
    ax.vlines(x, sub["ci95_low"], sub["ci95_high"], color=colors, alpha=0.55, linewidth=1.2)
    ax.scatter(x, sub["estimate"], c=colors, s=14, zorder=3)
    ax.set_ylabel(f"{dv}\npooled estimate")
    ax.set_title(f"Specification curve: {dv} ({len(sub)} specifications)", fontsize=9, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    labels = []
    for i, (choice, options) in enumerate(CHOICES.items()):
        for j, option in enumerate(options):
            y = -(i * 2 + j)
            labels.append((y, f"{choice}: {option}"))
            mask = (sub[choice] == option).to_numpy()
            axc.scatter(x[mask], np.full(mask.sum(), y), marker="|", s=40, color="#333333")
    axc.set_yticks([y for y, _ in labels])
    axc.set_yticklabels([label for _, label in labels], fontsize=6.5)
    axc.set_xticks([])
    for spine in ("top", "right", "bottom"):
        axc.spines[spine].set_visible(False)
    fig.savefig(output_dir / f"spec_curve_{dv}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", required=True, help="standardized/ directory of a batch run.")
    parser.add_argument("--meta-view", default=None, help="Defaults to <input-dir>/../meta_view.csv.")
    parser.add_argument("--output-dir", default="analyses/output_multiverse")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    meta_view = Path(args.meta_view) if args.meta_view else input_dir.parent / "meta_view.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = run(input_dir, meta_view)
    specs.to_csv(output_dir / "multiverse_specifications.csv", index=False)
    summary = summarize(specs)
    summary.to_csv(output_dir / "multiverse_summary.csv", index=False)
    for dv in summary.loc[summary["k_max"] >= 3, "dv"]:
        plot_curve(specs, dv, output_dir)
    (output_dir / "multiverse_choices.json").write_text(json.dumps(CHOICES, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
