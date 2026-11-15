"""What the OpenDV mapping adds, measured on the whole corpus.

Reads one batch run of the corpus (``meta_view.csv``: one row per column of every standardized
table), the corpus analysis (``opendv analyze``), the cross-study relations and
``corpus_studies.csv``, and writes to ``--output-dir``:

DV columns are the columns mapped to a schema DV or to an item of ``schemas/instruments.yaml``.

- ``harmonization_by_dv.csv``: per canonical DV, the studies, columns and distinct raw labels it
  unites, and whether the study mappings or the shared schema's aliases matched them.
- ``poolability.csv``: the same DV columns identified four ways (raw label, normalized raw label,
  the shared schema's aliases alone, the full OpenDV mapping), and for each how many measures
  recur in at least 2, 3, 5 and 10 studies, how many study pairs share a measure and the mean
  pairwise Jaccard overlap. For all studies and for the SIGCHI subset.
- ``schema_growth.csv``: per study, in the order its mapping was committed, the schema DVs it uses
  and how many of them the schema already had (from the git history of the schema).
- ``label_ambiguity.csv``: raw names used by at least two studies and the DVs they stand for.
- ``scale_audit.csv``: per study x DV, the value range against the DV's canonical range and how the
  analysis pooled it (in range, rescaled, not pooled, or to review for a narrower scale).
- ``reference_values.csv``: pooled random-effects means of every DV reported by at least five
  studies, with the number of studies in the 39-study development analysis for comparison.
- ``value_summary.json``: the headline numbers; ``value_figure.pdf/png``: four panels.

    python -m analyses.opendv_corpus.value --batch-dir data/processed/corpus88
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCHEMA = "schemas/standard_dv_mapping.yaml"
STRATEGIES = ("raw_label", "normalized_label", "schema_aliases", "opendv")
STRATEGY_LABEL = {
    "raw_label": "Raw column name",
    "normalized_label": "Normalized name",
    "schema_aliases": "Shared schema aliases",
    "opendv": "OpenDV mapping",
}
MIN_STUDIES = (2, 3, 5, 10)


def normalize_label(label: object) -> str:
    """Lower case, letters and digits only: 'Mental Demand' and 'mental_demand' become one name."""
    return re.sub(r"[^0-9a-z]+", "", str(label).lower())


def base_dv(dv: str) -> str:
    return re.sub(r"__dup_\d+$", "", str(dv))


def instrument_items() -> set[str]:
    """Item ids of ``schemas/instruments.yaml`` that are not DVs themselves (``ueq_s_1``, ``tia_k09``,
    ...): the batch files their columns as custom, and the analysis scores the instrument from them."""
    doc = yaml.safe_load((REPO / "schemas" / "instruments.yaml").read_text(encoding="utf-8"))
    return {str(item["id"]) for inst in doc.get("instruments", {}).values()
            for item in inst.get("items") or [] if isinstance(item, dict) and item.get("id")}


def dv_columns(meta_view: pd.DataFrame) -> pd.DataFrame:
    """The columns the mapping assigns to a schema DV or to an instrument item, with their
    identifier under each strategy."""
    is_item = meta_view["canonical_dv"].map(base_dv).isin(instrument_items())
    dv = meta_view[(meta_view["mapping_domain"] == "dv") | is_item].copy()
    dv["opendv"] = dv["canonical_dv"].map(base_dv)
    dv["raw_label"] = dv["original_alias"].astype(str)
    dv["normalized_label"] = dv["original_alias"].map(normalize_label)
    shared = dv["mapping_source"].astype(str).str.replace("\\", "/").str.endswith(SCHEMA)
    dv["schema_aliases"] = np.where(shared, dv["opendv"], "raw:" + dv["normalized_label"])
    dv["matched_by"] = np.where(shared, "shared_schema", "study_mapping")
    return dv


def harmonization_by_dv(dv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, grp in dv.groupby("opendv"):
        labels = grp["normalized_label"].value_counts()
        raw = grp.drop_duplicates("normalized_label").set_index("normalized_label")["raw_label"]
        rows.append({
            "dv": name,
            "studies": grp["source_id"].nunique(),
            "columns": len(grp),
            "distinct_raw_labels": int(grp["normalized_label"].nunique()),
            "share_matched_by_study_mappings": round(float((grp["matched_by"] == "study_mapping").mean()), 3),
            "example_labels": "; ".join(raw[lab] for lab in labels.index[:6]),
        })
    return pd.DataFrame(rows).sort_values(["studies", "columns"], ascending=False)


def _overlap(study_sets: dict[str, set[str]]) -> tuple[int, int, float]:
    pairs = list(itertools.combinations(sorted(study_sets), 2))
    sharing = 0
    jaccard = []
    for a, b in pairs:
        inter = len(study_sets[a] & study_sets[b])
        union = len(study_sets[a] | study_sets[b])
        sharing += inter > 0
        jaccard.append(inter / union if union else 0.0)
    return len(pairs), sharing, float(np.mean(jaccard)) if jaccard else float("nan")


def poolability(dv: pd.DataFrame, studies: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all": set(dv["source_id"]),
        "sigchi": set(studies.loc[studies["venue_group"] == "sigchi", "source_id"]) & set(dv["source_id"]),
        "acm": set(studies.loc[studies["venue_group"].isin(["sigchi", "acm_other"]), "source_id"])
        & set(dv["source_id"]),
    }
    rows = []
    for subset, ids in subsets.items():
        part = dv[dv["source_id"].isin(ids)]
        for strategy in STRATEGIES:
            per_study = part.groupby("source_id")[strategy].agg(set).to_dict()
            counts = part.groupby(strategy)["source_id"].nunique()
            n_pairs, sharing, jaccard = _overlap(per_study)
            row = {
                "subset": subset, "strategy": strategy, "studies": len(per_study),
                "distinct_measures": int(counts.size),
                "study_pairs": n_pairs, "pairs_sharing_a_measure": sharing,
                "share_of_pairs_sharing": round(sharing / n_pairs, 3) if n_pairs else float("nan"),
                "mean_jaccard": round(jaccard, 4),
            }
            for k in MIN_STUDIES:
                row[f"measures_in_{k}plus_studies"] = int((counts >= k).sum())
            rows.append(row)
    return pd.DataFrame(rows)


# ── Schema growth from the git history ───────────────────────────────────────
def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True,
                          encoding="utf-8").stdout


def _utc(date: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(date).tz_convert("UTC") if date else None


def _schema_versions() -> list[tuple[str, pd.Timestamp, set[str]]]:
    """(commit, date, DV ids) for every commit that changed the schema, oldest first."""
    versions = []
    for line in _git("log", "--reverse", "--format=%H %ad", "--date=iso-strict", "--", SCHEMA).splitlines():
        commit, date = line.split(" ", 1)
        try:
            doc = yaml.safe_load(_git("show", f"{commit}:{SCHEMA}"))
        except (subprocess.CalledProcessError, yaml.YAMLError):
            continue
        entries = doc.get("dvs") if isinstance(doc, dict) else None
        if entries is None and isinstance(doc, dict):
            entries = doc.get("dependent_variables") or []
        ids = set()
        for entry in entries or []:
            if isinstance(entry, dict):
                ids.add(str(entry.get("id") or entry.get("name")))
        versions.append((commit, _utc(date), ids))
    return versions


def _first_commit_date(path: str) -> str | None:
    out = _git("log", "--follow", "--diff-filter=A", "--format=%ad", "--date=iso-strict", "--", path).split()
    return out[-1] if out else None


def _manifest_date(source_id: str, manifest: str) -> str | None:
    out = _git("log", "--reverse", "--format=%ad", "--date=iso-strict", "-S", source_id, "--", manifest).split()
    return out[0] if out else None


def schema_growth(dv: pd.DataFrame, studies: pd.DataFrame) -> pd.DataFrame:
    versions = _schema_versions()
    # schema DVs only: instrument items (ueq_s_1, tia_k09, ...) belong to schemas/instruments.yaml
    current = versions[-1][2] if versions else set()
    used = dv[dv["opendv"].isin(current)].groupby("source_id")["opendv"].agg(lambda s: set(s))
    rows = []
    for _, st in studies.iterrows():
        # a corrected corpus copy dates from its frozen original (the corrections add no DV)
        path = next((st[c] for c in ("mapping_corrected_from", "mapping_path")
                     if c in st and isinstance(st[c], str)), None)
        date = _first_commit_date(path) if path and not path.startswith("data/") else None
        if date is None:
            date = _manifest_date(st["source_id"], "sources_manifest_no4tu.yaml")
        date = _utc(date)
        dvs = used.get(st["source_id"], set())
        if date is None:
            raise SystemExit(f"{st['source_id']}: no commit date for its mapping; commit it first")
        before = set()
        for _commit, vdate, ids in versions:
            if vdate < date:
                before = ids
        rows.append({
            "source_id": st["source_id"], "study_set": st["study_set"], "venue_group": st["venue_group"],
            "mapping_committed": date.isoformat() if date is not None else None, "dvs_used": len(dvs),
            "dvs_already_in_schema": len(dvs & before), "dvs_new_to_schema": len(dvs - before),
            "new_dvs": "; ".join(sorted(dvs - before)),
        })
    out = pd.DataFrame(rows).sort_values("mapping_committed").reset_index(drop=True)
    seen: set[str] = set()
    cumulative = []
    for sid in out["source_id"]:
        seen |= used.get(sid, set())
        cumulative.append(len(seen))
    out["cumulative_dvs_used"] = cumulative
    out["schema_dvs_at_commit"] = [
        len(next((ids for _c, d, ids in reversed(versions) if d < _utc(date)), set()))
        for date in out["mapping_committed"]
    ]
    return out


# ── Pooled reference values ──────────────────────────────────────────────────
def reference_values(analysis_dir: Path, dev_analysis_dir: Path | None, min_k: int = 5) -> pd.DataFrame:
    meta = pd.read_csv(analysis_dir / "meta_analysis_summary.csv")
    kcol = next(c for c in ("k_studies", "k", "n_studies") if c in meta.columns)
    keep = [c for c in ("dv", kcol, "random_effects_mean", "hksj_ci95_low", "hksj_ci95_high",
                        "prediction_interval_low", "prediction_interval_high", "heterogeneity_i2_pct",
                        "tau2", "unit", "primary_unit") if c in meta.columns]
    out = meta.loc[meta[kcol] >= min_k, keep].rename(columns={kcol: "k"})
    if dev_analysis_dir is not None:
        dev_file = dev_analysis_dir / "meta_analysis_summary_llm_excluded.csv"
        if dev_file.exists():
            dev = pd.read_csv(dev_file)
            dkcol = next(c for c in ("k_studies", "k", "n_studies") if c in dev.columns)
            out = out.merge(dev[["dv", dkcol]].rename(columns={dkcol: "k_development"}), on="dv", how="left")
            out["k_development"] = out["k_development"].fillna(0).astype(int)
    return out.sort_values("k", ascending=False).reset_index(drop=True)


# ── Scale audit ──────────────────────────────────────────────────────────────
def scale_audit(dv: pd.DataFrame, analysis_dir: Path) -> pd.DataFrame:
    """Per study x DV with a canonical range: the standardized value range and how the analysis
    pooled it. ``status`` is

    - ``in_range``: the values lie in the DV's range;
    - ``rescaled``: they lie on a common scale outside it, and the analysis rescaled them;
    - ``not_pooled``: they fit no common scale and are left out (scale_note);
    - ``review_narrower_scale``: they stay in the bottom half of the range (n >= 10, a range of at
      least 4 points). The analysis cannot tell a narrower recording scale (1-10 inside 0-20) from
      low values, so such pairs need a look at the study's documentation.
    """
    schema = yaml.safe_load((REPO / SCHEMA).read_text(encoding="utf-8"))
    ranges = {d["id"]: tuple(d["measurement"]["canonical_range"]) for d in schema["dvs"]
              if (d.get("measurement") or {}).get("canonical_range")}
    part = dv[dv["opendv"].isin(ranges) & (pd.to_numeric(dv["n"], errors="coerce") > 0)]
    g = part.groupby(["source_id", "opendv"]).agg(
        n=("n", "sum"), min=("min", "min"), max=("max", "max")).reset_index()
    g["range_lo"] = g["opendv"].map(lambda d: ranges[d][0])
    g["range_hi"] = g["opendv"].map(lambda d: ranges[d][1])
    notes = {}
    harm_file = analysis_dir / "harmonized_dv_summary.csv"
    if harm_file.exists():
        harm = pd.read_csv(harm_file)
        notes = {(r.study, r.dv): r.scale_note for r in harm.itertuples() if isinstance(r.scale_note, str)}
    g["scale_note"] = [notes.get((sid, d), "") for sid, d in zip(g["source_id"], g["opendv"])]
    span = g["range_hi"] - g["range_lo"]
    narrow = ((g["max"] <= g["range_lo"] + 0.5 * span) & (g["min"] >= g["range_lo"] - 0.15 * span)
              & (g["n"] >= 10) & (span >= 4))
    g["status"] = np.select(
        [g["scale_note"].str.startswith("not pooled"), g["scale_note"].str.startswith("rescaled"), narrow],
        ["not_pooled", "rescaled", "review_narrower_scale"], default="in_range")
    return g.rename(columns={"source_id": "study", "opendv": "dv"}).sort_values(["status", "study", "dv"])


# ── Raw-name ambiguity ───────────────────────────────────────────────────────
def label_ambiguity(dv: pd.DataFrame) -> pd.DataFrame:
    """Raw names (normalized) used by at least two studies, with the DVs the mapping assigns them.
    A name with more than one DV would pool different measures if studies were matched by name."""
    g = dv.groupby("normalized_label").agg(
        studies=("source_id", "nunique"), dvs=("opendv", lambda s: sorted(set(s))),
        example=("raw_label", "first"))
    g = g[g["studies"] >= 2].copy()
    g["n_dvs"] = g["dvs"].map(len)
    g["dvs"] = g["dvs"].map("; ".join)
    return g.reset_index().sort_values(["n_dvs", "studies"], ascending=False)


# ── Figure ───────────────────────────────────────────────────────────────────
_ITEM = re.compile(r"^(SUS|AOA)\d+$|_item_\d+$|^ueq_s_\d+$|^tia_k\d+$|^ssq_(?!total|nausea$|oculomotor|disorientation)")


def _is_item(dv: str) -> bool:
    """Questionnaire items (SUS1, AOA3, perceived_safety_item_02, ...), left out of the figure's DV panels."""
    return bool(_ITEM.search(str(dv)))


LEVEL_NAMES = {"names": "Raw column name", "automatic": "Shared schema aliases", "curated": "OpenDV mapping"}
LADDER_METRICS = (("pairs_pooled_k3", "measure pairs,\n>= 3 studies"),
                  ("cross_construct_pairs_k3", "cross-construct\npairs, >= 3 studies"),
                  ("findings", "findings"))


def figure(pool: pd.DataFrame, harm: pd.DataFrame, ref: pd.DataFrame, ladder: pd.DataFrame | None,
           out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"
    # one hue, light -> dark: more harmonization is darker
    colors = {"raw_label": "#cbd5e1", "normalized_label": "#94a3b8",
              "schema_aliases": "#60a5fa", "opendv": "#1d4ed8"}
    level_color = {"names": colors["raw_label"], "automatic": colors["schema_aliases"],
                   "curated": colors["opendv"]}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

    def tidy(ax, grid_axis):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        getattr(ax, f"{grid_axis}axis").grid(True, color=grid, linewidth=0.6)
        ax.set_axisbelow(True)

    # (a) measures shared by >= k studies under each identifier
    ax = axes[0, 0]
    allp = pool[pool["subset"] == "all"].set_index("strategy")
    ks = [3, 5, 10]  # >= 2 is left out: raw-name matches there are often coincidences ("A1", "Q1")
    x = np.arange(len(ks))
    width = 0.2
    for i, strategy in enumerate(STRATEGIES):
        vals = [int(allp.loc[strategy, f"measures_in_{k}plus_studies"]) for k in ks]
        bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.9, color=colors[strategy],
                      label=STRATEGY_LABEL[strategy])
        if strategy == "opendv":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", va="bottom",
                        fontsize=6, color=ink)
    ax.set_xticks(x, [f"\u2265 {k}" for k in ks])
    ax.set_xlabel("studies reporting the measure")
    ax.set_ylabel("measures")
    ax.set_title("(a) Measures shared across studies", loc="left", fontsize=8, color=ink)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    tidy(ax, "y")

    # (b) between-person relations pooled over >= 3 studies at each harmonization level
    ax = axes[0, 1]
    if ladder is not None and not ladder.empty:
        levels = [lv for lv in ("names", "automatic", "curated") if lv in set(ladder["level"])]
        between = ladder[ladder["association"] == "between_person"].set_index("level")
        x = np.arange(len(LADDER_METRICS))
        width = 0.26
        for i, lv in enumerate(levels):
            vals = [int(between.loc[lv, m]) for m, _ in LADDER_METRICS]
            bars = ax.bar(x + (i - (len(levels) - 1) / 2) * width, vals, width * 0.9,
                          color=level_color[lv], label=LEVEL_NAMES[lv])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", va="bottom",
                        fontsize=5.6, color=ink if lv == "curated" else muted)
        ax.set_xticks(x, [label.replace(">=", "\u2265") for _, label in LADDER_METRICS])
        ax.legend(frameon=False, fontsize=6, loc="upper right")
        ax.set_ylabel("between-person correlations")
    ax.set_title("(b) Cross-study relations that can be pooled", loc="left", fontsize=8, color=ink)
    tidy(ax, "y")

    # (c) distinct raw names behind the most widely reported DVs (constructs, not questionnaire items)
    ax = axes[1, 0]
    top = harm[~harm["dv"].map(_is_item)].head(12).iloc[::-1]
    y = np.arange(len(top))
    ax.barh(y, top["distinct_raw_labels"], color=colors["opendv"], height=0.62)
    for yi, (labels, studies) in enumerate(zip(top["distinct_raw_labels"], top["studies"])):
        ax.text(labels + 0.6, yi, f"{studies} studies", va="center", fontsize=5.6, color=muted)
    ax.set_yticks(y, top["dv"], fontsize=6)
    ax.set_xlabel("distinct raw column names (normalized)")
    ax.set_xlim(0, top["distinct_raw_labels"].max() * 1.35)
    ax.set_title("(c) Names one DV goes by", loc="left", fontsize=8, color=ink)
    tidy(ax, "x")

    # (d) studies per pooled DV: development set vs corpus
    ax = axes[1, 1]
    if "k_development" in ref.columns:
        top = ref[~ref["dv"].map(_is_item)].head(12).iloc[::-1]
        y = np.arange(len(top))
        ax.barh(y, top["k"], color=colors["opendv"], height=0.62, label="88-study corpus")
        ax.barh(y, top["k_development"], color=colors["schema_aliases"], height=0.62,
                label="39 development studies")
        ax.set_yticks(y, top["dv"], fontsize=6)
        ax.set_xlabel("studies pooled (k)")
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_title("(d) Studies per pooled DV", loc="left", fontsize=8, color=ink)
    tidy(ax, "x")
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"value_figure.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=HERE / "analysis")
    parser.add_argument("--dev-analysis-dir", type=Path, default=REPO / "analyses" / "output_python_full")
    parser.add_argument("--relations-dir", type=Path, default=HERE / "relations")
    parser.add_argument("--studies", type=Path, default=HERE / "corpus_studies.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "value")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta_view = pd.read_csv(args.batch_dir / "meta_view.csv", low_memory=False)
    studies = pd.read_csv(args.studies)
    dv = dv_columns(meta_view)
    harm = harmonization_by_dv(dv)
    pool = poolability(dv, studies)
    growth = schema_growth(dv, studies)
    ref = reference_values(args.analysis_dir, args.dev_analysis_dir)
    ambiguity = label_ambiguity(dv)
    audit = scale_audit(dv, args.analysis_dir)
    ladder_file = args.relations_dir / "harmonization_ladder.csv"
    ladder = pd.read_csv(ladder_file) if ladder_file.exists() else None
    harm.to_csv(args.output_dir / "harmonization_by_dv.csv", index=False)
    pool.to_csv(args.output_dir / "poolability.csv", index=False)
    growth.to_csv(args.output_dir / "schema_growth.csv", index=False)
    ref.to_csv(args.output_dir / "reference_values.csv", index=False)
    ambiguity.to_csv(args.output_dir / "label_ambiguity.csv", index=False)
    audit.to_csv(args.output_dir / "scale_audit.csv", index=False)
    figure(pool, harm, ref, ladder, args.output_dir)

    allp = pool[pool["subset"] == "all"].set_index("strategy")
    summary = {
        "studies": int(studies.shape[0]),
        "studies_with_dv_columns": int(dv["source_id"].nunique()),
        "studies_by_venue_group": studies["venue_group"].value_counts().to_dict(),
        "studies_by_set": studies["study_set"].value_counts().to_dict(),
        "columns_read": int(len(meta_view)),
        "dv_columns": int(len(dv)),
        "canonical_dvs_used": int(dv["opendv"].nunique()),
        "distinct_raw_labels_of_dv_columns": int(dv["normalized_label"].nunique()),
        "share_dv_columns_matched_by_study_mappings": round(float((dv["matched_by"] == "study_mapping").mean()), 3),
        "measures_in_3plus_studies": {s: int(allp.loc[s, "measures_in_3plus_studies"]) for s in STRATEGIES},
        "share_of_study_pairs_sharing_a_measure": {s: float(allp.loc[s, "share_of_pairs_sharing"]) for s in STRATEGIES},
        "heldout_studies_needing_new_schema_dvs": int(
            (growth.loc[growth["study_set"] != "development", "dvs_new_to_schema"] > 0).sum()),
        "pooled_dvs_k5plus": int(len(ref)),
        "raw_names_in_2plus_studies": int(len(ambiguity)),
        "raw_names_in_2plus_studies_meaning_different_dvs": int((ambiguity["n_dvs"] > 1).sum()),
        "scale_audit": audit["status"].value_counts().to_dict(),
    }
    if ladder is not None:
        summary["harmonization_ladder"] = ladder.to_dict(orient="records")
    rel = args.relations_dir / "relations_summary.json"
    if rel.exists():
        summary["relations_summary"] = json.loads(rel.read_text(encoding="utf-8"))
    (args.output_dir / "value_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "relations_summary"}, indent=2))


if __name__ == "__main__":
    main()
