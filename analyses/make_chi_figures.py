#!/usr/bin/env python3
"""Publication-quality figures for the OpenDV-HCI cross-study analysis (CHI format).

Reads the artifacts produced by ``run_catalog_meta_analysis.py`` (batch
standardization + per-source provenance) and ``multi_study_analysis.py`` (the
full meta-analysis suite) and renders a cohesive figure set sized for the ACM
CHI two-column template. Every figure is written as both a vector PDF (for
LaTeX inclusion) and a 300-dpi PNG (for preview).

Run:
    python analyses/make_chi_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analyses.multi_study_analysis import _load_dv_measurement_metadata  # noqa: E402

ANALYSIS_DIR = REPO / "analyses" / "output_python_full"
BATCH_DIR = REPO / "data" / "processed" / "catalog_meta_analysis"
OUT = REPO / "analyses" / "figures_chi"
OUT.mkdir(parents=True, exist_ok=True)

# ── ACM column geometry (inches) ─────────────────────────────────────────────
COL1 = 3.33   # single column
COL2 = 7.00   # full width

# ── Style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "axes.grid": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
})

INK = "#222222"
BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#9aa0a6"

# ── Cluster palette (from schemas/thematic_clusters.yaml) ─────────────────────
_clusters = yaml.safe_load((REPO / "schemas" / "thematic_clusters.yaml").read_text(encoding="utf-8"))
CLUSTER_COLOR = {c["id"]: c["color"] for c in _clusters["clusters"]}
CLUSTER_LABEL = {c["id"]: c["label"] for c in _clusters["clusters"]}
CLUSTER_COLOR["unknown"] = GREY
CLUSTER_LABEL["unknown"] = "Unclassified"

DV_META = _load_dv_measurement_metadata()


def dv_cluster(dv: str) -> str:
    return DV_META.get(dv, {}).get("cluster", "") or "unknown"


# ── Friendly labels ──────────────────────────────────────────────────────────
STUDY_LABEL = {
    "roads_chi25": "ROADS",
    "ehmi_for_all_chi26": "eHMI-for-All",
    "ehmi_optimization_chi25": "eHMI-Opt",
    "fact_av": "FACT-AV",
    "longitudinal_usa_germany_ehmi": "Longit. US/DE",
    "3d_display_comp_vehicle": "3D-Display",
    "fourtu_critical_ehmi": "4TU-Critical",
    "osf_cwd6h": "OSF-cwd6h",
    "road_bumps_touch": "Road-Bumps",
}
DV_LABEL = {
    "mental_demand": "Mental Demand", "physical_demand": "Physical Demand",
    "temporal_demand": "Temporal Demand", "effort": "Effort",
    "frustration": "Frustration", "performance": "Performance (TLX)",
    "TLX_SCORE": "NASA-TLX (overall)", "trust_rating": "Trust",
    "understanding_rating": "Understanding", "perceived_safety": "Perceived Safety",
    "acceptance_rating": "Acceptance", "usability": "Usability (SUS)",
    "situational_awareness": "Situational Aware.",
    "AOA_USEFULNESS": "AOA Usefulness", "AOA_SATISFYING": "AOA Satisfying",
    "ueq_hedonic": "UEQ Hedonic", "ueq_pragmatic": "UEQ Pragmatic",
}


def dv_label(dv: str) -> str:
    return DV_LABEL.get(dv, dv)


def study_label(s: str) -> str:
    return STUDY_LABEL.get(s, s)


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ── Load artifacts ───────────────────────────────────────────────────────────
presence = pd.read_csv(ANALYSIS_DIR / "dv_presence_matrix.csv", index_col=0)
overlap = pd.read_csv(ANALYSIS_DIR / "dv_overlap_matrix.csv", index_col=0)
summary = pd.read_csv(ANALYSIS_DIR / "harmonized_dv_summary.csv")
meta = pd.read_csv(ANALYSIS_DIR / "meta_analysis_summary.csv")
freq = pd.read_csv(ANALYSIS_DIR / "dv_frequency.csv")
run_summary = json.loads((BATCH_DIR / "run_summary.json").read_text(encoding="utf-8"))
analysis_summary = json.loads((BATCH_DIR / "analysis" / "analysis_summary.json").read_text(encoding="utf-8"))
results = pd.DataFrame(run_summary["results"])
causal_edges = pd.read_csv(ANALYSIS_DIR / "causal_edges.csv")
causal = json.loads((ANALYSIS_DIR / "analysis_results.json").read_text(encoding="utf-8")).get("causal_discovery")

AVG_JACCARD = analysis_summary["average_pairwise_jaccard"]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Pipeline & corpus attrition (full width, single row funnel)
# ══════════════════════════════════════════════════════════════════════════════
def fig_pipeline():
    n_sources = int(run_summary["total_sources"])
    n_retrieved = int(results[results["status"] != "access_restricted"].shape[0])
    n_files = int(results["processed_files"].sum())
    n_cols = int(results["total_columns"].sum())
    n_mapped = int(results["mapped_columns"].sum())
    n_dv = presence.shape[1]
    n_meta = int(meta.shape[0])
    n_multi = int((meta["k_studies"] >= 3).sum())

    stages = [
        ("Catalog\nsources", n_sources, "datasets", BLUE),
        ("Retrieved", n_retrieved, "1 ACM blocked", BLUE),
        ("Files\nstandardized", n_files, "tables", "#3a7abd"),
        ("Columns\nmapped", n_mapped, f"of {n_cols:,} ({n_mapped / n_cols * 100:.0f}%)", "#5b96c9"),
        ("Canonical\nDVs", n_dv, "harmonized", "#c77b3a"),
        ("Meta-\nanalyzable", n_meta, "DVs, k≥2", RED),
        ("Multi-study\npooled", n_multi, "DV, k≥3", "#7a1320"),
    ]

    fig, ax = plt.subplots(figsize=(COL2, 1.95))
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h, y0 = 0.82, 0.5, 0.30
    for i, (title, val, sub, color) in enumerate(stages):
        x = i + 0.5
        box = FancyBboxPatch(
            (x - box_w / 2, y0), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.06",
            linewidth=1.1, edgecolor=color, facecolor=color + "22", zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y0 + box_h * 0.66, f"{val:,}", ha="center", va="center",
                fontsize=15, fontweight="bold", color=color)
        ax.text(x, y0 + box_h * 0.22, title, ha="center", va="center",
                fontsize=7, color=INK, linespacing=0.95)
        ax.text(x, y0 - 0.10, sub, ha="center", va="center",
                fontsize=6, color="#666666", style="italic")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + box_w / 2 + 0.005, y0 + box_h / 2),
                (x + 1 - box_w / 2 - 0.005, y0 + box_h / 2),
                arrowstyle="-|>", mutation_scale=9, linewidth=1.1,
                color="#999999", zorder=1,
            ))
    ax.text(0.02, 0.96,
            "From open catalog to comparable evidence: the OpenDV-HCI standardization pipeline",
            ha="left", va="top", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.06,
            "Deterministic schema mapping (no LLM). 9 datasets retrieved; 6 contribute canonical "
            "questionnaire DVs after harmonization.",
            ha="left", va="bottom", fontsize=6.3, color="#666666", transform=ax.transAxes)
    save(fig, "fig1_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — DV presence matrix with marginals (the fragmentation figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_presence():
    # order DVs by frequency desc, then cluster; order studies by #DVs desc
    dv_order = (freq.sort_values(["n_studies", "dv"], ascending=[False, True])["dv"].tolist())
    dv_order = [d for d in dv_order if d in presence.columns]
    study_order = presence.sum(axis=1).sort_values(ascending=False).index.tolist()

    P = presence.loc[study_order, dv_order]
    nS, nD = P.shape
    colors = [CLUSTER_COLOR[dv_cluster(d)] for d in dv_order]

    fig = plt.figure(figsize=(COL2, 3.05))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[nD, 3.2], height_ratios=[1.6, nS],
        wspace=0.03, hspace=0.04,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # main presence grid
    for j, d in enumerate(dv_order):
        for i, s in enumerate(study_order):
            if P.iloc[i, j]:
                ax.add_patch(plt.Rectangle((j + 0.08, i + 0.08), 0.84, 0.84,
                                           facecolor=colors[j], edgecolor="none", alpha=0.92))
    ax.set_xlim(0, nD)
    ax.set_ylim(0, nS)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(nD) + 0.5)
    ax.set_xticklabels([dv_label(d) for d in dv_order], rotation=90, fontsize=5.6)
    ax.set_yticks(np.arange(nS) + 0.5)
    ax.set_yticklabels([study_label(s) for s in study_order], fontsize=7)
    ax.set_xticks(np.arange(nD + 1), minor=True)
    ax.set_yticks(np.arange(nS + 1), minor=True)
    ax.grid(which="minor", color="#e6e6e6", linewidth=0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")

    # top marginal: per-DV study count
    counts = P.sum(axis=0).values
    ax_top.bar(np.arange(nD) + 0.5, counts, width=0.84, color=colors, alpha=0.92)
    ax_top.set_ylim(0, counts.max() + 0.5)
    ax_top.set_yticks([0, counts.max()])
    ax_top.tick_params(labelbottom=False, length=0, labelsize=6)
    for sp in ("top", "right", "left"):
        ax_top.spines[sp].set_visible(False)
    ax_top.set_ylabel("# studies", fontsize=6)
    for j, c in enumerate(counts):
        if c >= 2:
            ax_top.text(j + 0.5, c + 0.05, str(int(c)), ha="center", va="bottom", fontsize=5.5)

    # right marginal: per-study DV count
    dv_counts = P.sum(axis=1).values
    ax_right.barh(np.arange(nS) + 0.5, dv_counts, height=0.84, color="#555a60", alpha=0.85)
    ax_right.set_ylim(0, nS)
    ax_right.invert_yaxis()
    ax_right.tick_params(labelleft=False, length=0, labelsize=6)
    for sp in ("top", "right", "bottom"):
        ax_right.spines[sp].set_visible(False)
    ax_right.set_xlabel("# DVs", fontsize=6)
    for i, c in enumerate(dv_counts):
        ax_right.text(c + 0.3, i + 0.5, str(int(c)), ha="left", va="center", fontsize=5.8)

    # cluster legend
    used = []
    for d in dv_order:
        cl = dv_cluster(d)
        if cl not in used:
            used.append(cl)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CLUSTER_COLOR[c], alpha=0.92) for c in used]
    ax_top.legend(handles, [CLUSTER_LABEL[c] for c in used], loc="upper right",
                  bbox_to_anchor=(1.46, 1.25), fontsize=6, handlelength=1.0,
                  handleheight=1.0, labelspacing=0.25, borderaxespad=0)

    fig.suptitle("Dependent-variable coverage across 9 open HCI datasets",
                 x=0.045, y=1.06, ha="left", fontsize=9, fontweight="bold")
    fig.text(0.045, -0.20,
             f"After schema-driven standardization, mean pairwise DV overlap is only "
             f"Jaccard = {AVG_JACCARD:.3f}. Only Mental Demand (NASA-TLX) is shared by "
             f">2 studies;\n3 sensor/log datasets expose no canonical questionnaire DV.",
             ha="left", va="top", fontsize=6.2, color="#666666")
    save(fig, "fig2_presence_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Cross-study DV overlap heatmap (Jaccard)
# ══════════════════════════════════════════════════════════════════════════════
def fig_overlap():
    order = presence.sum(axis=1).sort_values(ascending=False).index.tolist()
    M = overlap.loc[order, order].astype(float).values
    n = len(order)

    fig, ax = plt.subplots(figsize=(COL1 + 0.7, COL1 + 0.55))
    cmap = mpl.colormaps["YlGnBu"].copy()
    cmap.set_bad("#f3f3f3")
    masked = np.ma.masked_invalid(M)
    # blank the diagonal for legibility
    for i in range(n):
        masked[i, i] = np.ma.masked
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=0.7)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([study_label(s) for s in order], rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels([study_label(s) for s in order], fontsize=6.5)
    ax.tick_params(length=0)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#bbbbbb", fontsize=6)
            elif not np.isnan(M[i, j]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}".lstrip("0") if v > 0 else "0",
                        ha="center", va="center", fontsize=5.6,
                        color="white" if v > 0.42 else "#333333")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Jaccard DV overlap", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    ax.set_title("Pairwise DV-vocabulary overlap between datasets", fontsize=8.5, pad=6)
    save(fig, "fig3_overlap_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Forest plot: NASA-TLX Mental Demand (k = 6)
# ══════════════════════════════════════════════════════════════════════════════
def fig_forest(dv="mental_demand", fname="fig4_forest_mental_demand"):
    sub = summary[(summary["dv"] == dv) & (summary["n"] > 1) & (summary["sd"] > 0)].copy()
    sub = sub.sort_values("mean").reset_index(drop=True)
    mrow = meta[meta["dv"] == dv].iloc[0]

    means = sub["mean"].values
    ses = sub["sd"].values / np.sqrt(sub["n"].values)
    lo, hi = means - 1.96 * ses, means + 1.96 * ses
    w = 1.0 / ses ** 2
    wpct = 100 * w / w.sum()
    k = len(sub)

    pm, pse = mrow["random_effects_mean"], mrow["random_effects_se"]
    plo, phi = mrow["ci95_low"], mrow["ci95_high"]
    pi_lo, pi_hi = mrow.get("prediction_interval_low"), mrow.get("prediction_interval_high")

    fig, (axf, axt) = plt.subplots(
        1, 2, figsize=(COL2, 0.55 * k + 1.7),
        gridspec_kw={"width_ratios": [3.1, 2.0], "wspace": 0.02})

    ys = np.arange(k, 0, -1)
    cl_color = CLUSTER_COLOR[dv_cluster(dv)]
    for i, y in enumerate(ys):
        axf.hlines(y, lo[i], hi[i], color=BLUE, linewidth=1.3, zorder=2)
        axf.plot(means[i], y, "s", color=BLUE, markersize=4 + wpct[i] * 0.18, zorder=3)
    # pooled diamond
    dy = 0.0
    axf.fill([plo, pm, phi, pm], [dy, dy + 0.32, dy, dy - 0.32], color=RED, alpha=0.85, zorder=4)
    if pd.notna(pi_lo) and pd.notna(pi_hi):
        axf.hlines(dy, pi_lo, pi_hi, color=RED, linewidth=1.0, linestyle=(0, (4, 2)), alpha=0.6, zorder=3)
    axf.axvline(pm, color=RED, linewidth=0.7, linestyle="--", alpha=0.45)

    axf.set_yticks(list(ys) + [0])
    axf.set_yticklabels([study_label(s) for s in sub["study"]] + ["Random-effects pooled"], fontsize=7)
    axf.get_yticklabels()[-1].set_fontweight("bold")
    axf.set_ylim(-0.7, k + 0.7)
    axf.set_xlabel(f"{dv_label(dv)}  —  group mean (0–20 NASA-TLX scale)", fontsize=7.5)
    for sp in ("top", "right"):
        axf.spines[sp].set_visible(False)
    axf.tick_params(length=2)

    # annotation table
    axt.axis("off")
    axt.set_xlim(0, 1)
    axt.set_ylim(-0.7, k + 0.7)
    cx = {"n": 0.02, "mean": 0.30, "w": 0.55, "ci": 0.74}
    hf = {"family": "monospace", "size": 6.6, "weight": "bold"}
    mf = {"family": "monospace", "size": 6.6}
    axt.text(cx["n"], k + 0.55, "N obs", fontdict=hf)
    axt.text(cx["mean"], k + 0.55, "Mean", fontdict=hf)
    axt.text(cx["w"], k + 0.55, "Wt", fontdict=hf)
    axt.text(cx["ci"], k + 0.55, "95% CI", fontdict=hf)
    axt.axhline(k + 0.28, color="#999", linewidth=0.5)
    for i, y in enumerate(ys):
        axt.text(cx["n"], y, f"{int(sub['n'].iloc[i]):>6,}", fontdict=mf, va="center")
        axt.text(cx["mean"], y, f"{means[i]:>5.2f}", fontdict=mf, va="center")
        axt.text(cx["w"], y, f"{wpct[i]:>4.1f}%", fontdict=mf, va="center")
        axt.text(cx["ci"], y, f"[{lo[i]:.1f}, {hi[i]:.1f}]", fontdict=mf, va="center")
    axt.axhline(0.45, color="#999", linewidth=0.5)
    axt.text(cx["n"], 0, f"{int(sub['n'].sum()):>6,}", fontdict={**mf, "weight": "bold"}, va="center")
    axt.text(cx["mean"], 0, f"{pm:>5.2f}", fontdict={**mf, "weight": "bold"}, va="center")
    axt.text(cx["w"], 0, "100%", fontdict={**mf, "weight": "bold"}, va="center")
    axt.text(cx["ci"], 0, f"[{plo:.1f}, {phi:.1f}]", fontdict={**mf, "weight": "bold"}, va="center")

    i2, tau2, q, qp = mrow["heterogeneity_i2_pct"], mrow["tau2"], mrow["heterogeneity_q"], mrow["q_pvalue"]
    pstr = "p < 0.001" if qp < 0.001 else f"p = {qp:.3f}"
    het = (f"Random-effects (DerSimonian–Laird), k = {k} studies.   "
           f"Heterogeneity: I² = {i2:.1f}%,  τ² = {tau2:.2f},  Q({k-1}) = {q:.0f}, {pstr}")
    fig.text(0.5, -0.015, het, ha="center", va="top", fontsize=6.4, color="#555555",
             family="monospace")
    fig.suptitle("Evidence synthesis unlocked by standardization: NASA-TLX Mental Demand",
                 x=0.045, y=1.02, ha="left", fontsize=9, fontweight="bold")
    fig.text(0.045, 0.965,
             "Pooling is only possible because heterogeneous source labels were mapped to one canonical DV. "
             "Very high I² shows that shared names still mask scale/population differences.",
             ha="left", va="top", fontsize=6.2, color="#777777")
    save(fig, fname)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Cross-study causal structure (LiNGAM DAG)
# ══════════════════════════════════════════════════════════════════════════════
def fig_causal():
    order = causal["causal_order"]
    edges = causal_edges
    # layout: causal order left -> right, alternating vertical offset
    pos = {}
    for i, dv in enumerate(order):
        pos[dv] = (i * 1.0, 0.32 if i % 2 else -0.32)

    fig, ax = plt.subplots(figsize=(COL2, 2.25))
    ax.set_xlim(-0.6, (len(order) - 1) + 0.6)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")

    # edges
    emax = edges["coefficient"].abs().max()
    for _, e in edges.iterrows():
        if e["from"] not in pos or e["to"] not in pos:
            continue
        x0, y0 = pos[e["from"]]
        x1, y1 = pos[e["to"]]
        coef = e["coefficient"]
        col = BLUE if coef > 0 else RED
        lw = 1.0 + 3.2 * abs(coef) / emax
        arr = FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
            linewidth=lw, color=col, alpha=0.85,
            connectionstyle="arc3,rad=0.18", shrinkA=20, shrinkB=20, zorder=1)
        ax.add_patch(arr)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + (0.16 if abs(y1 - y0) < 0.1 else 0.0)
        ax.text(mx, my + 0.10, f"{coef:+.2f}", ha="center", va="center", fontsize=7,
                color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # nodes
    for dv, (x, y) in pos.items():
        c = CLUSTER_COLOR[dv_cluster(dv)]
        ax.scatter([x], [y], s=2300, color=c, alpha=0.9, edgecolor="white", linewidth=1.5, zorder=2)
        ax.text(x, y, dv_label(dv).replace(" ", "\n"), ha="center", va="center",
                fontsize=6.6, color="white", fontweight="bold", zorder=3, linespacing=0.9)

    # causal-order baseline annotation
    for i in range(len(order) - 1):
        ax.annotate("", xy=(i + 0.62, -0.78), xytext=(i + 0.38, -0.78),
                    arrowprops=dict(arrowstyle="-|>", color="#bbbbbb", lw=0.8))
    ax.text((len(order) - 1) / 2, -0.93, "inferred causal order (DirectLiNGAM)",
            ha="center", va="center", fontsize=6, color="#999999", style="italic")

    legend = [
        Line2D([0], [0], color=BLUE, lw=2.4, label="positive effect"),
        Line2D([0], [0], color=RED, lw=2.4, label="negative effect"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=6.5, bbox_to_anchor=(1.0, 1.02))
    ax.set_title("Cross-study causal structure among standardized DVs",
                 x=0.0, ha="left", fontsize=9, loc="left", pad=2)
    fig.text(0.045, 0.02,
             f"DirectLiNGAM on pooled within-study z-scores (complete-case, non-synthetic; "
             f"n = {causal['n_observations']:,} obs). Edge weights are structural coefficients.",
             ha="left", va="bottom", fontsize=6.2, color="#777777")
    save(fig, "fig5_causal_dag")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Schema mapping coverage by source (mapped vs unknown columns)
# ══════════════════════════════════════════════════════════════════════════════
def fig_coverage():
    r = results[results["status"] != "access_restricted"].copy()
    r["mapped_ratio"] = r["mapped_columns"] / r["total_columns"].replace(0, np.nan)
    r = r.sort_values("mapped_ratio", ascending=True)
    y = np.arange(len(r))

    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    mapped = r["mapped_columns"].values
    unknown = (r["total_columns"] - r["mapped_columns"]).values
    ax.barh(y, mapped, color=BLUE, alpha=0.9, label="mapped to schema")
    ax.barh(y, unknown, left=mapped, color="#e0e0e0", label="unmapped / unknown")
    ax.set_yticks(y)
    ax.set_yticklabels([study_label(s) for s in r["source_id"]], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1, r["total_columns"].max() * 1.6)
    ax.set_xlabel("Columns encountered (log scale)", fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for i, (_, row) in enumerate(r.iterrows()):
        tot = row["total_columns"]
        ratio = row["mapped_columns"] / tot if tot else 0
        ax.text(tot * 1.07, i, f"{ratio*100:.0f}%  ({int(row['mapped_columns']):,}/{int(tot):,})",
                va="center", ha="left", fontsize=6, color="#444")
    ax.legend(loc="lower right", fontsize=6.5, ncol=1)
    ax.set_title("Schema coverage by dataset (all mapping families)", fontsize=8.5)
    fig.text(0.045, -0.03,
             "Coverage tracks data type: questionnaire/telemetry exports map well; raw simulator-log "
             "and touch-sensor dumps expose mostly study-specific columns.",
             ha="left", va="top", fontsize=6.2, color="#777777")
    save(fig, "fig6_mapping_coverage")


# ══════════════════════════════════════════════════════════════════════════════
# TEASER — combined 2x2 (presence, overlap, forest, causal) for page 1
# ══════════════════════════════════════════════════════════════════════════════
def fig_teaser():
    # Reuse standalone renders by stitching the saved PNGs into one figure.
    import matplotlib.image as mpimg
    panels = [
        ("fig2_presence_matrix.png", "(a) DV coverage across 9 datasets"),
        ("fig3_overlap_heatmap.png", "(b) Pairwise overlap (Jaccard)"),
        ("fig4_forest_mental_demand.png", "(c) Meta-analysis: Mental Demand"),
        ("fig5_causal_dag.png", "(d) Cross-study causal structure"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.5))
    for ax, (png, cap) in zip(axes.ravel(), panels):
        p = OUT / png
        if p.exists():
            ax.imshow(mpimg.imread(p), aspect="equal")
        ax.axis("off")
        ax.set_title(cap, fontsize=8, fontweight="bold", loc="left", pad=1)
    fig.suptitle("OpenDV-HCI: standardizing dependent variables turns fragmented open "
                 "datasets into comparable, synthesizable evidence",
                 y=0.995, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.03, hspace=0.10)
    save(fig, "fig0_teaser")


if __name__ == "__main__":
    print(f"Output -> {OUT}")
    fig_pipeline()
    fig_presence()
    fig_overlap()
    fig_forest()
    fig_causal()
    fig_coverage()
    fig_teaser()
    print("Done.")
