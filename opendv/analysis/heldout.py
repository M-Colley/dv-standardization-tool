"""Pre-registered tests on the held-out confirmation set (docs/preregistration_heldout_relations.md).

Reads the output of ``construct_relations`` run on the held-out batch, and the E1 run of
``evaluate_llm_mapping`` on the same batch, and decides the hypotheses as pre-registered:

* **H1** (primary): the pooled within-person correlation of ``mental_demand`` and
  ``perceived_safety`` at the curated level is negative. One-sided HKSJ test, alpha 0.05.
  Needs at least two held-out studies; with fewer it is *not testable on this set*.
* **H2** (primary): effort x frustration, effort x mental demand and frustration x mental
  demand each have a positive pooled within-person correlation. One-sided HKSJ test per pair,
  Holm-corrected over the three pairs; H2 holds if all three are significant. A pair needs at
  least two studies.
* **H2b** (secondary, descriptive): the NASA-TLX performance subscale has the lowest mean
  pooled within-person r with the other five subscales.
* **H3** (secondary): on held-out E1 DV columns, header matching first and then the LLM maps
  more columns correctly than the LLM alone. Paired exact McNemar, two-sided, alpha 0.05;
  precision of accepted DV mappings with Wilson 95% CIs.

    python -m opendv.analysis.heldout --relations analyses/heldout/relations \\
        --e1 analyses/heldout/runs/definition --output analyses/heldout/results.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from opendv._paths import REPO_ROOT

LEVEL = "curated"
ASSOCIATION = "within_person"
ALPHA = 0.05
MIN_K = 2
H1_PAIR = ("mental_demand", "perceived_safety")
H2_PAIRS = (("effort", "frustration"), ("effort", "mental_demand"), ("frustration", "mental_demand"))
TLX_SUBSCALES = ("mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration")
HELDOUT_DIR = REPO_ROOT / "analyses" / "heldout"
MANIFEST_PATH = HELDOUT_DIR / "sources_manifest.yaml"
DATA_DIR = REPO_ROOT / "data" / "raw" / "heldout"


def _pair(x: str, y: str) -> tuple[str, str]:
    return (x, y) if x <= y else (y, x)


def _rows(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    a, b = _pair(x, y)
    return frame[(frame["level"] == LEVEL) & (frame["association"] == ASSOCIATION)
                 & (frame["dv_x"] == a) & (frame["dv_y"] == b)]


def one_sided_p(p_two_sided: float, estimate: float, direction: int) -> float:
    """One-sided p of a symmetric two-sided test, for the predicted sign ``direction`` (+1 / -1)."""
    if not np.isfinite(p_two_sided):
        return float("nan")
    return p_two_sided / 2 if np.sign(estimate) == direction else 1 - p_two_sided / 2


def test_pair(pooled: pd.DataFrame, by_study: pd.DataFrame, x: str, y: str, direction: int) -> dict[str, Any]:
    """Pooled within-person r of one pair at the curated level, with its one-sided HKSJ test."""
    studies = _rows(by_study, x, y)
    out: dict[str, Any] = {
        "pair": list(_pair(x, y)),
        "predicted_sign": "positive" if direction > 0 else "negative",
        "k_studies": int(len(studies)),
        "studies": [
            {"study": r.study, "domain": r.domain, "n_participants": int(r.n_participants), "n_rows": int(r.n),
             "r": float(r.r), "ci95": [float(r.ci95_low), float(r.ci95_high)], "p_value": float(r.p_value)}
            for r in studies.sort_values("study").itertuples()
        ],
        "testable": bool(len(studies) >= MIN_K),
    }
    if not out["testable"]:
        return out
    row = _rows(pooled, x, y).iloc[0]
    out.update({
        "pooled_r": float(row["pooled_r"]),
        "hksj_ci95": [float(row["hksj_ci95_low"]), float(row["hksj_ci95_high"])],
        "prediction_interval": [float(row["prediction_interval_low"]), float(row["prediction_interval_high"])],
        "heterogeneity_i2_pct": float(row["heterogeneity_i2_pct"]),
        "tau": float(row["tau"]),
        "n_total": int(row["n_total"]),
        "p_hksj_two_sided": float(row["p_hksj"]),
        "p_one_sided": one_sided_p(float(row["p_hksj"]), float(row["pooled_r"]), direction),
    })
    return out


def h1(pooled: pd.DataFrame, by_study: pd.DataFrame) -> dict[str, Any]:
    result = test_pair(pooled, by_study, *H1_PAIR, direction=-1)
    if not result["testable"]:
        result["decision"] = "not testable on this set"
    else:
        result["decision"] = "supported" if result["p_one_sided"] < ALPHA else "not supported"
    return result


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted, running, m = {}, 0.0, len(ordered)
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def h2(pooled: pd.DataFrame, by_study: pd.DataFrame) -> dict[str, Any]:
    pairs = {f"{x} x {y}": test_pair(pooled, by_study, x, y, direction=1) for x, y in H2_PAIRS}
    # Holm over the three pre-registered pairs; an untestable pair counts as p = 1.
    p = {name: (res["p_one_sided"] if res["testable"] else 1.0) for name, res in pairs.items()}
    for name, adjusted in holm(p).items():
        pairs[name]["p_holm"] = adjusted
        pairs[name]["significant"] = bool(pairs[name]["testable"] and adjusted < ALPHA)
    if not any(res["testable"] for res in pairs.values()):
        decision = "not testable on this set"
    elif all(res["significant"] for res in pairs.values()):
        decision = "supported"
    elif all(res["testable"] for res in pairs.values()):
        decision = "not supported"
    else:
        decision = "not supported (not every pair testable)"
    return {"pairs": pairs, "decision": decision}


def h2b(pooled: pd.DataFrame, by_study: pd.DataFrame) -> dict[str, Any]:
    """Mean pooled within-person r of each TLX subscale with the other five (k = 1: that study's r)."""
    means: dict[str, Any] = {}
    for subscale in TLX_SUBSCALES:
        values, missing = [], []
        for other in TLX_SUBSCALES:
            if other == subscale:
                continue
            pooled_row = _rows(pooled, subscale, other)
            if len(pooled_row):
                values.append(float(pooled_row.iloc[0]["pooled_r"]))
                continue
            studies = _rows(by_study, subscale, other)
            if len(studies) == 1:
                values.append(float(studies.iloc[0]["r"]))
            else:
                missing.append(other)
        means[subscale] = {"mean_r": float(np.mean(values)) if values else None,
                           "n_pairs": len(values), "missing_pairs": missing}
    available = {s: v["mean_r"] for s, v in means.items() if v["mean_r"] is not None}
    lowest = min(available, key=available.get) if available else None
    complete = all(not v["missing_pairs"] for v in means.values())
    return {
        "mean_r_with_other_subscales": means,
        "lowest": lowest,
        "decision": ("met" if lowest == "performance" else "not met") + ("" if complete else " (pairs missing)"),
    }


def table_options_by_source(manifest_path: Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    sources = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["sources"]
    return {s["source_id"]: dict(s.get("table_options") or {}) for s in sources}


def h3(e1_dir: Path, data_dir: Path = DATA_DIR, manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Header matching first, then the LLM, against the LLM alone, on held-out E1 DV columns."""
    from opendv._paths import SCHEMAS_DIR
    from opendv.analysis.confirmatory import _e1_items, _mcnemar, header_match_answers
    from opendv.analysis.evaluate_llm_mapping import summarize
    from opendv.run_batch_standardization import SchemaFamilies

    items = _e1_items(e1_dir)
    header = pd.Series(
        header_match_answers(items, data_dir, table_options_by_source(manifest_path)), dtype=object
    ).reindex(items.index)
    families = SchemaFamilies.load(SCHEMAS_DIR / "standard_dv_mapping.yaml")
    dv_ids = {canonical for canonical, domain in families.canonical_domain_lookup.items() if domain == "dv"}
    pos = items["label_kind"] == "dv"
    llm_correct = (items["final_checked"] == items["gold"])[pos]
    answer = header.combine_first(items["final_checked"])
    combined_correct = (answer == items["gold"])[pos]
    llm_rates = summarize(items.reset_index(), dv_ids)["E1"]["checked"]
    combined_rates = summarize(items.assign(final_checked=answer).reset_index(), dv_ids)["E1"]["checked"]
    test = _mcnemar(combined_correct.to_numpy(), llm_correct.to_numpy())
    return {
        "n_dv_columns": int(pos.sum()),
        "n_non_dv_columns": int((~pos).sum()),
        "llm_only": {"correct": int(llm_correct.sum()),
                     "precision_of_accepted_dv_mappings": llm_rates["precision_of_accepted_dv_mappings"]},
        "header_matching_then_llm": {
            "correct": int(combined_correct.sum()),
            "precision_of_accepted_dv_mappings": combined_rates["precision_of_accepted_dv_mappings"],
            "header_matched_dv_columns": int(header[pos].notna().sum()),
            "header_matched_correct": int((header == items["gold"])[pos].sum()),
            "header_matched_non_dv_columns": int(header[~pos].notna().sum()),
        },
        "mcnemar": test,
        "decision": (
            "supported" if test["new_only"] > test["baseline_only"] and test["p_two_sided"] < ALPHA
            else "not supported"
        ),
    }


def run(relations_dir: Path, e1_dir: Path | None, data_dir: Path = DATA_DIR,
        manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    pooled = pd.read_csv(relations_dir / "relations_pooled.csv")
    by_study = pd.read_csv(relations_dir / "relations_by_study.csv")
    result: dict[str, Any] = {
        "preregistration": "docs/preregistration_heldout_relations.md",
        "level": LEVEL, "association": ASSOCIATION, "alpha": ALPHA, "min_studies": MIN_K,
        "H1": h1(pooled, by_study),
        "H2": h2(pooled, by_study),
        "H2b": h2b(pooled, by_study),
    }
    if e1_dir is not None:
        result["H3"] = h3(e1_dir, data_dir, manifest_path)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--relations", required=True, help="output directory of construct_relations")
    parser.add_argument("--e1", default=None, help="output directory of the E1 run (definition shortlist)")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output", default=str(HELDOUT_DIR / "results.json"))
    args = parser.parse_args(argv)
    result = run(Path(args.relations), Path(args.e1) if args.e1 else None,
                 Path(args.data_dir), Path(args.manifest))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({k: v.get("decision") for k, v in result.items() if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
