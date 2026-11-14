"""Pre-registered decisions of the second held-out test (docs/preregistration_heldout_relations_2.md).

The second held-out test asks whether the cross-study relations found so far recur on studies
that played no part in finding them or in the first held-out test. Its replication targets are
every curated-level *finding* of the development analysis (``analyses/output_relations``, 39
studies) and of the first held-out test (``analyses/heldout/relations``): at least three studies,
a pair not scored from shared items, an HKSJ 95% interval that excludes 0 and BH q < .05. Each
target carries the sign it was found with.

Hypotheses, all at the curated level:

* **H1** (primary) and **H2** (primary) and **H2b** (secondary, descriptive) are those of the
  first held-out test, decided by the same code (``opendv.analysis.heldout``): mental demand x
  perceived safety is negative within persons; the three core NASA-TLX pairs are positive within
  persons (Holm over the three, all must hold); performance is the weakest-linked TLX subscale.
* **H4** (secondary): every other target. A target is *testable* when at least two held-out
  studies estimate it; it *replicates* when its one-sided HKSJ test in the predicted direction
  survives Benjamini-Hochberg at q < .05 over all testable H4 targets. The number of testable and
  replicated targets is the outcome; no threshold on that number is claimed.
* **H3** (secondary): header matching before the LLM, decided as in the first test.

Two subcommands::

    python -m opendv.analysis.heldout_replication targets \\
        --development analyses/output_relations --first-test analyses/heldout/relations \\
        --output analyses/heldout_2/targets.csv
    python -m opendv.analysis.heldout_replication decide --relations analyses/heldout_2/relations \\
        --targets analyses/heldout_2/targets.csv --e1 analyses/heldout_2/runs/definition \\
        --manifest analyses/heldout_2/sources_manifest.yaml --data-dir data/raw/heldout_2 \\
        --output analyses/heldout_2/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from opendv._paths import REPO_ROOT
from opendv.analysis import heldout as first

LEVEL = first.LEVEL
ALPHA = first.ALPHA
MIN_K = first.MIN_K
H1_PAIR = first.H1_PAIR
H2_PAIRS = first.H2_PAIRS
OUT_DIR = REPO_ROOT / "analyses" / "heldout_2"
PREREGISTRATION = "docs/preregistration_heldout_relations_2.md"
TARGET_COLUMNS = [
    "target", "family", "association", "dv_x", "dv_y", "pair_type", "predicted_sign", "found_in",
    "development_k", "development_r", "development_q", "first_test_k", "first_test_r", "first_test_q",
]


def _pair(x: str, y: str) -> tuple[str, str]:
    return first._pair(x, y)


def _findings(pooled: pd.DataFrame) -> pd.DataFrame:
    found = pooled[(pooled["level"] == LEVEL) & (pooled["finding"].astype(str) == "True")]
    return found[["association", "dv_x", "dv_y", "pair_type", "k_studies", "pooled_r", "q_bh"]]


def family_of(association: str, x: str, y: str) -> str:
    pair = _pair(x, y)
    if association == "within_person" and pair == _pair(*H1_PAIR):
        return "H1"
    if association == "within_person" and pair in {_pair(a, b) for a, b in H2_PAIRS}:
        return "H2"
    return "H4"


def build_targets(development: pd.DataFrame, first_test: pd.DataFrame) -> pd.DataFrame:
    """Replication targets: the union of the curated findings of both analyses, with their sign.

    A pair found in both analyses with opposite signs would have no prediction; it is refused.
    """
    keys = ["association", "dv_x", "dv_y"]
    dev = _findings(development).rename(columns={"k_studies": "development_k", "pooled_r": "development_r",
                                                 "q_bh": "development_q", "pair_type": "pair_type_dev"})
    t1 = _findings(first_test).rename(columns={"k_studies": "first_test_k", "pooled_r": "first_test_r",
                                               "q_bh": "first_test_q", "pair_type": "pair_type_t1"})
    merged = dev.merge(t1, on=keys, how="outer")
    signs = pd.DataFrame({"dev": np.sign(merged["development_r"]), "t1": np.sign(merged["first_test_r"])})
    conflict = signs.notna().all(axis=1) & (signs["dev"] != signs["t1"])
    if conflict.any():
        raise ValueError(f"findings with opposite signs: {merged.loc[conflict, keys].to_dict('records')}")
    merged["predicted_sign"] = signs["dev"].fillna(signs["t1"]).astype(int)
    merged["pair_type"] = merged["pair_type_dev"].fillna(merged["pair_type_t1"])
    merged["found_in"] = np.select(
        [merged["development_r"].notna() & merged["first_test_r"].notna(), merged["development_r"].notna()],
        ["both", "development"], default="first test")
    pairs = list(zip(merged["association"], merged["dv_x"], merged["dv_y"], strict=True))
    merged["family"] = [family_of(a, x, y) for a, x, y in pairs]
    merged = merged.sort_values(["family", "association", "dv_x", "dv_y"]).reset_index(drop=True)
    merged["target"] = [f"{a.split('_')[0]}:{x}~{y}" for a, x, y in
                        zip(merged["association"], merged["dv_x"], merged["dv_y"], strict=True)]
    return merged[TARGET_COLUMNS]


def _rows(frame: pd.DataFrame, association: str, x: str, y: str) -> pd.DataFrame:
    a, b = _pair(x, y)
    return frame[(frame["level"] == LEVEL) & (frame["association"] == association)
                 & (frame["dv_x"] == a) & (frame["dv_y"] == b)]


def evaluate_target(pooled: pd.DataFrame, by_study: pd.DataFrame, association: str, x: str, y: str,
                direction: int) -> dict[str, Any]:
    """Pooled r of one pair at the curated level and its one-sided HKSJ test.

    As ``heldout.test_pair``, with the association as a parameter.
    """
    studies = _rows(by_study, association, x, y)
    out: dict[str, Any] = {
        "association": association,
        "pair": list(_pair(x, y)),
        "predicted_sign": "positive" if direction > 0 else "negative",
        "k_studies": int(len(studies)),
        "studies": [
            {"study": r.study, "domain": r.domain, "n_participants": int(r.n_participants), "n_rows": int(r.n),
             "r": float(r.r), "ci95": [float(r.ci95_low), float(r.ci95_high)], "p_value": float(r.p_value)}
            for r in studies.sort_values("study").itertuples()
        ],
        "k_predicted_sign": int((np.sign(studies["r"]) == direction).sum()),
        "testable": bool(len(studies) >= MIN_K),
    }
    if not out["testable"]:
        return out
    row = _rows(pooled, association, x, y).iloc[0]
    out.update({
        "pooled_r": float(row["pooled_r"]),
        "hksj_ci95": [float(row["hksj_ci95_low"]), float(row["hksj_ci95_high"])],
        "prediction_interval": [float(row["prediction_interval_low"]), float(row["prediction_interval_high"])],
        "heterogeneity_i2_pct": float(row["heterogeneity_i2_pct"]),
        "tau": float(row["tau"]),
        "n_total": int(row["n_total"]),
        "p_hksj_two_sided": float(row["p_hksj"]),
        "p_one_sided": first.one_sided_p(float(row["p_hksj"]), float(row["pooled_r"]), direction),
    })
    return out


def h4(pooled: pd.DataFrame, by_study: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
    """Every H4 target: testable with k >= 2, replicated when BH-adjusted one-sided p < alpha."""
    results = {}
    for t in targets[targets["family"] == "H4"].itertuples():
        result = evaluate_target(pooled, by_study, t.association, t.dv_x, t.dv_y, int(t.predicted_sign))
        result.update({"found_in": t.found_in, "pair_type": t.pair_type})
        results[t.target] = result
    testable = [name for name, res in results.items() if res["testable"]]
    p = np.array([results[name]["p_one_sided"] for name in testable], dtype=float)
    q = false_discovery_control(np.nan_to_num(p, nan=1.0), method="bh") if len(p) else np.array([])
    for name, adjusted in zip(testable, q, strict=True):
        res = results[name]
        res["q_bh"] = float(adjusted)
        res["replicated"] = bool(adjusted < ALPHA)
        res["same_sign"] = bool(np.sign(res["pooled_r"]) == (1 if res["predicted_sign"] == "positive" else -1))
    for res in results.values():
        res.setdefault("replicated", False)
    by_family = {}
    for kind in ("within_person", "between_person"):
        names = [n for n in testable if results[n]["association"] == kind]
        by_family[kind] = {
            "targets": int(sum(res["association"] == kind for res in results.values())),
            "testable": len(names),
            "replicated": int(sum(results[n]["replicated"] for n in names)),
            "same_sign": int(sum(results[n]["same_sign"] for n in names)),
        }
    return {
        "targets": len(results),
        "testable": len(testable),
        "replicated": int(sum(res["replicated"] for res in results.values())),
        "same_sign": int(sum(results[n]["same_sign"] for n in testable)),
        "by_association": by_family,
        "results": results,
        "decision": (f"{sum(res['replicated'] for res in results.values())} of {len(testable)} testable "
                     f"targets replicated ({len(results)} targets)") if testable else "not testable on this set",
    }


def new_findings(pooled: pd.DataFrame, targets: pd.DataFrame) -> list[dict[str, Any]]:
    """Curated findings of this set that are no target (exploratory candidates for a later test)."""
    found = _findings(pooled)
    known = set(zip(targets["association"], targets["dv_x"], targets["dv_y"], strict=True))
    return [
        {"association": r.association, "pair": [r.dv_x, r.dv_y], "pair_type": r.pair_type,
         "k_studies": int(r.k_studies), "pooled_r": float(r.pooled_r), "q_bh": float(r.q_bh)}
        for r in found.itertuples() if (r.association, r.dv_x, r.dv_y) not in known
    ]


def decide(relations_dir: Path, targets_path: Path, e1_dir: Path | None = None,
           data_dir: Path | None = None, manifest_path: Path | None = None) -> dict[str, Any]:
    pooled = pd.read_csv(relations_dir / "relations_pooled.csv")
    by_study = pd.read_csv(relations_dir / "relations_by_study.csv")
    targets = pd.read_csv(targets_path)
    result: dict[str, Any] = {
        "preregistration": PREREGISTRATION,
        "level": LEVEL, "alpha": ALPHA, "min_studies": MIN_K,
        "studies": sorted(set(by_study.loc[by_study["level"] == LEVEL, "study"])),
        "H1": first.h1(pooled, by_study),
        "H2": first.h2(pooled, by_study),
        "H2b": first.h2b(pooled, by_study),
        "H4": h4(pooled, by_study, targets),
        "new_findings_exploratory": new_findings(pooled, targets),
    }
    if e1_dir is not None:
        result["H3"] = first.h3(e1_dir, data_dir, manifest_path)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("targets", help="derive the replication targets from the earlier analyses")
    t.add_argument("--development", default=str(REPO_ROOT / "analyses" / "output_relations"))
    t.add_argument("--first-test", default=str(REPO_ROOT / "analyses" / "heldout" / "relations"))
    t.add_argument("--output", default=str(OUT_DIR / "targets.csv"))
    d = sub.add_parser("decide", help="decide the pre-registered hypotheses")
    d.add_argument("--relations", required=True, help="output directory of construct_relations")
    d.add_argument("--targets", default=str(OUT_DIR / "targets.csv"))
    d.add_argument("--e1", default=None, help="output directory of the E1 run (definition shortlist)")
    d.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw" / "heldout_2"))
    d.add_argument("--manifest", default=str(OUT_DIR / "sources_manifest.yaml"))
    d.add_argument("--output", default=str(OUT_DIR / "results.json"))
    args = parser.parse_args(argv)
    if args.command == "targets":
        frame = build_targets(pd.read_csv(Path(args.development) / "relations_pooled.csv"),
                              pd.read_csv(Path(args.first_test) / "relations_pooled.csv"))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(frame.groupby(["family", "association"]).size().to_string())
        return
    result = decide(Path(args.relations), Path(args.targets), Path(args.e1) if args.e1 else None,
                    Path(args.data_dir), Path(args.manifest))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({k: v.get("decision") for k, v in result.items() if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
