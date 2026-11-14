"""Exploratory analyses of the second held-out test (docs/preregistration_heldout_relations_2.md).

Labelled exploratory in the pre-registration; none of them decides a hypothesis.

1. H2 leaving out one study at a time (this set).
2. Every target pooled over the first and second held-out sets together. This follows the first
   test's narrow miss, so it is a form of optional continuation.
3. Every target pooled over all studies: development, first and second held-out sets.

All three re-pool the committed per-study correlations (``relations_by_study.csv``, curated level)
with ``construct_relations.pool_relations``, the function that pooled them in each analysis.

    python -m analyses.heldout_2.exploratory
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from opendv._paths import REPO_ROOT
from opendv.analysis import construct_relations as cr
from opendv.analysis import heldout as first
from opendv.analysis import heldout_replication as hr

OUT = REPO_ROOT / "analyses" / "heldout_2" / "exploratory"
SETS = {
    "development": REPO_ROOT / "analyses" / "output_relations" / "relations_by_study.csv",
    "heldout_1": REPO_ROOT / "analyses" / "heldout" / "relations" / "relations_by_study.csv",
    "heldout_2": REPO_ROOT / "analyses" / "heldout_2" / "relations" / "relations_by_study.csv",
}


def curated(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    return rows[rows["level"] == hr.LEVEL]


def target_table(pooled: pd.DataFrame, targets: pd.DataFrame, label: str) -> pd.DataFrame:
    keys = ["association", "dv_x", "dv_y"]
    merged = targets.merge(pooled, on=keys, how="left", suffixes=("", "_pooled"))
    out = merged[["target", "family", "association", "dv_x", "dv_y", "predicted_sign", "k_studies", "pooled_r",
                  "hksj_ci95_low", "hksj_ci95_high", "prediction_interval_low", "prediction_interval_high",
                  "heterogeneity_i2_pct", "p_hksj"]].copy()
    out["p_one_sided"] = [
        first.one_sided_p(p, r, s) if np.isfinite(p) else np.nan
        for p, r, s in zip(out["p_hksj"], out["pooled_r"], out["predicted_sign"], strict=True)
    ]
    out.insert(0, "pooled_over", label)
    return out


def leave_one_out(rows: pd.DataFrame) -> pd.DataFrame:
    """H2's three pairs, within persons, pooled with each study left out in turn."""
    within = rows[rows["association"] == "within_person"]
    out = []
    for x, y in hr.H2_PAIRS:
        a, b = sorted((x, y))
        pair = within[(within["dv_x"] == a) & (within["dv_y"] == b)]
        for study in sorted(pair["study"]):
            rest = pair[pair["study"] != study]
            if len(rest) < 2:
                continue
            pooled = cr.pool_relations(rest).iloc[0]
            out.append({"pair": f"{a} x {b}", "left_out": study, "k_studies": len(rest),
                        "pooled_r": pooled["pooled_r"], "hksj_ci95_low": pooled["hksj_ci95_low"],
                        "hksj_ci95_high": pooled["hksj_ci95_high"],
                        "p_one_sided": first.one_sided_p(pooled["p_hksj"], pooled["pooled_r"], 1)})
    return pd.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(hr.OUT_DIR / "targets.csv")
    rows = {name: curated(path) for name, path in SETS.items()}
    tables = []
    for label, names in (("heldout_1+heldout_2", ["heldout_1", "heldout_2"]),
                         ("all", ["development", "heldout_1", "heldout_2"])):
        pooled = cr.pool_relations(pd.concat([rows[n] for n in names], ignore_index=True))
        pooled.to_csv(OUT / f"relations_pooled_{label.replace('+', '_')}.csv", index=False)
        tables.append(target_table(pooled, targets, label))
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUT / "targets_pooled.csv", index=False)
    loo = leave_one_out(rows["heldout_2"])
    loo.to_csv(OUT / "h2_leave_one_out.csv", index=False)
    summary = {
        label: {
            "targets_estimable": int(group["k_studies"].ge(2).sum()),
            "same_sign": int((np.sign(group["pooled_r"]) == group["predicted_sign"]).sum()),
            "one_sided_p_below_05": int((group["p_one_sided"] < 0.05).sum()),
        }
        for label, group in combined.groupby("pooled_over")
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
