"""Audit table and in-situ precision of the LLM-mapped columns that fed a pooled result.

Joins ``inventory.csv`` (one row per pooled LLM-mapped standardized column, from
``inventory.py``) with the verdicts in ``verdicts.yaml`` and writes

- ``audit.csv``: one row per audited column (study x raw column x LLM DV), with the
  files it occurs in, the verdict, the right DV, the evidence, and the decision group
  (one entry of verdicts.yaml; repeated items share it);
- ``precision.csv``: the share of audited columns whose LLM DV is ``correct``, overall,
  by study and for the ten DVs the LLM assigned most often, with Wilson 95% intervals
  (``statsmodels.stats.proportion.proportion_confint``). Two variants: ``unclear``
  counted as wrong, and ``unclear`` left out. The same is reported with decision groups
  as the unit, since columns of one group share one decision and are not independent.

Every inventory row must match exactly one verdict entry; the script stops otherwise.

Usage::

    python -I analyses/llm_mapping_audit/audit.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
VERDICTS = ("correct", "wrong_dv", "not_a_dv", "unclear")


def load_verdicts(path: Path) -> list[dict]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    for index, entry in enumerate(entries):
        entry["decision_group"] = f"{entry['study']}#{index + 1:02d}"
        if entry["verdict"] not in VERDICTS:
            raise ValueError(f"{entry['decision_group']}: verdict {entry['verdict']!r} is not one of {VERDICTS}")
        if entry["verdict"] in ("correct", "wrong_dv") and not entry.get("target"):
            raise ValueError(f"{entry['decision_group']}: a {entry['verdict']} verdict needs a target")
        if entry["verdict"] == "unclear" and not entry.get("missing"):
            raise ValueError(f"{entry['decision_group']}: an unclear verdict needs `missing`")
        if not entry.get("evidence"):
            raise ValueError(f"{entry['decision_group']}: no evidence")
    return entries


def _matches(entry: dict, study: str, column: str, llm_dv: str) -> bool:
    if entry["study"] != study:
        return False
    if entry.get("llm_dv") and entry["llm_dv"] != llm_dv:
        return False
    if "columns" in entry:
        return column in [str(c) for c in entry["columns"]]
    return re.fullmatch(entry["pattern"], column) is not None


def build_audit(inventory: pd.DataFrame, entries: list[dict]) -> pd.DataFrame:
    rows = []
    keys = ["study", "column", "llm_dv"]
    used: set[str] = set()
    for (study, column, llm_dv), group in inventory.groupby(keys, sort=True):
        hits = [entry for entry in entries if _matches(entry, study, column, llm_dv)]
        if len(hits) != 1:
            raise ValueError(f"{study} / {column!r} / {llm_dv}: {len(hits)} verdict entries match")
        entry = hits[0]
        used.add(entry["decision_group"])
        files = sorted(group["file"].unique())
        findings = sorted({f for value in group["relations_finding"] for f in value.split("; ") if f})
        rows.append({
            "study": study,
            "file": files[0] if len(files) == 1 else f"{len(files)} files, e.g. {files[0]}",
            "column": column,
            "llm_dv": llm_dv,
            "verdict": entry["verdict"],
            "correct_target": entry.get("target", "") if entry["verdict"] in ("correct", "wrong_dv") else "",
            "llm_dv_ruled_out": bool(entry.get("llm_dv_ruled_out", entry["verdict"] in ("wrong_dv", "not_a_dv"))),
            "evidence": " ".join(str(entry["evidence"]).split()),
            "missing": " ".join(str(entry.get("missing", "")).split()),
            "decision_group": entry["decision_group"],
            "n_files": len(files),
            "n_standardized_columns": len(group),
            "relations_findings": "; ".join(findings),
        })
    unused = [entry["decision_group"] for entry in entries if entry["decision_group"] not in used]
    if unused:
        raise ValueError(f"verdict entries that match no inventory column: {unused}")
    return pd.DataFrame(rows)


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(low), float(high)


def _precision_rows(frame: pd.DataFrame, scope: str, level: str, unit: str) -> list[dict]:
    out = []
    for variant in ("unclear_as_wrong", "unclear_excluded"):
        data = frame if variant == "unclear_as_wrong" else frame[frame["verdict"] != "unclear"]
        n = len(data)
        k = int((data["verdict"] == "correct").sum())
        low, high = _wilson(k, n)
        out.append({
            "unit": unit, "scope": scope, "level": level, "variant": variant,
            "n": n, "correct": k, "precision": k / n if n else float("nan"),
            "wilson_low": low, "wilson_high": high,
            "n_wrong_dv": int((data["verdict"] == "wrong_dv").sum()),
            "n_not_a_dv": int((data["verdict"] == "not_a_dv").sum()),
            "n_unclear": int((data["verdict"] == "unclear").sum()),
        })
    return out


def precision_table(audit: pd.DataFrame, n_dvs: int = 10) -> pd.DataFrame:
    rows: list[dict] = []
    # Decision groups as the unit: a group is correct when its decision is (one verdict per group).
    groups = audit.groupby("decision_group", as_index=False).agg(
        study=("study", "first"), verdict=("verdict", "first"), llm_dv=("llm_dv", lambda s: s.mode().iat[0]),
    )
    for unit, frame in (("column", audit), ("decision_group", groups)):
        rows += _precision_rows(frame, "overall", "all", unit)
        for study, part in frame.groupby("study"):
            rows += _precision_rows(part, "study", study, unit)
        if unit == "column":
            top = audit["llm_dv"].value_counts()
            top = top.sort_values(ascending=False, kind="stable").head(n_dvs).index
            for dv in top:
                rows += _precision_rows(frame[frame["llm_dv"] == dv], "llm_dv", dv, unit)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inventory", type=Path, default=HERE / "inventory.csv")
    parser.add_argument("--verdicts", type=Path, default=HERE / "verdicts.yaml")
    parser.add_argument("--audit", type=Path, default=HERE / "audit.csv")
    parser.add_argument("--precision", type=Path, default=HERE / "precision.csv")
    args = parser.parse_args()
    inventory = pd.read_csv(args.inventory, dtype=str, keep_default_na=False)
    audit = build_audit(inventory, load_verdicts(args.verdicts))
    audit.to_csv(args.audit, index=False)
    precision = precision_table(audit)
    precision.to_csv(args.precision, index=False, float_format="%.4f")
    print(f"{len(audit)} audited columns in {audit['decision_group'].nunique()} decision groups -> {args.audit}")
    print(audit["verdict"].value_counts().to_string())
    overall = precision[(precision["scope"] == "overall")]
    print(overall[["unit", "variant", "n", "correct", "precision", "wilson_low", "wilson_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
