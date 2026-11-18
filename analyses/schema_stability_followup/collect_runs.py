"""Collect the follow-up's run records into its outputs folder (run from the repository root, after `followup`).

dry_run_prompt_changes.csv and generations.csv come from the dry-run and run logs in
data/processed/schema_stability_followup/; provenance.json records the frozen mapping files
and the response caches read; exploratory.json holds the counts added after the results were
seen (not in docs/schema_stability_followup_design.md).
"""
import json
import re
from pathlib import Path

import pandas as pd

from opendv.analysis import schema_stability as ss

W = Path("data/processed/schema_stability_followup")
OUT = Path("analyses/output_llm_evaluation/schema_stability_followup")
LINE = re.compile(r"^(?P<time>\S+ \S+) INFO schema_stability: (?P<condition>\S+) (?P<perturbation>\S+): "
                  r"(?P<items>\d+) items, (?P<generated>\d+) generated, (?P<errors>\d+) errors, (?P<seconds>\d+) s")


def parse(log: Path, **extra) -> pd.DataFrame:
    rows = [m.groupdict() for line in log.read_text(encoding="utf-8").splitlines() if (m := LINE.match(line))]
    frame = pd.DataFrame(rows).assign(**extra)
    for c in ("items", "generated", "errors", "seconds"):
        frame[c] = frame[c].astype(int)
    return frame


dry = []
for condition in ("e2b_definition_nopad", "e2b_definition_vote3"):
    counts = parse(W / f"dry_run_{condition}.log")[["perturbation", "generated"]]
    counts = counts.rename(columns={"generated": "new_prompts"})
    table = pd.read_csv(W / f"dry_run_{condition}.csv").rename(columns={
        "prompt_changed": "prompt_changed_rate", "set_changed": "shortlist_set_changed_rate", "n": "n_items"})
    dry.append(table.merge(counts, on="perturbation", how="outer").assign(condition=condition))
dry = pd.concat(dry, ignore_index=True)
dry["family"] = dry["family"].fillna("base")
for column in ("k", "n_items"):
    dry[column] = dry[column].astype("Int64")
dry = dry[["condition", "perturbation", "family", "k", "prompt_changed_rate", "shortlist_set_changed_rate", "n_items",
           "new_prompts"]]
dry.to_csv(OUT / "dry_run_prompt_changes.csv", index=False)

runs = pd.concat([parse(W / f"run_{c}.log") for c in
                  ("e2b_definition_nopad", "e2b_definition_vote3", "e2b_definition_vote3u")], ignore_index=True)
runs.to_csv(OUT / "generations.csv", index=False)

provenance = {
    "frozen_mappings": json.loads((W / "frozen_mappings.json").read_text(encoding="utf-8")),
    "extra_caches": {c: json.loads((W / c / "extra_caches.json").read_text(encoding="utf-8"))
                     for c in ("e2b_definition_nopad", "e2b_definition_vote3", "e2b_definition_vote3u")},
    "chain_progress": (W / "chain_progress.txt").read_text(encoding="utf-8").splitlines(),
    "generated": runs.groupby("condition")["generated"].sum().to_dict(),
    "generation_errors_logged": runs.groupby("condition")["errors"].sum().to_dict(),
    "dry_run_new_prompts": dry.groupby("condition")["new_prompts"].sum().to_dict(),
}
(OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, default=int), encoding="utf-8")
print(json.dumps({k: provenance[k] for k in ("generated", "generation_errors_logged", "dry_run_new_prompts")},
                 default=int))
print(json.dumps(provenance["extra_caches"], indent=1))

# Exploratory, added after the results were seen (not in the design note).

ref = ss.base_answers(ss.read_answers_file(ss.OUTPUT_DIR / "answers_e2b_definition.csv.gz"))
conditions = {name: ss.base_answers(ss.read_answers_file(OUT / f"answers_{name}.csv.gz"))
              for name in ss.FOLLOWUP_CONDITIONS}
exploratory = {"rank1_answer_share": {}, "accepted_mappings": {}}
for name, frame in {"e2b_definition": ref, **conditions}.items():
    asked = frame[frame["prompt_sha256"].notna()]
    first = [s.split("|")[0] if isinstance(s, str) else None for s in asked["shortlist"]]
    exploratory["rank1_answer_share"][name] = {
        "n_asked": int(len(asked)),
        "share": float(sum(a == f for a, f in zip(asked["suggestion"], first, strict=True)) / len(asked)),
    }
    accepted = frame[frame["final_checked"].notna()]
    correct = ss._correct(accepted["final_checked"], accepted["gold"])
    exploratory["accepted_mappings"][name] = {
        "accepted": int(len(accepted)), "correct": int(correct.sum()),
        "wrong_dv_on_dv_columns": int(((accepted["label_kind"] == "dv") & ~correct).sum()),
        "curated_non_dv": int((accepted["label_kind"] != "dv").sum()),
    }
paired = conditions["e2b_definition_vote3"].merge(ref, on=ss.ITEM_KEY, suffixes=("", "_ref"))
dv = paired[paired["label_kind"] == "dv"]
delta = (ss._correct(dv["final_checked"], dv["gold"]).astype(int)
         - ss._correct(dv["final_checked_ref"], dv["gold"]).astype(int))
by_study = delta.groupby(dv["source_id"]).agg(net="sum", gained=lambda x: int((x > 0).sum()),
                                              lost=lambda x: int((x < 0).sum()))
exploratory["vote3_dv_accuracy_by_study"] = by_study[by_study["net"] != 0].sort_values("net").reset_index().to_dict(
    orient="records")
(OUT / "exploratory.json").write_text(json.dumps(exploratory, indent=2, default=int), encoding="utf-8")
print(json.dumps(exploratory["rank1_answer_share"]), json.dumps(exploratory["accepted_mappings"]))
