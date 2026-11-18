#!/usr/bin/env bash
# Schema-stability follow-up: no zero-score padding and order votes
# (docs/schema_stability_followup_design.md). Run from the repository root.
#
# Inputs: those of analyses/schema_stability/run.sh (det23l, det16l, manifest23.yaml,
# manifest16.yaml, the download caches) and the reference's response cache, which
# that run wrote to data/processed/schema_stability/e2b_definition/response_cache.jsonl
# (SHA-256 75fb660d...aa95c for the committed reference answers).
# Answers go to data/processed/schema_stability_followup/<condition>/<perturbation>.csv;
# a rerun skips the perturbations already written, so the script can be restarted
# (a perturbation with a failed generation is kept as *.errors.csv and redone).
set -euo pipefail
PY=${PYTHON:-python}
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg PYTHONUTF8=1
W=data/processed/schema_stability_followup
REF_CACHE=${REF_CACHE:-data/processed/schema_stability/e2b_definition/response_cache.jsonl}
RUN="$PY -m opendv.analysis.schema_stability"
mkdir -p "$W"

# 1. The mapping files the reference read (commit ffabda7), and manifests that point at them.
$RUN freeze-mappings

# 2. Prompts only, no LLM: how many prompts each condition asks beyond the reference's cache.
for condition in e2b_definition_nopad e2b_definition_vote3; do
  $RUN dry-run --condition "$condition" --extra-cache "$REF_CACHE" > "$W/dry_run_$condition.log" 2>&1
done

# 3. The conditions in the pre-specified order; the vote's order seeds (sample_seed) last.
$RUN run --condition e2b_definition_nopad --extra-cache "$REF_CACHE" >> "$W/run_e2b_definition_nopad.log" 2>&1
$RUN run --condition e2b_definition_vote3 --extra-cache "$REF_CACHE" \
  --families base,add,remove,reword,schema_order,real_3_2 >> "$W/run_e2b_definition_vote3.log" 2>&1
$RUN run --condition e2b_definition_vote3 --extra-cache "$REF_CACHE" --families sample_seed \
  >> "$W/run_e2b_definition_vote3.log" 2>&1
# The unanimous rule asks the same prompts: every answer comes from the two caches.
$RUN run --condition e2b_definition_vote3u --extra-cache "$REF_CACHE" \
  --extra-cache "$W/e2b_definition_vote3/response_cache.jsonl" >> "$W/run_e2b_definition_vote3u.log" 2>&1

# 4. Outcomes against the committed reference answers, integrity checks, secondary tables.
$RUN followup

# 5. Dry-run and generation counts from the logs, provenance (frozen mapping files, caches
#    read), and the exploratory counts added after the results were seen.
PYTHONPATH=. $PY analyses/schema_stability_followup/collect_runs.py
