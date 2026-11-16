#!/usr/bin/env bash
# Schema-stability follow-up: no zero-score padding and order votes
# (docs/schema_stability_followup_design.md). Run from the repository root.
#
# Inputs: those of analyses/schema_stability/run.sh (det23l, det16l, manifest23.yaml,
# manifest16.yaml, the download caches) and the reference's response cache, which
# that run wrote to data/processed/schema_stability/e2b_definition/response_cache.jsonl
# (SHA-256 75fb660d...aa95c for the committed reference answers).
# Answers go to data/processed/schema_stability_followup/<condition>/<perturbation>.csv;
# a rerun skips the perturbations already written, so the script can be restarted.
set -euo pipefail
PY=${PYTHON:-python}
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg PYTHONUTF8=1
W=data/processed/schema_stability_followup
REF_CACHE=${REF_CACHE:-data/processed/schema_stability/e2b_definition/response_cache.jsonl}
RUN="$PY -m opendv.analysis.schema_stability"

# 1. The mapping files the reference read (commit ffabda7), and manifests that point at them.
$RUN freeze-mappings

# 2. Prompts only, no LLM: how many prompts each condition asks beyond the reference's cache.
for condition in e2b_definition_nopad e2b_definition_vote3; do
  $RUN dry-run --condition "$condition" --extra-cache "$REF_CACHE"
done

# 3. The conditions in the pre-specified order; the vote's order seeds (sample_seed) last.
$RUN run --condition e2b_definition_nopad --extra-cache "$REF_CACHE"
$RUN run --condition e2b_definition_vote3 --extra-cache "$REF_CACHE" \
  --families base,add,remove,reword,schema_order,real_3_2
$RUN run --condition e2b_definition_vote3 --extra-cache "$REF_CACHE" --families sample_seed
# The unanimous rule asks the same prompts: every answer comes from the two caches.
$RUN run --condition e2b_definition_vote3u --extra-cache "$REF_CACHE" \
  --extra-cache "$W/e2b_definition_vote3/response_cache.jsonl"

# 4. Outcomes against the committed reference answers, integrity checks, secondary tables.
$RUN followup
