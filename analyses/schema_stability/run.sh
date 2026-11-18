#!/usr/bin/env bash
# Schema stability of the LLM mappings (docs/schema_stability_design.md).
# Run from the repository root. Inputs: the benchmark's two runs without the LLM and their
# manifests and download caches (docs/reproducibility.md, "Rerunning"):
#   data/processed/det23l, data/processed/det16l       (meta_view.csv, run_summary.json)
#   data/processed/manifest23.yaml, manifest16.yaml    (the first 23 / last 16 sources)
#   data/processed/cache, data/processed/cache16       (download caches of 2026-11-04)
# Each perturbation's answers go to data/processed/schema_stability/<condition>/<perturbation>.csv;
# a rerun skips the perturbations already written, so the script can be restarted.
set -euo pipefail
PY=${PYTHON:-python}
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg PYTHONUTF8=1
HUB=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
E4B_WEIGHTS=$HUB/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187
E4B_TOKENIZER=$HUB/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2

# 1. The draws (design.json) and the perturbed schema files; then the prompts alone, no LLM.
$PY -m opendv.analysis.schema_stability plan
$PY -m opendv.analysis.schema_stability dry-run --condition e2b_definition
$PY -m opendv.analysis.schema_stability dry-run --condition e2b_dense_h

# 2. The conditions, in the pre-specified order.
for condition in e2b_definition e2b_dense_h e2b_definition_sc5 e2b_definition_alpha; do
  $PY -m opendv.analysis.schema_stability run --condition "$condition"
done
$PY -m opendv.analysis.schema_stability run --condition e4b_definition --cpu-ple "$E4B_WEIGHTS" "$E4B_TOKENIZER"

# 3. Flip rates, accuracy changes and the paired comparisons.
$PY -m opendv.analysis.schema_stability analyze

# 4. Exploratory mechanism analyses, added after the runs (not in the design note).
$PY -m opendv.analysis.schema_stability explore
