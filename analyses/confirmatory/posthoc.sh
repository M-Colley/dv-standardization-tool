#!/usr/bin/env bash
# Post hoc rerun of the confirmatory E1 after the pipeline gaps the test found
# were fixed (answer-code recoding, header matching, indicator rule); see the
# "Post hoc" section of docs/confirmatory_shortlist_results.md. Not part of the
# pre-registered test: run.sh and analyses/confirmatory/runs/ stay as they were.
# Run from the repository root after run.sh.
set -euo pipefail
PY=${PYTHON:-python}
DET=data/processed/confirmatory_det
RUNS=analyses/confirmatory/runs
POSTHOC=analyses/confirmatory/posthoc
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg

rm -rf "$DET"
$PY -m opendv.run_batch_standardization --manifest analyses/confirmatory/sources_manifest.yaml \
  --output-dir "$DET" --disable-llm-deduction --workers 4 --no-fair-export

for shortlist in definition dense_h mixed_h; do
  $PY -m opendv.analysis.evaluate_llm_mapping --manifest analyses/confirmatory/sources_manifest.yaml \
    --batch-dir "$DET" --cache-dir data/processed/confirmatory_cache --output-dir "$POSTHOC/$shortlist" \
    --shortlist "$shortlist" --max-per-source 40 --seed 20261017 --e2-samples 0 --models google/gemma-4-E2B-it
done

$PY -m opendv.analysis.confirmatory posthoc \
  --runs definition="$POSTHOC/definition" dense_h="$POSTHOC/dense_h" mixed_h="$POSTHOC/mixed_h" \
  --preregistered-runs definition="$RUNS/definition" dense_h="$RUNS/dense_h" mixed_h="$RUNS/mixed_h" \
  --output "$POSTHOC/results.json"
