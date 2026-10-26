#!/usr/bin/env bash
# Confirmatory shortlist test (docs/preregistration_confirmatory_shortlist.md).
# Run from the repository root after `python -m opendv.analysis.confirmatory fetch`
# and with the frozen gold labels (analyses/confirmatory/mappings/) in place.
set -euo pipefail
PY=${PYTHON:-python}
DET=data/processed/confirmatory_det
RUNS=analyses/confirmatory/runs
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg

$PY -m opendv.analysis.confirmatory manifest
rm -rf "$DET"
# Gold items: the frozen labels applied as curated mappings, no LLM.
$PY -m opendv.run_batch_standardization --manifest analyses/confirmatory/sources_manifest.yaml \
  --output-dir "$DET" --disable-llm-deduction --workers 4 --no-fair-export

for shortlist in definition dense_h mixed_h; do
  $PY -m opendv.analysis.evaluate_llm_mapping --manifest analyses/confirmatory/sources_manifest.yaml \
    --batch-dir "$DET" --cache-dir data/processed/confirmatory_cache --output-dir "$RUNS/$shortlist" \
    --shortlist "$shortlist" --max-per-source 40 --seed 20261017 --e2-samples 0 --models google/gemma-4-E2B-it
done

$PY -m opendv.analysis.confirmatory analyze \
  --runs definition="$RUNS/definition" dense_h="$RUNS/dense_h" mixed_h="$RUNS/mixed_h" \
  --output analyses/confirmatory/results.json
