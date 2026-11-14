#!/usr/bin/env bash
# Second held-out test (docs/preregistration_heldout_relations_2.md), procedure step 4.
# Mirrors analyses/heldout/run.sh. Run from the repository root after
#   python -m opendv.analysis.confirmatory fetch --sources analyses/heldout_2/sources.yaml --data-dir data/raw/heldout_2
# with the frozen mappings (analyses/heldout_2/mappings/) and study designs in place.
set -euo pipefail
PY=${PYTHON:-python}
DET=data/processed/heldout_2_det
OUT=analyses/heldout_2
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg PYTHONUTF8=1

# Batch without the LLM.
rm -rf "$DET"
$PY -m opendv.run_batch_standardization --manifest "$OUT/sources_manifest.yaml" \
  --output-dir "$DET" --disable-llm-deduction --workers 4 --no-fair-export

# Construct relations on this set's studies only (levels names, automatic, curated, llm;
# defaults: >= 10 participants, within-person df >= 10, REML with HKSJ).
$PY -m opendv.analysis.construct_relations --input-dir "$DET/standardized" --output-dir "$OUT/relations"

# Mapping benchmark (E1) with the settings of the first held-out test (H3).
$PY -m opendv.analysis.evaluate_llm_mapping --manifest "$OUT/sources_manifest.yaml" \
  --batch-dir "$DET" --cache-dir data/processed/heldout_2_cache --output-dir "$OUT/runs/definition" \
  --shortlist definition --max-per-source 40 --seed 20261017 --e2-samples 0 --models google/gemma-4-E2B-it

# Decisions on H1, H2, H2b, H3 and H4.
$PY -m opendv.analysis.heldout_replication decide --relations "$OUT/relations" \
  --targets "$OUT/targets.csv" --e1 "$OUT/runs/definition" \
  --manifest "$OUT/sources_manifest.yaml" --data-dir data/raw/heldout_2 --output "$OUT/results.json"
