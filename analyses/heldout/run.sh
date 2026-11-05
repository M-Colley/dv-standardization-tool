#!/usr/bin/env bash
# Held-out confirmation test (docs/preregistration_heldout_relations.md), procedure steps 3-5.
# Run from the repository root after
#   python -m opendv.analysis.confirmatory fetch --sources analyses/heldout/sources.yaml --data-dir data/raw/heldout
# with the frozen mappings (analyses/heldout/mappings/) and study designs in place.
set -euo pipefail
PY=${PYTHON:-python}
DET=data/processed/heldout_det
OUT=analyses/heldout
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg

# 3. Batch without the LLM.
rm -rf "$DET"
$PY -m opendv.run_batch_standardization --manifest "$OUT/sources_manifest.yaml" \
  --output-dir "$DET" --disable-llm-deduction --workers 4 --no-fair-export

# 4. Construct relations on the held-out studies only (levels names, automatic, curated, llm;
#    defaults: >= 10 participants, within-person df >= 10, REML with HKSJ).
$PY -m opendv.analysis.construct_relations --input-dir "$DET/standardized" --output-dir "$OUT/relations"

# 5. Mapping benchmark (E1) with the settings of the confirmatory shortlist test.
$PY -m opendv.analysis.evaluate_llm_mapping --manifest "$OUT/sources_manifest.yaml" \
  --batch-dir "$DET" --cache-dir data/processed/heldout_cache --output-dir "$OUT/runs/definition" \
  --shortlist definition --max-per-source 40 --seed 20261017 --e2-samples 0 --models google/gemma-4-E2B-it

# Decisions on H1, H2, H2b and H3.
$PY -m opendv.analysis.heldout --relations "$OUT/relations" --e1 "$OUT/runs/definition" \
  --output "$OUT/results.json"
