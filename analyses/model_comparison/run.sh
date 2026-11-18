#!/usr/bin/env bash
# E4B against E2B on the full LLM mapping benchmark (docs/model_comparison_design.md).
# Run from the repository root. Inputs (docs/reproducibility.md, "Rerunning"):
#   $RUNS/det23b, $RUNS/det16b   the runs without the LLM of audit39b (meta_view.csv, run_summary.json)
#   $CACHE, $CACHE16             the download caches of 2026-11-04 (first 23, added 16 sources)
#   $HF_HUB_CACHE                the Hugging Face cache with the snapshots of docs/reproducibility.md
# Each run goes to analyses/output_llm_evaluation/model_comparison/<model>_<shortlist>/<half>/;
# a run whose summary exists without generation errors is skipped, so the script can be restarted.
set -euo pipefail
PY=${PYTHON:-python}
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg
RUNS=${RUNS:-data/processed}
CACHE=${CACHE:-data/processed/cache}
CACHE16=${CACHE16:-data/processed/cache16}
HUB=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
# E4B: weights from the cached snapshot fee6332, tokenizer and chat template from the pinned revision ee0ef60.
E4B_WEIGHTS=$HUB/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187
E4B_TOKENIZER=$HUB/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
OUT=analyses/output_llm_evaluation/model_comparison
ROOT=$(pwd)
# On a shared GPU: before each run, wait until this much GPU memory (MiB) is free (0: do not wait).
# E4B needs 10.4 GB with its per-layer embeddings on the CPU, E2B about 10 GB.
WAIT_FOR_GPU_MB=${WAIT_FOR_GPU_MB:-0}

wait_for_gpu() {
  [ "$WAIT_FOR_GPU_MB" -gt 0 ] || return 0
  local free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')
    [ "${free:-0}" -ge "$WAIT_FOR_GPU_MB" ] && return 0
    sleep 60
  done
}

# Python isolated (-I, -X utf8): no file in a data folder can shadow a module; only the repository root is added.
runmod() {
  "$PY" -I -X utf8 -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)' "$ROOT" "$@"
}

# The added 16 sources' manifest: the last 16 entries of sources_manifest_no4tu.yaml (as manifest16.yaml of audit39b).
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$PY" -I -X utf8 -c 'import sys, yaml; m = yaml.safe_load(open(sys.argv[1], encoding="utf-8")); yaml.safe_dump({"sources": m["sources"][-16:]}, open(sys.argv[2], "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)' \
  sources_manifest_no4tu.yaml "$TMP/manifest16.yaml"

run_cell() {  # MODEL SHORTLIST HALF
  local model=$1 shortlist=$2 half=$3
  local dir=$OUT/${model}_${shortlist}/$half
  if [ -f "$dir/llm_mapping_evaluation_summary.json" ] \
     && grep -q '"llm_generation_errors": 0,' "$dir/llm_mapping_evaluation_summary.json"; then
    echo "skip $dir (done)"
    return
  fi
  local args=(--max-per-source 1000 --seed 20261017 --shortlist "$shortlist" --output-dir "$dir")
  case $model in
    e2b) args+=(--models google/gemma-4-E2B-it) ;;
    e4b) args+=(--models google/gemma-4-E4B-it --cpu-ple "$E4B_WEIGHTS" "$E4B_TOKENIZER") ;;
  esac
  case $half in
    first_23) args+=(--manifest sources_manifest_no4tu.yaml --batch-dir "$RUNS/det23b" --cache-dir "$CACHE"
                     --subset added_2026_11_04=analyses/development_sources/sources.yaml) ;;
    added_16) args+=(--manifest "$TMP/manifest16.yaml" --batch-dir "$RUNS/det16b" --cache-dir "$CACHE16"
                     --e2-samples 0) ;;
  esac
  wait_for_gpu
  echo "$(date '+%F %T') START $model $shortlist $half"
  runmod opendv.analysis.evaluate_llm_mapping "${args[@]}"
  echo "$(date '+%F %T') DONE  $model $shortlist $half"
}

# The pre-specified order: the four E2B runs, then E4B with definition, then with dense_h. One at a time.
for model in e2b e4b; do
  for shortlist in definition dense_h; do
    for half in first_23 added_16; do
      run_cell "$model" "$shortlist" "$half"
    done
  done
done

# Paired comparison, reproduction and determinism checks.
runmod opendv.analysis.model_comparison

# Exploratory, added after the runs (not in the design note): dense_h against definition within each model.
runmod opendv.analysis.model_comparison --explore
