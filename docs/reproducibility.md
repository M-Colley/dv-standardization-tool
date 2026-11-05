# Reproducing the results

The committed outputs (`analyses/output_python_full/`, `analyses/output_multiverse/`, `analyses/output_llm_evaluation/`, `analyses/figures_chi/`) were produced with the exact environment recorded in the repository. `pyproject.toml` only sets minimum versions; the lockfiles pin everything.

## Python

`uv.lock` pins every package, including indirect dependencies, for all platforms and Python ≥ 3.12. It was resolved from the versions installed when the committed outputs were generated (Python 3.13, Windows), so it reproduces that environment rather than the newest releases.

```bash
uv sync --frozen --all-extras          # with uv
pip install -r requirements-lock.txt   # with pip (same pins, exported from uv.lock)
pip install --no-deps -e .
```

`torch` from PyPI is the CPU build on Windows. For a GPU, install the CUDA wheel of the same version first (`torch==2.14.1`, see `requirements-llm.txt`); the committed runs used `2.14.1+cu130`.

After changing dependencies in `pyproject.toml`, run `uv lock` and `uv export --frozen --all-extras --no-default-groups --no-hashes --no-emit-project --format requirements-txt -o requirements-lock.txt`; CI checks that both files match `pyproject.toml`.

## R

The measurement-invariance step calls `Rscript` with lavaan (`opendv/analysis/measurement_invariance.R`). `renv.lock` pins R 4.6.1, lavaan 0.7-3, jsonlite 2.0.0 and their dependencies:

```r
install.packages("renv")
renv::restore(lockfile = "renv.lock")
```

Without R the analysis still runs and reports invariance as `not_tested`, and the lavaan simulation tests are skipped. The effect-size and estimator checks in `tests/test_meta_analysis_reference.py` compare against a stored metafor 5.2.1 fixture, so they do not need R.

## Models

LLM mapping and dense retrieval load these Hugging Face snapshots (`~/.cache/huggingface/hub`); every committed run used them:

| Model | Use | Revision |
|---|---|---|
| `google/gemma-4-E2B-it` | LLM mapping (all committed runs) | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| `google/gemma-4-E4B-it` | first-choice LLM where memory allows | `ee0ef6023621cff504d758262d4e04895a5af4a2` |
| `BAAI/bge-small-en-v1.5` | dense retrieval (`--llm-shortlist dense`, `hierarchical`, `mixed`) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| `intfloat/multilingual-e5-small` | dense retrieval of German labels | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |

```bash
hf download google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7
```

Generation is greedy, so with the same snapshot and inputs the LLM's choices repeat; `run_summary.json` lists the models a run used. Set `HF_HUB_OFFLINE=1` to make sure nothing newer is fetched.

## Data

`sources_manifest_no4tu.yaml` lists the 39 development studies.

- **The first 11** are fetched from their repositories by the batch itself.
- **The 12 added on 2026-11-04 and the 16 added on 2026-11-05** are read from `data/raw/development/`. Download them first; each file is checked against its pinned checksum:

    ```bash
    python -m opendv.analysis.confirmatory fetch --sources analyses/development_sources/sources.yaml --data-dir data/raw/development
    python -m opendv.analysis.confirmatory fetch --sources analyses/development_sources/sources_search.yaml --data-dir data/raw/development
    ```

Each run records the source revision it read (git commit, or the repository record's version and file checksums) in `meta_view.csv` (`commit`) and in the RO-Crate and data packages written next to the outputs.

## Rerunning

```bash
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg
opendv batch --manifest sources_manifest_no4tu.yaml --output-dir data/processed/catalog_meta_analysis --snapshot-manifest --workers 8
opendv analyze --input-dir data/processed/catalog_meta_analysis/standardized --output-dir analyses/output_python_full --exclude-llm-deduced
python -m opendv.analysis.multiverse_analysis --input-dir data/processed/catalog_meta_analysis/standardized --output-dir analyses/output_multiverse
python -m opendv.analysis.make_chi_figures

# LLM benchmark: a run without LLM deductions gives the gold labels; every
# curated column is asked about (--max-per-source 1000) for the replay below.
opendv batch --manifest sources_manifest_no4tu.yaml --output-dir data/processed/det23 --disable-llm-deduction --workers 8 --no-fair-export
for s in definition dense_h mixed_h; do
  out=analyses/output_llm_evaluation; [ "$s" = definition ] || out=$out/postfix/shortlist_$s
  python -m opendv.analysis.evaluate_llm_mapping --manifest sources_manifest_no4tu.yaml --batch-dir data/processed/det23 \
    --cache-dir data/processed/cache --max-per-source 1000 --seed 20261017 --models google/gemma-4-E2B-it \
    --subset added_2026_11_04=analyses/development_sources/sources.yaml --shortlist "$s" --output-dir "$out"
done

# The same benchmark (E1 only) on the 16 studies added on 2026-11-05: MANIFEST_ADDED holds their
# manifest entries (the last 16 of sources_manifest_no4tu.yaml), DET_ADDED a run of them without LLM.
python -m opendv.analysis.evaluate_llm_mapping --manifest "$MANIFEST_ADDED" --batch-dir "$DET_ADDED" \
  --cache-dir data/processed/cache16 --max-per-source 1000 --seed 20261017 --models google/gemma-4-E2B-it \
  --e2-samples 0 --shortlist definition --output-dir analyses/output_llm_evaluation/added_2026_11_05

# Cross-study relations, with the LLM standing in for the curated mappings
python -m opendv.analysis.construct_relations --input-dir data/processed/catalog_meta_analysis/standardized \
  --output-dir analyses/output_relations \
  --llm-replay-items analyses/output_llm_evaluation/llm_mapping_evaluation_items.csv \
                     analyses/output_llm_evaluation/added_2026_11_05/llm_mapping_evaluation_items.csv
```

On the same environment this reproduces the committed tables exactly; figure PDFs differ only in their embedded timestamps. The relation tables of 2026-11-05 combine the 2026-11-04 batch run of the first 23 studies with a run of the 16 added ones (same code, settings and model; see `docs/cross_study_insights.md`). The other analyses (`analyses/output_python_full/`, `output_multiverse/`, `figures_chi/`) are still those of the 23 studies.
