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

`sources_manifest_no4tu.yaml` lists the 11 benchmark studies. Each run records the source revision it read (git commit, or the repository record's version and file checksums) in `meta_view.csv` (`commit`) and in the RO-Crate and data packages written next to the outputs.

## Rerunning

```bash
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg
opendv batch --manifest sources_manifest_no4tu.yaml --output-dir data/processed/catalog_meta_analysis --snapshot-manifest --workers 8
opendv analyze --input-dir data/processed/catalog_meta_analysis/standardized --output-dir analyses/output_python_full --exclude-llm-deduced
python -m opendv.analysis.multiverse_analysis --input-dir data/processed/catalog_meta_analysis/standardized --output-dir analyses/output_multiverse
python -m opendv.analysis.make_chi_figures
```

On the same environment this reproduces the committed tables exactly; figure PDFs differ only in their embedded timestamps.
