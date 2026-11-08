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

## Pickled data files

Downloaded data is untrusted, and unpickling runs code: a pickle names functions that `pickle.load` imports and calls, so loading a `.pkl` from a repository runs whatever its author put there. The pipeline therefore never unpickles by default.

- **Batch.** `.pkl` and `.pickle` files are discovered but not loaded. The run logs a warning naming them and lists them under `refused_pickle_files` in `run_summary.json`; a source whose files are all pickles is reported as `not_available`. This holds for every source type, local folders included.
- **Opt-in.** `allow_pickle: true` on a manifest source unpickles its pickles (`opendv convert --allow-pickle` for one file, `opendv.standardize(..., allow_pickle=True)` in Python). Set it only for files you created yourself. No shipped manifest sets it, and a test checks that none does.
- **Outputs.** The standardized table of a pickle input is written as Parquet (it keeps the column types), so nothing downstream needs to unpickle. The analysis skips pickles left in an output folder from older runs, with a warning.

When a study's tables exist only as pickles, convert them once and pin the result:

1. Record the checksum of each pickle as published (the git blob hash at a pinned commit, or the repository's SHA-256 or MD5).
2. Optionally list what the file would import: `python -m pickletools file.pkl` disassembles it without running it. Globals other than pandas, NumPy and builtins constructors are a reason to stop.
3. Convert in a disposable environment with no network, no credentials and none of your files (a fresh container or VM with pandas installed): `python -I -c "import sys, pandas as pd; pd.read_pickle(sys.argv[1]).to_parquet(sys.argv[2])" file.pkl file.parquet`. Pickled DataFrames depend on the pandas version that wrote them, so note the version that read them.
4. Deposit the Parquet files where they can be fetched by checksum (an OSF or Zenodo record, or a commit in a repository you control), list them with their SHA-256 in `analyses/development_sources/sources.yaml`, fetch them with the `confirmatory fetch` command above, and read them as a `local_path` source. Keep the input checksums from step 1 next to them so the conversion can be traced.

`road_bumps_touch` needs none of this. Its `data/` folder holds the four tables the paper's ANOVAs read (`R_Count_Down.csv`, `R_Count_Move.csv`, `R_Offset.csv`, `R_TCT.csv`: touch counts and touch offset per participant and condition, task completion time also per stimulus, computed by the authors from the pickles) and four pickles with the raw touch and accelerometer logs. The manifest reads the CSV tables only, mapped by `schemas/road_bumps_touch_mapping.yaml`.

## Rerunning

```bash
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg
# --cache-dir keeps the 11 remote downloads (1.6 GB) for later runs; a run writes about 1 GB.
# Eight workers took about 35 GB of memory (commit charge on Windows); with less free, use
# fewer: a worker that runs out drops the file it was reading, and the number of workers does
# not change the results otherwise.
opendv batch --manifest sources_manifest_no4tu.yaml --output-dir data/processed/catalog_meta_analysis --snapshot-manifest --workers 8 --cache-dir data/processed/cache
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

On the same environment this reproduces the committed tables exactly; figure PDFs differ only in their embedded timestamps.

The committed `output_python_full/`, `output_multiverse/`, `figures_chi/` and `output_relations/` (2026-11-06) come from one batch run of all 39 studies with the commands above (3 workers, download cache of 2026-11-04), written to `data/processed/catalog39_toast` of the main checkout (hence the `input_dir` in `analysis_results.json`). It reads NL2TAP's TOAST items through the per-file `rename` option. Apart from that one added table, its 1,806 standardized tables have the same contents as a run of the same day without the TOAST items, which in turn reproduces the two runs behind the relation tables of 2026-11-05 (the 2026-11-04 run of the first 23 studies, 8 workers, and the 2026-11-05 run of the 16 added ones): identical table contents, `meta_view.csv` differing only in absolute paths and run timestamps, and the relation tables byte for byte. The benchmark of the added studies was rerun with the TOAST items too: the nine new items are the only change, every other answer repeats.

**The output path does not matter.** The batch shortens a standardized file's name with a hash when its full path would exceed 240 characters, which happens for the longest names below an output folder of more than about 90 characters (e.g. inside a `.claude/worktrees/<name>` checkout). The analyses do not rely on those names: they identify each file by its path in the source (the `dataset_id` in its `-quality.json`), read participant ids declared with `participant_from_file` from that path, and read files in the order of the names a short output path gives them. eHMI Optimization, whose questionnaire files take their participant ids from their folder names, gives byte-identical results from a 71- and a 130-character output folder. Before 2026-11-06, half of its rows had no participant id when read from the longer one (see the changelog).
