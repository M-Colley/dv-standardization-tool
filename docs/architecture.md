# System Architecture

## Overview
The tool is a layered pipeline. Each layer has one entry script and can be used on its own:

1. **Schema layer** (`schemas/`). YAML files describe canonical DV names, sensor streams, object-detection fields, and metadata/process columns. `scripts/schema_utils.py` flattens and validates them; `scripts/validate_schema.py` is a standalone sanity check.
2. **Single-file conversion** (`scripts/convert_dv.py`). Reads a `.csv`/`.xlsx`, classifies the dataset type, applies the relevant schema family, and writes a standardized file (+ optional sidecars: `_metadata.json`, `_schema_suggestions.yaml`, `*-quality.json`).
3. **Batch standardization** (`scripts/run_batch_standardization.py`). Iterates a manifest of sources (`local_path`, `github_repo`, `osf_project`, `web_dataset`), downloads/extracts archives, runs conversion per dataset, and writes cross-source artifacts: `meta_view.csv/json`, `run_summary.json`, `unknown_alias_summary.json`, `mapping_debug_summary.json`. Optional local LLM deduction (`scripts/llm_utils.py`) is a fallback after explicit mappings.
4. **Catalog orchestrator** (`scripts/run_catalog_meta_analysis.py`). Reads a CSV/Excel catalog of dataset URLs, deduplicates, generates a manifest, invokes batch standardization, then runs the cross-study meta-analysis.
5. **Multi-study analysis** (`analyses/multi_study_analysis.py`). Consumes the `standardized/` tree produced above. Computes DV overlap, harmonized per-(study, DV) summaries, random-effects pooled estimates, standardized study-vs-pool deviations, a cross-study composite, and LiNGAM causal discovery. Uses `meta_view.csv` for mapping-provenance categories (`schema` / `repo_mapping` / `llm_deduced` / `blocked` / `unknown`) and supports sensitivity passes via `--exclude-llm-deduced`.
6. **UI layer** (`ui/`). Streamlit app covering single-file standardization and catalog-driven exploration of overlap/meta-analysis artifacts.

## Provenance and fallbacks
At every boundary the pipeline records *why* each value exists:

- **Alias resolution**: `mapping_source` per (source_id, canonical_dv) in `meta_view.csv` — `schema`, `repo_mapping`, `llm_deduced`, or `blocked`.
- **Source availability**: `run_summary.json` statuses include `ok`, `not_available`, `access_restricted` (e.g. Cloudflare-blocked ACM downloads), and `failed`.
- **Meta-analysis pool membership**: `meta_analysis_summary.csv` columns `mapping_source_categories`, `k_llm_deduced`, `includes_llm_deduced` flag pooled estimates that lean on LLM inference.
- **Causal discovery**: `discover_causal_structure` reports `used_method` ∈ {`complete_case`, `clique_complete_case`, `synthetic_cholesky`} and a `synthetic_fallback` flag.

## Batch execution model
`run_batch` materializes one source at a time; within a source, every dataset file becomes a `_DatasetTask` handled by `_process_dataset` (load → classify → map → standardize → compose → summarize).

- `--workers N` fans the tasks out to a process pool. The built-in schema families (`SchemaFamilies`) are sent to each worker once, and alias mappings are memoized per (schema family, repository mapping YAML).
- A dataset that needs LLM deduction comes back from its worker as `needs_llm`. The main process, which owns the local model and its caches, deduces the aliases in file order and resubmits the task with the deduced aliases. LLM calls therefore happen in the same order as in a sequential run.
- Outcomes are aggregated in file order, so `meta_view.csv`, `run_summary.json` and all sidecars match `--workers 1`.

Extension points: dataset types (and their schema family, `llm_deduction`, `item_composition`) live in `schemas/dataset_type_profiles.yaml`; source types are registered with `@register_source_type("<name>")` in `scripts/run_batch_standardization.py`.

## LLM plausibility layer
`scripts/llm_plausibility.py` screens what the local LLM sees and what it may return, using the rules in `schemas/llm_plausibility_rules.yaml`.

- **Before the call:** name rules (demographics, identifiers, seeds, timestamps, survey timings, coordinates, conditions, rankings) and a numeric-data rule.
- **After the call:** a range rule that compares observed values with the canonical DV's range and standard response scales.

Column profiles are computed in the worker that loaded the data. The parent process only sees the surviving candidate columns. Rejections are aggregated into `llm_rejected_suggestions.json`, and the rules version is recorded in `run_summary.json`. `analyses/evaluate_llm_mapping.py` measures the layer against curated mappings (leave-alias-out).

## Candidate retrieval for the LLM
`scripts/dv_retrieval.py` ranks canonical DVs for an unmapped column by TF-IDF similarity to their definitions in `standard_dv_mapping.yaml`. The LLM then chooses among the top 8. The fuzzy alias score is still used to decide whether to ask the LLM at all. `--llm-shortlist fuzzy` switches back to alias-string candidates.

## Response recoding before standardization
Before any numeric step, `_process_dataset` applies two recodings (`scripts/value_labels.py`), and every recoded cell is counted in the dataset's `-quality.json`:

1. text labels declared under `value_labels` in the repository mapping;
2. for survey-like tables, LimeSurvey `negN` answer codes.

## Self-describing outputs

After the run summary is written, `scripts/fair_export.py` adds a Frictionless Data Package (`datapackage.json`) to every source directory and an RO-Crate (`ro-crate-metadata.json`) to the run directory, using the `frictionless` and `rocrate` libraries. Column types and row counts are recorded in each file's quality JSON when the table is written, so the export does not read the tables again (about 8 s for the 11-study run). Field metadata comes from `meta_view.csv` (raw names, mapping source), the DV schema (label, range, direction, unit) and the quality JSON (value-label and negative-code recoding). Mapping files appear by file name only, so no local paths are published. The exporter also runs on older runs, which lack the recorded types: it then reads each table.

## Validation of the statistics
`tests/test_meta_analysis_reference.py` checks DerSimonian-Laird pooling, REML, Egger's regression test, trim-and-fill (L0/R0, FE/DL) and Hartung-Knapp-Sidik-Jonkman intervals against R's `metafor` 5.2.1. It uses two published datasets (Hackshaw 1998; Colditz et al. 1994), stored with metafor's results in `tests/fixtures/metafor_reference.json`. `analyses/multiverse_analysis.py` quantifies how much pooled estimates depend on harmonization choices.

## Configuration files and their validation

`scripts/validate_schema.py` validates every configuration file, and CI runs it on all of them: the four standard schemas, each dataset mapping (`schemas/*_mapping.yaml`: targets, `scales`, `value_labels`; unknown targets are warnings), the instrument registry (`schemas/instruments.yaml`) and the study designs (`schemas/study_designs.yaml`).

## Technologies Used
- YAML for schema and manifest representation
- Python (pandas, numpy, matplotlib, seaborn, pydantic) for transformation and analysis
- Optional `transformers` + `torch` for local LLM deduction (`requirements-llm.txt`)
- Optional `streamlit` for the UI (`requirements-ui.txt`)
- Unittest / pytest suite under `tests/`

## Example end-to-end flow
**Input**: `data/raw/study_catalog_example.csv`
```
python scripts/run_catalog_meta_analysis.py \
  --catalog data/raw/study_catalog_example.csv \
  --url-column dataset_url \
  --source-id-column study_name \
  --source-type-column source_type \
  --output-dir data/processed/catalog_meta_analysis
```
**Output**:
- `generated_sources_manifest.yaml` + `catalog_source_summary.csv`
- `standardized/<source_id>/*-standardized.csv` per ingested study
- `meta_view.csv` (mapping provenance)
- `analysis/meta_analysis_summary.csv`, `dv_overlap_matrix.csv`, `dv_presence_matrix.csv`, `dv_overlap_details.csv`, `study_vs_pool_standardized_deviation.csv`
- `analysis/*_llm_excluded.csv` sensitivity siblings when any LLM-deduced rows are present
