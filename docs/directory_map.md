# Repository Directory Map

This document outlines the folder structure and purpose of each key directory.

```
.github/                 → GitHub workflow & community templates
opendv/                  → The Python package (installed with `pip install .`)
    __init__.py          → Public API: standardize, run_batch, analyze, review_queue, score, load_plugins
    cli.py               → `opendv` command line
    plugins.py           → Entry-point plugins (opendv.sources, opendv.loaders, opendv.instruments)
    _paths.py            → Where schemas are found (package data, or schemas/ in a checkout)
    sources/             → Repository sources (OSF, GitHub, Zenodo, figshare, 4TU, Dataverse)
    analysis/            → Multi-study analysis, condition contrasts, measurement invariance (+ .R),
                           multiverse, LLM benchmarks, CHI figures
    ui/                  → Streamlit app (`opendv ui`), components and assets
scripts/, analyses/*.py  → Aliases of the pre-3.1 module paths (source checkouts only)
analyses/                → Committed analysis outputs (output_*), figures (figures_chi) and
                           run_latest_multi_study_analysis.sh
data/
    raw/                 → Unprocessed CSVs and the example catalog file
    processed/           → Standardized datasets and batch outputs (gitignored)
docs/                    → Project documentation (this folder)
notebooks/               → Prototyping and testing notebooks
schemas/                 → Canonical schemas, instrument registry, study designs, dataset mappings
                           (installed as opendv.schemas)
tests/                   → Pytest suite
troubleshooting/         → Logs, mismatches, kernel and schema issues
sources_manifest_example.yaml → Example batch manifest covering GitHub / OSF / 4TU / ACM
```

## Key CLI entrypoints

| script | role |
| --- | --- |
| `opendv/convert_dv.py` | Standardize a single dataset against the schema(s). |
| `opendv/run_batch_standardization.py` | Standardize many sources from a manifest; writes `meta_view.csv` and per-source standardized trees. |
| `opendv/run_catalog_meta_analysis.py` | End-to-end: CSV/Excel catalog → batch standardization → meta-analysis artifacts. |
| `opendv/analysis/multi_study_analysis.py` | Overlap / pooled-mean / random-effects meta-analysis on a folder of standardized files. |
| `opendv/validate_schema.py` | Schema sanity check. |
| `opendv/ui/app.py` | Streamlit entry point (single-file + catalog modes). |
