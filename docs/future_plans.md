# Future Plans & Integration Roadmap

This document outlines the forward-looking roadmap for enhancing the DV Standardization Tool. These potential improvements aim to broaden the tool's applicability, improve usability, and align with best practices in responsible HCI research, reproducibility, and open science infrastructure.

## LLM Integration

Local-LLM alias deduction is now implemented in `scripts/llm_utils.py` and wired into the batch runner as a fallback after explicit mappings and built-in schema families (see `scripts/run_batch_standardization.py`). Remaining directions include confidence scoring for suggested mappings, retrieval-augmented suggestion that reuses past mapping decisions, and better support for multilingual datasets or domain-specific constructs not yet represented in the canonical schema. Additionally, the use of semantic embeddings could allow for unsupervised clustering of DV names, facilitating schema refinement and cross-study harmonization at scale.

## Enhanced Visualization

Improved visualization capabilities are envisioned to assist with both development and interpretation of the tool's output. In particular, we plan to expand the column comparison visualizations implemented in `scripts/visual_helpers.py` to support more flexible plotting, heatmaps, and schema overlay views. These visualizations would be instrumental in validating fuzzy matches and in conveying the standardization impact to users and reviewers. A secondary goal is to explore visual schema mapping interfaces that dynamically reflect the transformation from raw to standardized formats.

## UI Roadmap

A Streamlit frontend now ships in `ui/app.py`, covering upload-to-export standardization of single files and catalog-driven overlap/meta-analysis exploration, with modular components under `ui/components/` and styling via a minimal CSS layer (`ui/assets/style.css`). Future iterations will explore configuration of schema mappings from within the UI and themes such as dark mode. These interface features, while not core to the thesis, serve to demonstrate the extensibility and adoption potential of the tool in open research environments.

## Analysis Roadmap

- **R parity for multi-study analysis.** Ship an `analyses/multi_study_analysis.R` that mirrors the Python script so users can run DV overlap, harmonized summaries, and `metafor`-style random-effects pooling from R directly. Until then, users can load `harmonized_dv_summary.csv` into their preferred R meta-analysis package.
- **Forest plots annotated with provenance.** Highlight pooled estimates whose `includes_llm_deduced` flag is true, so readers can discount accordingly.
- **Publication-ready effect tables.** An export path that writes LaTeX/HTML summaries from `meta_analysis_summary.csv` plus its `_llm_excluded` sibling.
