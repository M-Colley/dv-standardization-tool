# Repository Changelog

This document logs human-readable updates to the repo's functionality, structure, and features.

---

### [2026-10-18]

- **Condition contrasts from declared study designs.** New `schemas/study_designs.yaml` declares the participant column and manipulated factors of each study and maps factor levels to canonical independent variables (`ehmi_presence`, `display_dimensionality`). New `analyses/contrast_meta_analysis.py` computes condition-vs-reference effects on participant means (Hedges' g between subjects, metafor `SMCRPH` within subjects), checks the declared design against the data, combines dependent contrasts by GLS and pools them across studies. All formulas match metafor 5.2.1 (`tests/test_contrast_meta_analysis.py`). `multi_study_analysis.py` runs it by default (`--study-designs`). Five of the 11 benchmark studies declare factors; eHMI-for-All's survey condition codes A–R are mapped to its eHMI × road user × distraction cells.
- **Instrument registry.** New `schemas/instruments.yaml` and `scripts/instruments.py` define items, keys, response codings, scoring rules and citations for SUS, raw NASA-TLX, Van der Laan acceptance, UEQ-S, SSQ and Körber's TiA. The SUS, TLX and AOA scores that were hard-coded in `multi_study_analysis.py` now come from the registry (the 11-study outputs are byte-identical), and UEQ-S, SSQ and TiA scores are new. Answers outside every declared coding are no longer scored. Tests check worked examples and that each score's range equals its DV's canonical range. `validate_schema.py` also validates the registry and the design file, and CI runs both.
- **DV schema 3.1.** `mental_demand` is the NASA-TLX subscale only; the Paas scale (`paas_mental_effort`, 1–9) and RSME (`rsme`, 0–150) are separate DVs because their ranges differ.
- **Self-describing outputs.** Batch runs write a Frictionless Data Package per source (column-level canonical DV, raw names, mapping source, range, direction, unit, recoding, item membership) and an RO-Crate for the run (sources with URL and DOI, code version and commit, schema and rule versions, LLM model, outputs), via the `frictionless` and `rocrate` libraries (`scripts/fair_export.py`; `--no-fair-export` to skip; `python -m scripts.fair_export <dir>` for existing runs). All 11 benchmark packages validate. The run summary now records `schema_version`, `llm_models_used` and `output_format`.
- **Parquet output.** `--output-format parquet` writes every standardized table as Parquet; the analysis reads Parquet and gives results identical to the native formats.
- **Release basics.** `CITATION.cff`; `validate_schema.py` checks dataset mapping YAMLs (targets, `scales`, `value_labels`), and CI validates every configuration file. `frictionless`, `rocrate` and `pyarrow` are core dependencies.
- **Shared pooling code.** `multi_study_analysis.pool_effects` now holds the pooling formerly inside `_run_meta_for_dv`; existing outputs are unchanged.

---

### [2026-10-17]

- **Parallel batch standardization.** `run_batch_standardization.py --workers N` (also `run_catalog_meta_analysis.py`) processes datasets in worker processes. Outputs match a sequential run file for file; LLM deduction stays in the main process and runs in file order. Mappings are now built once per (schema family, repository mapping) rather than once per file, numeric columns skip a redundant `to_numeric` pass during summarization, and delimited files are opened twice instead of three times.
- **Faster analysis loading.** `multi_study_analysis.py` reads files in a thread pool and skips files that have no canonical DV column (`load_studies(..., skip_files_without_dvs=True)`, on by default in both CLIs). Studies whose files are all skipped are kept as empty frames, so study counts and coverage percentages do not change. Schema YAML is parsed once with libyaml's C loader.
- **Trim-and-fill fixed.** `trim_and_fill` previously mirrored every study on one side of the pooled mean instead of estimating the number of missing studies. It now implements Duval & Tweedie's iterative L0/R0 estimator, matching `metafor::trimfill` (k0, side, adjusted estimate and SE) on two published datasets (`tests/test_meta_analysis_reference.py`). It uses the main analysis model (`--estimator`, default DL), picks the side automatically, and writes a row for every evaluated DV, including those with `k_imputed = 0`.
- **REML scoring step corrected.** The Fisher information in `_estimate_tau2_reml` was missing the `(Σw²/Σw)²` term. The converged estimate was already right, but convergence is now reliable. Estimates agree with direct maximisation of the restricted likelihood to within 1e-6.
- **Composite index no longer imputes empty rows.** Rows with none of the shared DVs observed (e.g. telemetry rows of a study whose questionnaire lives in another file) used to be median-imputed into all-zero z-rows. That inflated `n`, shrank SDs, and dominated the PCA. Such rows are now dropped before imputation.
- **Causal analysis.** Construct → item relationships now also come from the DV schema (`<construct>_item_NN` ids or an explicit `subitem_of`), so item → scale edges cannot enter the LiNGAM DAG. Bootstrap edge stability (via `lingam.DirectLiNGAM.bootstrap`, seeded) and the PC skeleton now run on the same observations as the reported DAG; before, they silently returned nothing whenever DVs only partly overlapped across studies.
- **Extension points.** Dataset-type profiles drive schema-family selection, LLM eligibility and item composition. Source types are registered with `register_source_type`. HTTP 502/503/504 outages are reported as `temporarily_unavailable`.
- **LLM plausibility check.** New `scripts/llm_plausibility.py` with `schemas/llm_plausibility_rules.yaml` (v1.1). It keeps demographic, ID, seed, timestamp, survey-timing, coordinate, condition, ranking and non-numeric columns away from the LLM, and rejects suggestions whose values fit no plausible scale. Rejections are logged to `llm_rejected_suggestions.json`; `--no-llm-plausibility-check` turns the check off for ablation.
- **LLM mapping benchmark.** `analyses/evaluate_llm_mapping.py` provides a leave-alias-out evaluation against curated mappings (E1: real dataset columns; E2: held-out schema aliases), with Wilson CIs. Results are in `analyses/output_llm_evaluation/`.
- **Small-sample intervals and a multiverse.** `meta_analysis_summary.csv` gains Hartung-Knapp-Sidik-Jonkman columns (`hksj_se`, `hksj_ci95_low`, `hksj_ci95_high`), validated against metafor. `analyses/multiverse_analysis.py` writes 32-specification robustness summaries and specification curves.
- **Dependencies updated to the newest releases.** Includes pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, transformers 5.20.1, torch 2.14.1, openai 3.33, sentence-transformers 6.1, plotly 7.1, streamlit 1.66, pytest 9.1 and ruff 0.17 (`torch` is now declared in the LLM extra). scipy stays at 1.16.x and lingam at 1.12.2: lingam 1.13.0 pins `scipy<=1.13.1`, which cannot be installed with numpy ≥ 2.3, and its dependency pygam requires `scipy<1.17`. CI now tests Python 3.12 and 3.14 with `actions/checkout@v7` / `setup-python@v7`; Python 3.15 is left out until PyYAML ships 3.15 wheels. The LLM prototype notebook now uses the current OpenAI client API.
- **Text-label recoding.** Mapping YAMLs can declare `value_labels` (`scripts/value_labels.py`). `psych_conflict_handling`, whose cells are textual Likert labels, previously contributed no numeric DV; it now contributes 15. The figure script derives every caption number from the run, renders alternative runs via `OPENDV_ANALYSIS_DIR` / `OPENDV_BATCH_DIR` / `OPENDV_FIGURES_DIR`, labels the three newer studies, and no longer overlaps the Fig. 2 caption with DV labels.
- **Negative answer codes and AOA scoring.** LimeSurvey `negN` codes are recoded to −N. Previously eHMI-for-All and ROADS silently lost every negative AOA answer (about 30% of cells), biasing AOA means upward. Derived AOA usefulness/satisfying scores now use each study's coding midpoint (0 for −2…+2, 3 for 1…5) instead of a hard-coded 3.
- **Definition-aware LLM shortlist (new default).** `scripts/dv_retrieval.py` ranks candidate DVs by TF-IDF similarity to their schema definitions. On held-out real columns (final 11-study evaluation), recall@8 rose from 42% to 83%, accuracy from 18% to 49%, and precision of accepted LLM mappings from 23% to 59% (`analyses/evaluate_shortlist_retrieval.py`, `analyses/evaluate_llm_mapping.py`). `--llm-shortlist fuzzy` restores the previous shortlist.
- **Research agenda.** `docs/research_agenda.md` frames the paper (research questions, evidence, ten proposed studies, threats to validity).

---

### [2026-04-19]

- **Meta-analysis provenance tracking.** `harmonized_dv_summary.csv` now includes a `mapping_source` column (`schema` / `repo_mapping` / `llm_deduced` / `blocked` / `unknown`). `meta_analysis_summary.csv` gains `estimator`, `mapping_source_categories`, `k_llm_deduced`, and `includes_llm_deduced` to make it easy to flag pooled estimates that depend on LLM inferences.
- **Sensitivity analysis flag.** `analyses/multi_study_analysis.py` adds `--exclude-llm-deduced`, which writes `harmonized_dv_summary_llm_excluded.csv`, `meta_analysis_summary_llm_excluded.csv`, and `study_vs_pool_standardized_deviation_llm_excluded.csv`. The catalog orchestrator (`scripts/run_catalog_meta_analysis.py`) emits these automatically whenever any LLM-deduced rows are present.
- **Causal discovery transparency.** `discover_causal_structure` now reports `used_method` (`complete_case` / `clique_complete_case` / `synthetic_cholesky`), `n_complete_rows`, and `synthetic_fallback`. Passing `refuse_synthetic=True` (CLI: `--refuse-synthetic-causal`) blocks the Cholesky-synthesized fallback.
- **Renamed study-vs-pool effects output.** The legacy `standardized_effects.csv` is preserved for backward compatibility, and a clearer `study_vs_pool_standardized_deviation.csv` is now also emitted to make clear that the value is a descriptive standardized distance from the pooled mean, not a contrast-based Hedges' g.
- **Additional catalog sources.** The example catalog and `sources_manifest_example.yaml` now include the Touch-Interaction-with-Road-Bumps GitHub dataset, 4TU's critical-eHMI article, and the ACM CHI 2026 supplemental ZIP. ACM downloads behind Cloudflare are now recorded as `access_restricted` rather than `failed`.

---

### [2024-07-01]

- Added `ui/` interface with uploader, download button, and column preview
- Created full schema suite: `standard_dv_mapping.yaml`, `schema_meta.yaml`, etc.
- Prototyped LLM-based suggestions via `scripts/llm_utils.py`
- Established full pipeline validation through `convert_dv.py` and `visual_validation.ipynb`
- Populated the `docs/` directory for transparency

Schema-specific updates: see `schemas/schema_changelog.yaml`
