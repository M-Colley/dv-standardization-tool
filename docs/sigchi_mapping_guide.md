# Mapping a study for OpenDV (CHI and other SIGCHI venues)

OpenDV maps the columns of a study's open data to one vocabulary of dependent variables (DVs). Once a study is mapped, its measures can be pooled and compared with every other mapped study, whatever the original column names were. This guide shows how to map a new study. It also lists the instruments common at CHI, CSCW, UIST, AutomotiveUI, MobileHCI, CHI PLAY, IUI, DIS, HRI and other SIGCHI venues, and the pitfalls found while mapping 88 studies.

- **Vocabulary:** 206 DVs in schema 3.4 ([schema reference](schema_reference.md); also `schemas/export/opendv_dvs.csv` and `.json`).
- **Evidence that it transfers:** the 49 studies of the two held-out tests were mapped with the schema as it stood, without adding a DV. Across the 88 studies of the corpus, the mapping multiplies the measures and relations that can be pooled ([corpus analysis](opendv_corpus.md)).

## 1. Prepare the data

What makes a dataset easy to map (see also [dataset guidelines](dataset_guidelines.md)):

- **One row per participant × condition** (or × trial), with a participant column and one column per factor. Wide exports (one column per condition) can be reshaped with the manifest's `wide_to_long` table option.
- **Item-level questionnaire answers**, not only the scores. The analysis computes SUS, raw NASA-TLX, UEQ-S, SSQ, Van der Laan and TiA scores from their items (`schemas/instruments.yaml`), so they are scored the same way in every study.
- **A codebook** that gives, for every measure: the instrument, the item wording, the response range and anchors, reverse-keyed items, units, and how scores were computed.

## 2. Add the study to a manifest

```yaml
sources:
  - source_id: my_study_chi27          # lower_snake_case, unique
    source_type: local_path            # or github_repo, osf_project, web_dataset
    location: data/raw/my_study_chi27
    publication_doi: 10.1145/xxxxxxx.xxxxxxx
    license: CC-BY-4.0
    include_globs: ['**/results.csv']
    mapping_path: schemas/my_study_chi27_mapping.yaml
    # table_options: {'**/results.xlsx': {sheet: Data, header: 1}}
```

`analyses/heldout_2/sources_manifest.yaml` has examples of every table option (sheets, header rows, `wide_to_long`, `pivot`, `rename`, MATLAB variables).

## 3. Write the mapping

Copy `schemas/templates/study_mapping_template.yaml` to `schemas/<source_id>_mapping.yaml`. Each line maps a raw column name (case-insensitive) to a target:

| Target | Use for |
|---|---|
| a DV id from the [schema reference](schema_reference.md), e.g. `mental_demand`, `usability`, `task_completion_time` | dependent measures on the DV's metric |
| an item id, e.g. `SUS1`, `ueq_s_1`, `tia_k09` | questionnaire items; the analysis scores the instrument when all items are mapped |
| `meta_*`, e.g. `meta_participant_id`, `meta_condition_display`, `meta_demographic_age`, `meta_derived_<name>` | everything else: identifiers, design factors, demographics, traits, checks, and measures that have no schema DV |

Write in the file header what you looked up: the paper, the data record, the files read and why others were left out, and for each scale its range, anchors and keying. The mappings in `schemas/` and `analyses/heldout*/mappings/` are worked examples.

Also lock every measure that has no schema DV as `meta_*`. A column left unmapped is offered to the LLM fallback, if it is enabled. In the benchmark only 21–26% of the DV mappings the LLM accepted were right (`analyses/output_llm_evaluation/`).

## 4. Check, run, inspect

```bash
opendv check-mapping schemas/my_study_chi27_mapping.yaml
opendv batch --manifest my_manifest.yaml --output-dir data/processed/my_run --disable-llm-deduction
opendv analyze --input-dir data/processed/my_run/standardized --output-dir results/my_run
```

`check-mapping` reports:

- targets that are not in the schema (usually a typo; the closest DV ids are listed);
- names that a shared alias or a metadata or sensor alias would capture first;
- names on the batch's never-map list;
- which instrument scores the analysis will compute.

After the batch, look at `meta_view.csv`, which has one row per column with its DV, mapping source and value range, and at `unknown_alias_summary.json`, which lists the columns nothing matched.

## 5. Common SIGCHI instruments and their DVs

| Instrument | DV ids (range) |
|---|---|
| SUS | `SUS1`–`SUS10` (1–5) → `usability` (0–100, scored by the analysis) |
| UMUX, UMUX-Lite | `umux_score`, `umux_lite_score` (0–100) |
| Single Ease Question | `seq_rating` (1–7) |
| NASA-TLX (raw) | `mental_demand`, `physical_demand`, `temporal_demand`, `performance`, `effort`, `frustration` (0–20 marks; 1–21 and 0–100 are recognised) → `TLX_SCORE` (0–100) |
| Paas mental effort, RSME | `paas_mental_effort` (1–9), `rsme` (0–150) |
| UEQ, UEQ-S | `ueq_attractiveness` … `ueq_novelty`, `ueq_pragmatic`, `ueq_hedonic` (−3..3); UEQ-S items `ueq_s_1`–`ueq_s_8` |
| AttrakDiff | `attrakdiff_pq`, `attrakdiff_hqs`, `attrakdiff_hqi`, `attrakdiff_att` |
| User Engagement Scale (SF) | `engagement_level` (1–5) with `ues_focused_attention`, `ues_perceived_usability`, `ues_aesthetic_appeal`, `ues_reward` |
| Creativity Support Index | `csi_score` (0–100) |
| IPQ | `ipq_general_presence` (0–6), `ipq_spatial_presence`, `ipq_involvement`, `ipq_experienced_realism` |
| Slater–Usoh–Steed presence | `slater_usoh_steed_presence` (mean, 1–7) |
| Avatar embodiment | `embodiment_score` (1–7) |
| SSQ, VRSQ, FMS, MISC | `ssq_total`, `ssq_nausea`, `ssq_oculomotor`, `ssq_disorientation`; `vrsq_score`; `fms_score` (0–20); `misc_rating` (0–10) |
| PXI | 11 constructs `pxi_*` (−3..3), e.g. `pxi_mastery`, `pxi_immersion`, `pxi_enjoyment` |
| GEQ (core) | `geq_competence`, `geq_immersion`, `geq_flow`, `geq_tension`, `geq_challenge`, `geq_negative_affect`, `geq_positive_affect` (0–4) |
| IMI | `imi_interest_enjoyment`, `imi_perceived_competence`, `imi_effort_importance`, `imi_pressure_tension`, `imi_perceived_choice`, `imi_value_usefulness` (1–7) |
| Flow Short Scale | `flow_fks_total`, `flow_fks_fluency`, `flow_fks_absorption` (1–7) |
| PANAS, SAM | `panas_positive`, `panas_negative` (1–5); `valence`, `arousal`, `dominance` |
| Godspeed | `godspeed_anthropomorphism`, `godspeed_animacy`, `godspeed_likeability`, `godspeed_perceived_intelligence`, `godspeed_perceived_safety` |
| TAM, UTAUT | `tam_perceived_usefulness`, `tam_perceived_ease_of_use`; `utaut_*` |
| Trust in automation | `jian_trust`; TiA (Körber) `tia_*`; `trust_rating` for single items |
| Situation awareness, workload (driving) | `sart_*`, `situational_awareness`, `dali_*` |
| Text entry | `words_per_minute`, `kspc`, `msd_error_rate`, `total_error_rate` |
| Pointing | `fitts_throughput` (bits/s), `task_completion_time`, `error_rate` |
| Generic performance | `task_completion_time` (s), `task_success` (0–1), `error_rate` (%), `efficiency` |

## 6. Pitfalls seen in 88 studies

- **Raw names mislead.** Across the corpus, 84 normalized raw names appear in at least two studies; 13 of them mean different DVs in different studies. For example, `A1` is a Van der Laan item in one study and an efficiency measure in another, and `TLX 1.2` is mental demand in one and physical demand in another. Mental demand alone goes by 34 different names in 40 studies. Map every measure explicitly.
- **Shared aliases win.** A name that is also an alias in the shared DV schema is mapped by the schema unless the manifest entry sets `override_standard_aliases: true`. Metadata and sensor aliases can match first too: osf_cwd6h's `ArduinoData1` is mapped to perceived risk in its mapping, but the batch reads it as a sensor channel. Of the 85 study mappings in the corpus, this is the only DV lost this way; `check-mapping` flags it. Run `check-mapping`.
- **Scores off the canonical metric.** Map an authors' score to the score DV only when it is on that DV's metric. Otherwise map the items and let the analysis score them, and lock the authors' score as `meta_derived_*`. For example, ARTV's UES scores were computed on 1–7 with a negated perceived-usability subscale.
- **Ranges and codings.** NASA-TLX subscales come as 0–20, 1–21 or 0–100. UEQ items come as −3..3 or 1–7. A "SUS" column may be the System Usability Scale or the Slater–Usoh–Steed presence questionnaire. State the coding in the mapping header. The analysis flags values outside each DV's canonical range in `data_quality_report.csv`.
- **Reverse-keyed items.** Declare them with `reverse_items` (columns, `scale_range`, reason), as `schemas/artv_ornaments_2025_mapping.yaml` does.
- **Wide and stacked tables.** Use the manifest's table options rather than editing the data.

## 7. Proposing a new DV

Add a DV when a validated instrument or a common measure has none, as schema 3.4 did for 43 instruments of SIGCHI venues. Each entry in `schemas/standard_dv_mapping.yaml` gives:

- an `id` and a `label`;
- a `cluster` from `schemas/thematic_clusters.yaml`;
- `measurement`: category, unit, scale type, direction and canonical range;
- the `instruments` with citations;
- `notes` on scoring;
- for subscales, `subitem_of`.

Keep its aliases specific to the instrument (`umux_lite`, not `ul`). Shared aliases override study mappings, so a short alias can capture unrelated columns in other studies. Before committing:

1. `opendv validate schemas/standard_dv_mapping.yaml`;
2. `python -m opendv.schema_export` to regenerate the reference;
3. add an entry to `schemas/schema_changelog.yaml`;
4. a batch run of the corpus before and after: no existing column may change its DV (see [the corpus analysis](opendv_corpus.md)).
