# Pre-registration: confirmatory test of the hierarchical shortlist

Written 2026-10-26, before any of the datasets below was downloaded, labelled or passed to a retrieval model or the LLM. The commit that adds this file precedes every commit that adds labels or results; `git log` is the timestamp.

## Background

When a column has no curated or schema alias, OpenDV-HCI asks a local LLM to choose a DV from a shortlist of candidates. On the 11 development studies (E1, 101 real DV columns, plausibility check on):

- the default `definition` shortlist (TF-IDF over DV definitions) mapped 52 columns correctly;
- `hierarchical` (`dense_h`: dense + TF-IDF fusion that never lists a scale together with its own items) mapped 65 (paired exact McNemar p = 0.035);
- `mixed_h` mapped 69 (p < 0.001).

These numbers are post hoc: the pre-specified test of 2026-10-25 failed on its "no lost mappings" criterion, a data bug was then fixed, and the shortlists were designed on the same columns (`docs/research_agenda.md`, section 2.4). This test replicates the comparison on studies the method has never seen.

## Data

Seven open datasets (`analyses/confirmatory/sources.yaml`, with download URLs pinned to a commit or repository file and the published checksums), all CC BY 4.0, none used in developing OpenDV-HCI, none from automotive research:

| source_id | study | domain |
|---|---|---|
| `agency_ux_tochi22` | Sense of Agency and UX (TOCHI 2022) | input devices |
| `pxi_validation_chi24` | Player Experience Inventory validation (CHI 2024) | games |
| `hiaudio_eval_2026` | Hi-Audio platform evaluation (2026) | music software |
| `xai_trust_scales_chi23` | Trust scales for AI (CHI EA 2023) | explainable AI |
| `sonomancer_chiplay25` | Sonomancer audio-AR game (CHI PLAY 2025) | audio AR games |
| `vr_grab_techniques_icom20` | VR selection and manipulation techniques (Zenodo 4553165) | VR interaction |
| `tamago_supervision_2026` | TAMAGO gamified supervision interfaces (Zenodo 23273694) | education software |

Only the listed files are downloaded. If a file is unavailable or fails its checksum, the source is dropped and the drop is reported.

## Gold labels

- **Who.** Claude (the AI assistant operating this repository) labels every column of the data files, using each dataset's codebook, analysis script, questionnaire printout or data dictionary, and the values. The user chose this over independent human labelling; it is the main limitation (see below).
- **When.** Labels are written before any shortlist or LLM is run on these data, and committed as one commit ("Freeze confirmatory gold labels") before the evaluation starts. No retrieval ranking or LLM answer for these columns is looked at before that commit.
- **Where.** One mapping file per source, `analyses/confirmatory/mappings/<source_id>_mapping.yaml`, in the format of the repository's dataset mappings. The batch runner applies them as curated mappings, so the harness reads them as gold labels exactly as in E1.
- **Rules.**
  - A DV column gets the most specific schema DV id (`schemas/standard_dv_mapping.yaml`): the item-level id when the schema itemizes the instrument (e.g. `SUS3`, `tia_item_02`), otherwise the construct.
  - Columns that are not DVs (ids, conditions, demographics, timing, attention checks, free text, derived ranks) get a `meta_*` label, as curated non-DV labels in E1.
  - DV columns with no fitting schema DV are left unlabelled and listed in the results; they are not E1 items.
  - The schema is not changed during the test.
  - Labels are not revised after the runs. Label errors found later are reported, not corrected.

## Procedure

1. `python -m opendv.analysis.confirmatory fetch` downloads and verifies the files.
2. Labels are written and frozen (commit).
3. A batch run without the LLM over `analyses/confirmatory/sources_manifest.yaml` (`local_path` sources) provides the gold items.
4. `opendv.analysis.evaluate_llm_mapping` (E1, leave-alias-out) is run three times, once per shortlist: `definition`, `dense_h`, `mixed_h`. Fixed settings:
   - `--max-per-source 40` (per source and label kind; this caps the large PXI set) and `--seed 20261017`, both as in E1;
   - `--e2-samples 0`: E2 uses only the schema and is not part of this test;
   - `--models google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`;
   - plausibility rules version 1.1;
   - code at the commit that freezes the labels.

   Each shortlist is run once. If the harness fails, the fault is fixed and all three runs are repeated, and this is reported.

## Hypotheses and analysis

- **Primary (H1).** With the plausibility check on, `dense_h` maps more DV columns correctly than `definition`.
  - Unit: E1 DV columns (gold label is a DV).
  - Correct means the checked answer equals the gold label.
  - Test: paired exact McNemar test on the discordant columns, two-sided, α = 0.05.
- **Secondary (H2).** The same comparison for `mixed_h` vs. `definition`. H1 and H2 are corrected together with Holm's method; H1 is decided at its own α as the primary test.
- **Decision rule.** The pipeline default becomes `hierarchical` if `dense_h` has more correct mappings than `definition`, the primary test gives p < 0.05, and the precision of accepted DV mappings for `dense_h` is at least that of `definition`. Otherwise `definition` stays the default and the result is reported as a failed replication.
- **Also reported for each shortlist** (descriptive, with Wilson 95% CIs):
  - gold DV among the candidates (recall);
  - precision of accepted DV mappings;
  - abstentions;
  - curated non-DV columns mapped to a DV;
  - true mappings lost to the check, counted only among columns that reach the LLM, because columns screened out before the LLM do so under every shortlist.
- **Exploratory, labelled as such:**
  - results per dataset;
  - correctness at the instrument level (right instrument, wrong item or scale level).

## Power

On the development columns, about a third of the columns were discordant between `dense_h` and `definition`, split 23 : 10. With about 150 DV columns, about 50 discordant pairs are expected. At a 0.7 : 0.3 split, the exact two-sided test then has power of roughly 0.85. A smaller true effect may be missed, which the report will state.

## Limitations stated in advance

- **Labeller.** The gold labels come from the same AI assistant that ran the development experiments, not from independent human raters. Label quality is therefore unverified, and agreement (Cohen's κ) cannot be reported. A second, human rater on a sample is recommended before publication.
- **One model.** One small model (Gemma 4 E2B) is tested, and the results may not transfer to larger models.
- **Instrument mix.** The studies lean on NASA-TLX, SUS, AttrakDiff and PXI, so instruments the schema does not cover produce unlabelled columns.
