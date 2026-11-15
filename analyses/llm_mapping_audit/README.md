# Audit of the LLM-mapped columns in the pooled results (2026-11-15)

The batch maps a study's columns to schema DVs with the study's curated mapping file, the
shared schema or header matches; a local LLM (gemma-4-E2B-it, greedy) maps what is left.
The primary analysis leaves LLM-mapped columns out. The all-mappings analysis keeps an
LLM-mapped column where the study has no column of that DV mapped without the LLM, and the
`llm` level of the cross-study relations follows the same rule. Until this audit, columns
were locked as `meta_*` only after a wrong row showed up in a figure (Curved Dragging,
FOV Snap-Turn, Gaze Platformer), which picks out the conspicuous errors. This audit checks
every LLM-mapped column the all-mappings analysis still pooled.

The verdicts (`verdicts.yaml`) and this table were written and committed before any
mapping file was changed.

## Files

| File | Content |
|---|---|
| `inventory.py` | Lists the pooled LLM-mapped columns of a batch run with the analysis's own functions (`load_column_provenance`, `_file_column_categories`, `_llm_columns_to_drop`), traces each to its raw column through `meta_view.csv`, and flags the columns behind an `llm`-level relations finding. |
| `inventory.csv` | Its output for the run `lock39` of 2026-11-14 (the run behind the outputs committed then): one row per standardized column. |
| `verdicts.yaml` | One entry per decision: the columns it covers, the verdict, the right DV and the evidence (document and section). |
| `audit.py` | Joins the two (every column must match exactly one entry), writes `audit.csv` and `precision.csv`. |
| `audit.csv` | One row per audited column: study, file(s), raw column, LLM DV, verdict, right DV, evidence, decision group. |
| `precision.csv` | In-situ precision with Wilson 95% intervals (`statsmodels` `proportion_confint`). |
| `round2/` | The same for the columns the round-1 fixes exposed (see below). |

```bash
python analyses/llm_mapping_audit/inventory.py --run-dir data/processed/lock39
python analyses/llm_mapping_audit/audit.py
```

Run from the repository root with the root on the import path. The audit read the
downloaded data with pandas, pyreadstat and pypdf in isolated interpreters (`python -I`),
zip members in memory, and ran no code shipped with the data; the R scripts were read, not
run.

## Verdicts

- `correct`: the LLM's DV is the column's construct.
- `wrong_dv`: the column measures another schema DV (`correct_target`).
- `not_a_dv`: no schema DV fits.
- `unclear`: the documentation does not settle it (`missing` says what is missing).
  `llm_dv_ruled_out` marks the unclear columns whose documentation still rules the LLM's DV
  out (e.g. SenseCobot's `NASA_3` is one NASA-TLX subscale, so not the TLX total, but which
  one is not documented).

A single questionnaire item mapped to the scale DV of its own construct counts as
`correct` (the pipeline averages a study's columns of one DV, which reproduces the
published scale mean), an item mapped to another construct's DV as `wrong_dv`, e.g. a SART
demand item mapped to the SART total or a TiA understanding item to TiA trust.

## Results

**Inventory.** The all-mappings analysis of the run `lock39` pooled 1,361 LLM-mapped
standardized columns from 22 studies. They come from 378 distinct raw columns (study x raw
column x LLM DV): eHMI-for-All's `total time` repeats in 720 per-run log files and
eHMI Optimization's six questionnaire items in 37 per-participant files, and one decision
covers each such column. 73 of the 378 (82 standardized columns) sit behind a finding of the
relations' `llm` level that includes LLM-mapped columns. The 378 columns fall into 86
decision groups (one entry of `verdicts.yaml` each; e.g. the 18 intuitiveness items of the AR
pedestrian study, the 53 EEG features of SenseCobot).

**Verdicts.** 34 correct, 49 wrong DV, 276 no schema DV, 19 unclear
(8 of the unclear ones still rule the LLM's DV out).

**In-situ precision** (share of audited columns whose LLM DV is right, Wilson 95% interval):

| Unit | Unclear counted as wrong | Unclear left out |
|---|---|---|
| Columns (378) | 34/378 = 9.0% [6.5, 12.3] | 34/359 = 9.5% [6.9, 12.9] |
| Decision groups (86) | 12/86 = 14.0% [8.2, 22.8] | 12/78 = 15.4% [9.0, 25.0] |

Columns of one decision group share one decision, so the column-level interval is too
narrow; the decision-group row is the conservative reading. Counted by standardized column
the share is 768/1,361, but 720 of those are the copies of eHMI-for-All's `total time`, so
that count says nothing about the LLM.

By study (columns):

| Study | Columns | Correct | Wrong DV | No DV | Unclear | Precision, unclear as wrong | Unclear left out |
|---|---|---|---|---|---|---|---|
| 3d_display_comp_vehicle | 25 | 5 | 16 | 4 | 0 | 5/25 = 20.0% [8.9, 39.1] | 5/25 = 20.0% [8.9, 39.1] |
| ar_pedestrian_online_2023 | 59 | 0 | 0 | 59 | 0 | 0/59 = 0.0% [0.0, 6.1] | 0/59 = 0.0% [0.0, 6.1] |
| av_acceptance_vr_ijhci24 | 9 | 0 | 0 | 9 | 0 | 0/9 = 0.0% [0.0, 29.9] | 0/9 = 0.0% [0.0, 29.9] |
| av_prevalence_ehmi_autoui24 | 1 | 0 | 0 | 1 | 0 | 0/1 = 0.0% [0.0, 79.3] | 0/1 = 0.0% [0.0, 79.3] |
| bvi_auditory_hav_autoui26 | 4 | 0 | 0 | 1 | 3 | 0/4 = 0.0% [0.0, 49.0] | 0/1 = 0.0% [0.0, 79.3] |
| concurrent_speech_2022 | 2 | 0 | 0 | 2 | 0 | 0/2 = 0.0% [0.0, 65.8] | 0/2 = 0.0% [0.0, 65.8] |
| driving_style_automated_2021 | 18 | 0 | 0 | 18 | 0 | 0/18 = 0.0% [0.0, 17.6] | 0/18 = 0.0% [0.0, 17.6] |
| ehmi_car_body_2020 | 8 | 0 | 0 | 8 | 0 | 0/8 = 0.0% [0.0, 32.4] | 0/8 = 0.0% [0.0, 32.4] |
| ehmi_for_all_chi26 | 2 | 1 | 0 | 0 | 1 | 1/2 = 50.0% [9.4, 90.5] | 1/1 = 100.0% [20.6, 100.0] |
| ehmi_optimization_chi25 | 5 | 0 | 5 | 0 | 0 | 0/5 = 0.0% [0.0, 43.5] | 0/5 = 0.0% [0.0, 43.5] |
| fact_av | 28 | 10 | 13 | 4 | 1 | 10/28 = 35.7% [20.7, 54.2] | 10/27 = 37.0% [21.5, 55.8] |
| incar_vr_gain_chi24 | 10 | 0 | 0 | 9 | 1 | 0/10 = 0.0% [0.0, 27.8] | 0/9 = 0.0% [0.0, 29.9] |
| longitudinal_usa_germany_ehmi | 37 | 18 | 15 | 4 | 0 | 18/37 = 48.6% [33.5, 64.1] | 18/37 = 48.6% [33.5, 64.1] |
| music_tempo_taskload_2022 | 22 | 0 | 0 | 16 | 6 | 0/22 = 0.0% [0.0, 14.9] | 0/16 = 0.0% [0.0, 19.4] |
| myocontrol_intent_2025 | 5 | 0 | 0 | 0 | 5 | 0/5 = 0.0% [0.0, 43.5] | – |
| nl2tap_filter_2026 | 1 | 0 | 0 | 1 | 0 | 0/1 = 0.0% [0.0, 79.3] | 0/1 = 0.0% [0.0, 79.3] |
| osf_cwd6h | 8 | 0 | 0 | 8 | 0 | 0/8 = 0.0% [0.0, 32.4] | 0/8 = 0.0% [0.0, 32.4] |
| pairsonic_cscw24 | 3 | 0 | 0 | 3 | 0 | 0/3 = 0.0% [0.0, 56.1] | 0/3 = 0.0% [0.0, 56.1] |
| perceived_risk_merging_2022 | 1 | 0 | 0 | 1 | 0 | 0/1 = 0.0% [0.0, 79.3] | 0/1 = 0.0% [0.0, 79.3] |
| roads_chi25 | 2 | 0 | 0 | 1 | 1 | 0/2 = 0.0% [0.0, 65.8] | 0/1 = 0.0% [0.0, 79.3] |
| sensecobot_fusion_2024 | 70 | 0 | 0 | 69 | 1 | 0/70 = 0.0% [0.0, 5.2] | 0/69 = 0.0% [0.0, 5.3] |
| woz_adaptation_ijhcs24 | 58 | 0 | 0 | 58 | 0 | 0/58 = 0.0% [0.0, 6.2] | 0/58 = 0.0% [0.0, 6.2] |

The ten DVs the LLM assigned most often:

| LLM DV | Columns | Precision, unclear as wrong | Unclear left out |
|---|---|---|---|
| performance | 38 | 0/38 = 0.0% [0.0, 9.2] | 0/38 = 0.0% [0.0, 9.2] |
| SUS1 | 31 | 0/31 = 0.0% [0.0, 11.0] | 0/31 = 0.0% [0.0, 11.0] |
| perceived_safety | 21 | 16/21 = 76.2% [54.9, 89.4] | 16/21 = 76.2% [54.9, 89.4] |
| tia_trust_in_automation | 18 | 4/18 = 22.2% [9.0, 45.2] | 4/18 = 22.2% [9.0, 45.2] |
| sart_understanding | 18 | 3/18 = 16.7% [5.8, 39.2] | 3/18 = 16.7% [5.8, 39.2] |
| trust_rating | 16 | 0/16 = 0.0% [0.0, 19.4] | 0/15 = 0.0% [0.0, 20.4] |
| ueq_novelty | 16 | 0/16 = 0.0% [0.0, 19.4] | 0/16 = 0.0% [0.0, 19.4] |
| efficiency | 15 | 0/15 = 0.0% [0.0, 20.4] | 0/15 = 0.0% [0.0, 20.4] |
| sts_ad_trust | 14 | 0/14 = 0.0% [0.0, 21.5] | 0/13 = 0.0% [0.0, 22.8] |
| ueq_pragmatic | 14 | 8/14 = 57.1% [32.6, 78.6] | 8/14 = 57.1% [32.6, 78.6] |

**Columns behind `llm`-level relations findings** (73): 20 correct, 8 wrong DV,
43 no schema DV, 2 unclear. The STS-AD trust that carried the within-person
finding mental demand x STS-AD trust was the 3D display study's driving-style rating (1 =
safe to 7 = dangerous), FACT-AV's driving-style rating and `complex` (a usability item in the
lab's other studies) and in-car VR gain's maximum Misery Scale rating: none is an STS-AD item.
SenseCobot's `performance` behind between-person frustration x performance was 36 EEG band
powers.

**What the LLM gets right.** Its correct answers are columns whose names already say what
they are: Faas perceived-safety items (`perceivedSafety1-4`, 16 of its 21 perceived-safety
columns), UEQ-S items 1-4 under pragmatic quality, TiA trust items, SART understanding items,
eHMI-for-All's total time. None of the columns it assigned to a SUS item, NASA-TLX
performance, efficiency, UEQ novelty, STS-AD trust or generic trust was right. Where a
block of numbered items mixes subscales (SART 1-10, TiA understanding and trust, UEQ-S pragmatic and hedonic) it put
the whole block under one subscale.

## Round 2: columns the fixes exposed

The fixes of round 1 were run through the batch (`audit39`) and the inventory repeated
before any analysis. 40 LLM-mapped standardized columns (38 raw columns) of two studies
would now have been pooled, none of which the LLM had mapped in `lock39`: locking some of a
table's columns changes the prompt the LLM sees for the others, and it answered for columns
it had left unmapped or mapped elsewhere before. All 38 have no schema DV
(`round2/verdicts.yaml`, `round2/audit.csv`): 34 EEG coherence and band-power features of
SenseCobot read as IPQ spatial presence, DALI interference, Jian trust, STS-AD item 2 and
NASA-TLX performance, and eHMI on the car body's "I have the feeling that the vehicle
detected me" read as IPQ presence and Van der Laan item 1. Precision 0 of 38, Wilson
[0.0, 9.2]%.

So a fix that locks only the columns the LLM got wrong is not stable while the table still
offers the LLM other columns. Round 2 therefore takes the LLM out of both studies: eHMI on
the car body's 16 remaining unmapped columns are locked (its curated mapping then covers
every column), and SenseCobot, whose curated mapping covers every schema DV it measured
(NASA-TLX, heart rate, RMSSD, skin conductance) and whose ~630 other columns are
physiological and facial-expression features without a schema DV, is excluded from the LLM
step in the manifest (`use_llm_deduction: false`) instead of locking each feature, which
would also have added them all to the benchmark's curated non-DV items.

## After the fixes

The rerun of 2026-11-15 (`data/processed/audit39`; `docs/reproducibility.md`) repeated the
inventory after the batch: no LLM-mapped column is pooled. The 17 LLM answers left are all
set aside beside a curated column of their DV, so the all-mappings analysis equals the
LLM-excluded one and the relations' `llm` level equals the curated one. What moved (primary
pools gained by the newly curated columns, all-mappings pools that disappeared, relations,
benchmark) is in the changelog entry of 2026-11-15.

## Not in the audit

- Columns the all-mappings analysis set aside (`analyses/output_python_full/llm_columns_set_aside.csv`)
  did not feed a pooled result and were not audited.
- Curated (non-LLM) mappings were not audited. These came up while reading the sources:
  - FACT-AV: `schemas/fact_av_mapping.yaml` maps a column `understanding_rating` to
    `sart_understanding`, but the file's column is `Understanding` (the SART understanding
    sum, 3-21), which the shared schema's alias sends to `understanding_rating`, the eHMI
    comprehension rating (1-5) the mapping meant to keep it out of. Its `SA` (the SART total
    U - (D - S)) goes to `situational_awareness` rather than `sart_total`.
  - FACT-AV's `trust` and eHMI Optimization's logged `Trust` are the two-item TiA trust
    subscale (checked on the data: the mean of the first two items), mapped by the generic
    alias `Trust` to `trust_rating`, which the schema reserves for single-item trust; the
    lab's other studies put the same subscale under `tia_trust_in_automation`.
  - eHMI Optimization's logged `Understanding` is the unreversed mean of the four TiA
    understanding/predictability items (checked on the data), mapped by the generic alias to
    `understanding_rating`. Its questionnaire items 2 and 4 correlate positively with items 1
    and 3, so whether they were reverse-keyed in this study is open (the paper names two
    inverse items, the analysis script reverses items 2 and 4).
