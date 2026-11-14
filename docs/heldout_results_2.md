# Results: second held-out test of the cross-study relations

Run on 2026-11-14, as pre-registered in `docs/preregistration_heldout_relations_2.md`. Order of commits:

1. `4f4d5d5`: MATLAB reader (`.mat` files in the reader registry).
2. `0a27274`: pre-registration, screening decisions, pinned files, targets and decision code, before any download.
3. Download of the pinned files (not committed; `data/raw/heldout_2/` is ignored).
4. `aba7c62`: three reader options needed by the downloaded files' layouts, added before the freeze (allowed by the pre-registration).
5. `31b9a18`: mappings, study designs, table options and the batch manifest frozen ("Freeze second held-out mappings").
6. `294d548`: a fault fix in the analysis code (standardized `.tsv` files), after which the run was repeated (see Deviations).
7. Then the batch, the relations, the LLM benchmark and the decisions (`analyses/heldout_2/run.sh`).

Outputs:

- `analyses/heldout_2/results.json` (decisions);
- `analyses/heldout_2/relations/` (relations per study and pooled, harmonization ladder);
- `analyses/heldout_2/runs/definition/` (E1 benchmark for H3);
- `analyses/heldout_2/exploratory/` (the analyses labelled exploratory in the pre-registration).

To rerun: `bash analyses/heldout_2/run.sh`, then `python -m analyses.heldout_2.exploratory`.

## In short

| Hypothesis | Outcome |
|---|---|
| H1 (primary): mental demand × perceived safety is negative within persons | **Not testable:** no study of this set measures both (expected) |
| H2 (primary): the three core NASA-TLX pairs are positive within persons | **Supported:** r = 0.46 to 0.59, k = 7 to 8, Holm-corrected one-sided p ≤ .005 |
| H2b: performance is the weakest-linked TLX subscale | **Not met:** physical demand is (mean r 0.22 vs. performance 0.31) |
| H3: header matching before the LLM beats the LLM alone | **Supported:** 84 vs. 65 of 273 DV columns, 19 vs. 0 discordant, p = 4 × 10⁻⁶ |
| H4: the other 33 findings recur | **27 of 32 testable targets replicate** (BH q < .05); all 32 have the predicted sign |

H2 failed narrowly in the first held-out test (Holm p = .052 for each pair). On this independent set, with more studies, it holds clearly.

## Data

### Screening

- The 79 "to check" datasets of the 2026-11-05 search and the five datasets the first test dropped before download were checked from metadata, READMEs, zip listings and papers only (`analyses/candidate_search/check_decisions.csv`).
- **36 qualified, 48 did not.** None of the five first-test drops qualified: their data hold no readable table of schema DVs (PNG figures, raw time series), or no participants of their own.
- The pinned files of the 36 (42.4 MB) were downloaded and verified against their repository checksums; every file but one dataset's arrived.

### Dropped after download (6 of 36)

Six datasets were dropped before the freeze. Each decision rests on the files' structure and documentation, never on a statistic between columns. The reasons are in the header of `analyses/heldout_2/sources_manifest.yaml`.

| Dataset | Reason | Rule |
|---|---|---|
| eeg_sonification_2025 | Dryad answered HTTP 401 for every file | pre-registered (cannot be downloaded) |
| immersive_analytics_scale_2026 | the table has no participant identifier | pre-registered (no participant-level data) |
| cyclist_overtaking_2025 | the documented end-of-study perceived-safety and stress items are not in the pinned Appen export (they are in an unpinned 137 MB file); the simulator questionnaire adds only an acceptance item | pre-registered (fewer than two schema DVs, contrary to the documentation) |
| ceap_360vr_2021 | the NASA-TLX is stored as seven numbers per block whose items are documented nowhere, so no subscale can be labelled; the other measures are space-separated strings in nested JSON | deviation (see below) |
| child_pedestrian_vr_2025 | the questionnaire items (T2–T13) are undocumented; the motion-capture rows code participant and trial together | deviation (see below) |
| audiobook_driving_2024 | the lane-keeping and distance measures are pivot-table blocks side by side, each with its own participant column; the one plain table holds a single DV | deviation (see below) |

The **30 analysed datasets** have 7 automotive and 23 other-domain studies, 2,422 participants with between-person relations and 949 in the 18 studies with repeated measures.

### Columns

- The 30 datasets have 2,238 columns in the files the manifest reads.
- The frozen mappings label 363 as DVs and 977 as non-DV (`meta_*`); 42 identifier and condition columns are caught by the never-map blocklist. The rest are left unlabelled: DVs without a fitting schema DV, items whose keying or direction is undocumented, derived scores and figure data.

## Pre-registered hypotheses

All tests use the `curated` level and within-person correlations (repeated-measures r, Bakdash & Marusich 2017), pooled on Fisher's z with REML and the Hartung-Knapp-Sidik-Jonkman interval, decided by `opendv.analysis.heldout_replication` (H1, H2, H2b and H3 by the first test's own functions).

### H1: workload and perceived safety: not testable on this set

No study of this set measures `mental_demand` together with `perceived_safety` (k = 0), as the pre-registration expected from the documentation. The one dataset that documented perceived safety repeatedly (child_pedestrian_vr_2025) has no NASA-TLX and was dropped. F1 remains an exploratory finding of the development studies (k = 4).

### H2: structure of the NASA-TLX: supported

Eight studies measure at least two of the three core subscales repeatedly: exergames in children, human-swarm supervision, layer interaction on elastic displays, the navigable world-in-miniature, steering gain, the VR classroom under noise, VR shopping, and (effort × mental demand only) the MR airplane display layouts.

| Pair | k | Pooled r | HKSJ 95% CI | 95% PI | I² | p (one-sided) | p (Holm) |
|---|---|---|---|---|---|---|---|
| effort × frustration | 7 | 0.48 | [0.24, 0.67] | [−0.26, 0.87] | 90% | 0.0018 | 0.0036 |
| effort × mental demand | 8 | 0.59 | [0.39, 0.74] | [−0.13, 0.91] | 92% | 0.0003 | 0.0009 |
| frustration × mental demand | 7 | 0.46 | [0.17, 0.67] | [−0.40, 0.89] | 93% | 0.0046 | 0.0046 |

Per study (r, n participants):

| Study | effort × frustration | effort × mental demand | frustration × mental demand |
|---|---|---|---|
| exergame_children_2025 (21) | 0.46 | 0.58 | 0.48 |
| human_swarm_pfm_2025 (180) | 0.36 | 0.41 | 0.18 |
| layer_interaction_elastic_2022 (23) | 0.55 | 0.49 | 0.32 |
| mr_airplane_layouts_2021 (24) | – | 0.78 | – |
| navigable_wim_2026 (36) | 0.56 | 0.50 | 0.50 |
| steering_gain_2022 (24) | 0.14 | 0.27 | 0.14 |
| vr_classroom_noise_2025 (24) | 0.33 | 0.59 | 0.49 |
| vr_shopping_demands_2026 (22) | 0.79 | 0.85 | 0.82 |

**All three pairs are significant after Holm, so H2 is supported.** All 22 study-level correlations are positive. The studies still disagree on the size (I² = 90 to 93%), and the prediction intervals include 0, as in the first test. With seven or eight studies the HKSJ interval has six or seven degrees of freedom, enough to establish the sign despite that disagreement.

### H2b: performance is the weakest-linked subscale: not met

Mean pooled within-person r of each subscale with the other five:

| Subscale | Mean r |
|---|---|
| physical demand | 0.22 |
| temporal demand | 0.31 |
| performance | 0.31 |
| frustration | 0.36 |
| mental demand | 0.41 |
| effort | 0.44 |

Physical demand, not performance, is the weakest-linked subscale on this set; performance ties with temporal demand. In the development set and the first held-out test performance was clearly last (0.29–0.45 and 0.01). Two things weigh on this set's value:

- performance was left unlabelled wherever its direction was undocumented (four studies), so it rests on fewer studies (k = 5 or 6 per pair) than the other subscales;
- several tasks of this set (VR shopping, steering, the noise classroom) vary physical effort little, so physical demand varies little within persons.

### H4: replication of the other 33 targets: 27 of 32 testable targets replicate

One target was not testable: Van der Laan satisfying × usefulness, which only the human-swarm study measured. Of the 32 testable targets, 27 replicate (one-sided HKSJ test in the predicted direction, Benjamini-Hochberg q < .05 over the 32), and **all 32 pooled correlations have the predicted sign**.

| Association | Targets | Testable | Replicated | Predicted sign |
|---|---|---|---|---|
| within persons (12 NASA-TLX pairs) | 12 | 12 | 11 | 12 |
| between persons (TLX pairs, SUS × workload, UEQ, Van der Laan) | 21 | 20 | 16 | 20 |

The five that do not replicate:

| Target | k | Pooled r | HKSJ 95% CI | q |
|---|---|---|---|---|
| between: effort × usability | 3 | −0.25 | [−0.82, 0.58] | 0.18 |
| between: mental demand × usability | 3 | −0.18 | [−0.50, 0.18] | 0.085 |
| between: performance × physical demand | 7 | 0.13 | [−0.08, 0.34] | 0.091 |
| between: mental demand × performance | 8 | 0.21 | [−0.05, 0.44] | 0.058 |
| within: performance × physical demand | 5 | 0.08 | [−0.04, 0.20] | 0.077 |

- **Workload and usability.** Three of the five SUS-by-workload targets replicate (TLX total, frustration, temporal demand); effort and mental demand do not. They rest on three studies (human-swarm supervision, PipeBlocks, navigable WIM; the TLX total also on debugging CPS), so the HKSJ test has two degrees of freedom. The development finding that generalized (mental demand × usability, r = −0.42, prediction interval excluding 0) is weaker here: r = −0.18.
- **Performance.** Three of the five misses involve the performance subscale, which links weakly to physical demand and mental demand, as in the development set (there between persons r = 0.15 with physical demand and 0.13 with mental demand, within persons 0.27 with physical demand).
- **UEQ hedonic × pragmatic** replicates over four studies (r = 0.44, I² = 0%), so does the TLX total × usability (r = −0.39, k = 4, I² = 0%).
- Several within-person targets replicate under the one-sided BH rule although their two-sided 95% interval touches 0 (e.g. effort × performance, [−0.01, 0.68]); the pre-registered test is one-sided.

Every target with its estimate, studies and q is in `analyses/heldout_2/results.json`.

### H3: header matching before the LLM: supported

E1 has **273 DV columns** (at most 40 per source) and 198 non-DV columns. The settings were those of the first test: `definition` shortlist, `google/gemma-4-E2B-it`, plausibility rules 1.3.

| | LLM alone | Header matching, then the LLM |
|---|---|---|
| Correct mappings (of 273) | 65 | 84 |
| Precision of accepted DV mappings | 26.9% [21.7, 32.8] | 34.7% [29.0, 40.9] |

- **Test.** Paired exact McNemar: 19 columns correct only with header matching, 0 only without it, p = 3.8 × 10⁻⁶. **H3 is supported.**
- **What header matching did.**
  - It answered 28 DV columns, all correctly, and no non-DV column.
  - The 28 are question-text or labelled headers in three studies: the SUS statements of the ChatGPT role-play study (10) and of PipeBlocks (6), PipeBlocks' NASA-TLX questions (6), and the "Mental Demand (NASA TLX)"-style headers of the P300 BCI study (6).
  - The LLM alone got 9 of these 28 right, and none of the 16 SUS statements: it read 7 as TAM perceived ease of use, 3 as the SUS total and 6 as unrelated DVs.
- **The LLM alone remains weak on new studies**: 65 of 273 DV columns (24%); in the first test 77 of 238 (32%).
- One paper that the LLM context fetches by DOI arrived as an HTML page instead of a PDF (pypdf warning in the run log, during the risky-driving study), so that study's E1 prompts probably carried no paper text. Both arms share the same LLM answers, so this does not bias the H3 comparison.

## Exploratory (not pre-registered as tests)

- **H2 without any single study** (`exploratory/h2_leave_one_out.csv`). The one-sided p of each pair stays at or below .013 whichever study is left out; no single study carries the result. VR shopping has the largest correlations (0.79–0.85), steering gain the smallest (0.14–0.27).
- **No new finding.** Every curated finding of this set is one of the targets; the set produced no finding outside them (`results.json`, `new_findings_exploratory`).
- **Both held-out sets pooled** (k = 12–13 per core pair; `exploratory/targets_pooled.csv`). This follows the first test's narrow miss, so it is optional continuation and decides nothing.

  | Pair | Pooled r | HKSJ 95% CI | 95% PI |
  |---|---|---|---|
  | effort × frustration | 0.47 | [0.30, 0.61] | [−0.19, 0.84] |
  | effort × mental demand | 0.60 | [0.43, 0.73] | [−0.15, 0.91] |
  | frustration × mental demand | 0.44 | [0.24, 0.61] | [−0.34, 0.86] |

  Over both held-out sets, 36 targets are estimable, all with the predicted sign, 31 with one-sided p < .05.
- **All studies pooled** (development, first and second held-out sets; 20–21 studies per core pair). All 37 targets have the predicted sign, 36 with one-sided p < .05 (the exception is within-person performance × physical demand, r = 0.13). The core pairs: effort × frustration r = 0.52 [0.43, 0.61], effort × mental demand 0.59 [0.49, 0.68], frustration × mental demand 0.47 [0.36, 0.57]. The prediction intervals of effort × frustration and effort × mental demand now exclude 0 (0.03 to 0.81, 0.05 to 0.87). Workload × usability between persons, over seven to eight studies: TLX total −0.41 [−0.48, −0.34], frustration −0.51, temporal demand −0.36, effort −0.37, mental demand −0.33; the TLX total, frustration and temporal demand generalize. Mental demand × perceived safety within persons stays at its four development studies (r = −0.44).
- **The harmonization ladder recurs.** With identical raw column names the 30 studies pool 16 pairs over three or more studies, none of them across constructs; the shared schema and header matching reach almost nothing beyond (one pair). With the study mappings, 32 between-person and 22 within-person pairs pool over three or more studies, 10 of them across constructs, and 22 are findings (`relations/harmonization_ladder.csv`). As on the development and first held-out sets, the curated mappings make the cross-construct relations estimable.

## Deviations and decisions made during the test

- **Reader options added before the freeze** (`aba7c62`), as the pre-registration allows, with tests:
    - `wide_to_long`: repeated measures stored side by side (one column per condition) become one row per participant and condition. Without it, nine datasets (steering gain, debugging CPS, exergames, VR shopping, VR classroom noise, navigable WIM, subtitles, AR telepresence, Spot) would have entered only between persons.
    - `sheet` may list several sheets, which are stacked (VR shopping, the prompting study, VR-SPIRIT, subtitles).
    - `row_id` numbers rows for tables with one row per participant and no identifier (the prompting study's UEQ tool sheets; SUS-DE-Pos, whose survey serial numbers restart between its desktop and mobile runs).
- **Three datasets dropped where a reader would have been needed for DVs that bear on no target** (CEAP-360VR, child pedestrian VR, audiobook driving). The pre-registration allows reader features but lists no rule for when not to add one. The rule applied, decided from file structures and target relevance before any statistic: add a reader feature when DVs relevant to a pre-registered target need it, not otherwise. CEAP-360VR's NASA-TLX could not have been labelled anyway (undocumented item order); child pedestrian's and audiobook's relevant DVs (perceived safety, lane keeping) are no target pair of this set.
- **A fault fixed in the analysis code, and the run repeated** (`294d548`). `multi_study_analysis` read every standardized file that was not `.csv` or `.parquet` as Excel, so `construct_relations` stopped at the teleoperation study's `.tsv` output. `_read_file` and `_read_header` now read `.tsv` as tab-separated text (with a test); `run.sh` was then run again from the start. This is the procedure's "fault fixed, step repeated"; the pre-registration otherwise froze the analysis code.
- **Mapping judgments that limit what some studies contribute** (each documented in its mapping file):
    - the NASA-TLX performance item was left unlabelled where its direction is undocumented (Build PC, Build 3D, citizen science, exergames); it was reversed where its wording and anchors say high = good (human-swarm, PipeBlocks, VR classroom noise);
    - the VR classroom noise study's item order is inferred from the paper's dimension list and factor structure, not stated;
    - SUS-DE-Pos: the paper analyses 877 of 902 respondents, but no column flags the 25 excluded, so all 902 enter;
    - IPQ, AttrakDiff and other items whose keying is undocumented were left unlabelled.
- **Performance in the H2b comparison** rests on fewer studies than the other subscales because of the rule above.
- **The archive limit for whole downloads** was 100 MB instead of the first test's 1 GB, as pre-registered.
- **A batch run before the freeze** checked that every mapping key reached its column and every study resolved its participant ids (as in the first test). It found summary rows (means, SDs, sample sizes) in three workbooks, which the mappings exclude by `exclude_rows`, and seven participant columns on the never-map list, added to the designs under their raw names. No relation between columns was computed in that run.

## Limitations

- **Who mapped.** The mappings come from the AI assistant that ran the exploratory analysis; the screening used AI subagents reviewed by the same assistant. No human rater checked either.
- **The targets are not independent.** The fifteen NASA-TLX pairs share studies; the replication count is descriptive.
- **Heterogeneity remains large.** The NASA-TLX subscales covary in every study, but by how much differs widely (I² ≈ 90%), and the prediction intervals of most pairs on this set still include 0.
- **Domain.** 23 of the 30 studies come from outside the automotive domain, and most NASA-TLX studies are VR, HRI or interaction-technique studies; the first held-out set was mostly automotive. The core TLX relation recurs across both.
