# What harmonization reveals: relations between constructs across studies

`opendv/analysis/construct_relations.py` asks what synchronizing the dependent variables of open studies makes visible that no single study shows, and which step of the pipeline makes it visible.

## Why relations, not means

Pooling study *means* assumes every study measures a DV on the same metric (scalar measurement invariance). None of the scales shared by the development studies reaches it (`research_agenda.md` §4), and pooled means move with every scale-harmonization choice (§2.3).

Correlations need less. They do not change under a linear rescaling, so they stay comparable when anchors, ranges and units differ. Relations between constructs are therefore the cross-study quantity that harmonized open data can support today.

## Method

1. **Provenance of every column.** Each DV column of each standardized file is traced back to its raw column through `meta_view.csv`. It is assigned the step that mapped it:

    | Level | A DV column is usable when it was mapped by ... |
    |---|---|
    | `names` | nothing: studies are pooled by identical raw column names (the baseline without harmonization) |
    | `automatic` | the shared DV schema (the raw name is a schema alias of the same DV), question-text header matching, or the instrument registry's scoring of such items |
    | `curated` | a study's own mapping file, including the value labels, reverse keying, missing-value codes and unit and scale conversions only a mapping file declares |
    | `llm` | the local-LLM fallback, for the columns no mapping covers |

    Each level includes the ones above it. At `names` the baseline is told which columns are DVs (those the deterministic pipeline maps), which favours it.
2. **Participants.** Columns mapped to the same DV within a study (wide repeated measures, items of one construct) are averaged per row, then per participant. Participant columns for all 39 studies are declared in `schemas/study_designs.yaml`. At the `llm` level this also averages a study's LLM-mapped column with its curated column of the same DV. The pooled-means analysis stopped doing that on 2026-11-14, when it began to exclude LLM-mapped columns one by one, as the levels here always have; that change left every relation table unchanged (see the changelog).
3. **Per study.** The analysis computes two correlations for every pair of construct-level DVs:
    - the Pearson correlation over participants (between-person r, `scipy.stats.pearsonr`);
    - for repeated-measures studies, the repeated-measures correlation of Bakdash & Marusich (2017) on the rows (within-person r).

    Single questionnaire items are left out; NASA-TLX subscales count as constructs.
4. **Across studies.** Correlations are pooled per pair on Fisher's z with the random-effects code the other analyses use (REML, Hartung-Knapp-Sidik-Jonkman interval, prediction interval) and back-transformed.
5. **Findings.** A pooled relation is a *finding* when all of the following hold:
    - it rests on at least three studies;
    - its HKSJ 95% interval excludes 0;
    - its Benjamini-Hochberg q < .05 among all pooled pairs of the same level and association type;
    - the two DVs are not scored from shared items (a TLX subscale and the TLX total).

    A finding *generalizes* when its 95% prediction interval excludes 0 as well, so a new study is expected to show the same sign. Pairs from one instrument (two UEQ-S scales) are findings too but are reported apart from cross-construct relations.

## Does the LLM stand in for curation?

The ladder above shows what the LLM adds on top of curated mappings. A tool for new datasets needs a different answer: what happens without curated mappings, with the LLM in their place.

`--llm-replay-items` answers it from the leave-alias-out benchmark (`evaluate_llm_mapping.py`, E1). There, every column a study mapping decided is shown to the pipeline's LLM step with that mapping hidden. The *replay* keeps the automatic and LLM columns. Every column a study mapping decided takes the LLM's answer: a DV column the DV the LLM chose, or nothing when it abstained or the plausibility check rejected the answer; a column the mapping locked as non-DV (`meta_*`) the DV the LLM gave it. The replay's findings are then compared with the curated findings (`llm_replay_conclusions.csv`):

- **confirmed:** the same pair is a curated finding with the same sign;
- **sign flipped:** the same pair is a curated finding with the opposite sign;
- **unsupported:** the pair is not a curated finding.

This evaluates LLM harmonization by the conclusions it leads to, not by column accuracy.

## Running it

```bash
python -m opendv.analysis.construct_relations \
    --input-dir data/processed/catalog_meta_analysis/standardized \
    --output-dir analyses/output_relations \
    --llm-replay-items analyses/output_llm_evaluation/llm_mapping_evaluation_items.csv \
                       analyses/output_llm_evaluation/added_2026_11_05/llm_mapping_evaluation_items.csv
```

`--llm-replay-items` takes several item files (benchmark runs on disjoint sets of studies) and concatenates them.

Outputs in `analyses/output_relations/`:

| File | Content |
|---|---|
| `relations_by_study.csv` | every study correlation, per level and association |
| `relations_pooled.csv` | pooled relations with HKSJ and prediction intervals, BH q, heterogeneity, the domain split and the single-study comparison |
| `harmonization_ladder.csv` / `.png` | pairs pooled and findings per level |
| `relations_attribution.csv` | per pair: k at each level and the first level at which it is pooled and becomes a finding |
| `relations_findings.png` | per-study and pooled r of the curated cross-construct relations pooled over at least three studies, findings highlighted |
| `llm_replay_conclusions.csv` | replay and automatic findings against the curated ones |
| `relations_summary.json` | the ladder, the finding rule and the replay's column counts |

**Inputs of the committed outputs (2026-11-06).** One batch run of all 39 studies, the same run as `analyses/output_python_full/` (`docs/reproducibility.md`), including NL2TAP's TOAST trust items. The replay uses the 23-study benchmark items of 2026-11-04 and a benchmark of the 16 added studies with the same settings (E1 only: `analyses/output_llm_evaluation/added_2026_11_05/`). Without the TOAST items, this run reproduces the relation tables of 2026-11-05, which combined a run of the first 23 studies with one of the 16 added ones, byte for byte.

## Results with 39 studies

All 39 development studies declare their participant columns. 31 of them contribute relations from the deterministic levels (2,853 participants; 16 and 1,233 with 23 studies), and 37 once LLM mappings are included (21 before). Two of the 31, road_bumps_touch and Handsize, contribute since 2026-11-08 only one pair, task completion time × the new `touch_offset`: participants who took longer touched closer to the target in both (between persons r = −0.43 each; two studies are too few for a finding). No finding changed in that rerun (see the changelog). Of the 16 studies added on 2026-11-05, 13 contribute between-person relations; eHMI on the car body and the eHMI crowd surveys measure one DV per file, and the myocontrol study has eight participants, below the minimum of ten (it contributes within-person relations).

This is exploratory: the finding rule and the false-discovery family were fixed after a first look at the first 23 studies. The 16 added studies were mapped from their codebooks and papers before any cross-study relation or LLM answer on them was seen (within-study item correlations informed two mapping decisions, documented in `schemas/sensecobot_fusion_2024_mapping.yaml`), but the rule was not re-registered for them, so the new findings below are exploratory as well. The held-out confirmation set (`docs/heldout_results.md`) is the confirmatory test.

### What each step makes estimable

Pairs of construct-level DVs measured together in at least two and at least three studies (`harmonization_ladder.csv`); the 23-study values in brackets:

| Level | Between persons: k ≥ 2 | k ≥ 3 | findings | Within persons: k ≥ 2 | k ≥ 3 | findings |
|---|---|---|---|---|---|---|
| Identical raw names | 31 (16) | 14 (0) | 4 (0) | 16 (1) | 12 (0) | 9 (0) |
| + shared schema, header matching | 26 (9) | 21 (0) | 6 (0) | 23 (6) | 21 (0) | 14 (0) |
| + study mapping files | 84 (52) | 35 (29) | 19 (0) | 46 (44) | 24 (23) | 16 (16) |
| + local-LLM fallback | 226 (132) | 82 (54) | 20 (0) | 119 (88) | 49 (41) | 17 (10) |

The new studies change the lower rungs. With 23 studies, pooling by column names reached no pair in three studies; now 14 pairs reach three, and the shared schema 21. **All of them are pairs of NASA-TLX subscales.** Four of the added studies (TLX formats, NL2TAP, concurrent speech, visual fidelity) name the subscales by their own words (`mental_demand`, "Mental Demand"), which the names baseline and the schema both recognize.

Relations between different constructs still come only from curated mappings:

| Level | Cross-construct pairs in ≥ 3 studies (between / within persons) |
|---|---|
| Identical raw names | 0 / 0 |
| + shared schema, header matching | 0 / 0 |
| + study mapping files | 12 / 2 |
| + local-LLM fallback | 59 / 27 |

Naming conventions decide what the cheap steps reach: a measure that studies already call by one name pools without harmonization, a construct that each study names its own way does not.

Figure: `analyses/output_relations/harmonization_ladder.png`.

### Relations that hold across studies

Figure: `analyses/output_relations/relations_findings.png` (per-study and pooled r of the curated cross-construct relations in at least three studies).

- **Higher workload goes with lower usability, between persons (new).** In the four studies that measured NASA-TLX mental demand and the SUS (eHMI for All, NL2TAP, Reflexion mapping, ROADS; 162 participants), participants who rated the task more demanding rated the system less usable:
    - pooled r = −0.42, HKSJ 95% CI [−0.51, −0.31], BH q = .004;
    - heterogeneity is nil (I² = 0%), all four studies negative (−0.33 to −0.51), and the prediction interval [−0.66, −0.09] excludes 0, so a new study is expected to show the same sign;
    - frustration (−0.56), effort (−0.49), the TLX total (−0.48) and temporal demand (−0.47) show the same in three studies (NL2TAP, Reflexion mapping, ROADS), each a finding, with prediction intervals that still include 0 at k = 3.

  With 23 studies only two studies measured both (r = −0.35). The relation needs the curated mappings: NL2TAP stores the SUS items as `item_1`–`item_10` and Reflexion mapping the TLX as `tlx_1_1`–`tlx_2_6`, so the shared schema reaches two studies and the replay (below) reaches two. Both are post-task self-reports, so common-method variance is part of the correlation.
- **Workload goes with lower perceived safety, within persons (unchanged).** No added study measured both repeatedly, so the relation rests on the same four studies (BVI auditory HAV, eHMI for All, in-car VR gain, Running into Traffic):
    - pooled within-person r = −0.44, HKSJ 95% CI [−0.72, −0.05], BH q = .04;
    - all four negative (−0.14 to −0.61), I² = 94%, prediction interval (−0.94 to 0.65) includes 0.

  Between persons (five studies) the relation stays weak and uncertain: r = −0.27 [−0.62, 0.16].
- **The NASA-TLX structure now holds between persons too.** Within persons the subscales covary in all eight studies that measured them repeatedly (four before: concurrent speech, myocontrol, TLX formats and visual fidelity added): all 15 pairs are findings, r = 0.27–0.60, and nine generalize. Between persons, over 10–12 studies, 12 of the 15 pairs are findings (none were with 23 studies):
    - effort, mental and temporal demand are the core (r = 0.63–0.78; effort × mental demand 0.78 with a prediction interval from 0.37);
    - *performance* is again the weakest-linked subscale: r = 0.13–0.24 with the others, three of its five pairs no finding;
    - heterogeneity is lower between persons (median I² 67%) than within (median 87%).

  SenseCobot's cumulative performance item is left unlabelled because its scale direction is undocumented (see the mapping file).
- **Same-instrument pairs.** Van der Laan satisfying × usefulness now rests on four studies (AR pedestrian online adds 1,164 respondents): r = 0.84 [0.67, 0.92], prediction interval above 0. UEQ-S hedonic × pragmatic: r = 0.50 [0.24, 0.70] over five studies. These are reported apart from cross-construct relations.
- **Not established.** Van der Laan satisfying × SUS (r = 0.68, three studies, q = .052) and trust × understanding (r = 0.67, three studies, HKSJ interval including 0) gained no study: none of the added studies measured both. NL2TAP's TOAST trust scale adds a fourth study to mental demand × trust, which stays weak and uncertain (r = −0.23, q = .25), and pairs trust with the other TLX scales in a second study.

### What the LLM fallback adds

The LLM extends coverage to 37 studies and 226 pairs. Beyond the curated findings it adds three cross-construct findings; the fourth of the 23-study analysis (performance × STS-AD trust, from wrong mappings) no longer passes (q = .31):

- **Perceived safety × trust in automation** (TiA): r = 0.63 between and 0.61 within persons over the same four studies as before, which include the wrong mappings described in the 23-study analysis (TiA understanding items averaged into trust in 3D Display, the ride's relaxation rating as trust in in-car VR).
- **Workload × perceived safety, within persons**, over eight studies: r = −0.38 [−0.59, −0.13], as before.
- **Trust × UEQ-S hedonic quality** (new, three studies, r = 0.28, HKSJ [0.22, 0.35], q = .02): BVI auditory HAV, eHMI state and intent, and Running into Traffic. It rested on four until 2026-11-08, when ARTV ornaments' unmapped measures were marked never-map: there the LLM had mapped the Film-IEQ *transportation* score, an immersion measure, to trust. The LLM still maps minimum heart rate to trust in the music-tempo study, which measures no UEQ-S.

On the 16 added studies the LLM is weaker than on the first 23. In the leave-alias-out benchmark (`analyses/output_llm_evaluation/added_2026_11_05/`) it gives the curated DV for 71 of 423 curated DV columns (17%; 41% on the first 23). Where it fails:

- the Qualtrics codes of AR pedestrian online (`Q39_3`, `Q40_1`, ...): 0 of 99 right, Van der Laan items read as SUS items;
- the Godspeed differentials of driving style: 0 of 90 right (2 before schema 3.3 added a DV to the candidates), e.g. "Mechanical vs organic" read as skin conductance;
- the NASA-TLX subscales: most read as mental demand, as before.

### Does the LLM stand in for curation?

Still no. In the replay (`llm_replay_conclusions.csv`; definition shortlist, plausibility check on):

- **Columns.** Of the 314 DV columns the study mappings decide, the LLM answers 241 and gives the curated DV for 67 (21%).
- **Conclusions, within persons.** The replay has 15 findings; 13 confirm curated findings, 2 are unsupported. This is no gain of the LLM: the shared schema alone confirms 14 of the 16 curated within-person findings, because the added TLX studies name their subscales plainly. The one cross-construct pair, workload × perceived safety, is a replay finding now (r = −0.39 over eight studies, q = .03; q = .09 with 23 studies), because the false-discovery family holds more true TLX relations.
- **Conclusions, between persons.** The replay has 1 finding against 19 curated ones, a confirmed TLX pair (mental × temporal demand). Until 2026-11-08 it had a second, unsupported one, SSQ oculomotor × STS-AD trust, which rested on wrong mappings; the LLM's answers for FOV snap-turn changed with schema 3.3, and the pair now pools only two studies. It recovers none of the five usability × workload findings, because it reaches only two of the studies that measured both.
- **Estimates.** Where both pool the same pair, the pooled correlations agree closely (median |Δr| = 0.07 within, 0.06 between persons).

So the cheap steps reach a construct when the studies already name it alike; the LLM does not add reliable relations on top, and the cross-construct findings need the curated (or reviewed) mappings.

**Limits of the replay.** The values keep the study mappings' value transforms (reverse keying, unit and scale conversions), which a fresh pipeline would lack, so the replay is optimistic. Composite columns take the LLM's answer for their first item. The benchmark asks about each distinct column name once per study, so one answer stands for that name in every file.

### The 23-study analysis (2026-11-04)

Superseded by the results above; kept for the record. With 23 studies, 16 contributed relations at the deterministic levels (1,233 participants). No pair was pooled over three studies by names or the shared schema. The curated level had 16 within-person findings (workload × perceived safety and the NASA-TLX subscales in four studies) and no between-person finding. The replay recovered none of the 16 within-person findings; with the dense shortlist (`dense_h`) it recovered one (frustration × mental demand, three studies). The committed tables of that analysis are in the git history (`aad5db9`).
