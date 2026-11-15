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
    | `llm` | the local-LLM fallback, for the DVs a study has no column of from the steps above |

    Each level includes the ones above it. At `names` the baseline is told which columns are DVs (those the deterministic pipeline maps), which favours it.

    At `llm` the LLM fills gaps only. A study's LLM-mapped column of a DV is set aside when the study has a column of that DV mapped by its study mapping, the shared schema or a header match, whether in the same table (a `__dup_N` column) or in another table. A registry score computed from such columns is kept before an LLM-mapped column of the score's DV. This is the rule the pooled-means analysis applies with all mappings (`llm_columns="prefer_curated"`). The columns set aside are listed in `relations_llm_columns_set_aside.csv`. Until 2026-11-14 they were averaged with the curated column row by row, or stacked below it, so an LLM column could change a DV the study had measured without it (see the changelog).
2. **Participants.** Columns mapped to the same DV within a study (wide repeated measures, items of one construct) are averaged per row, then per participant. Participant columns for all 39 studies are declared in `schemas/study_designs.yaml`.
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

`--llm-replay-items` answers it from the leave-alias-out benchmark (`evaluate_llm_mapping.py`, E1). There, every column a study mapping decided is shown to the pipeline's LLM step with that mapping hidden. The *replay* keeps the automatic and LLM columns. Every column a study mapping decided takes the LLM's answer: a DV column the DV the LLM chose, or nothing when it abstained or the plausibility check rejected the answer; a column the mapping locked as non-DV (`meta_*`) the DV the LLM gave it. As at the `llm` level, the LLM's columns of a DV are then set aside where the study has an automatic (schema or header-match) column of it (since 2026-11-14). The replay's findings are compared with the curated findings (`llm_replay_conclusions.csv`):

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
| `relations_llm_columns_set_aside.csv` | the LLM-mapped columns the `llm` level and the replay set aside, with the table, the raw column and what was kept instead |
| `relations_findings.png` | per-study and pooled r of the curated cross-construct relations pooled over at least three studies, findings highlighted |
| `llm_replay_conclusions.csv` | replay and automatic findings against the curated ones |
| `relations_summary.json` | the ladder, the finding rule, the counts of LLM columns set aside and the replay's column counts |

**Inputs of the committed outputs (2026-11-15).** The batch run of all 39 studies behind `analyses/output_python_full/`, after the LLM-mapping audit fixed the mappings of 22 studies (`data/processed/audit39`, `docs/reproducibility.md`, `analyses/llm_mapping_audit/`). The replay uses the benchmark items of the first 23 studies, rerun on the same day with those fixes, and a benchmark of the 16 added studies with the same settings (E1 only: `analyses/output_llm_evaluation/added_2026_11_05/`). The run of 2026-11-06 added NL2TAP's TOAST trust items; without them it reproduced the relation tables of 2026-11-05, which combined a run of the first 23 studies with one of the 16 added ones, byte for byte.

## Results with 39 studies

All 39 development studies declare their participant columns. 33 of them contribute between-person relations from the deterministic levels (2,923 participants; 31 and 2,853 until the audit of 2026-11-15 curated the TiA, SART, perceived-safety and STS-AD items of 3D display, FACT-AV and Longitudinal US/DE; 16 and 1,233 with 23 studies). Since the audit the LLM adds no study: no LLM-mapped column is pooled (36 studies with LLM mappings before; 37 until Gaze Platformer's questionnaire items, all LLM-mapped, were locked on 2026-11-14). Two of the 31, road_bumps_touch and Handsize, contribute since 2026-11-08 only one pair, task completion time × the new `touch_offset`: participants who took longer touched closer to the target in both (between persons r = −0.43 each; two studies are too few for a finding). No finding changed in that rerun (see the changelog). Of the 16 studies added on 2026-11-05, 13 contribute between-person relations; eHMI on the car body and the eHMI crowd surveys measure one DV per file, and the myocontrol study has eight participants, below the minimum of ten (it contributes within-person relations).

This is exploratory: the finding rule and the false-discovery family were fixed after a first look at the first 23 studies. The 16 added studies were mapped from their codebooks and papers before any cross-study relation or LLM answer on them was seen (within-study item correlations informed two mapping decisions, documented in `schemas/sensecobot_fusion_2024_mapping.yaml`), but the rule was not re-registered for them, so the new findings below are exploratory as well. The two held-out sets are the confirmatory tests (`docs/heldout_results.md`, `docs/heldout_results_2.md`).

### What each step makes estimable

Pairs of construct-level DVs measured together in at least two and at least three studies (`harmonization_ladder.csv`); the 23-study values in brackets:

| Level | Between persons: k ≥ 2 | k ≥ 3 | findings | Within persons: k ≥ 2 | k ≥ 3 | findings |
|---|---|---|---|---|---|---|
| Identical raw names | 36 (16) | 17 (0) | 4 (0) | 21 (1) | 15 (0) | 10 (0) |
| + shared schema, header matching | 24 (9) | 21 (0) | 6 (0) | 22 (6) | 21 (0) | 14 (0) |
| + study mapping files | 108 (52) | 51 (29) | 21 (0) | 67 (44) | 41 (23) | 25 (16) |
| + local-LLM fallback | 108 (132) | 51 (54) | 21 (0) | 67 (88) | 41 (41) | 25 (10) |

Since the audit of 2026-11-15 the LLM level equals the curated one, because no LLM-mapped column is pooled; until then it had 191 / 76 / 20 between and 90 / 41 / 19 within persons, and the curated level 84 / 35 / 19 and 46 / 24 / 16.

The new studies change the lower rungs. With 23 studies, pooling by column names reached no pair in three studies; now 17 pairs reach three, and the shared schema 21. **All the schema's, and 14 of the names level's, are pairs of NASA-TLX subscales.** Four of the added studies (TLX formats, NL2TAP, concurrent speech, visual fidelity) name the subscales by their own words (`mental_demand`, "Mental Demand"), which the names baseline and the schema both recognize. The names level's other three are SART subscale pairs: the lab behind 3D display, BVI auditory HAV and FACT-AV names its ten SART items `SART1`–`SART10` in all three, and since the audit the items are DV columns.

Relations between different constructs still come only from curated mappings:

| Level | Cross-construct pairs in ≥ 3 studies (between / within persons) |
|---|---|
| Identical raw names | 3 / 3 (SART subscales) |
| + shared schema, header matching | 0 / 0 |
| + study mapping files | 28 / 19 (12 / 2 before the audit) |
| + local-LLM fallback | 28 / 19 (53 / 19 before the audit) |

The SART subscale pairs count as different constructs because the instrument registry has no SART entry; they are subscales of one instrument.

Naming conventions decide what the cheap steps reach: a measure that studies already call by one name pools without harmonization, a construct that each study names its own way does not.

Figure: `analyses/output_relations/harmonization_ladder.png`.

### Relations that hold across studies

Figure: `analyses/output_relations/relations_findings.png` (per-study and pooled r of the curated cross-construct relations in at least three studies).

- **Higher workload goes with lower usability, between persons (new).** In the four studies that measured NASA-TLX mental demand and the SUS (eHMI for All, NL2TAP, Reflexion mapping, ROADS; 162 participants), participants who rated the task more demanding rated the system less usable:
    - pooled r = −0.42, HKSJ 95% CI [−0.51, −0.31], BH q = .006 (.004 before the audit enlarged the false-discovery family);
    - heterogeneity is nil (I² = 0%), all four studies negative (−0.33 to −0.51), and the prediction interval [−0.66, −0.09] excludes 0, so a new study is expected to show the same sign;
    - frustration (−0.56), effort (−0.49), the TLX total (−0.48) and temporal demand (−0.47) show the same in three studies (NL2TAP, Reflexion mapping, ROADS), each a finding, with prediction intervals that still include 0 at k = 3.

  With 23 studies only two studies measured both (r = −0.35). The relation needs the curated mappings: NL2TAP stores the SUS items as `item_1`–`item_10` and Reflexion mapping the TLX as `tlx_1_1`–`tlx_2_6`, so the shared schema reaches two studies and the replay (below) reaches two. Both are post-task self-reports, so common-method variance is part of the correlation.
- **Workload goes with lower perceived safety, within persons.** Since the audit (2026-11-15) the relation rests on seven curated studies: BVI auditory HAV, eHMI for All, in-car VR gain and Running into Traffic as before, and 3D display, FACT-AV and Longitudinal US/DE, whose Faas perceived-safety items were LLM-mapped until then:
    - pooled within-person r = −0.40, HKSJ 95% CI [−0.58, −0.18], BH q = .009 (four studies: −0.44 [−0.72, −0.05], q = .04);
    - I² = 98%, prediction interval (−0.81 to 0.27) includes 0.

  Between persons (eight studies) the relation stays weak and uncertain: r = −0.27 [−0.48, −0.02], q = .08 (five studies: −0.27 [−0.62, 0.16]).
- **Trust in automation with perceived safety and workload (new, 2026-11-15).** The audit scored the Koerber TiA subscales of 3D display and Longitudinal US/DE from their items (until then the LLM had averaged 3D display's understanding items into trust and mapped in-car VR gain's ride-relaxation rating to trust). With BVI auditory HAV, three studies measure TiA trust and perceived safety:
    - perceived safety × TiA trust, between persons r = 0.69 [0.62, 0.74], q = .004, and within persons r = 0.62 [0.49, 0.72], q = .007; within persons the prediction interval [0.02, 0.89] excludes 0;
    - mental demand × TiA trust, within persons r = −0.38 [−0.62, −0.08], q = .047, prediction interval including 0.

  All three studies come from one lab with the same questionnaire set, and the prediction intervals at k = 3 are wide, so this needs studies from elsewhere.
- **SART (new, 2026-11-15).** 3D display, BVI auditory HAV and FACT-AV give the ten SART items, which take their subscale ids: within persons mental demand goes with SART demand (r = 0.27, prediction interval above 0), weakly with supply (0.12) and against understanding (−0.11); SART demand × supply between persons (0.52) and supply × understanding within persons (0.15) are relations inside the instrument.
- **The NASA-TLX structure now holds between persons too.** Within persons the subscales covary in all eight studies that measured them repeatedly (four before: concurrent speech, myocontrol, TLX formats and visual fidelity added): all 15 pairs are findings, r = 0.27–0.60, and nine generalize. Between persons, over 10–12 studies, 12 of the 15 pairs are findings (none were with 23 studies):
    - effort, mental and temporal demand are the core (r = 0.63–0.78; effort × mental demand 0.78 with a prediction interval from 0.37);
    - *performance* is again the weakest-linked subscale: r = 0.13–0.24 with the others, three of its five pairs no finding;
    - heterogeneity is lower between persons (median I² 67%) than within (median 87%).

  SenseCobot's cumulative performance item is left unlabelled because its scale direction is undocumented (see the mapping file).
- **Same-instrument pairs.** Van der Laan satisfying × usefulness now rests on four studies (AR pedestrian online adds 1,164 respondents): r = 0.84 [0.67, 0.92], prediction interval above 0. UEQ-S hedonic × pragmatic: r = 0.50 [0.30, 0.65] over six studies between persons (five, [0.24, 0.70], until the audit scored Longitudinal US/DE's UEQ-S from its items), and since then a within-person finding too (four studies, r = 0.59 [0.27, 0.79]). These are reported apart from cross-construct relations.
- **Not established.** Van der Laan satisfying × SUS (r = 0.68, three studies, q = .074; .052 before the audit changed the false-discovery family) gained no study: none of the added studies measured both. Trust × understanding pools two studies since the audit (r = 0.77): its third, FACT-AV, contributed its SART understanding sum, which the shared alias `Understanding` had sent to the eHMI comprehension rating (three studies, r = 0.67, HKSJ interval including 0 before). NL2TAP's TOAST trust scale adds a fourth study to mental demand × trust, which stays weak and uncertain (r = −0.23, q = .28), and pairs trust with the other TLX scales in a second study.

### What the LLM fallback adds

**Since the audit of 2026-11-15, nothing that is pooled.** Its findings had needed review, and the audit (`analyses/llm_mapping_audit/`) checked every LLM-mapped column the all-mappings analysis pooled against the studies' documentation: 34 of 378 had the right DV (9.0%, Wilson [6.5, 12.3]), 20 of the 73 behind `llm`-level relations findings. The mappings were fixed (right and wrong-DV columns curated, the rest locked), SenseCobot was taken out of the LLM step, and the 17 LLM answers left are all set aside beside a curated column of their DV (153 columns of 6 studies in `relations_llm_columns_set_aside.csv`). The `llm` level therefore equals the curated one. What became of the five cross-construct findings the LLM level had beyond the curated ones:

- **Perceived safety × trust in automation** (TiA; r = 0.63 between, 0.61 within over four studies) rested on correct mappings in two studies, TiA understanding items averaged into trust in 3D display and the ride's relaxation rating as trust in in-car VR gain. With the TiA subscales scored from their items it is a curated finding over three studies (0.69 and 0.62; see above), without in-car VR gain.
- **Workload × perceived safety, within persons** (seven studies, r = −0.40): three of the seven had LLM-mapped Faas items, which were right; it is now curated over the same seven.
- **Trust × UEQ-S hedonic quality** (three studies, r = 0.28): BVI auditory HAV's trust column was its undocumented `Trust[SQ003]` (now locked), so the pair pools two studies.
- **Mental demand × STS-AD trust, within persons** (three studies, r = 0.23): 3D display's and FACT-AV's `drivingstyle` are a self-developed rating of the vehicle's driving style (1 = completely safe to 7 = completely dangerous), FACT-AV's `complex` a 1–5 item ("I found the visualizations unnecessarily complex" in the lab's other studies), and in-car VR gain's `MaxMS` the maximum Misery Scale rating. None is an STS-AD item; the studies' real STS-AD items had been read as NASA-TLX performance, perceived safety and TiA propensity to trust. The pair is gone.
- **Mental demand × trust in automation, within persons** (four studies, r = −0.35): without in-car VR gain's relaxation rating and with 3D display's understanding items out of its trust score, a curated finding over three studies (r = −0.38, q = .047).

Before the audit, locking FOV Snap-Turn's and Gaze Platformer's questionnaire items (VRNQ, PXI, IMI, eye strain; 2026-11-14) had removed 35 between-person and 18 within-person pairs, and setting aside LLM columns beside curated ones (2026-11-14) 213 columns of 11 studies. Mental demand × SSQ oculomotor (between persons) was a finding for part of 2026-11-14 on FOV Snap-Turn's VRNQ items read as mental demand; SenseCobot's SSQ oculomotor was an EEG band power (`O2_theta Max`), and its NASA-TLX performance 36 EEG features.

On the 16 added studies the LLM is weaker than on the first 23. In the leave-alias-out benchmark (`analyses/output_llm_evaluation/added_2026_11_05/`) it gives the curated DV for 71 of 423 curated DV columns (17%; 36% of 300 on the first 23 since the audit added curated columns, 41% of 258 before). Where it fails:

- the Qualtrics codes of AR pedestrian online (`Q39_3`, `Q40_1`, ...): 0 of 99 right, Van der Laan items read as SUS items;
- the Godspeed differentials of driving style: 0 of 90 right (2 before schema 3.3 added a DV to the candidates), e.g. "Mechanical vs organic" read as skin conductance;
- the NASA-TLX subscales: most read as mental demand, as before.

### Does the LLM stand in for curation?

Still no. In the replay (`llm_replay_conclusions.csv`; definition shortlist, plausibility check on):

- **Columns.** Of the 1,114 DV columns the study mappings decide, the LLM answers 989 and gives the curated DV for 783 (720 of them eHMI-for-All's `total time` in its per-run logs, one benchmark answer; before the audit 67 of 314, 21%).
- **Conclusions, within persons.** The replay has 11 findings; 10 confirm curated findings, 1 is unsupported (performance × STS-AD trust, from the LLM's answers for 3D display's, FACT-AV's and SenseCobot's columns). This is no gain of the LLM: the shared schema alone confirms 14 of the 25 curated within-person findings, because the added TLX studies name their subscales plainly. Since the audit the replay replays the LLM's answers on the newly curated columns too: it reads the STS-AD Performance items of 3D display and FACT-AV as NASA-TLX performance, which dilutes mental demand × performance (r 0.23, HKSJ including 0), and it has no answer for the TiA items, which map to registry items the benchmark does not ask about, so the TiA findings are missing from it (as BVI's TiA items always were). Before the audit it had 15 findings (13 confirmed). The one confirmed cross-construct pair, workload × perceived safety, stays a replay finding (r = −0.42 over seven studies, q = .02).
- **Conclusions, between persons.** The replay has no finding against 21 curated ones. Its one finding before the audit, mental × temporal demand, now misses (r = 0.67 over six studies, q = .10). It recovers none of the five usability × workload findings, because it reaches only two of the studies that measured both.
- **Estimates.** Where both pool the same pair, the pooled correlations agree closely (median |Δr| = 0.04 within and 0.06 between persons; 0.05 and 0.05 before the audit, 0.07 and 0.06 before the LLM's columns of schema-mapped DVs were set aside on 2026-11-14).
- **The locked columns are in the replay.** The replay stands for the LLM in place of every curated mapping, so the locked columns the LLM maps in the benchmark enter it as the LLM's answers: FOV Snap-Turn's and Gaze Platformer's (2026-11-14; the VRNQ game-mechanics and PXI enjoyment items as mental demand, eye-strain symptoms as pedestrian crossing time and others) and since 2026-11-15 the audit's locks (1,438 curated non-DV columns mapped by the LLM, 1,109 before).

So the cheap steps reach a construct when the studies already name it alike; the LLM does not add reliable relations on top, and the cross-construct findings need the curated (or reviewed) mappings.

**Limits of the replay.** The values keep the study mappings' value transforms (reverse keying, unit and scale conversions), which a fresh pipeline would lack, so the replay is optimistic. Composite columns take the LLM's answer for their first item. The benchmark asks about each distinct column name once per study, so one answer stands for that name in every file.

### The 23-study analysis (2026-11-04)

Superseded by the results above; kept for the record. With 23 studies, 16 contributed relations at the deterministic levels (1,233 participants). No pair was pooled over three studies by names or the shared schema. The curated level had 16 within-person findings (workload × perceived safety and the NASA-TLX subscales in four studies) and no between-person finding. The replay recovered none of the 16 within-person findings; with the dense shortlist (`dense_h`) it recovered one (frustration × mental demand, three studies). The committed tables of that analysis are in the git history (`aad5db9`).
