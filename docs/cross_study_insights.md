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
2. **Participants.** Columns mapped to the same DV within a study (wide repeated measures, items of one construct) are averaged per row, then per participant. Participant columns for all 23 studies are declared in `schemas/study_designs.yaml`.
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
    --llm-replay-items analyses/output_llm_evaluation/llm_mapping_evaluation_items.csv
```

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

## Results with 23 studies

All 23 development studies declare their participant columns. 16 of them contribute relations from the deterministic levels (1,233 participants), and 21 once LLM mappings are included. This is exploratory: the finding rule and the false-discovery family were fixed after a first look at these data. The first draft controlled the false-discovery rate over all pooled pairs, including pairs of two studies and pairs scored from shared items; under that family the workload–safety relation below had q = .07 and would not have counted as a finding.

### What each step makes estimable

Pairs of construct-level DVs measured together in at least two and at least three studies (`harmonization_ladder.csv`):

| Level | Between persons: k ≥ 2 | k ≥ 3 | findings | Within persons: k ≥ 2 | k ≥ 3 | findings |
|---|---|---|---|---|---|---|
| Identical raw names | 16 | 0 | 0 | 1 | 0 | 0 |
| + shared schema, header matching | 9 | 0 | 0 | 6 | 0 | 0 |
| + study mapping files | 52 | 29 | 0 | 44 | 23 | 16 |
| + local-LLM fallback | 132 | 54 | 0 | 88 | 41 | 10 |

The `names` row is not a rung below the schema: it is told which columns are DVs (including the curated ones) but pools them only by identical names. Pooling open data by column names gives no relation shared by three studies. Even the shared schema with header matching and instrument scoring reaches only two studies per pair, because the 12 added studies name their columns their own way. The curated study mappings are what make relations comparable across studies.

Figure: `analyses/output_relations/harmonization_ladder.png`.

### Relations that hold across studies

Figure: `analyses/output_relations/relations_findings.png` (per-study and pooled r of the curated cross-construct relations in at least three studies).

- **Workload goes with lower perceived safety, within persons.** In the four studies that measured both repeatedly (BVI auditory HAV, eHMI for All, in-car VR gain, Running into Traffic), the occasions with higher NASA-TLX mental demand were rated less safe:
    - pooled within-person r = −0.44, HKSJ 95% CI [−0.72, −0.05], BH q = .04;
    - all four studies negative (−0.14 to −0.61), three significant alone;
    - heterogeneity is large (I² = 94%), so the size differs by setting, and the prediction interval (−0.94 to 0.65) does not rule out a null in a new study.

  Between persons (participant means, five studies) the relation is weaker and uncertain: r = −0.27 [−0.62, 0.16]. The coupling is a state-level one, between conditions, not a trait-level one between people. A single study shows one value; only the harmonized set shows that the within-person relation recurs across populations and tasks (blind and visually impaired passengers, pedestrians in VR, passengers in an in-car VR ride) while the between-person one does not. Two of the four studies contribute perceived safety only because the instrument registry now scores it from its four items.
- **NASA-TLX subscales covary within persons in every study**, in automotive and VR-interaction research alike (four studies). Effort, frustration and mental demand are the core (r = 0.56–0.65, prediction intervals above 0). *Performance* is the weakest-linked subscale (r = 0.29–0.45 with the others), consistent with the TLX failing a one-factor model in the invariance test (`research_agenda.md` §4). Between persons the same pairs are much more heterogeneous (I² 80–90%).
- **Not established.** No between-person relation passes the rule. The strongest are Van der Laan satisfying × SUS (r = 0.68 [0.13, 0.91], three studies, q = .08) and trust × understanding (r = 0.67, three studies, HKSJ interval including 0). With k = 3, the HKSJ interval uses t with 2 degrees of freedom, so only very consistent relations pass.

### What the LLM fallback adds

The LLM extends coverage to 21 studies and 132 pairs. Three of its ten within-person findings are cross-construct:

- **Workload × perceived safety** is reproduced in eight studies (r = −0.38 [−0.59, −0.13]), seven of them negative.
- **Perceived safety × trust in automation** (TiA): r = 0.60 [0.53, 0.67] in four studies, with a prediction interval above 0. Two of the four studies have no curated mapping at all; the LLM is the only route to them. Inspecting the mapped columns:
    - correct in two studies: BVI's (curated) and Longitudinal eHMI's (`TiA_Trust1`, `TiA_Trust2`, `perceivedSafety1`–`4`, mapped by the LLM);
    - partly wrong in 3D Display: the four TiA *understanding/predictability* items (`TiA_U1`–`U4`) were averaged into trust next to the two trust items, and three other questions into perceived safety;
    - wrong in In-car VR: the ride's *relaxation* rating was mapped to trust.

    The relation is homogeneous anyway (r = 0.58–0.69 in all four), because relaxation also tracks perceived safety. Homogeneity does not validate a mapping.
- **Performance × STS-AD trust** (three studies, r = 0.21) rests entirely on wrong mappings: a motion-sickness maximum (`MaxMS`) and a driving-style rating were mapped to STS-AD trust.

The LLM thus finds candidate relations in studies no rule can reach, but every one of them needs its mappings reviewed (`opendv review`) before it counts as evidence. The keying check caught one more case: in FOV Snap-Turn the LLM mapped VR-navigation items to Van der Laan items whose answers run against the registry's keys.

### Does the LLM stand in for curation?

No. In the replay (`llm_replay_conclusions.csv`; answers from the leave-alias-out benchmark of 2026-11-04 with the definition shortlist, plausibility check on):

- **Columns.** Of the 161 DV columns the study mappings decide, the LLM answers 95 and gives the curated DV for 46 (29%); the other 66 are abstentions or answers the plausibility check rejects. It also maps 192 of the 319 distinct columns the mappings lock as non-DV.
- **Conclusions.** It recovers none of the 16 within-person findings of the curated pipeline:
    - The NASA-TLX structure disappears because the LLM reads most subscale columns other than mental demand as mental demand (16 of 35) or as the TLX total (9); 6 get their own subscale.
    - The workload–safety relation is reproduced over eight studies with almost the same size (r = −0.39, HKSJ [−0.60, −0.14], p = .009), but it is no longer a finding (q = .09): the replay's other relations are mostly null, so the false-discovery control is stricter.
    - Its one finding, perceived safety × TiA trust (three studies, r = 0.59), rests on studies the curated level does not cover, and on the partly wrong mappings described above.
- **Estimates.** Where both pool the same pair, the pooled correlations agree closely (median |Δr| = 0.06 within persons, 0.09 between persons). The LLM's cost is coverage and misassignment of columns, not distortion of the relations it does recover.

**With the dense shortlist** (`dense_h`, which maps more columns correctly: 69 of the 161 keep their DV), the replay recovers one of the 16 findings (frustration × mental demand, three studies); the workload–safety relation again keeps its size (r = −0.39 over eight studies, q = .06).

So the LLM can propose where to look, but the insights above need the curated (or reviewed) mappings. The deterministic steps alone (shared schema, header matching, instrument scoring) recover no finding either, because the studies name their columns in their own way.

**Limits of the replay.** The values keep the study mappings' value transforms (reverse keying, unit and scale conversions), which a fresh pipeline would lack, so the replay is optimistic. Composite columns take the LLM's answer for their first item. The benchmark asks about each distinct column name once per study, so one answer stands for that name in every file.
