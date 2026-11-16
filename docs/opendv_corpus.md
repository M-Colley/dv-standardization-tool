# OpenDV corpus: one analysis of 88 studies with open data

The OpenDV mapping is meant to be reused for new studies from CHI and other SIGCHI venues. This page tests that claim on every study the repository maps: 88 studies with open data, read with their curated mappings only, with no LLM involved. It asks four questions:

1. How many different names does one measure go by?
2. What becomes comparable once the names are mapped?
3. Does the vocabulary transfer to studies it was not built on?
4. Where do mapped values still go wrong?

Outputs are in `analyses/opendv_corpus/`, which also has the commands to reproduce them.

## Corpus

| Venue group | Development (39) | Held-out 1 (19) | Held-out 2 (30) | All |
|---|---|---|---|---|
| SIGCHI conferences and journals (CHI 11, AutomotiveUI 4, ACM/IEEE HRI 3, CSCW, MobileHCI, IMWUT) | 17 | 3 | 1 | **21** |
| Other ACM venues (Mensch und Computer, AVI, MUM, IMX, CHASE, THRI) | 3 | 2 | 5 | 10 |
| IEEE venues (TVCG, ISMAR, THMS, VL/HCC, RO-MAN, ...) | 5 | 4 | 7 | 16 |
| Other journals and conferences (IJHCI, IJHCS, Transportation Research F, Ergonomics, ...) | 12 | 10 | 17 | 39 |
| Data repository without a paper | 2 | 0 | 0 | 2 |

- **Years:** the papers appeared between 2010 and 2026 (median 2024).
- **Domains:** 36 automotive studies, 7 on human-robot interaction, and the rest on VR and AR, games, usable security, software tools, input, media and education.
- **Sets:** the development studies are the ones the schema and tooling were built on. The two held-out sets were searched and mapped later, each under a pre-registration (`docs/preregistration_heldout_relations*.md`).
- **Columns:** the batch read 31,758 columns from 1,944 tables, and every study standardized without error. Of those columns, 6,326 measure a DV: 5,530 map to a schema DV, and the rest are items that the analysis scores into one. They come from 87 studies. The 88th, ai_label_trust_2025, has no DV column: its Likert items are not described in its files, so its mapping leaves them out.
- **Development mappings:** as corrected by the audit of every LLM-mapped column the development analysis pooled (PR #51, `analyses/llm_mapping_audit/`). The corpus gained 1,036 DV columns: eHMI-for-All's task completion time in its 720 log files, osf_cwd6h's perceived risk in 216 drive logs (which a sensor alias had captured), and 100 questionnaire items of three studies by one automotive lab.

## 1. One measure, many names

The 6,326 DV columns carry 1,161 distinct names, even after case, spaces and punctuation are ignored. The mapping reduces them to 155 measures: 125 schema DVs and 30 questionnaire items.

| DV | Studies | Columns | Distinct names | Examples |
|---|---|---|---|---|
| NASA-TLX mental demand | 40 | 99 | 34 | `MentalLoad`, `mental_demand`, `TLX1`, `MD`, `Mental`, `TLX 1.1` |
| NASA-TLX effort | 34 | 63 | 37 | `Effort`, `EF`, `TLX5`, `tlx_1_5`, `TASK_1_B`, `HLFM_TLX_Eff` |
| NASA-TLX frustration | 33 | 56 | 34 | `Frustration`, `fr`, `TLX6`, `LyrLouNASA6` |
| Task completion time | 19 | 753 | 21 | `total time`, `completion_time_s`, `TaskCompletionTime.A`, `MT`, `rt` |
| SUS item 1 | 12 | 19 | 11 | `SUS1`, `u1`, `a_sus0`, `Mid-Air Gesture_Q1`, `item_1` |

(`value/harmonization_by_dv.csv` has every DV. 720 of the task-completion-time columns are eHMI-for-All's log files.)

Raw names also mislead. 110 normalized names occur in at least two studies, and 17 of them map to different DVs in different studies. Most stand for different measures: `A1` is a Van der Laan item in one study, an efficiency measure in a second and a task-feasibility item in a third. `TLX 1.2` is mental demand in one study and physical demand in another. `mental effort` is a NASA-TLX subscale in one study and the Paas scale in another. Others are the same item mapped at another grain: `perceivedSafety1` is the first Faas perceived-safety item in four studies and is averaged into the scale in a fifth (`value/label_ambiguity.csv`).

## 2. What becomes comparable

The table identifies the same 5,290 columns in four ways:

- **Raw column name:** the name as it appears in the data.
- **Normalized name:** the name with case, spaces and punctuation ignored.
- **Shared schema aliases:** the shared schema's alias lists alone, without study mappings.
- **OpenDV mapping:** the full mapping.

| Identifier | Measures in ≥ 3 studies | in ≥ 10 studies | Study pairs that share a measure | Mean Jaccard overlap |
|---|---|---|---|---|
| Raw column name | 50 | 0 | 1.5% | 0.003 |
| Normalized name | 52 | 1 | 3.1% | 0.007 |
| Shared schema aliases | 51 | 1 | 2.8% | 0.006 |
| **OpenDV mapping** | **78** | **20** | **31.1%** | **0.094** |
| *SIGCHI studies only (21):* raw / normalized / OpenDV | 0 / 11 / **37** | 0 / 0 / 0 | 1.0% / 3.8% / **24.3%** | |

Two things stand out:

- **The OpenDV mapping makes about twenty times as many study pairs comparable as raw names do, and ten times as many as normalized names.** It is also the only identifier under which any measure is reported by ten or more studies.
- **The shared alias lists alone add almost nothing over normalized names.** The study mappings, which match half of the DV columns, do most of the work.
- **Raw names do best within one lab.** Of the 50 raw names in three or more studies, 24 are items that one automotive lab names alike across its studies (`SART4`, `perceivedSafety2`, TiA and STS-AD items). The other 26 are NASA-TLX subscale and SUS item names (`effort`, `md`, `pf`, `sus1`–`sus10`) and `trust`. Before the audit curated that lab's items there were 27 in all.

(`value/poolability.csv`; the "measures" of the raw and normalized rows include false matches such as `A1`.)

### Pooled values

44 DVs are reported by at least five studies. All their pools rest on curated mappings only. Selected pools (random effects, Hartung–Knapp–Sidik–Jonkman interval; k in the 39 development studies alone in brackets):

| DV | k (development) | Pooled mean | HKSJ 95% CI | 95% prediction interval | I² |
|---|---|---|---|---|---|
| NASA-TLX mental demand (0–20) | 40 (19) | 8.80 | [7.96, 9.64] | [1.69, 15.91] | 99.5% |
| NASA-TLX effort (0–20) | 34 (13) | 8.95 | [7.94, 9.97] | [0.54, 17.37] | 99.5% |
| NASA-TLX frustration (0–20) | 33 (13) | 6.46 | [5.73, 7.18] | [−0.26, 13.18] | 99.2% |
| NASA-TLX total, raw (0–100) | 31 (13) | 39.27 | [35.76, 42.79] | [16.56, 61.98] | 98.8% |
| SUS usability (0–100) | 15 (6) | 71.32 | [67.52, 75.12] | [53.46, 89.18] | 96.9% |
| Perceived safety (Faas, −3..3) | 12 (10) | 1.20 | [0.70, 1.70] | [−1.64, 4.04] | 99.8% |
| UEQ pragmatic quality (−3..3) | 10 (6) | 1.37 | [0.99, 1.76] | [−0.21, 2.96] | 98.4% |
| UEQ hedonic quality (−3..3) | 10 (6) | 1.06 | [0.68, 1.43] | [−0.49, 2.61] | 98.1% |
| Trust rating (1–5) | 10 (6) | 3.73 | [3.40, 4.06] | [2.78, 4.69] | 98.6% |

- **More studies per pool:** across the corpus, 77 DVs are pooled from two or more studies, 67 from three or more and 23 from ten or more. The development studies alone give 62, 49 and 8.
- **Typical values:** the pooled SUS of 71 is close to the averages that large SUS collections report (about 68–70).
- **Wide prediction intervals:** a new study's mean can lie far from these values, so they describe where published studies sit rather than a norm.
- **Measurement invariance:** no shared scale reaches it (`analysis/measurement_invariance.csv`). Pooled means describe where studies sit, not comparable measurements.
- **Task completion time** (k = 19) mixes unrelated tasks, so its pooled mean has no meaning.

## 3. Relations that can be pooled

`construct_relations` pools the correlations between DVs across studies. It does this at three levels of harmonization: identical raw names, the shared schema with header matching and instrument scoring, and the curated study mappings. Between-person correlations:

| Harmonization level | Studies with relations | Measure pairs in ≥ 2 studies | in ≥ 3 studies | Cross-construct pairs in ≥ 3 studies | Findings | Cross-construct findings |
|---|---|---|---|---|---|---|
| Identical raw column names | 72 | 71 | 45 | 3 | 2 | 0 |
| + shared schema, header matching, instrument scoring | 21 | 25 | 21 | 0 | 11 | 0 |
| **+ study mappings (OpenDV)** | **77** | **202** | **129** | **100** | **25** | **8** |

Within persons (across conditions), the curated level has 74 pairs in ≥ 3 studies, 51 of them cross-construct, and 33 findings, 19 of them cross-construct.

The pair type counts as "same instrument" the scores and subscales of one instrument in the registry (`schemas/instruments.yaml`). The registry has no SART entry, so pairs of SART items or subscales count as cross-construct: the three such pairs at the raw-name level, one of the eight cross-construct findings between persons (SART demand × supply) and one of the nineteen within persons (supply × understanding).

A finding is a pooled correlation over at least three studies, not computed from shared items, whose Hartung–Knapp–Sidik–Jonkman interval excludes zero at Benjamini–Hochberg q < .05. It generalizes when its 95% prediction interval excludes zero as well. The cross-construct findings at the curated level are about how workload, usability, acceptance and trust go together. Some examples:

| Relation | Level | k | Pooled r [HKSJ 95% CI] | I² | Generalizes |
|---|---|---|---|---|---|
| NASA-TLX total × SUS usability | between persons | 8 | −0.41 [−0.48, −0.34] | 0% | yes |
| NASA-TLX frustration × SUS usability | between persons | 7 | −0.51 [−0.60, −0.41] | 0% | yes |
| NASA-TLX temporal demand × SUS usability | between persons | 7 | −0.36 [−0.48, −0.23] | 0% | yes |
| Van der Laan satisfying × SUS usability | between persons | 4 | 0.69 [0.46, 0.84] | 69% | no |
| NASA-TLX total × SUS usability | within persons | 5 | −0.60 [−0.77, −0.34] | 88% | no |
| Van der Laan usefulness × NASA-TLX performance | within persons | 4 | −0.23 [−0.35, −0.11] | 0% | yes |
| Perceived risk × trust | within persons | 3 | −0.62 [−0.71, −0.51] | 10% | yes |
| NASA-TLX mental demand × perceived safety | within persons | 7 | −0.40 [−0.58, −0.18] | 98% | no |

Participants who report more workload rate a system as less usable, with a consistent strength across the eight studies that measured both. The same holds within participants across conditions. A single study cannot show such a regularity, and raw column names cannot find it. The six NASA-TLX subscales go by 30 to 37 names each, and the SUS items by 11 names each. Most studies' SUS and TLX totals are computed from those items by the analysis.

Without the curated mappings, no correlation between two different constructs (say, workload and trust) can be pooled over three studies. Every pair found at the lower levels is two parts of one instrument: NASA-TLX subscales, SUS items, or the three SART items that one lab names alike in three studies. (`relations/harmonization_ladder.csv`, `relations/relations_pooled.csv`.)

## 4. Does the vocabulary transfer?

- **The held-out studies needed no new DV.** The 49 held-out studies were mapped after the schema was built, each set frozen before its data were analysed. Git history shows no DV was added for them (`value/schema_growth.csv`). Their mappings use schema DVs 319 times, all of which already existed. The schema grew from 54 to 163 DVs while the development studies were mapped. Since 2026-11-07 (`touch_offset`, for two touch-input studies) it has not changed, until the instrument additions of 3.4.
- **What the held-out mappings could not map.** The held-out mappings list the measures they left without a schema DV. Most are custom single items, plus a few recurring instruments: the Misery Scale (MISC; four held-out studies and one development study), the Single Ease Question, UMUX, the Slater–Usoh–Steed presence questionnaire, the Flow Short Scale and embodiment ratings. Schema 3.4 adds DVs for these and for other standard instruments of SIGCHI venues (PXI, GEQ, IMI, UES-SF subscales, CSI, text entry, Fitts throughput). The additions were checked by a batch run of all 88 studies before and after: every one of the 31,976 columns kept its mapping and every table its content.
- **`opendv check-mapping`** checks a new mapping against the schema. Run on the 87 study mappings of the corpus with their manifest flags, it finds no error and no target outside the vocabulary apart from four study-specific measures, plus two identifier columns mapped twice under different case. Before its manifest entry set `override_standard_aliases`, it flagged the one curated DV the batch lost: osf_cwd6h's perceived risk (`ArduinoData1`), captured by a shared sensor alias.

## 5. Where mapped values still go wrong: scales

The mapping solves naming, but a DV column can still be on another scale than the DV. A scale audit compares each study × DV's values with the DV's canonical range (`value/scale_audit.csv`, 579 pairs):

| Status | Pairs |
|---|---|
| in the canonical range | 398 |
| on a common scale outside it, rescaled by the analysis (e.g. NASA-TLX on 0–100, Van der Laan items on 1–5, UEQ items on 1–7) | 158 |
| in the bottom half of the range: review for a narrower recording scale | 23 |

The first run of the audit found two code defects and eleven mappings with undeclared scales:

- **A wrong coding (fixed).** The instrument registry scored Running into Traffic's NASA-TLX subscales, recorded 0..21, on 0–100. Its raw TLX came out 3.89 instead of 18.52, and the development pool of the TLX total 38.02 [29.15, 46.89] instead of 39.13 [31.75, 46.50] (k = 13). The study's mapping now declares the 0–21 scale too (PR #51); the rule covers the next study whose scale nobody declared.
- **A scale read off the data (fixed).** Values that fit no common scale were rescaled from their observed minimum and maximum. This turned Music Intensity's TLX totals (40–1720, formula unstated) into an arbitrary 0–100 mean. Such values are no longer pooled.
- **Undeclared narrower scales in development mappings (fixed).** Six development study × DV pairs sat in the bottom half of their range: SART items on 1–7 in three studies, a 5-point SAM valence, and perceived-risk ratings on 1–5 and 0–10. Each study's documentation confirmed the narrower scale, and their mappings now declare `scale_conversions` (PR #51).
- **Undeclared narrower scales in held-out mappings (corrected for the corpus).** Six held-out mappings note a NASA-TLX scale of 1–10, 1–9, 1–7 or 0–1 in their headers but declare no `scale_conversions`. One partial TLX was also captured by the shared alias `tlxscore`. The analysis cannot see a narrower scale inside the DV's range, so these were pooled as 0–20 subscales and 0–100 totals. The corpus uses corrected copies (`analyses/opendv_corpus/corrections.py`, `mappings/`); the frozen held-out mappings and their pre-registered results stay as they were. The held-out results are correlations, which a linear rescaling does not change. Together with the code fixes, the corrections raise the corpus's pooled TLX total from 34.76 [29.35, 40.17] (k = 32) to 39.27 [35.76, 42.79] (k = 31), and mental demand from 8.04 to 8.80.
- **Pairs to review.** Of the 23 pairs flagged now, 22 are plausibly low values: SSQ, FMS and error-rate columns, two studies whose TLX is documented on 1–20, a reverse-keyed SUS item and one all-zero physical demand. The remaining one is ehmi_infrastructure_2024's perceived risk, whose header says 1–10 although the data contain zeros.

For new mappings, this means: declare the recording scale of every rating (`scale_conversions`) and run `check-mapping`. The [mapping guide](sigchi_mapping_guide.md) lists these and the other pitfalls seen in the corpus.

## Limitations

- **Domain:** the corpus leans to automotive studies (36 of 88), and 21 of its papers are from SIGCHI venues. The instruments common there (NASA-TLX, SUS, UEQ, Van der Laan, trust, SSQ) are the ones the pooled values cover.
- **Mapping authors:** the held-out mappings were written by the team that built the schema. They show that the vocabulary covers new studies, not that other researchers would map them the same way.
- **Heterogeneity:** the pooled means come from very heterogeneous studies (I² ≈ 99%). Their weights rest on observation counts, so large exports weigh heavily.
- **Scale audit:** the audit is a heuristic. It can only flag scales whose values stay low, not a narrower scale whose values fill it.
