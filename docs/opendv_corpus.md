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
- **Columns:** the batch read 31,975 columns from 1,944 tables, and every study standardized without error. Of those columns, 5,290 measure a DV: 4,544 map to a schema DV, and the rest are items that the analysis scores into one. They come from 86 studies. Two studies have no DV column: ai_label_trust_2025's Likert items are not described in its files, so its mapping leaves them out, and osf_cwd6h's only DV column is captured by a sensor alias (see §4).

## 1. One measure, many names

The 5,290 DV columns carry 1,129 distinct names, even after case, spaces and punctuation are ignored. The mapping reduces them to 151 measures: 125 schema DVs and 26 questionnaire items.

| DV | Studies | Columns | Distinct names | Examples |
|---|---|---|---|---|
| NASA-TLX mental demand | 40 | 99 | 34 | `MentalLoad`, `mental_demand`, `TLX1`, `MD`, `Mental`, `TLX 1.1` |
| NASA-TLX effort | 34 | 63 | 37 | `Effort`, `EF`, `TLX5`, `tlx_1_5`, `TASK_1_B`, `HLFM_TLX_Eff` |
| NASA-TLX frustration | 33 | 56 | 34 | `Frustration`, `fr`, `TLX6`, `LyrLouNASA6` |
| Task completion time | 18 | 33 | 20 | `completion_time_s`, `TaskCompletionTime.A`, `MT`, `rt` |
| SUS item 1 | 12 | 19 | 11 | `SUS1`, `u1`, `a_sus0`, `Mid-Air Gesture_Q1`, `item_1` |

(`value/harmonization_by_dv.csv` has every DV.)

Raw names also mislead. 92 normalized names occur in at least two studies, and 13 of them stand for different measures in different studies. `A1` is a Van der Laan item in one study, an efficiency measure in a second and a task-feasibility item in a third. `TLX 1.2` is mental demand in one study and physical demand in another. `mental effort` is a NASA-TLX subscale in one study and the Paas scale in another (`value/label_ambiguity.csv`).

## 2. What becomes comparable

The table identifies the same 5,290 columns in four ways:

- **Raw column name:** the name as it appears in the data.
- **Normalized name:** the name with case, spaces and punctuation ignored.
- **Shared schema aliases:** the shared schema's alias lists alone, without study mappings.
- **OpenDV mapping:** the full mapping.

| Identifier | Measures in ≥ 3 studies | in ≥ 10 studies | Study pairs that share a measure | Mean Jaccard overlap |
|---|---|---|---|---|
| Raw column name | 27 | 0 | 1.4% | 0.003 |
| Normalized name | 29 | 1 | 3.0% | 0.007 |
| Shared schema aliases | 28 | 1 | 2.7% | 0.006 |
| **OpenDV mapping** | **62** | **20** | **31.4%** | **0.098** |
| *SIGCHI studies only (20 with DV columns):* raw / normalized / OpenDV | 0 / 11 / **37** | 0 / 0 / 0 | 1.1% / 4.2% / **24.7%** | |

Two things stand out:

- **The OpenDV mapping makes about ten times as many study pairs comparable as raw names do.** It is also the only identifier under which any measure is reported by ten or more studies.
- **The shared alias lists alone add almost nothing over normalized names.** The study mappings, which match 40% of the DV columns, do most of the work.

(`value/poolability.csv`; the "measures" of the raw and normalized rows include false matches such as `A1`.)

### Pooled values

40 DVs are reported by at least five studies. All their pools rest on curated mappings only. Selected pools (random effects, Hartung–Knapp–Sidik–Jonkman interval; k in the 39 development studies alone in brackets):

| DV | k (development) | Pooled mean | HKSJ 95% CI | 95% prediction interval | I² |
|---|---|---|---|---|---|
| NASA-TLX mental demand (0–20) | 40 (19) | 8.81 | [7.97, 9.65] | [1.70, 15.91] | 99.5% |
| NASA-TLX effort (0–20) | 34 (13) | 8.96 | [7.95, 9.97] | [0.60, 17.32] | 99.5% |
| NASA-TLX frustration (0–20) | 33 (13) | 6.46 | [5.74, 7.18] | [−0.26, 13.18] | 99.2% |
| NASA-TLX total, raw (0–100) | 31 (13) | 39.27 | [35.76, 42.79] | [16.56, 61.98] | 98.8% |
| SUS usability (0–100) | 15 (6) | 71.32 | [67.52, 75.12] | [53.46, 89.18] | 96.9% |
| UEQ pragmatic quality (−3..3) | 10 (5) | 1.44 | [1.06, 1.82] | [−0.17, 3.04] | 98.4% |
| UEQ hedonic quality (−3..3) | 10 (5) | 1.06 | [0.68, 1.43] | [−0.49, 2.61] | 98.1% |
| Trust rating (1–5) | 10 (6) | 3.73 | [3.40, 4.06] | [2.78, 4.69] | 98.6% |

- **More studies per pool:** across the corpus, 69 DVs are pooled from two or more studies, 55 from three or more and 23 from ten or more. The development studies alone gave 54, 36 and 7.
- **Typical values:** the pooled SUS of 71 is close to the averages that large SUS collections report (about 68–70).
- **Wide prediction intervals:** a new study's mean can lie far from these values, so they describe where published studies sit rather than a norm.
- **Measurement invariance:** no shared scale reaches it (`analysis/measurement_invariance.csv`). Pooled means describe where studies sit, not comparable measurements.
- **Task completion time** (k = 18) mixes unrelated tasks, so its pooled mean has no meaning.

## 3. Relations that can be pooled

`construct_relations` pools the correlations between DVs across studies. It does this at three levels of harmonization: identical raw names, the shared schema with header matching and instrument scoring, and the curated study mappings. Between-person correlations:

| Harmonization level | Studies with relations | Measure pairs in ≥ 2 studies | in ≥ 3 studies | Cross-construct pairs in ≥ 3 studies | Findings | Cross-construct findings |
|---|---|---|---|---|---|---|
| Identical raw column names | 71 | 66 | 42 | 0 | 2 | 0 |
| + shared schema, header matching, instrument scoring | 21 | 27 | 21 | 0 | 11 | 0 |
| **+ study mappings (OpenDV)** | **75** | **179** | **114** | **85** | **25** | **8** |

Within persons (across conditions), the curated level adds 59 pairs in ≥ 3 studies, 36 of them cross-construct, and 29 findings, 15 of them cross-construct.

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

Participants who report more workload rate a system as less usable, with a consistent strength across the eight studies that measured both. The same holds within participants across conditions. A single study cannot show such a regularity, and raw column names cannot find it. The six NASA-TLX subscales go by 30 to 37 names each, and the SUS items by 11 names each. Most studies' SUS and TLX totals are computed from those items by the analysis.

Without the curated mappings, no correlation between two different constructs (say, workload and trust) can be pooled over three studies. Every pair found at the lower levels is two subscales of one instrument. (`relations/harmonization_ladder.csv`, `relations/relations_pooled.csv`.)

## 4. Does the vocabulary transfer?

- **The held-out studies needed no new DV.** The 49 held-out studies were mapped after the schema was built, each set frozen before its data were analysed. Git history shows no DV was added for them (`value/schema_growth.csv`). Their mappings use schema DVs 319 times, all of which already existed. The schema grew from 54 to 163 DVs while the development studies were mapped. Since 2026-11-07 (`touch_offset`, for two touch-input studies) it has not changed, until the instrument additions of 3.4.
- **What the held-out mappings could not map.** The held-out mappings list the measures they left without a schema DV. Most are custom single items, plus a few recurring instruments: the Misery Scale (MISC; four held-out studies and one development study), the Single Ease Question, UMUX, the Slater–Usoh–Steed presence questionnaire, the Flow Short Scale and embodiment ratings. Schema 3.4 adds DVs for these and for other standard instruments of SIGCHI venues (PXI, GEQ, IMI, UES-SF subscales, CSI, text entry, Fitts throughput). The additions were checked by a batch run of all 88 studies before and after: every one of the 31,976 columns kept its mapping and every table its content.
- **`opendv check-mapping`** checks a new mapping against the schema. Run on the 85 study mappings of the corpus, it finds no target outside the vocabulary apart from four study-specific measures. It flags one curated DV that the batch loses: osf_cwd6h's perceived risk (`ArduinoData1`) is captured by a shared sensor alias.

## 5. Where mapped values still go wrong: scales

The mapping solves naming, but a DV column can still be on another scale than the DV. A scale audit compares each study × DV's values with the DV's canonical range (`value/scale_audit.csv`, 555 pairs):

| Status | Pairs |
|---|---|
| in the canonical range | 367 |
| on a common scale outside it, rescaled by the analysis (e.g. NASA-TLX on 0–100, Van der Laan items on 1–5, UEQ items on 1–7) | 159 |
| in the bottom half of the range: review for a narrower recording scale | 29 |

The audit found two code defects and six held-out mappings with undeclared scales:

- **A wrong coding (fixed).** The instrument registry scored Running into Traffic's NASA-TLX subscales, recorded 0..21, on 0–100. Its raw TLX came out 3.89 instead of 18.5. The development pool of the TLX total moves from 38.02 [29.15, 46.89] to 39.13 [31.75, 46.50] (k = 13).
- **A scale read off the data (fixed).** Values that fit no common scale were rescaled from their observed minimum and maximum. This turned Music Intensity's TLX totals (40–1720, formula unstated) into an arbitrary 0–100 mean. Such values are no longer pooled.
- **Undeclared narrower scales (corrected for the corpus).** Six held-out mappings note a NASA-TLX scale of 1–10, 1–9, 1–7 or 0–1 in their headers but declare no `scale_conversions`. One partial TLX was also captured by the shared alias `tlxscore`. The analysis cannot see a narrower scale inside the DV's range, so these were pooled as 0–20 subscales and 0–100 totals. The corpus uses corrected copies (`analyses/opendv_corpus/corrections.py`, `mappings/`); the frozen held-out mappings and their pre-registered results stay as they were. The held-out results are correlations, which a linear rescaling does not change. Together with the two fixes above, the corrections raise the corpus's pooled TLX total from 34.76 [29.35, 40.17] (k = 32) to 39.27 [35.76, 42.79] (k = 31), and mental demand from 8.04 to 8.81.
- **Pairs to review.** Of the 29 flagged pairs, 22 are plausibly low values: SSQ, FMS and error-rate columns, two studies whose TLX is documented on 1–20, a reverse-keyed SUS item and one all-zero physical demand. Seven need a look at their documentation, all perceived-risk, SART and valence columns. Six of these are development studies and are with the development mapping audit. The seventh is ehmi_infrastructure_2024's perceived risk, whose header says 1–10 although the data contain zeros.

For new mappings, this means: declare the recording scale of every rating (`scale_conversions`) and run `check-mapping`. The [mapping guide](sigchi_mapping_guide.md) lists these and the other pitfalls seen in the corpus.

## Limitations

- **Domain:** the corpus leans to automotive studies (36 of 88), and 21 of its papers are from SIGCHI venues. The instruments common there (NASA-TLX, SUS, UEQ, Van der Laan, trust, SSQ) are the ones the pooled values cover.
- **Mapping authors:** the held-out mappings were written by the team that built the schema. They show that the vocabulary covers new studies, not that other researchers would map them the same way.
- **Heterogeneity:** the pooled means come from very heterogeneous studies (I² ≈ 99%). Their weights rest on observation counts, so large exports weigh heavily.
- **Scale audit:** the audit is a heuristic. It can only flag scales whose values stay low, not a narrower scale whose values fill it.
