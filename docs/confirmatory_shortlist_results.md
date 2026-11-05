# Results: confirmatory test of the hierarchical shortlist

Run on 2026-10-26, as pre-registered in `docs/preregistration_confirmatory_shortlist.md`. The pre-registration was committed (33161d8) before any data were downloaded. The gold labels were frozen (e625980) before any shortlist or LLM run. The runs used the pre-registered settings without deviation. Outputs: `analyses/confirmatory/results.json` and `analyses/confirmatory/runs/<shortlist>/`.

> **Note (2026-11-04).** A code review found that the benchmark prompt listed the held-out alias among the candidate DVs' aliases when it was one of their first three schema aliases (fixed in `llm_utils._format_candidate_reference`). In this test that applies to 1 of the 168 DV columns, so the results below stand; they were not rerun.

## Data

The seven datasets were downloaded and verified against their published checksums, and none was dropped. Of the 522 columns:

- 173 got a DV label, 210 a non-DV label, and 139 are DVs with no fitting schema DV (PXI, PENS, IMI, GAMEX, agency, custom items).
- No column name had to be left out as a conflict.
- 21 ID and condition columns are caught by the never-map blocklist before mapping and enter E1 as blocklisted non-DV items.

E1 has **168 DV columns** (the cap of 40 per source applied to Hi-Audio), plus 144 curated non-DV and 9 blocklisted columns.

## Primary and secondary outcomes (check on)

| | `definition` (default) | `dense_h` (primary) | `mixed_h` (secondary) |
|---|---|---|---|
| Correct mappings (of 168) | 55 | 53 | 56 |
| vs. `definition`, paired exact McNemar | | 8 vs. 10 discordant, p = 0.81 (Holm 1.0) | 5 vs. 4, p = 1.0 (Holm 1.0) |
| Precision of accepted DV mappings | 34.6% [27.6, 42.3] | 32.9% [26.1, 40.5] | 34.6% [27.7, 42.2] |
| Gold DV among the candidates | 114 | 117 | 108 |
| Abstained on DV columns | 35 | 40 | 38 |
| Curated non-DV columns mapped to a DV (of 144) | 26 | 33 | 32 |
| True mappings lost to the check (columns that reached the LLM) | 0 | 0 | 0 |

**H1 is not supported.** `dense_h` maps no more columns correctly than `definition` (53 vs. 55, p = 0.81), and its precision is lower. The decision rule is not met, so `definition` stays the default. H2 (`mixed_h`) is not supported either. The post hoc advantage on the development studies (65 vs. 52, then 69 vs. 52) does not replicate on new studies. It was most likely specific to those studies' instruments, mainly the scale/item conflicts of SUS and perceived safety that the filter was designed around.

## Exploratory (not pre-registered as tests)

- **Accuracy drops for every shortlist on new data.** Development studies: 52 of 101 correct (51%), 59% precision, gold among candidates for 82%. New studies: 55 of 168 (33%), 35% precision, 68%. This gap is larger than the difference between any two shortlists.
- **Where the definition shortlist fails** (168 DV columns): 55 correct, 78 wrong DV, 31 screened out before the LLM, 4 abstentions.
  - **Screened out.** All 20 XAI-trust items and 9 Hi-Audio columns were screened as non-numeric. The LimeSurvey answers are text codes, and the Hi-Audio task outcomes are Y/N. The pipeline does not recode either, so these columns never reach the LLM under any shortlist.
  - **Wrong despite the right candidate.** 44 of the 78 wrong answers had the gold DV among the candidates. The model's choice, not retrieval, is the main bottleneck.
  - **Long question-text headers.** In the VR study (full question sentences as headers) only 2 of 35 were correct. Several SUS items went to automotive DVs such as `pedestrian_crossing_time` or `interaction_events`, which suggests a bias from the schema's automotive-heavy aliases.
  - **Common confusions.** Within NASA-TLX (physical or temporal demand → `mental_demand`), within AttrakDiff (`attrakdiff_pq` / `hqs` / `hqi` → `attrakdiff_att`), and SUS items → TAM ease of use.
- **Correct instrument** (right questionnaire, possibly wrong item or level): `definition` 82, `dense_h` 76, `mixed_h` 85 of 168.
- **Per dataset**, correct with `definition`:

  | Dataset | Correct |
  |---|---|
  | PXI | 30 / 31 |
  | TAMAGO | 14 / 17 |
  | AgencyUX | 4 / 18 |
  | Hi-Audio | 3 / 40 |
  | Sonomancer | 2 / 7 |
  | VR | 2 / 35 |
  | XAI trust | 0 / 20 |

  Short, conventional item codes (PXI `PQ_3`, TAMAGO `sus_q4`) are mapped well. Question-text, coded or non-English headers are not.

## Limitations

- The gold labels are Claude's, made from the datasets' documentation. No independent human rater checked them. The labelling notes record a few debatable labels:
  - TAMAGO's 3-subscale TLX mean → `TLX_SCORE`;
  - Hi-Audio's self-reported total time → `task_completion_time`;
  - AgencyUX hit counts → `task_success`.
- One small model (Gemma 4 E2B) was tested.

## Consequences

1. `definition` stays the default shortlist; `hierarchical` stays opt-in and is documented as not replicated.
2. Accuracy on unseen studies is low (about a third of DV columns), so LLM-deduced mappings must stay quarantined from the main analyses (the `*_llm_excluded` outputs) or be reviewed by a person.
3. The largest gains on new data are outside the shortlist:
   - recode text answer codes and Y/N outcomes before screening;
   - handle question-text headers, e.g. by matching against instrument item wording;
   - reduce the automotive bias in the shortlist and prompt.

## Post hoc: after fixing the gaps (2026-11-03, updated 2026-11-04)

**Not part of the pre-registered test.** The first two gaps of consequence 3 were fixed after seeing these data, and header matching was designed while looking at these studies' headers. The numbers below are therefore optimistic and need new studies to confirm. The pre-registered results above are unchanged.

What changed in the pipeline:
- survey answer codes (LimeSurvey `A1`/`A01`, Y/N, Yes/No, Ja/Nein) are recoded before the non-numeric screen;
- header matching assigns blocks of question-text headers to registered instrument items by their wording;
- the post-LLM check never accepts a 0/1 column for a rating-scale DV;
- since 2026-11-04: `True`/`False` columns are recoded to 1/0, and semicolon-separated files with decimal commas are read as numbers.

The E1 runs were repeated with the pre-registered settings on the same 168 DV columns (`analyses/confirmatory/posthoc.sh`; outputs in `analyses/confirmatory/posthoc/`). The three shortlists were run one after another with the same cached manuscript context (identical context fingerprints in each summary) and no failed model call.

| | `definition` | `dense_h` | `mixed_h` |
|---|---|---|---|
| Correct, LLM only (pre-registered run) | 64 (55) | 54 (53) | 57 (56) |
| vs. the pre-registered run, paired exact McNemar | 9 vs. 0, p = 0.004 | 1 vs. 0, p = 1.0 | 1 vs. 0, p = 1.0 |
| Precision of accepted DV mappings | 36.4% [29.6, 43.7] | 29.8% [23.6, 36.9] | 31.0% [24.7, 38.0] |
| Abstained on DV columns (pre-registered run) | 14 (35) | 17 (40) | 14 (38) |
| Curated non-DV columns mapped to a DV, of 144 (pre-registered run) | 22 (26) | 30 (33) | 30 (32) |
| DV columns screened out before the LLM | 2 (31) | 2 (31) | 2 (31) |
| **Header matching first, LLM for the rest: correct** | **106** | **93** | **99** |
| Precision of accepted DV mappings | 59.9% [52.5, 66.8] | 50.5% [43.4, 57.7] | 53.8% [46.6, 60.9] |
| vs. LLM only, paired exact McNemar | 42 vs. 0, p < 0.001 | 39 vs. 0, p < 0.001 | 42 vs. 0, p < 0.001 |

- **The shortlist conclusion holds.** On the rerun `dense_h` vs. `definition` is 8 vs. 18 discordant (p = 0.076, Holm 0.15) and `mixed_h` 5 vs. 12 (p = 0.14), so `definition` stays the default.
- **Answer-code recoding** lets 28 of the 31 screened DV columns reach the LLM. With `definition`:
  - 8 XAI Hoffman trust items (`TrustHoffman[SQ001]`…) are mapped correctly to `trust_rating`;
  - the 12 Jian items go to `trust_rating` or TiA items instead of `jian_trust`;
  - of the 8 Hi-Audio task outcomes (Y/N), 7 get no answer, and for 1 the LLM proposed a rating-scale DV, which the indicator rule rejected.

  With the hierarchical shortlists none of the 28 is correct, and their precision falls, because more columns now reach the LLM and get wrong answers.
- **`True`/`False` recoding** lets the VR study's per-trial `Success` reach the LLM; all three shortlists map it correctly to `task_success` (the only change from the 2026-11-03 rerun, which these runs otherwise reproduce exactly).
- **Header matching** assigns 45 DV columns: Hi-Audio's TLX and SUS question headers, and the VR study's TLX and SUS headers in English and German. All 45 are correct, and no non-DV column is matched. With `definition`, the LLM alone got 3 of the 45 right. The matcher uses only the standard aliases and the instrument wording in `schemas/instruments.yaml`, never the gold labels or the studies' mapping files (`confirmatory.header_match_answers`).
- **Still screened out (2):**
  - Hi-Audio's free-text total time;
  - the VR study's `Time`, screened by name as an event time (the plausibility rule `event_time: exact [time]`). Its values are now read as numbers.
- **Decimal commas biased summaries, not only screening.** The VR study's semicolon-separated measurements use decimal commas (`24,825`). Before the fix numeric coercion kept only the integer-valued cells, so the catalog summarized `task_completion_time` from 259 of 8,640 manipulation trials (mean 29.3 s, mostly 30-s timeouts) instead of all of them (mean 9.3 s), and from 26 of 8,100 selection trials (14.6 s instead of 1.6 s).
- **Keying check** (registered items whose first-principal-component loading runs against their key):
  - **AgencyUX, TLX performance** (`NASA4`, loading −0.45, n = 126). This is a true positive. The study's codebook words the item "How successful were you in accomplishing what you were asked to do? low–high", the reverse of the TLX's perfect–failure scale.
  - **Hi-Audio, physical demand** (−0.59, n = 22; the same answers in two tables). This is probably not a keying error. With 22 people, physical demand correlates −0.55 with temporal demand. Hi-Audio's data dictionary also contradicts itself on the direction of its performance item, and the check cannot settle that either.

  The check is a warning, not a correction. At small n it can flag an item for reasons other than keying and miss a weakly reversed one.
