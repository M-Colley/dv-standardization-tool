# Results: confirmatory test of the hierarchical shortlist

Run on 2026-10-26, as pre-registered in `docs/preregistration_confirmatory_shortlist.md`. The pre-registration was committed (33161d8) before any data were downloaded. The gold labels were frozen (e625980) before any shortlist or LLM run. The runs used the pre-registered settings without deviation. Outputs: `analyses/confirmatory/results.json` and `analyses/confirmatory/runs/<shortlist>/`.

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
