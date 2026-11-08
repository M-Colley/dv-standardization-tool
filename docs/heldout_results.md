# Results: held-out confirmation of the cross-study relations

Run on 2026-11-05, as pre-registered in `docs/preregistration_heldout_relations.md`. Order of commits:

1. `bc60cdb`: pre-registration and split, before any download.
2. `5992d53`: pinned files and download tooling.
3. `58f260a`: mappings, study designs and test code frozen ("Freeze held-out mappings").
4. Then the batch, the relations and the LLM benchmark.

Outputs:

- `analyses/heldout/results.json` (decisions);
- `analyses/heldout/relations/` (relations per study and pooled, harmonization ladder);
- `analyses/heldout/runs/definition/` (E1 benchmark).

To rerun: `bash analyses/heldout/run.sh`.

## In short

| Hypothesis | Outcome |
|---|---|
| H1 (primary): mental demand × perceived safety is negative within persons | **Not testable:** no held-out study measures both |
| H2 (primary): the three core NASA-TLX pairs are positive within persons | **Not supported, narrowly.** All three pooled r are positive (0.42 to 0.60, k = 5) but Holm-corrected p = 0.052, because the studies disagree on the size |
| H2b: performance is the weakest-linked TLX subscale | **Met** (mean r 0.01 vs. 0.36 to 0.44) |
| H3: header matching before the LLM beats the LLM alone | **Supported:** 93 vs. 77 of 238 DV columns, 16 vs. 0 discordant, p = 3 × 10⁻⁵ |

## Data

Of the 29 held-out datasets, **19 were analysed** and 10 were dropped under the pre-registered rules (reasons in `analyses/heldout/sources_manifest.yaml`):

| Reason | Datasets |
|---|---|
| No table the pipeline reads (data only in `.mat`, `.edf`, figures or per-trial logs); decided before download | driver_trust_index_2026, ehmi_flash_text_2023, ehmi_gimmick_2022, ehmi_location_2019, partial_automation_ui_2024 |
| Cannot be downloaded (Dryad: HTTP 401 from the API, 403 from the public link) | tactile_dvi_braking_2026 |
| No DV in the schema in the downloaded tables | cyclist_av_decisions_2022, ehmi_behaviour_crowd_2021, ehmi_colours_crowd_2020, ped_gesture_vr_2021 |

Eight of the ten dropped datasets are TU Delft studies (4TU.ResearchData), whose participant data are mostly in MATLAB files or raw experiment exports. The held-out set that remains has 9 automotive and 10 other-domain studies.

Columns:

- The 19 datasets have 1,559 columns in the files the manifest reads.
- The frozen mappings label 279 as DVs and 317 as non-DV (`meta_*`). The rest are DVs with no fitting schema DV, or derived columns (z-scores, residuals, figure data).
- 73 ID and condition columns are caught by the never-map blocklist.

## Pre-registered hypotheses

All tests use the `curated` level and within-person correlations (repeated-measures r, Bakdash & Marusich 2017), pooled on Fisher's z with REML and the Hartung-Knapp-Sidik-Jonkman interval.

### H1: workload and perceived safety: not testable on this set

No held-out study measures both `mental_demand` and `perceived_safety` at least twice per participant (k = 0). The pre-registration expected this.

- The studies with perceived safety (coupled simulator, anticipated acceptance) have no NASA-TLX.
- The study with both workload and a safety-related rating (eHMI on the vehicle or the infrastructure) measured *perceived criticality*. That is a risk rating, which the pre-registration says is not reversed into perceived safety.

F1 therefore remains an exploratory finding of the development studies (k = 4).

### H2: structure of the NASA-TLX: not supported (narrowly)

Five held-out studies measure the six subscales repeatedly: eHMI infrastructure, function allocation, Physio-HRC, shared autonomy and tool embodiment.

| Pair | k | Pooled r | HKSJ 95% CI | 95% PI | I² | p (one-sided) | p (Holm) |
|---|---|---|---|---|---|---|---|
| effort × frustration | 5 | 0.44 | [0.02, 0.73] | [−0.65, 0.94] | 94% | 0.022 | 0.052 |
| effort × mental demand | 5 | 0.60 | [0.08, 0.86] | [−0.76, 0.98] | 97% | 0.017 | 0.052 |
| frustration × mental demand | 5 | 0.42 | [−0.10, 0.76] | [−0.79, 0.96] | 96% | 0.043 | 0.052 |

Per study (r, n participants):

| Study | effort × frustration | effort × mental demand | frustration × mental demand |
|---|---|---|---|
| ehmi_infrastructure_2024 (43) | 0.34 | 0.60 | 0.39 |
| function_allocation_hrc_2026 (26) | 0.21 | 0.42 | 0.19 |
| physio_hrc_2025 (87) | 0.80 | 0.91 | 0.83 |
| shared_autonomy_2026 (11) | 0.37 | 0.53 | 0.34 |
| tool_embodiment_2025 (69–70) | 0.26 | 0.21 | 0.06 |

All three pooled correlations are positive, and each is significant on its own (one-sided p = .017 to .043). After the pre-registered Holm correction over the three pairs, all three come to p = 0.052, just above α = .05. **H2, which needs all three to be significant, is not supported.** All 15 study-level correlations are positive.

The test misses because the studies disagree on the size of the correlation (I² = 94 to 97%), not because of any negative result. With k = 5, the HKSJ interval has 4 degrees of freedom and widens with that disagreement.

### H2b: performance is the weakest-linked subscale: met

Mean pooled within-person r of each subscale with the other five:

| Subscale | Mean r |
|---|---|
| performance | 0.01 |
| temporal demand | 0.36 |
| frustration | 0.37 |
| mental demand | 0.40 |
| physical demand | 0.43 |
| effort | 0.44 |

Performance is far below the others, as in the development studies (r = 0.29 to 0.45 there). Its link with the other subscales is weaker still on the held-out set.

### H3: header matching before the LLM: supported

E1 has **238 DV columns** (at most 40 per source) and 227 non-DV columns. The settings were those of the confirmatory shortlist test: `definition` shortlist, `google/gemma-4-E2B-it` at revision 3e22461f, plausibility rules 1.3.

| | LLM alone | Header matching, then the LLM |
|---|---|---|
| Correct mappings (of 238) | 77 | 93 |
| Precision of accepted DV mappings | 26.2% [21.5, 31.5] | 31.5% [26.5, 37.0] |

- **Test.** Paired exact McNemar: 16 columns correct only with header matching, 0 only without it, p = 3.1 × 10⁻⁵. **H3 is supported.**
- **What header matching did.**
  - It answered 21 DV columns, all correctly, and no non-DV column.
  - The 21 columns are question-text or labelled headers in three studies: the NASA-TLX questions of the developer and robot-feeding studies, and the "Mental demand (1 to 10)"-style headers of the eHMI infrastructure study.
  - The LLM alone got 5 of these 21 right.
- **Size of the effect.** The gain is smaller than in the post hoc rerun (42 discordant columns on the confirmatory studies), because fewer held-out headers are question texts. The pre-registration anticipated this.

## Exploratory (not pre-registered as tests)

- **The LLM alone remains weak on new studies.**
  - It mapped 77 of 238 DV columns correctly (32%), close to the 33% of the confirmatory shortlist test.
  - The gold DV was among its candidates for 192 (81%), so the model's choice, not retrieval, is the bottleneck again: 131 wrong DVs and 30 abstentions.
  - It mapped 86 of 192 curated non-DV columns to a DV.
  - Two failures stand out:
    - **Physio-HRC.** Columns named exactly as schema DVs (`mental_demand`, `effort`, ...): 6 of 13 correct once their own alias is hidden.
    - **The Van der Laan items of the coupled-HMD study.** 0 of 10; the gold item was never among the candidates.
- **Without Physio-HRC (leave-one-out).**
  - Physio-HRC's subscales correlate at r = 0.80 to 0.91 within persons, far above the other studies.
  - Leaving it out lowers the pooled estimates to r = 0.30, 0.45 and 0.25, and the heterogeneity with them.
  - The three one-sided p are then .002, .009 and .023. All three are below .05 after Holm.
  - Leaving out any other single study leaves the one-sided p between .02 and .09.
- **Held-out and development studies pooled** (k = 9 per pair):

  | Pair | Pooled r | HKSJ 95% CI | 95% PI |
  |---|---|---|---|
  | effort × frustration | 0.55 | [0.36, 0.69] | [−0.12, 0.88] |
  | effort × mental demand | 0.61 | [0.41, 0.76] | [−0.16, 0.92] |
  | frustration × mental demand | 0.49 | [0.28, 0.66] | [−0.24, 0.87] |

  Holm-corrected one-sided p ≤ 0.0006. The prediction intervals still include 0: a new study is expected to find a positive correlation of uncertain size, and a small chance of none.
- **The harmonization ladder recurs on new data.**
  - With identical raw column names, the 19 held-out studies share only 3 construct pairs across two studies, and none across three.
  - With the shared schema and header matching, also 3.
  - With the study mappings, 39 within-person pairs pool across at least two studies, and 22 across at least three.
  - Most of the gain on these studies comes from the curated mappings, not from the automatic steps (`analyses/heldout/relations/harmonization_ladder.csv`).

## Deviations and decisions made during the test

- **Pinned files committed after the download.** `analyses/heldout/sources.yaml` was written before any download, as pre-registered, but committed afterwards (`5992d53`). Its checksums are the ones the repositories publish, so they could not have been adapted to the data.
- **Download retries.** 4TU answered HTTP 429 (rate limit) on the first run. The fetch was given retries with back-off and run again, which fetched every 4TU file. This is a fault fixed and a step repeated, as the procedure allows.
- **No `.mat` files read.** The pre-registration allows `scipy.io.loadmat`, but the pipeline reads no MATLAB files. Datasets whose participant data exist only as `.mat` were dropped (see above); none of them names the NASA-TLX in its metadata.
- **A batch run before the freeze.** Before `58f260a`, one batch run without the LLM checked that every mapping key reached its column and that every study resolved its participant column. It found that the pipeline strips surrounding spaces from headers. Twelve keys with trailing spaces were stripped, and seven participant columns on the never-map list were added to the designs under their raw names. No relation between columns was computed in that run.
- **Physio-HRC counts as one study.** The pipeline treats a source as one study, so its five experiments enter as one study with ids kept apart per file.
- **A reader library added.** `pyreadstat` was installed to read the SPSS file of the music study (the pipeline's `.sav` loader needs it). `remotezip` was added as the optional `fetch` extra.
