# Pre-registration: second held-out test of the cross-study relations

Written on 2026-11-14, before any participant data of the datasets below were downloaded. The commit that adds this file, the screening decisions, the pinned file list and the decision code comes before every commit that adds data, mappings or results, so `git log` is the timestamp.

## Background

`docs/cross_study_insights.md` reports relations between constructs pooled across the 39 development studies. They are exploratory: the finding rule and its false-discovery family were fixed after a first look at the first 23 studies.

The first held-out test (`docs/preregistration_heldout_relations.md`, results in `docs/heldout_results.md`, outputs at `5e266f5`) confirmed part of them on 19 unseen studies:

- **H1** (mental demand × perceived safety, within persons) was **not testable**: no held-out study measured both.
- **H2** (the three core NASA-TLX pairs, within persons) was **not supported, narrowly.** All three pooled correlations were positive (r = 0.42 to 0.60, k = 5), but the Holm-corrected one-sided p was 0.052 for each, because the studies disagreed on the size (I² = 94 to 97%).
- **H2b** (performance is the weakest-linked TLX subscale) was **met**.
- **H3** (header matching before the LLM) was **supported**.

This second test asks the same questions, and whether every other finding recurs, on studies that played no part in either analysis.

## Data

### Candidates

The 2026-11-05 search (`docs/open_data_candidates.md`) left 79 datasets "to check": plausible user studies whose paper or measures had not been verified. They belong neither to the development set nor to the first held-out test. To them come the five datasets that the first test dropped before download because their tables seemed to exist only as `.mat` or `.edf` files.

**May the five first-test drops join? Yes, under the same criteria as the 79.**

- Their participant data were never downloaded or opened: the first test read only their file listings and READMEs (`analyses/candidate_search/file_selection.csv`), and `data/raw/heldout/` holds no folder for them.
- They contributed nothing to the first test's results and played no part in the development analysis. A second test that includes them is independent of both.
- They were dropped for a technical reason decided from file listings (no reader for the format), not for anything in their data, so letting them in introduces no outcome-dependent selection.
- The first test's records stay as they are; only its unused datasets are reused.

### Screening

Each of the 84 datasets was checked from metadata, READMEs, archive listings and papers only. No participant data were opened.

1. `analyses/candidate_search/check_candidates.py` collected each dataset's file listing, the member list of every zip archive (from its central directory, by HTTP range requests) and the text of its READMEs, codebooks and questionnaires (at most 30 kB, PDFs at most 2 MB).
2. Four AI subagents read these dossiers and each dataset's paper under fixed criteria and proposed a verdict. Claude reviewed every verdict and decided.
3. Two rules were sharpened while screening, before any decision was final, and applied to all 84:
    - each NASA-TLX subscale, UEQ scale and Van der Laan scale counts as its own DV, as in the relation analysis;
    - two DVs scored from shared items (the SSQ subscales and their total) do not make a pair.

A dataset **qualifies** when all of the following hold:

| | Criterion |
|---|---|
| C1 | Participant-level data (one row per participant, trial or condition) in a file the pipeline reads (CSV, TSV, Excel, SPSS, JSON, MATLAB v4–v7.2, or a delimited `.txt` named explicitly), downloadable without registration |
| C2 | A user study: participants interacted with a system, task or stimuli during the study. Surveys about technology in general or about products recalled from daily use, clinical efficacy trials and psychophysics are excluded |
| C3 | At least two schema DVs measured on the same participants and present in the files, forming at least one pair not scored from shared items |
| C4 | At least 10 participants |
| C5 | A peer-reviewed paper that is the paper of these data |
| C6 | No participants shared with a development or first-test study |

**Outcome:** 36 qualify, 48 do not. Every decision and its reason is in `analyses/candidate_search/check_decisions.csv`.

| Reason for exclusion | Datasets |
|---|---|
| C3: fewer than two schema DVs in the files, or only raw time series | 30 |
| C2: no user study (surveys, a clinical trial, psychophysics, a perspective article) | 10 |
| C1: no readable participant-level table (no data shared, PNG figures, scanned questionnaires, statistics only, headerless event logs) | 5 |
| C4: fewer than 10 participants | 2 |
| C6: the dataset paper of the development study SenseCobot | 1 |

**None of the five first-test drops qualifies.** On closer reading:

- `driver_trust_index_2026` shares its trust scores only as PNG figures;
- `partial_automation_ui_2024` holds raw simulator time series in `.mat`; its trust, risk and acceptance ratings are not shared;
- `ehmi_flash_text_2023` and `ehmi_location_2019` give perceived safety only as raw keypress or eye-tracker time series, and no file is documented to hold their questionnaire ratings;
- `ehmi_gimmick_2022` is a perspective article that re-plots two earlier experiments.

**MATLAB files.** None of the 36 qualifying datasets keeps its DVs only in `.mat` files. `opendv/data_loaders.py` reads `.mat` files from this commit on, so a table that turns out to exist only as `.mat` can still be read.

**No split.** All 36 form the second held-out set. Nothing is set aside for development: the purpose is a test.

### Independence

- No dataset is a development or first-test study. A survey by the senior author of the first test's `av_acceptance_diverse_2025` with a separate sample is excluded anyway (C2).
- Possible overlaps among the 36 are reported, not resolved:
    - `build_pc_mar_2026` and `build_3d_mr_2025`: one group, two samples of 60 students with different gender compositions (9 vs 17 women);
    - `vr_classroom_asa_2022` and `vr_classroom_noise_2025`: one lab, a proof-of-concept and a later experiment.
- **One dataset is one study**, as in the first test. `human_swarm_pfm_2025` holds three experiments with 60 different participants each. It enters as one study, with participant ids kept apart per file. `sus_de_pos_2025` enters with its second study only: the first rated products from memory (C2).

### The 36 datasets

| Study | Domain | N | Design | DVs in the files | Relevant to |
|---|---|---|---|---|---|
| steering_gain_2022 | automotive | 24 | 4 steering conditions | NASA-TLX six subscales per condition | H2, H4 |
| remote_takeover_replay_2026 | automotive | 18 | 2 GUIs | TLX total, SART, takeover time | |
| audiobook_driving_2024 | automotive | ~40 | audiobook levels | SDLP, hazard reaction time, headway | |
| risky_driving_age_2017 | automotive | 115 | 3 scenarios | SDLP, crashes | |
| turn_by_turn_wayfinding_2010 | automotive | 50 | 2 drives | route time, recognition accuracy | |
| child_pedestrian_vr_2025 | automotive | 59 | 7 VR crossings | perceived safety, crossing initiation, dwell time | |
| cyclist_overtaking_2025 | automotive | 120 | crowd, once | perceived safety, acceptance, stress | |
| human_swarm_pfm_2025 | HRI | 180 | 2 scenarios, 3 experiments | NASA-TLX, SUS, Van der Laan, Jian trust | H2, H4 |
| handover_reaction_time_2025 | HRI | 20 | 10 handover blocks | trust, collaboration quality | |
| ar_telepresence_arm_2023 | HRI | 24 | 6 conditions | UMUX, blocks transferred | |
| spot_voice_gesture_2025 | HRI | 218 | 2×2 | number of commands, inter-command time | |
| teleoperation_latency_2026 | HRI | ~45 | 2 tasks × 5 latencies | completion time, blocks moved | |
| layer_interaction_elastic_2022 | interaction | 24 | 3 blocks | NASA-TLX, AttrakDiff, time, success | H2, H4 |
| vibrotactile_elastic_2023 | interaction | 23 | 4 feedback conditions | time, success | |
| debugging_cps_ar_vr_2026 | AR/VR | 36 | 3 conditions | NASA-TLX, SUS, UEQ-S, time, success | H2, H4 |
| mr_airplane_layouts_2021 | AR/VR | 24 | 3–6 conditions | TLX mental, physical demand, effort | H4 |
| navigable_wim_2026 | VR | ~35 | 3 techniques | task load, presence, hedonic quality, time | H2, H4 |
| vr_shopping_demands_2026 | VR | 22 | 7 demand conditions | NASA-TLX items | H2, H4 |
| vr_classroom_noise_2025 | VR | 24 | 4 noise blocks | modified TLX: mental demand, effort, frustration, performance | H2, H4 |
| ceap_360vr_2021 | VR | 32 | 2 blocks | NASA-TLX, SSQ, IPQ, valence, arousal | H2, H4 |
| vr_classroom_asa_2022 | VR | 21 | trials; SSQ pre/post | SSQ, IPQ, reaction time, errors | |
| immersive_analytics_scale_2026 | VR | ? | 4 dataset scales | accuracy, time, cognitive load, immersion | (H2 if NASA-TLX) |
| vr_propaganda_retention_2024 | VR | ? | environment × behaviour | valence, arousal, IPQ, retention | |
| teleport_distance_accuracy_2025 | VR | 60 | 30 conditions | failed teleports, throughput | |
| vr_spirit_ataxia_2023 | VR (patients) | 10 | one session | SUS, SSQ, PANAS, performance | |
| eeg_sonification_2025 | sonification | 40 | 2 blocks | NASA-TLX, kappa, response time, clicks | H2, H4 |
| exergame_children_2025 | exergames | 21 | 6 intensity levels | children's NASA-TLX, HRV, skin conductance, success | H2, H4 |
| build_pc_mar_2026 | AR education | 60 | once | NASA-TLX six subscales | H4 |
| build_3d_mr_2025 | MR education | 60 | once | NASA-TLX six subscales | H4 |
| p300_attention_bci_2024 | BCI | 47 | once | NASA-TLX six subscales | H4 |
| citizen_science_tutorial_2021 | visualization | 90 | once | NASA-TLX (five subscales), page times | H4 |
| pipeblocks_cicd_2025 | software tools | 10 | 2 tools | SUS, NASA-TLX | H4 |
| developers_prompt_docgen_2024 | software tools | 50 | once | UEQ | H4 |
| sus_de_pos_2025 | web usability | 877 | once | SUS, UEQ | H4 |
| chatgpt_roleplay_2025 | conversational agents | 20 | once | SUS, USE satisfaction | |
| simulst_subtitles_2026 | media accessibility | 40 | 2 videos | mental effort (Paas), satisfaction, dwell time | |

Data and paper DOIs, licences and files are in `analyses/heldout_2/sources.yaml`. Seven datasets are automotive, 29 from other domains. The first held-out set was mostly automotive (19 of 29 planned); this one is mostly not.

### What the documentation says will be testable

- **H1** is **expected not to be testable again.** No dataset documents NASA-TLX mental demand together with perceived safety, repeatedly or at all. The two that measure perceived safety (`child_pedestrian_vr_2025`, `cyclist_overtaking_2025`) have no NASA-TLX.
- **H2:** up to 11 datasets document NASA-TLX subscales measured repeatedly, with enough within-person degrees of freedom (`pipeblocks_cicd_2025` has df = 9 and drops out). Some may hold fewer subscales than documented.
- **H4, between persons:**
    - the NASA-TLX pairs: about 15 datasets;
    - SUS × workload: 3 (`human_swarm_pfm_2025`, `debugging_cps_ar_vr_2026`, `pipeblocks_cicd_2025`);
    - UEQ hedonic × pragmatic: 3 to 4 (`debugging_cps_ar_vr_2026`, `developers_prompt_docgen_2024`, `sus_de_pos_2025`, possibly `navigable_wim_2026`);
    - Van der Laan satisfying × usefulness: only `human_swarm_pfm_2025`, so it is not testable.

### What is downloaded

`python -m analyses.candidate_search.build_sources_2` pinned the files from the repositories' listings before any download. It writes `analyses/heldout_2/sources.yaml` and the role of every file and archive member in `analyses/heldout_2/file_selection.csv`.

- **Rules:** those of the first test (`build_sources.classify`), with two changes:
    - zip archives over 100 MB (1 GB in the first test) are read member by member, so a 300–800 MB archive is not downloaded for one small table;
    - per-dataset `pin` and `skip` patterns in `check_decisions.csv`, written from the documentation, add files the rules would leave out (per-participant JSON or questionnaire CSVs in one folder, results `.txt` files, questionnaire PDFs) and leave out exports that hold no DV of the study.
- **Total:** 36 datasets, 42.4 MB. The largest single download is an 18.8 MB archive (`vr_classroom_asa_2022`).

## Hypotheses

### Replication targets

The targets are every curated-level *finding* of the development analysis (`analyses/output_relations/relations_pooled.csv`, 39 studies) and of the first held-out test (`analyses/heldout/relations/relations_pooled.csv`). A finding there means:

- at least three studies;
- a pair not scored from shared items;
- an HKSJ 95% interval that excludes 0;
- BH q < .05 within its level and association.

Each target predicts the sign it was found with. No target was found with opposite signs in the two analyses.

`python -m opendv.analysis.heldout_replication targets` derives them. The list is committed with this file: `analyses/heldout_2/targets.csv`, 37 targets.

| Family | Association | Targets | Pairs |
|---|---|---|---|
| H1 | within persons | 1 | mental demand × perceived safety (−) |
| H2 | within persons | 3 | effort × frustration, effort × mental demand, frustration × mental demand (+) |
| H4 | within persons | 12 | the other 12 NASA-TLX subscale pairs (+) |
| H4 | between persons | 21 | 14 NASA-TLX subscale pairs (+); SUS usability with mental demand, effort, frustration, temporal demand and the TLX total (−); UEQ hedonic × pragmatic (+); Van der Laan satisfying × usefulness (+) |

Of the 21 between-person targets, 10 were found in both analyses, 9 only in the development analysis and 2 only in the first held-out test (effort × performance, mental demand × performance). All 16 within-person targets come from the development analysis: the first test had no within-person finding.

### Tests

All tests use the `curated` level: values that only a study's mapping file, the shared schema or header matching produced. No LLM-mapped column enters them. Within-person correlations are repeated-measures correlations (Bakdash & Marusich, 2017); between-person correlations are Pearson correlations of participant means. Both are pooled on Fisher's z with REML and the Hartung-Knapp-Sidik-Jonkman (HKSJ) interval, by `opendv.analysis.construct_relations` with its defaults:

- at least 10 participants per study;
- within-person df ≥ 10;
- a pair is pooled from two studies on.

A target is **testable** when at least two studies of this set estimate it.

- **H1 (primary): workload and perceived safety, within persons.** As in the first test.
    - *Prediction.* The pooled within-person correlation of `mental_demand` and `perceived_safety` is negative.
    - *Test.* One-sided HKSJ test of r < 0, α = .05. With fewer than two studies H1 is **not testable on this set**; a single study's r is then reported descriptively.
- **H2 (primary): the core of the NASA-TLX, within persons.** As in the first test.
    - *Prediction.* Effort × frustration, effort × mental demand and frustration × mental demand each have a positive pooled within-person correlation.
    - *Test.* One-sided HKSJ test of r > 0 per pair, α = .05, Holm-corrected over the three pairs. H2 holds if all three are significant. An untestable pair counts as p = 1.
- **H2b (secondary, descriptive): performance is the weakest-linked subscale, within persons.** As in the first test: its mean pooled within-person r with the other five subscales is the lowest of the six. Reported as met or not met.
- **H4 (secondary): replication of the other 33 targets.**
    - *Per target.* One-sided HKSJ test in the predicted direction. A testable target **replicates** when its p, Benjamini-Hochberg-adjusted over all testable H4 targets, is below .05.
    - *Outcome.* The number of testable targets that replicate, overall and per association, and the number whose pooled r has the predicted sign. No threshold on these numbers is claimed: the targets share studies and are not independent, so a binomial test across them would be wrong.
- **H3 (secondary): header matching before the LLM.** As in the first test: on the E1 DV columns of this set, header matching first and then the LLM maps more columns correctly than the LLM alone. Paired exact McNemar test, two-sided, α = .05; precision of accepted DV mappings with Wilson 95% intervals.
- **Multiplicity.** H1 and H2 are separate primary hypotheses, each decided at its own α, as in the first test. H2b, H3 and H4 are secondary.

`opendv.analysis.heldout_replication decide` makes every decision. For H1, H2, H2b and H3 it calls the functions that decided the first test (`opendv.analysis.heldout`), unchanged.

### Exploratory, labelled as such

- every pair that is no target, including new findings on this set (listed by the decision code as candidates for a later test);
- per-study correlations, prediction intervals, I², and a leave-one-study-out analysis of H2;
- the targets pooled over this set and the first held-out set together. This analysis follows a narrow miss of the first test, so it is a form of optional continuation; it is reported, never used to decide a hypothesis;
- the targets pooled over all studies (development, first and second held-out sets);
- the `llm` level and the harmonization ladder.

## Mappings and study designs

The rules of the first test apply unchanged:

- **Who.** Claude (the AI assistant operating this repository) writes one curated mapping file per dataset, `analyses/heldout_2/mappings/<source_id>_mapping.yaml`, from the dataset's README, codebook, questionnaire, paper and analysis code, plus the column headers and value ranges.
- **What.** The most specific schema DV; `meta_*` labels for non-DV columns; DV columns without a fitting schema DV are left unlabelled. The schema (3.3) is not changed during the test.
- **Perceived risk is not perceived safety.** A risk or anxiety rating is not reversed into `perceived_safety`.
- **Study designs.** Each dataset gets a `schemas/study_designs.yaml` entry: participant column, factors, domain.
- **Order.** Mappings, designs, table options and the batch manifest are committed in one commit ("Freeze second held-out mappings"). It comes before any correlation, pooled estimate or LLM answer is computed on these data. Looking at headers, value ranges and file structures to write mappings is allowed; computing statistics between columns is not.
- **Reader options.** A file whose table needs reshaping is read with the table options of `opendv/data_loaders.py` (`sheet`, `header`, `skip_rows`, `pivot`, `drop_columns`, `rename`, and for MATLAB files `variable`, `columns`, `dims`). If a file structure needs a reader feature that does not exist yet, the feature may be added before the freeze, with tests; the change is reported. Analysis code (`construct_relations`, `heldout`, `heldout_replication`, `multi_study_analysis`) is not changed after this commit.
- **No revision.** Mappings are not revised after the freeze. Errors found later are reported, not corrected.
- **Checks before the freeze.** As in the first test, one batch run without the LLM may check that every mapping key reaches its column and every study resolves its participant column. No relation between columns is computed in that run.

## Procedure

1. Ask the user for permission to download, listing files, sources and sizes.
2. Download and verify the pinned files: `python -m opendv.analysis.confirmatory fetch --sources analyses/heldout_2/sources.yaml --data-dir data/raw/heldout_2`.
3. Write and freeze the mappings and designs (one commit).
4. Run `analyses/heldout_2/run.sh`, which mirrors `analyses/heldout/run.sh`:
    - the batch without the LLM over `analyses/heldout_2/sources_manifest.yaml`;
    - `construct_relations` on this set's standardized files only, with the defaults above;
    - the E1 benchmark for H3 with the settings of the first test: `--shortlist definition --max-per-source 40 --seed 20261017 --e2-samples 0`, `--models google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, plausibility rules 1.3, header matching scored as in `confirmatory.header_match_answers`;
    - `heldout_replication decide`.
5. Report in `docs/heldout_results_2.md`, with every deviation from this plan.

Code is used as it stands at this commit, except for reader features added before the freeze (see above). If a step fails, the fault is fixed, the step is repeated, and both are reported.

## Downloads and untrusted data

- **What is downloaded.** Only the files pinned in `analyses/heldout_2/sources.yaml`: participant-level tables, codebooks, READMEs and questionnaires. Videos, Unity projects, ROS bags, raw physiological signals and code are left out.
- **Pinning.** Each file is pinned by its size and the checksum its repository publishes; each zip member by its size and CRC-32. From archives over 100 MB only the pinned members are fetched, by HTTP range requests.
- **Untrusted data.** Each dataset goes into its own directory. Files are read only with pandas, pyreadstat or `scipy.io.loadmat`, archives in memory. Pickles are never loaded, no code shipped with the data is run, and ad-hoc Python on the data runs isolated (`python -I`).

## Exclusion rules

**Datasets.** A selected dataset is dropped, and the drop reported with its reason, if:

- a pinned file fails its checksum or cannot be downloaded after retries;
- the downloaded files turn out to hold no participant-level data, or fewer than two schema DVs measured on the same participants (contrary to what the documentation said);
- the downloaded files show that its participants are those of a development or first-test study;
- its MATLAB file cannot be read by `scipy.io.loadmat` (v7.3/HDF5 files).

Each of these is decided from the files' structure and documentation before the freeze, never from a statistic between columns.

**Studies within an analysis.** `construct_relations` leaves a study out of a pair when fewer than 10 participants have both DVs, or when its within-person df is below 10. This is not decided by hand.

**Rows.** No outliers are removed. Rows that the authors' own documentation flags as excluded (failed attention checks, aborted sessions) are left out with the mapping option `exclude_rows`, decided before the freeze. Rows with missing values drop out pairwise, as in the earlier analyses.

## Power

- **H1.** Expected not testable (see Data).
- **H2.** A Monte-Carlo simulation (`python -m analyses.heldout_2.power_h2`, 2,000 runs per cell, seed 20261114) estimated power. It uses the repository's REML pooling and the HKSJ test, Holm over the three pairs, and requires all three to hold. Assumptions:
    - study effects drawn around the pooled r with between-study SD τ on Fisher's z;
    - the three pairs' study effects correlated 0.8;
    - 15–60 participants per study, three measurements each.

  Probability that H2 is supported:

  | Studies k | r = 0.30, τ = 0.25 | r = 0.30, τ = 0.40 | r = 0.45, τ = 0.25 | r = 0.45, τ = 0.40 | r = 0.55, τ = 0.40 |
  |---|---|---|---|---|---|
  | 6 | 0.49 | 0.23 | 0.90 | 0.56 | 0.80 |
  | 8 | 0.74 | 0.34 | 0.99 | 0.77 | 0.94 |
  | 10 | 0.84 | 0.47 | 1.00 | 0.88 | 0.99 |

  The development estimates were r = 0.52–0.60 with τ = 0.09–0.13. The first held-out test's were r = 0.42–0.60 with τ = 0.36–0.49, or 0.25–0.45 without Physio-HRC. With 8 to 10 testable studies, power is high if the core pairs correlate about 0.45 and modest if only about 0.30. A miss on this set is therefore informative about the size, not only the sign.
- **H4.** Many targets rest on two or three studies. The HKSJ test then has one or two degrees of freedom and is weak; such targets will often be testable but not replicate. The count of same-sign estimates is reported for this reason.
- **H3.** As in the first test, power depends on how many held-out headers are question texts, which is unknown before download.

## Limitations stated in advance

- **Labeller.** As in the first test, the mappings come from the AI assistant that ran the exploratory analysis, not from independent human raters.
- **Screener.** One rater (Claude, with AI subagents reading papers and READMEs) screened the candidates, with no second human rater. The decisions and reasons are in `analyses/candidate_search/check_decisions.csv`.
- **What the screen could see.** Measures were judged from READMEs, codebooks, archive listings and papers. Some files will hold less than their documentation suggests; the exclusion rules above handle that, and every such drop is reported.
- **Targets are not independent.** The 15 NASA-TLX pairs share studies, and the H4 targets share them with H2. The count of replicated targets is descriptive.
- **Same pipeline.** The test checks whether the relations recur when the same pipeline harmonizes new studies. It does not separate the constructs from the pipeline's decisions (schema, mapping rules).
