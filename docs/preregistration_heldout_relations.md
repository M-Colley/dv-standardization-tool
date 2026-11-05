# Pre-registration: held-out confirmation of the cross-study relations

Written on 2026-11-05, before any of the datasets below was downloaded. The commit that adds this file and `analyses/candidate_search/heldout_split.csv` comes before every commit that adds data, mappings or results, so `git log` is the timestamp.

## Background

The cross-study relations in `docs/cross_study_insights.md` are exploratory. The finding rule and its false-discovery family were fixed after a first look at the 23 development studies. Two results are to be confirmed on studies that played no part in finding them:

- **F1.** Within persons, NASA-TLX mental demand correlates negatively with perceived safety. Pooled within-person r = −0.44, HKSJ 95% CI [−0.72, −0.05], k = 4 (all negative), I² = 94%.
- **F2.** NASA-TLX subscales covary positively within persons:
    - effort, frustration and mental demand form the core (pooled r = 0.56–0.65);
    - performance is the weakest-linked subscale (r = 0.29–0.45 with the others).

A third result concerns the mapping pipeline. In the post hoc rerun of the shortlist test, matching question-text headers before asking the LLM raised correct mappings from 64 to 106 of 168 DV columns (`docs/confirmatory_shortlist_results.md`, post hoc section). That rerun was not pre-registered, because the matcher was designed while looking at those studies.

## Data

**Source.** The systematic repository search of 2026-11-05 (`docs/open_data_candidates.md`, second search) verified 57 datasets. They were split from metadata only, before anything was downloaded, by `analyses/candidate_search/split_heldout.py`:

- five strata: domain (automotive or other) crossed with the measures named in the metadata (NASA-TLX; safety, risk or trust; neither);
- within each stratum, sorted by DOI and permuted with the fixed seed 20261105, then half to each set;
- result: 29 datasets held out, 28 added to the development set.

The 79 datasets the search marked "to check" belong to neither set.

**Held-out datasets (29):**

| Stratum | Study | Paper | Data |
|---|---|---|---|
| automotive other | Tactile driver-vehicle interface for NDRTs and emergency braking | [10.1080/00140139.2026.2703690](https://doi.org/10.1080/00140139.2026.2703690) | [10.5061/dryad.vx0k6dk33](https://doi.org/10.5061/dryad.vx0k6dk33) |
| automotive other | Non-driving-related tasks and takeover in level 3 (simulator) | [10.1186/s12544-025-00745-6](https://doi.org/10.1186/s12544-025-00745-6) | [10.5281/zenodo.15281988](https://doi.org/10.5281/zenodo.15281988) |
| automotive other | Manual driving style and automated-driving takeover | [10.1016/j.apergo.2024.104366](https://doi.org/10.1016/j.apergo.2024.104366) | [10.4121/57e36cf2-4f74-4d2f-86e2-7fae7250438a.v1](https://doi.org/10.4121/57e36cf2-4f74-4d2f-86e2-7fae7250438a.v1) |
| automotive other | eHMI on the vehicle or the infrastructure, driving simulator | [10.59490/ejtir.2024.24.2.7273](https://doi.org/10.59490/ejtir.2024.24.2.7273) | [10.4121/8e1c6604-53c7-4905-8ef9-4a6b2acc4e7a.v1](https://doi.org/10.4121/8e1c6604-53c7-4905-8ef9-4a6b2acc4e7a.v1) |
| automotive other | AR for pedestrian-AV interaction, outdoor experiment | [10.3389/frobt.2024.1324060](https://doi.org/10.3389/frobt.2024.1324060) | [10.4121/a1f9f15c-1213-4657-8e4d-a154a725d747.v1](https://doi.org/10.4121/a1f9f15c-1213-4657-8e4d-a154a725d747.v1) |
| automotive other | eHMI flashing vs text with a gaze-contingent setup | [10.1016/j.trf.2023.07.005](https://doi.org/10.1016/j.trf.2023.07.005) | [10.4121/6dfe7bd5-16f7-4e8d-9ab7-f8203b902310.v1](https://doi.org/10.4121/6dfe7bd5-16f7-4e8d-9ab7-f8203b902310.v1) |
| automotive other | Coupled driver-pedestrian simulator | [10.1145/3543174.3546849](https://doi.org/10.1145/3543174.3546849) | [10.4121/20224281.v2](https://doi.org/10.4121/20224281.v2) |
| automotive other | eHMI: gimmick or necessity | [10.1016/j.trip.2022.100643](https://doi.org/10.1016/j.trip.2022.100643) | [10.4121/858e4b36-895b-4c84-928a-d1cdcf0920d6.v1](https://doi.org/10.4121/858e4b36-895b-4c84-928a-d1cdcf0920d6.v1) |
| automotive other | Coupled simulator with HMDs, driver-pedestrian | [10.1016/j.apergo.2022.103825](https://doi.org/10.1016/j.apergo.2022.103825) | [10.4121/20005565](https://doi.org/10.4121/20005565) |
| automotive other | Cyclists' decisions and AV characteristics | [10.1049/itr2.12235](https://doi.org/10.1049/itr2.12235) | [10.4121/20103188.v2](https://doi.org/10.4121/20103188.v2) |
| automotive other | Pedestrian-AV gesture communication in VR | [10.1155/2021/5573560](https://doi.org/10.1155/2021/5573560) | [10.4121/14406944.v1](https://doi.org/10.4121/14406944.v1) |
| automotive other | eHMI colours, crowd participants | [10.1109/smc42975.2020.9282998](https://doi.org/10.1109/smc42975.2020.9282998) | [10.4121/12948650.v3](https://doi.org/10.4121/12948650.v3) |
| automotive other | eHMI display location | [10.3390/info11010013](https://doi.org/10.3390/info11010013) | [10.4121/12682826](https://doi.org/10.4121/12682826) |
| automotive safety | Driver trust index | [10.1080/10447318.2026.2727771](https://doi.org/10.1080/10447318.2026.2727771) | [10.5281/zenodo.20627050](https://doi.org/10.5281/zenodo.20627050) |
| automotive safety | Anticipated acceptance of automated driving | [10.1145/3743049.3743051](https://doi.org/10.1145/3743049.3743051) | [10.17605/osf.io/dvsbh](https://doi.org/10.17605/osf.io/dvsbh) |
| automotive safety | Labelling automotive AI as trustworthy or reliable | [10.1038/s41598-025-85558-2](https://doi.org/10.1038/s41598-025-85558-2) | [10.17605/osf.io/6un4h](https://doi.org/10.17605/osf.io/6un4h) |
| automotive safety | Interfaces for partially automated vehicles | [10.1016/j.trf.2024.02.009](https://doi.org/10.1016/j.trf.2024.02.009) | [10.4121/18bb5d48-c1e5-4d50-ad71-d9010e4494d9.v2](https://doi.org/10.4121/18bb5d48-c1e5-4d50-ad71-d9010e4494d9.v2) |
| automotive safety | eHMI behaviour, 1,434 crowd participants | [10.1016/j.apergo.2021.103450](https://doi.org/10.1016/j.apergo.2021.103450) | [10.4121/14465715.v2](https://doi.org/10.4121/14465715.v2) |
| automotive tlx | Music intensity and lyrics in simulated urban driving | [10.1016/j.trf.2021.05.022](https://doi.org/10.1016/j.trf.2021.05.022) | [10.6084/m9.figshare.13385834](https://doi.org/10.6084/m9.figshare.13385834) |
| other no tlx | Mid-air gestures vs touch for map-making | [10.1080/15230406.2026.2619027](https://doi.org/10.1080/15230406.2026.2619027) | [10.6084/m9.figshare.30833309](https://doi.org/10.6084/m9.figshare.30833309) |
| other no tlx | Proxemics for human-agent interaction (CHI 2022) | [10.1145/3491102.3517593](https://doi.org/10.1145/3491102.3517593) | [10.18419/darus-2525](https://doi.org/10.18419/darus-2525) |
| other tlx | Shared-autonomy interface modalities | [10.1109/tnsre.2026.3657400](https://doi.org/10.1109/tnsre.2026.3657400) | [10.5281/zenodo.14198913](https://doi.org/10.5281/zenodo.14198913) |
| other tlx | Developers' eye tracking and creativity | [10.1145/3794860.3794891](https://doi.org/10.1145/3794860.3794891) | [10.6084/m9.figshare.31198108](https://doi.org/10.6084/m9.figshare.31198108) |
| other tlx | Function allocation in physical HRC | [10.1109/lra.2025.3632607](https://doi.org/10.1109/lra.2025.3632607) | [10.4121/d55393f4-f6b6-4076-86c1-7bf351f16787.v1](https://doi.org/10.4121/d55393f4-f6b6-4076-86c1-7bf351f16787.v1) |
| other tlx | Teleoperated robots | [10.1109/iros60139.2025.11245894](https://doi.org/10.1109/iros60139.2025.11245894) | [10.5281/zenodo.19040256](https://doi.org/10.5281/zenodo.19040256) |
| other tlx | Tool embodiment under task load, young and older adults | [10.3389/frvir.2025.1637212](https://doi.org/10.3389/frvir.2025.1637212) | [10.17605/osf.io/uxk58](https://doi.org/10.17605/osf.io/uxk58) |
| other tlx | Robot-assisted feeding outside the lab | [10.1109/hri61500.2025.10974182](https://doi.org/10.1109/hri61500.2025.10974182) | [10.17605/osf.io/prsxt](https://doi.org/10.17605/osf.io/prsxt) |
| other tlx | Multi-scale teleportation | [10.1109/tvcg.2024.3372043](https://doi.org/10.1109/tvcg.2024.3372043) | [10.5281/zenodo.10522829](https://doi.org/10.5281/zenodo.10522829) |
| other tlx | Robot programming by demonstration | [10.3850/978-981-18-8071-1_p241-cd](https://doi.org/10.3850/978-981-18-8071-1_p241-cd) | [10.5281/zenodo.23417050](https://doi.org/10.5281/zenodo.23417050) |

**What is downloaded.**
- Only participant-level tables, codebooks, READMEs and questionnaires.
- Videos, Unity projects, ROS bags and raw physiological signals are left out.
- Before download, each file is pinned with the checksum its repository publishes, in `analyses/heldout/sources.yaml`.
- From archives over 1 GB, only the needed members are fetched, by HTTP range requests where the server supports them.

**Dropping a dataset.** A dataset is dropped, and the drop reported, if:
- a file fails its checksum or cannot be downloaded;
- the files hold no participant-level data;
- every DV it measures is outside the schema.

**Untrusted data.** Downloaded files are untrusted:
- they are read with pandas or `scipy.io.loadmat`;
- archives are read in memory;
- pickles are never loaded, and no code shipped with the data is run.

## Mappings and study designs

- **Who and how.** Claude (the AI assistant operating this repository) writes one curated mapping file per held-out dataset, `analyses/heldout/mappings/<source_id>_mapping.yaml`. Sources are the dataset's README, codebook, questionnaire, paper and analysis code, plus the column headers and value ranges.
- **Rules.** The same as in `docs/preregistration_confirmatory_shortlist.md`:
    - the most specific schema DV;
    - `meta_*` labels for non-DV columns;
    - DV columns without a fitting schema DV are left unlabelled;
    - the schema is not changed during the test.
- **Perceived risk is not perceived safety.** A risk or anxiety rating is not reversed into `perceived_safety`.
- **Study designs.** Each dataset gets a `study_designs.yaml` entry: participant column, factors, domain.
- **Order.**
    - Mappings and designs are committed in one commit ("Freeze held-out mappings").
    - That commit comes before any correlation, pooled estimate or LLM answer is computed on these data.
    - Looking at headers and value ranges to write mappings is allowed. Computing statistics between columns is not.
- **No revision.** Mappings are not revised after that commit. Errors found later are reported, not corrected.

## Procedure

1. Download and verify the pinned files.
2. Write and freeze the mappings and designs (one commit).
3. Run the batch without the LLM over `analyses/heldout/sources_manifest.yaml`.
4. Run `python -m opendv.analysis.construct_relations` on the held-out standardized files only, with the defaults of the exploratory analysis:
    - at least 10 participants per study;
    - within-person df ≥ 10;
    - REML with HKSJ;
    - levels `names`, `automatic` and `curated`.

    The confirmatory tests use the `curated` level. No LLM-mapped column enters them.
5. Run the mapping benchmark (H3) with the settings of the confirmatory shortlist test:
    - `evaluate_llm_mapping` E1, `--shortlist definition --max-per-source 40 --seed 20261017 --e2-samples 0`;
    - `--models google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`;
    - plausibility rules 1.3;
    - header matching first, scored as in `confirmatory.header_match_answers`.

    Code is used as it stands at the freeze commit. If a step fails, the fault is fixed, the step is repeated, and both are reported.

## Hypotheses and analysis

- **H1 (primary): workload and perceived safety, within persons.**
    - *Prediction.* The pooled within-person correlation between `mental_demand` and `perceived_safety` is negative.
    - *Unit.* Held-out studies that measure both at least twice per participant and meet the inclusion thresholds above.
    - *Test.* Pooled r on Fisher's z (REML), one-sided HKSJ test of r < 0, α = 0.05.
    - *Testability.* H1 needs at least two such studies. With fewer, it is reported as **not testable on this set**, and any single-study estimate is reported descriptively. Metadata name both measures for only a few held-out datasets, so this outcome is likely and is stated now.
    - *Also reported.* Two-sided 95% HKSJ CI, prediction interval, I², and every study's own r.
- **H2 (primary): structure of the NASA-TLX.**
    - *Prediction.* Within persons, each of the three core pairs has a positive pooled correlation:
        - effort × frustration;
        - effort × mental demand;
        - frustration × mental demand.
    - *Test.* One-sided HKSJ test of r > 0 per pair, α = 0.05, Holm-corrected over the three pairs. H2 holds if all three are significant. A pair is tested if at least two held-out studies measure both subscales repeatedly.
    - *H2b (secondary, descriptive).* Performance is the weakest-linked subscale: its mean pooled r with the other five subscales is the lowest of the six subscales. This is reported as met or not met, without a test.
- **H3 (secondary): header matching before the LLM.**
    - *Prediction.* On held-out E1 DV columns, header matching first and then the LLM maps more columns correctly than the LLM alone.
    - *Test.* Paired exact McNemar, two-sided, α = 0.05.
    - *Also reported.* Precision of accepted DV mappings, with Wilson 95% CIs.
- **Multiplicity.** H1 and H2 are separate primary hypotheses, each decided at its own α. H3 is secondary.
- **Exploratory, labelled as such:**
    - all other construct pairs;
    - between-person relations;
    - the LLM level and the LLM replay;
    - results pooled with the development studies;
    - per-study results.

## Power

- **H1.** Power is low. With two or three studies, the one-sided HKSJ test has k − 1 = 1–2 degrees of freedom. The exploratory relation was very heterogeneous (I² = 94%), so even a true r = −0.4 may well be missed. A null result on this set is weak evidence against F1. Its role is to test whether F1 recurs where it can be measured.
- **H2.** Power is higher. The core pairs were 0.56–0.65 in every development study, and several held-out datasets name the NASA-TLX in their metadata.
- **H3.** The post hoc gain was 42 vs 0 discordant columns. If question-text headers are rare in the held-out data, there will be few discordant columns and the test will say little. In that case the number of columns header matching reaches is reported.

## Limitations stated in advance

- **Labeller.** The mappings come from the same AI assistant that ran the exploratory analysis, not from independent human raters.
- **Domain mix.** 19 of the 29 held-out datasets are automotive, 13 of them TU Delft studies of eHMIs and of pedestrian and cyclist interaction. Few of them use the NASA-TLX, and many are crowdsourced video studies with continuous keypress ratings.
- **Construct coverage.** Perceived safety is often measured as keypress percentages over videos. Whether such a percentage is the schema's `perceived_safety` will be decided when the mappings are written, before the freeze.
