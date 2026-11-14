# Open-data studies: candidates for new sources

Two searches: a first, agent-run search on 2026-11-04 (its 12 studies were added to the development set), and a systematic, scripted search of the research-data repositories on 2026-11-05 (57 datasets verified, 79 to check). The second search is described [below](#second-search-systematic-repository-search-2026-11-05).

The confirmatory test and its post hoc rerun (`docs/confirmatory_shortlist_results.md`) left gains that only new, unseen studies can confirm. This list collects peer-reviewed HCI studies whose participant-level data are openly licensed.

**Outcome (2026-11-04).** 12 of the 13 studies in the two tables were added to the development set. The exception is `acoustic_touch_blind_plosone23`, because Dryad refused automated downloads (HTTP 401/403). Their files are pinned in `analyses/development_sources/sources.yaml`, and each has a curated mapping, `schemas/<source_id>_mapping.yaml`. The mappings were written from the codebooks and papers before any LLM run on these studies. `sources_manifest_no4tu.yaml` lists them.

## Criteria

- Participant-level tabular data (CSV, TSV, XLSX, SAV) with DV columns, ideally instruments the schema covers (NASA-TLX, SUS, UEQ/UEQ-S, Van der Laan, SSQ, TiA, SART, TAM/UTAUT, perceived safety, presence) or performance measures.
- A licence permitting reuse and redistribution (CC BY 4.0, CC0, MIT).
- A codebook, README, questionnaire or analysis script to label columns from.
- A peer-reviewed paper with a DOI.
- Stable hosting (Zenodo, OSF, Dryad, or GitHub pinned to a commit).
- Not already in the development or confirmatory set.

## Search

Searched on 2026-11-04:

- **Repository indexes:**
  - the Zenodo REST API (datasets; instrument, topic, venue and creator terms);
  - the OSF SHARE index-card search, with files, licences and SHA-256 hashes taken from `api.osf.io/v2` including child components;
  - the Harvard Dataverse and Figshare APIs (Figshare also covers 4TU.ResearchData).
- **GitHub** repository and code search.
- **Papers:** OpenAlex and Crossref, to confirm each paper.

An AI research agent ran the search. Only metadata, READMEs, file listings and header rows of small files were read. Licences, files, sizes, checksums and paper DOIs of the candidates below were then re-checked against the Zenodo, OSF and Crossref APIs.

## Pool A: automotive (same domain as the development set)

| source_id (proposed) | Paper | Data | Licence | Files (bytes) | Instruments / DVs | Notes |
|---|---|---|---|---|---|---|
| `incar_vr_gain_chi24` | McGill et al., CHI 2024, [10.1145/3613904.3642298](https://doi.org/10.1145/3613904.3642298) | [Zenodo 10138112](https://zenodo.org/records/10138112) | CC BY 4.0 | `FINAL_Questionnaires.csv` 34,033; `FinalQuestions.csv` 87,736 (skip `FinalJSONMS.csv`, 5.3 MB of motion logs) | SSQ subscales, MSSQ, raw NASA-TLX (6 subscales), presence, perceived safety, rankings | On-road in-car VR; no codebook beyond the record |
| `bvi_auditory_hav_autoui26` | Colley et al., AutomotiveUI 2026, [10.1145/3828157.3828774](https://doi.org/10.1145/3828157.3828774) | [Zenodo 22145814](https://zenodo.org/records/22145814) | CC BY 4.0 | `bvi-auditory-hav.xlsx` 69,674 | NASA-TLX mental demand, TiA trust, SART (10 items), perceived safety, UEQ-S | The record lists every variable; use the `Results` sheet |
| `av_prevalence_ehmi_autoui24` | AutomotiveUI 2024, [10.1145/3640792.3675723](https://doi.org/10.1145/3640792.3675723) | [OSF cr3qm](https://osf.io/cr3qm/) (in bu3r2) | CC BY 4.0 | `data.csv` 212,166; `readme.html` 44,030 | UEQ-S per eHMI element, crossing-decision time (N = 64, 48 trials) | Full codebook; one semicolon file mixing three row types; skip the 2.3 GB of stimuli videos |
| `ehmi_state_intent_mtap24` | Multimedia Tools and Applications 2024, [10.1007/s11042-024-20497-0](https://doi.org/10.1007/s11042-024-20497-0) | [OSF w9mfv](https://osf.io/w9mfv/) (in uyqz6) | CC BY 4.0 (data component) | `data.csv` 19,918; `README.html` 44,606 | Crossing-initiation times, 7 repeated trust ratings, UEQ-S | Same group as the previous row; journal outside core HCI |
| `av_acceptance_vr_ijhci24` | IJHCI 2024, [10.1080/10447318.2024.2358577](https://doi.org/10.1080/10447318.2024.2358577) | [OSF y6fjr](https://osf.io/y6fjr/) | CC BY 4.0 | `Questionnaire_data/raw_data/EEDA_Qdata.csv` 119,089; `items/EEDA_questionnaire.xlsx` 14,167 | AV acceptance (UTAUT/TAM-style), perceived safety | Fetch only these two files; the project also holds about 6 GB of eye-tracking zips |

**Weaker (medium or low confidence, as reported by the agent; not re-checked):**

- **OptiCarVis** (CHI 2025): data split across about 120 per-participant files.
- **Japan–Germany eHMI** (IMWUT 2026): design close to `ehmi_optimization_chi25`.
- **Remote AV takeover** (AVI 2026): composite scores only; licence mismatch.
- **NDRT takeover** (ETRR 2025): performance only; possibly Czech labels.
- **Passenger HMI physiology** (IJHCI 2026): scale scores only.
- **Hit and Run** (AutomotiveUI 2026): zip contents unknown.
- **Cooperative AV AR trust** (AutomotiveUI 2025): Python pickle files only. Never load these unconverted: unpickling runs code.

## Pool B: other HCI domains

| source_id (proposed) | Paper | Data | Licence | Files (bytes) | Instruments / DVs | Notes |
|---|---|---|---|---|---|---|
| `fido2_auth_chi23` | CHI 2023, [10.1145/3544548.3580993](https://doi.org/10.1145/3544548.3580993) | [Zenodo 7572697](https://zenodo.org/records/7572697) | CC BY 4.0 | `fido2-lab-study-code-and-dataset.zip` 94,008 | SUS, Van der Laan acceptance, ATI (N = 87) | README and R Markdown inside; may hold scale scores rather than items |
| `pairsonic_cscw24` | PACM HCI (CSCW) 2024, [10.1145/3686964](https://doi.org/10.1145/3686964) | [Zenodo 13324112](https://zenodo.org/records/13324112) | CC BY 4.0 | `pairsonic-study-code-and-dataset.zip` 138,893 | SUS, ATI, completion time (N = 45) | Same lab as `fido2_auth_chi23` |
| `curved_dragging_vr_ijhci26` | IJHCI 2026, [10.1080/10447318.2026.2738094](https://doi.org/10.1080/10447318.2026.2738094) | [Zenodo 21589932](https://zenodo.org/records/21589932) | CC BY 4.0 | three `*_processed_condition_data.xlsx` (69,347; 138,639; 109,942); `README.md` 5,243 | Movement time, error rate, throughput, NASA-TLX subscales (N = 12) | Condition means per participant |
| `fov_snapturn_tvcg24` | IEEE TVCG 2025, [10.1109/TVCG.2024.3470214](https://doi.org/10.1109/TVCG.2024.3470214) | [OSF d4zyf](https://osf.io/d4zyf/) | CC BY 4.0 | `Data/FOV&SnapTurnDataOSF_Revision.csv` 108,810 | SSQ items pre/post, FMS, MSSQ, VIMSQ, VRNQ | No codebook beyond the supplementary PDF |
| `handsize_touch_mobilehci19` | MobileHCI 2019, [10.1145/3338286.3340115](https://doi.org/10.1145/3338286.3340115) | [hcidata @ fad8f707](https://github.com/henningpohl/hcidata) | CC BY 4.0 | `data-raw/HandSize.csv` 2,217,469; `R/HandSize.R` | Selection time, touch error (27k trials) | Same repository as `agency_ux_tochi22`; performance only |
| `acoustic_touch_blind_plosone23` | PLOS ONE 2023, [10.1371/journal.pone.0290431](https://doi.org/10.1371/journal.pone.0290431) | [Dryad qv9s4mwjp](https://doi.org/10.5061/dryad.qv9s4mwjp) | CC0 | about 8 small CSVs (e.g. `Fig_NASATLX.csv` 1,635); `README.md` | NASA-TLX, success rate, reaching time (N = 14) | One file per figure |
| `woz_adaptation_ijhcs24` | IJHCS 2025, [10.1016/j.ijhcs.2024.103440](https://doi.org/10.1016/j.ijhcs.2024.103440) | [Zenodo 10155676](https://zenodo.org/records/10155676) | CC BY 4.0 | `NSF2151464_Study2_ResearchData.xlsx` 40,580 | NASA-TLX (×3), IMI (N = 97) | Columns not yet seen |
| `gaze_platformer_chi24` | Colley et al., CHI 2024, [10.1145/3613904.3641909](https://doi.org/10.1145/3613904.3641909) | [M-Colley/shed-some-fear-data @ bd47389c](https://github.com/M-Colley/shed-some-fear-data) | MIT (LICENSE) vs. CC BY 4.0 (CITATION.cff) | `game-data.xlsx` 27,664; `game-demographic.xlsx` 7,339; `game-final.xlsx` 12,684; `CODEBOOK.md` 5,673 | PXI enjoyment, IMI competence, digital eye strain (N = 17, 5 days) | The two licence statements should be made consistent |

**Held back:**

- **Zenodo 20574145** (AR television in a simulated train, McGill/Brewster group): the record describes a field-of-view × ornamentation study, not the IMX 2024 ARTV paper the search linked it to, and no matching paper was found.
- **Not re-checked:**
  - MR airplane layouts (ISMAR 2021): partial NASA-TLX only.
  - VR tool embodiment (Frontiers in VR 2025): no data dictionary.
  - Treadmill exergame (Virtual Reality 2026): Spanish item headers.
  - NAO robot cognitive load (MDPI Information 2025).

## Rejected (summary)

- **No peer-reviewed paper:**
  - project reports (MEDIATOR, SAFER-SIM, METRANS);
  - preprints;
  - studies under review, including two unpublished Colley et al. repositories (revisit once published).
- **Licence:** CC BY-NC or BY-NC-ND, or no licence stated (several OSF projects and GitHub repositories).
- **Only multi-GB sensor archives:** e.g. TD2D, AdVitam, VR-CARLA, TimelyTale.
- **Design or measures:** qualitative only, custom single items, or surveys without a user study.

## Destination

The two options were:

- **The development set.** It grows the catalog and the meta-analysis, but the studies are then used up as test data.
- **A new held-out validation set.** It would confirm the post hoc gains, with gold labels frozen before any run, as in `docs/preregistration_confirmatory_shortlist.md`.

The studies went to the development set. Their first benchmark run is still informative, because none of the current methods was tuned on them; the harness reports them separately (`--subset`). Any change made after seeing that run uses them up like the other development studies. Confirming the post hoc gains therefore still needs a new held-out set.

That first run is in `docs/research_agenda.md` §2.4. Adding the studies also exposed four errors in the analyses, each now fixed (§2.1 and the changelog): scale detection, reversed Van der Laan keying in two original studies, mixed time units, and invariance fits that failed with between-subject studies.

## Second search: systematic repository search (2026-11-05)

The first search found 13 studies. To confirm the cross-study relations of `docs/cross_study_insights.md`, studies that played no part in finding them are needed. So the second search is scripted and reproducible: `analyses/candidate_search/search_open_data.py`.

### Method

1. **Sources.** The script queries two indexes:
    - the DataCite REST API (resource type Dataset), which indexes Zenodo, OSF, figshare (including university portals), Dryad, 4TU.ResearchData, Dataverse installations and Mendeley Data;
    - OSF's own SHARE index, for OSF projects that never minted a DOI.

    It sends 21 fixed queries: 14 for instruments the schema covers (NASA-TLX, SUS, UEQ, AttrakDiff, SSQ, FMS, trust in automation, perceived safety, Van der Laan, situation awareness, presence, PXI, TAM/UTAUT, ATI) and 7 for domains (eHMI, automated driving, takeover, driving simulators, VR/AR user studies, HCI venues, HRI). The queries and per-query counts are in `search_log.json`.
2. **Automatic screen.** Every record is kept in `hits.csv` with the stage it stopped at and why:
    - one record per dataset (its latest version);
    - an open licence (CC BY, CC0, MIT and similar; NC and ND excluded);
    - a peer-reviewed paper, either linked in the record's related identifiers or found on Crossref by a near-identical title (similarity ≥ 0.85, journal or proceedings article);
    - not already a source of this project;
    - a file listing from the repository's API with at least one table or archive.
3. **Manual screen.** I read the title, paper, description and file list of every candidate and kept user studies in HCI and human factors whose participant-level data hold DVs the schema covers. Decisions and reasons are in `screening_decisions.csv`.
4. **Verification of the included datasets:**
    - the dataset's creators were compared with the paper's authors on Crossref. They overlap for 52 of 57. The other five carry the paper's title, and two of them also link the paper's DOI;
    - README files and data dictionaries of up to 30 kB were read as text;
    - no data file was opened.

Only metadata, file listings and READMEs were read; nothing was downloaded into the project.

### Flow

| Stage | Datasets |
|---|---|
| Unique DOIs returned (DataCite 21 queries, OSF index 17 queries) | 5,427 |
| One record per dataset | 3,199 |
| Excluded: no licence stated (749), NC/ND licence (132), already a source (10) | −891 |
| Excluded: no peer-reviewed paper found | −1,652 |
| Excluded: no public file listing (restricted, embargoed, unsupported repository) | −102 |
| Excluded: no table or archive (PDFs, videos, code only) | −118 |
| Excluded: second record of the same dataset | −9 |
| **Candidates for manual screening** | **427** |
| Excluded after reading (reasons below) | −291 |
| **Included: verified** | **57** |
| **To check: plausible, contents or paper not yet verified** | **79** |

Reasons for the 291 manual exclusions:

| Reason | Datasets |
|---|---|
| Surveys or studies without an interactive user study measuring schema DVs (mostly TAM/UTAUT adoption surveys) | 158 |
| Sensor or log recordings without usable questionnaire or performance tables | 54 |
| Article tables (SciELO `Table_N.xls`), not participant data | 22 |
| Not human-computer interaction | 20 |
| Clinical, occupational or health-services studies | 14 |
| Second record of the same dataset under another DOI or repository | 12 |
| Reviews, meta-analyses, bibliometrics | 7 |
| Language-model studies without human participants | 2 |
| Preprint server; materials without participant data | 2 |

### Included datasets (57)

37 are automotive. 24 of those are on 4TU.ResearchData, under CC0 or CC BY 4.0, and come from TU Delft groups (mostly de Winter, Dodou and Bazilinskyy). They cover eHMIs, pedestrian and cyclist interaction, takeover and passenger uncertainty.

#### Automotive (37)

| Study, measures and files | Paper | Data | Licence |
|---|---|---|---|
| Perceived cooperativity scale in driving; trust, Van der Laan items; Dryad links an unrelated paper by the same authors, so the paper must be confirmed | Transportation Research Part F: Traffic Psychology and Behav 2026, [10.1016/j.trf.2026.103610](https://doi.org/10.1016/j.trf.2026.103610) | Dryad, [10.5061/dryad.5qfttdzpg](https://doi.org/10.5061/dryad.5qfttdzpg) | CC0 |
| Passengers' certainty and uncertainty, crowd study; one CSV | Frontiers in Psychology 2026, [10.3389/fpsyg.2026.1917625](https://doi.org/10.3389/fpsyg.2026.1917625) | 4TU.ResearchData, [10.4121/b61ddc8f-0c07-44b5-b570-6208edc2f4cb.v2](https://doi.org/10.4121/b61ddc8f-0c07-44b5-b570-6208edc2f4cb.v2) | CC BY 4.0 |
| Tactile driver-vehicle interface for NDRTs and emergency braking; experiment and questionnaire CSVs | Ergonomics 2026, [10.1080/00140139.2026.2703690](https://doi.org/10.1080/00140139.2026.2703690) | Dryad, [10.5061/dryad.vx0k6dk33](https://doi.org/10.5061/dryad.vx0k6dk33) | CC0 |
| Passengers' uncertainty in automated cars, crowd study; one workbook | Behaviour & Information Technology 2026, [10.1080/0144929x.2026.2746452](https://doi.org/10.1080/0144929x.2026.2746452) | 4TU.ResearchData, [10.4121/de7105b7-6200-4e9b-bc17-db3613bef6a7.v3](https://doi.org/10.4121/de7105b7-6200-4e9b-bc17-db3613bef6a7.v3) | CC BY 4.0 |
| Driver trust index; trust questionnaire scores (6.5 MB archive; authors not matched automatically) | International Journal of Human–Computer Interaction 2026, [10.1080/10447318.2026.2727771](https://doi.org/10.1080/10447318.2026.2727771) (title match) | Zenodo, [10.5281/zenodo.20627050](https://doi.org/10.5281/zenodo.20627050) | CC BY 4.0 |
| Anticipated acceptance of automated driving; perceived safety, acceptance (online survey) | Mensch und Computer 2025 2025, [10.1145/3743049.3743051](https://doi.org/10.1145/3743049.3743051) (title match) | OSF, [10.17605/osf.io/dvsbh](https://doi.org/10.17605/osf.io/dvsbh) | CC BY 4.0 |
| Non-driving-related tasks and takeover in level 3 (simulator); takeover performance (one workbook) | European Transport Research Review 2025, [10.1186/s12544-025-00745-6](https://doi.org/10.1186/s12544-025-00745-6) (title match) | Zenodo, [10.5281/zenodo.15281988](https://doi.org/10.5281/zenodo.15281988) | CC BY 4.0 |
| Labelling automotive AI as trustworthy or reliable; trust, acceptance (online experiment CSV) | Scientific Reports 2025, [10.1038/s41598-025-85558-2](https://doi.org/10.1038/s41598-025-85558-2) (title match) | OSF, [10.17605/osf.io/6un4h](https://doi.org/10.17605/osf.io/6un4h) | CC BY 4.0 |
| Manual driving style and automated-driving takeover (4.3 MB archive, contents not listed) | Applied Ergonomics 2024, [10.1016/j.apergo.2024.104366](https://doi.org/10.1016/j.apergo.2024.104366) | 4TU.ResearchData, [10.4121/57e36cf2-4f74-4d2f-86e2-7fae7250438a.v1](https://doi.org/10.4121/57e36cf2-4f74-4d2f-86e2-7fae7250438a.v1) | CC BY 4.0 |
| eHMI on the vehicle or the infrastructure, driving simulator; 'Open data.xlsx' with trial and participant data (117 MB archive) | European Journal of Transport and Infrastructure Research 2024, [10.59490/ejtir.2024.24.2.7273](https://doi.org/10.59490/ejtir.2024.24.2.7273) (title match) | 4TU.ResearchData, [10.4121/8e1c6604-53c7-4905-8ef9-4a6b2acc4e7a.v1](https://doi.org/10.4121/8e1c6604-53c7-4905-8ef9-4a6b2acc4e7a.v1) | CC BY 4.0 |
| Interfaces for partially automated vehicles; trust, acceptance; 6.7 GB simulator archive, questionnaire location unclear | Transportation Research Part F: Traffic Psychology and Behav 2024, [10.1016/j.trf.2024.02.009](https://doi.org/10.1016/j.trf.2024.02.009) (title match) | 4TU.ResearchData, [10.4121/18bb5d48-c1e5-4d50-ad71-d9010e4494d9.v2](https://doi.org/10.4121/18bb5d48-c1e5-4d50-ad71-d9010e4494d9.v2) | CC BY 4.0 |
| AR for pedestrian-AV interaction, outdoor experiment; MISC and questionnaire workbooks (688 MB archive) | Frontiers in Robotics and AI 2024, [10.3389/frobt.2024.1324060](https://doi.org/10.3389/frobt.2024.1324060) | 4TU.ResearchData, [10.4121/a1f9f15c-1213-4657-8e4d-a154a725d747.v1](https://doi.org/10.4121/a1f9f15c-1213-4657-8e4d-a154a725d747.v1) | CC BY 4.0 |
| Car sickness; FMS, MSSQ, SSQ workbooks (skip 3.5 GB of physiology) | 2023 IEEE Canadian Conference on Electrical and Computer Eng 2023, [10.1109/ccece58730.2023.10289000](https://doi.org/10.1109/ccece58730.2023.10289000) (title match) | Borealis, [10.5683/sp3/cinmpg](https://doi.org/10.5683/sp3/cinmpg) | CC0 |
| eHMI flashing vs text with a gaze-contingent setup; keypress data (6.3 GB archive) | Transportation Research Part F: Traffic Psychology and Behav 2023, [10.1016/j.trf.2023.07.005](https://doi.org/10.1016/j.trf.2023.07.005) (title match) | 4TU.ResearchData, [10.4121/6dfe7bd5-16f7-4e8d-9ab7-f8203b902310.v1](https://doi.org/10.4121/6dfe7bd5-16f7-4e8d-9ab7-f8203b902310.v1) | CC BY 4.0 |
| AR interfaces for pedestrian-vehicle interaction, online study; raw_data.xlsx | Transportation Research Part F: Traffic Psychology and Behav 2023, [10.1016/j.trf.2023.02.005](https://doi.org/10.1016/j.trf.2023.02.005) | 4TU.ResearchData, [10.4121/21603678.v1](https://doi.org/10.4121/21603678.v1) | CC0 |
| Tactile displays for situation awareness in automated vehicles; NASA-TLX, SART; one CSV of 1.4 kB (may hold condition means) | 2022 IEEE 9th International Conference on Computational Inte 2022, [10.1109/civemsa53371.2022.9853690](https://doi.org/10.1109/civemsa53371.2022.9853690) (title match) | DaRUS, [10.18419/darus-2824](https://doi.org/10.18419/darus-2824) | CC BY 4.0 |
| Level 2 transfer-of-control interface; SART and takeover responses (two CSVs) | 14th International Conference on Automotive User Interfaces  2022, [10.1145/3543174.3546839](https://doi.org/10.1145/3543174.3546839) (title match) | Harvard Dataverse, [10.7910/dvn/8srajs](https://doi.org/10.7910/dvn/8srajs) | CC0 |
| Task load and music tempo in simulated driving; NASA-TLX, repeated measures (SPSS file) | Ergonomics 2022, [10.1080/00140139.2021.2003872](https://doi.org/10.1080/00140139.2021.2003872) (title match) | figshare, [10.6084/m9.figshare.13348046](https://doi.org/10.6084/m9.figshare.13348046) | CC BY 4.0 |
| Drivers' eye contact and pedestrians' perceived safety; crowd keypresses and Appen survey (225 MB archive) | Transportation Research Part F: Traffic Psychology and Behav 2022, [10.1016/j.trf.2021.10.017](https://doi.org/10.1016/j.trf.2021.10.017) | 4TU.ResearchData, [10.4121/16866709.v3](https://doi.org/10.4121/16866709.v3) | CC0 |
| Perceived risk and trust ratings during merging and braking (SubjectiveRatings.xlsx in a 7.7 MB archive) | Transportation Research Part F: Traffic Psychology and Behav 2022, [10.1016/j.trf.2022.02.016](https://doi.org/10.1016/j.trf.2022.02.016) | 4TU.ResearchData, [10.4121/95a4bb4e-3ca4-4fcc-ba34-4be76a9ab578.v1](https://doi.org/10.4121/95a4bb4e-3ca4-4fcc-ba34-4be76a9ab578.v1) | CC BY 4.0 |
| Coupled simulator with HMDs, driver-pedestrian; questionnaire workbooks, acceptance (64 MB archive) | Applied Ergonomics 2022, [10.1016/j.apergo.2022.103825](https://doi.org/10.1016/j.apergo.2022.103825) | 4TU.ResearchData, [10.4121/20005565](https://doi.org/10.4121/20005565) | CC0 |
| Coupled driver-pedestrian simulator; intake, post-trial and post-experiment questionnaires (230 MB data archive; skip the 5.6 GB Unity archive) | 14th International Conference on Automotive User Interfaces  2022, [10.1145/3543174.3546849](https://doi.org/10.1145/3543174.3546849) | 4TU.ResearchData, [10.4121/20224281.v2](https://doi.org/10.4121/20224281.v2) | CC BY 4.0 |
| Cyclists' decisions and AV characteristics; two Appen surveys and keypress workbooks (205 MB archive) | IET Intelligent Transport Systems 2022, [10.1049/itr2.12235](https://doi.org/10.1049/itr2.12235) | 4TU.ResearchData, [10.4121/20103188.v2](https://doi.org/10.4121/20103188.v2) | CC0 |
| eHMI: gimmick or necessity (66 MB archive, contents not listed) | Transportation Research Interdisciplinary Perspectives 2022, [10.1016/j.trip.2022.100643](https://doi.org/10.1016/j.trip.2022.100643) | 4TU.ResearchData, [10.4121/858e4b36-895b-4c84-928a-d1cdcf0920d6.v1](https://doi.org/10.4121/858e4b36-895b-4c84-928a-d1cdcf0920d6.v1) | CC BY 4.0 |
| Music intensity and lyrics in simulated urban driving; NASA-TLX, repeated measures (SPSS file) | Transportation Research Part F: Traffic Psychology and Behav 2021, [10.1016/j.trf.2021.05.022](https://doi.org/10.1016/j.trf.2021.05.022) (title match) | figshare, [10.6084/m9.figshare.13385834](https://doi.org/10.6084/m9.figshare.13385834) | CC BY 4.0 |
| Pedestrian-AV gesture communication in VR (136 MB archive, contents not listed) | Journal of Advanced Transportation 2021, [10.1155/2021/5573560](https://doi.org/10.1155/2021/5573560) (title match) | 4TU.ResearchData, [10.4121/14406944.v1](https://doi.org/10.4121/14406944.v1) | CC0 |
| Bio-inspired intent communication (231 MB archive) | Transportation Research Part F: Traffic Psychology and Behav 2021, [10.1016/j.trf.2021.03.021](https://doi.org/10.1016/j.trf.2021.03.021) | 4TU.ResearchData, [10.4121/14096067.v2](https://doi.org/10.4121/14096067.v2) | CC0 |
| Implicit AV communication by lateral position; Appen survey CSV (722 MB archive with videos) | Ergonomics 2021, [10.1080/00140139.2021.1925353](https://doi.org/10.1080/00140139.2021.1925353) | 4TU.ResearchData, [10.4121/14046107.v2](https://doi.org/10.4121/14046107.v2) | CC0 |
| Driving style judged as automated; data.xlsx (0.4 MB archive) | Applied Ergonomics 2021, [10.1016/j.apergo.2021.103428](https://doi.org/10.1016/j.apergo.2021.103428) | 4TU.ResearchData, [10.4121/14077352.v2](https://doi.org/10.4121/14077352.v2) | CC0 |
| eHMI message perspective (2.6 GB archive) | Transportation Research Part F: Traffic Psychology and Behav 2021, [10.1016/j.trf.2021.01.013](https://doi.org/10.1016/j.trf.2021.01.013) | 4TU.ResearchData, [10.4121/14068553.v1](https://doi.org/10.4121/14068553.v1) | CC0 |
| eHMI behaviour, 1,434 crowd participants; perceived-safety keypresses and survey workbook (1.1 GB archive with videos) | Applied Ergonomics 2021, [10.1016/j.apergo.2021.103450](https://doi.org/10.1016/j.apergo.2021.103450) | 4TU.ResearchData, [10.4121/14465715.v2](https://doi.org/10.4121/14465715.v2) | CC0 |
| eHMI on the whole car body; clarity questionnaire (SPSS file, 6 kB) | Proceedings of Mensch und Computer 2020 2020, [10.1145/3404983.3410011](https://doi.org/10.1145/3404983.3410011) | OSF, [10.17605/osf.io/psmrx](https://doi.org/10.17605/osf.io/psmrx) | CC BY 4.0 |
| eHMI colours, crowd participants (355 MB archive) | 2020 IEEE International Conference on Systems, Man, and Cybe 2020, [10.1109/smc42975.2020.9282998](https://doi.org/10.1109/smc42975.2020.9282998) | 4TU.ResearchData, [10.4121/12948650.v3](https://doi.org/10.4121/12948650.v3) | CC0 |
| eHMI display location; crossing intentions, eye movements (4.1 GB archive) | Information 2019, [10.3390/info11010013](https://doi.org/10.3390/info11010013) | 4TU.ResearchData, [10.4121/12682826](https://doi.org/10.4121/12682826) | CC0 |
| Survey on eHMI concepts, crowd participants (36 MB archive) | Transportation Research Part F: Traffic Psychology and Behav 2019, [10.1016/j.trf.2019.10.013](https://doi.org/10.1016/j.trf.2019.10.013) | 4TU.ResearchData, [10.4121/uuid:9ea78136-3ffc-4194-84f7-3116b6a55758](https://doi.org/10.4121/uuid:9ea78136-3ffc-4194-84f7-3116b6a55758) | CC0 |
| Pedestrian-AV Wizard-of-Oz field experiment (1.1 MB archive) | Transportation Research Part F: Traffic Psychology and Behav 2018, [10.1016/j.trf.2018.07.020](https://doi.org/10.1016/j.trf.2018.07.020) | 4TU.ResearchData, [10.4121/uuid:4343cfa8-c246-4bac-bcfa-e3725aceacf7](https://doi.org/10.4121/uuid:4343cfa8-c246-4bac-bcfa-e3725aceacf7) | CC0 |
| Visual fidelity, curve negotiation and simulator discomfort; dependent-measure and questionnaire workbooks | Ergonomics 2015, [10.1080/00140139.2015.1005172](https://doi.org/10.1080/00140139.2015.1005172) | Taylor & Francis, [10.6084/m9.figshare.1311647.v1](https://doi.org/10.6084/m9.figshare.1311647.v1) | CC BY 4.0 |

#### Other HCI domains (20)

| Domain | Study, measures and files | Paper | Data | Licence |
|---|---|---|---|---|
| accessibility | Acoustic touch for blind and low-vision users; NASA-TLX and task CSVs (one per figure) | PLOS ONE 2023, [10.1371/journal.pone.0290431](https://doi.org/10.1371/journal.pone.0290431) | Dryad, [10.5061/dryad.qv9s4mwjp](https://doi.org/10.5061/dryad.qv9s4mwjp) | CC0 |
| attention and media | Short-form videos and prospective memory (CHI 2023); accuracy, RT, questionnaire CSVs | CHI Conference on Human Factors in Computing Systems 2023, [10.1145/3544548.3580778](https://doi.org/10.1145/3544548.3580778) (title match) | DaRUS, [10.18419/darus-3327](https://doi.org/10.18419/darus-3327) | CC BY 4.0 |
| audio interfaces | Concurrent speech-based information; NASA-TLX (4 MB CSV) | International Journal of Human-Computer Studies 2022, [10.1016/j.ijhcs.2021.102728](https://doi.org/10.1016/j.ijhcs.2021.102728) (title match) | figshare, [10.6084/m9.figshare.14743422](https://doi.org/10.6084/m9.figshare.14743422) | CC BY 4.0 |
| augmented reality | AR television ornamentations in a VR-simulated commute (IMX 2025); UEQ, questionnaire CSV | ACM International Conference on Interactive Media Experience 2025, [10.1145/3706370.3727854](https://doi.org/10.1145/3706370.3727854) | Zenodo, [10.5281/zenodo.20574410](https://doi.org/10.5281/zenodo.20574410) | CC BY 4.0 |
| augmented reality | Proxemics for human-agent interaction (CHI 2022); questionnaire CSVs | CHI Conference on Human Factors in Computing Systems 2022, [10.1145/3491102.3517593](https://doi.org/10.1145/3491102.3517593) | DaRUS, [10.18419/darus-2525](https://doi.org/10.18419/darus-2525) | CC BY 4.0 |
| health HCI | LLM primary-care chatbot evaluation; NASA-TLX, trust (fullData.csv) | JAMIA Open 2026, [10.1093/jamiaopen/ooag207](https://doi.org/10.1093/jamiaopen/ooag207) (title match) | Dryad, [10.5061/dryad.bcc2fqzw2](https://doi.org/10.5061/dryad.bcc2fqzw2) | CC0 |
| human factors | NASA-TLX with several response formats in a UAS task; data dictionary | IEEE Transactions on Human-Machine Systems 2023, [10.1109/thms.2023.3263482](https://doi.org/10.1109/thms.2023.3263482) | University of Virginia Dataverse, [10.18130/v3/97zjnl](https://doi.org/10.18130/v3/97zjnl) | CC BY 4.0 |
| human-robot interaction | Function allocation in physical HRC; questionnaire workbook (NASA-TLX, Van der Laan) inside a 7.2 GB archive of ROS bags | IEEE Robotics and Automation Letters 2026, [10.1109/lra.2025.3632607](https://doi.org/10.1109/lra.2025.3632607) | 4TU.ResearchData, [10.4121/d55393f4-f6b6-4076-86c1-7bf351f16787.v1](https://doi.org/10.4121/d55393f4-f6b6-4076-86c1-7bf351f16787.v1) | CC BY 4.0 |
| human-robot interaction | Shared-autonomy interface modalities; questionnaire CSV (skip 1.6 GB EMG); Zenodo links a related paper of the project | IEEE Transactions on Neural Systems and Rehabilitation Engin 2026, [10.1109/tnsre.2026.3657400](https://doi.org/10.1109/tnsre.2026.3657400) | Zenodo, [10.5281/zenodo.14198913](https://doi.org/10.5281/zenodo.14198913) | CC BY 4.0 |
| human-robot interaction | Myocontrol intent detection; NASA TLX.csv, test results | 2025 International Conference On Rehabilitation Robotics (IC 2025, [10.1109/icorr66766.2025.11063080](https://doi.org/10.1109/icorr66766.2025.11063080) | Zenodo, [10.5281/zenodo.18834139](https://doi.org/10.5281/zenodo.18834139) | CC BY 4.0 |
| human-robot interaction | Robot-assisted feeding outside the lab; NASA-TLX (29 CSVs) | 2025 20th ACM/IEEE International Conference on Human-Robot I 2025, [10.1109/hri61500.2025.10974182](https://doi.org/10.1109/hri61500.2025.10974182) (title match) | OSF, [10.17605/osf.io/prsxt](https://doi.org/10.17605/osf.io/prsxt) | CC BY 4.0 |
| human-robot interaction | Teleoperated robots; NASA-TLX, trust (46 MB archive) | 2025 IEEE/RSJ International Conference on Intelligent Robots 2025, [10.1109/iros60139.2025.11245894](https://doi.org/10.1109/iros60139.2025.11245894) | Zenodo, [10.5281/zenodo.19040256](https://doi.org/10.5281/zenodo.19040256) | CC BY 4.0 |
| human-robot interaction | SenseCobotFusion stress study; NASA-TLX (1 MB archive) | ACM/IEEE International Conference on Human-Robot Interaction 2024, [10.1145/3610977.3636440](https://doi.org/10.1145/3610977.3636440) | Zenodo, [10.5281/zenodo.14753191](https://doi.org/10.5281/zenodo.14753191) | CC BY 4.0 |
| human-robot interaction | Robot programming by demonstration; NASA-TLX, SUS (archive contents not listed) | Proceeding of the 33rd European Safety and Reliability Confe 2023, [10.3850/978-981-18-8071-1_p241-cd](https://doi.org/10.3850/978-981-18-8071-1_p241-cd) | Zenodo, [10.5281/zenodo.23417050](https://doi.org/10.5281/zenodo.23417050) | CC BY 4.0 |
| interaction techniques | Mid-air gestures vs touch for map-making; SUS items, task metrics | Cartography and Geographic Information Science 2026, [10.1080/15230406.2026.2619027](https://doi.org/10.1080/15230406.2026.2619027) (title match) | figshare, [10.6084/m9.figshare.30833309](https://doi.org/10.6084/m9.figshare.30833309) | CC BY 4.0 |
| software tools | LLM-assisted filter code, user study; NASA-TLX, SUS (7 MB archive) | Behaviour & Information Technology 2026, [10.1080/0144929x.2026.2711012](https://doi.org/10.1080/0144929x.2026.2711012) (title match) | Zenodo, [10.5281/zenodo.21411181](https://doi.org/10.5281/zenodo.21411181) | CC BY 4.0 |
| software tools | Developers' eye tracking and creativity; NASA_TLX_Scores.csv, post-questionnaire | 19th International Conference on Cooperative and Human Aspec 2026, [10.1145/3794860.3794891](https://doi.org/10.1145/3794860.3794891) (title match) | figshare, [10.6084/m9.figshare.31198108](https://doi.org/10.6084/m9.figshare.31198108) | CC BY 4.0 |
| software tools | Reflexion-mapping suggestions, controlled experiment; NASA-TLX, SUS (ProbandData.csv) | 2025 IEEE 22nd International Conference on Software Architec 2025, [10.1109/icsa-c65153.2025.00068](https://doi.org/10.1109/icsa-c65153.2025.00068) (title match) | Zenodo, [10.5281/zenodo.14718504](https://doi.org/10.5281/zenodo.14718504) | CC BY 4.0 |
| virtual reality | Tool embodiment under task load, young and older adults; NASA-TLX, performance CSV | Frontiers in Virtual Reality 2025, [10.3389/frvir.2025.1637212](https://doi.org/10.3389/frvir.2025.1637212) | OSF, [10.17605/osf.io/uxk58](https://doi.org/10.17605/osf.io/uxk58) | CC BY 4.0 |
| virtual reality | Multi-scale teleportation; tlx.csv, technique ratings, route-task logs | IEEE Transactions on Visualization and Computer Graphics 2024, [10.1109/tvcg.2024.3372043](https://doi.org/10.1109/tvcg.2024.3372043) (title match) | Zenodo, [10.5281/zenodo.10522829](https://doi.org/10.5281/zenodo.10522829) | CC BY 4.0 |

### To check (79)

These look like eligible user studies, but their archives were not listed or their paper or measures are unclear. Domains:
- virtual reality 17;
- automotive 15;
- human-robot interaction 11;
- conversational agents, interaction techniques and software tools, 4 each;
- 18 others.

They are in `screening_decisions.csv` with `decision = check`.

**Outcome (2026-11-14).** All 79, and the five datasets the first held-out test dropped before download, were checked from metadata, READMEs, archive listings and papers, without opening participant data (`analyses/candidate_search/check_candidates.py`). 36 qualify and form the second held-out set (`docs/preregistration_heldout_relations_2.md`); 48 do not, most because their files hold fewer than two schema DVs (30) or because they are surveys without a user study (10). Every decision and its reason is in `analyses/candidate_search/check_decisions.csv`. None of the five first-test drops qualifies. 30 of the 36 were downloaded, verified and analysed; results are in `docs/heldout_results_2.md`, and the six drops and their reasons are in `analyses/heldout_2/sources_manifest.yaml`.

### Limits

- **Recall.**
    - DataCite and OSF index only repositories that register DOIs or are hosted on OSF. Data on GitHub, lab pages or publisher supplements without a DataCite DOI are missed.
    - 1,652 datasets had no paper the script could find. Some of them do have one, for example when the dataset title differs from the paper's.
    - 749 had no licence in their metadata, mostly OSF projects whose licence was never set.
- **Title matches.** 22 of the 57 papers were found by title. The checks above make a wrong paper unlikely. Two linked papers are not the paper the data belong to:
    - Dryad links the perceived-cooperativity data to an earlier paper by the same authors;
    - Zenodo links the shared-autonomy data to a related paper of the same project.
- **Archives.** 30 of the 57 put their data inside zip archives, eight of them over 1 GB (ROS bags, videos, Unity projects). Their contents are known only from READMEs. Several hold MATLAB `.mat` files next to `.xlsx`/`.csv` tables. The tables are enough, and `.mat` files can be read without executing code.
- **One screener.** The manual screen was done by one rater (Claude), with no second rater.

### What happens next

**Split (2026-11-05).** The 57 were split half and half from metadata only, before any download, by `split_heldout.py`:
- stratified by domain and by the measures named in the metadata;
- seed 20261105;
- result in `heldout_split.csv`.

**The two halves.**
- *29 held-out datasets* test the exploratory relations of `docs/cross_study_insights.md` and the header-matching gain. The plan is pre-registered in `docs/preregistration_heldout_relations.md`.
- *28 development datasets* join the development set.

**Downloads.** Only tables, codebooks, READMEs and questionnaires are fetched, each pinned with its repository checksum as in `analyses/development_sources/sources.yaml`. From archives over 1 GB, only the needed members are fetched, by HTTP range requests where the server supports them.

`acoustic_touch_blind_plosone23` is among the 57 again. Dryad's API now lists its files, but the earlier download attempt was refused (HTTP 401/403), so it may still fail.

**Outcome (2026-11-05).**
- *Held-out.* 19 of the 29 datasets were downloaded, verified and analysed. Results are in `docs/heldout_results.md`; the 10 drops and their reasons are in `analyses/heldout/sources_manifest.yaml`.
- *Development.* 22 of the 28 datasets were downloaded and verified (`analyses/development_sources/sources_search.yaml`, files in `data/raw/development/`).
    - Two hold no table the pipeline reads.
    - Three are on Dryad, which refused the download again (HTTP 401), `acoustic_touch_blind_plosone23` among them.
    - One needs a Harvard Dataverse guestbook form.
    - 16 of the 22 have curated mappings (`schemas/<source_id>_mapping.yaml`) and joined `sources_manifest_no4tu.yaml` on 2026-11-05, written from the codebooks and papers before any LLM run on them. The other six hold no schema DV in the downloaded files; the manifest lists each with its reason.
