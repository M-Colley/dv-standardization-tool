# Open-data studies: candidates for new sources (2026-11-04)

The confirmatory test and its post hoc rerun (`docs/confirmatory_shortlist_results.md`) left gains that only new, unseen studies can confirm. This list collects peer-reviewed HCI studies whose participant-level data are openly licensed. Nothing listed here has been downloaded, labelled or run yet; which set they join (development or a new held-out validation set) is still open.

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

## Pool B: other HCI domains (candidates for a held-out set)

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

## Next steps

1. Decide the destination: the development set (grows the catalog and meta-analysis, but uses up the studies as test data) or a new held-out validation set (confirms the post hoc gains; gold labels frozen before any run, as in `docs/preregistration_confirmatory_shortlist.md`).
2. Download only the listed files, pinned by checksum, as in `analyses/confirmatory/sources.yaml`.
