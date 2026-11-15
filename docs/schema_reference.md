# DV schema reference

Generated from `schemas/standard_dv_mapping.yaml` (version 3.4) by `python -m opendv.schema_export`; do not edit by hand. 206 DVs in 9 clusters. The same records are in `schemas/export/opendv_dvs.csv` and `.json`.

Map a column to the **id**. *Range* is the canonical range the batch checks values against (blank: unbounded); *direction* says which end is better. Items of a scale name their scale under *item of*. How to write a study mapping: [Mapping a study for OpenDV](sigchi_mapping_guide.md).

## Task Performance (`performance`, 14)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `task_completion_time` | Task Completion Time |  | seconds | lower_is_better |  |  |
| `task_success` | Task Success Rate | 0 to 1 | proportion | higher_is_better |  |  |
| `error_rate` | Error Rate | 0 to 100 | percentage | lower_is_better |  |  |
| `efficiency` | Efficiency |  | ratio | higher_is_better |  |  |
| `touch_offset` | Touch Offset |  | millimeters | lower_is_better |  |  |
| `sdlp` | Standard Deviation of Lateral Position |  | meters | lower_is_better |  |  |
| `brake_reaction_time` | Brake Reaction Time |  | seconds | lower_is_better |  |  |
| `takeover_time` | Takeover Time |  | seconds | lower_is_better |  |  |
| `max_lateral_acceleration` | Maximum Lateral Acceleration |  | m/s² | lower_is_better |  |  |
| `words_per_minute` | Text Entry Rate (WPM) |  | words per minute | higher_is_better |  |  |
| `kspc` | Keystrokes per Character (KSPC) |  | keystrokes per character | lower_is_better |  |  |
| `msd_error_rate` | Uncorrected Error Rate (MSD) | 0 to 100 | percent | lower_is_better |  |  |
| `total_error_rate` | Total Error Rate (TER) | 0 to 100 | percent | lower_is_better |  |  |
| `fitts_throughput` | Throughput (Fitts' law, bits/s) |  | bits/s | higher_is_better |  |  |

## Behavioral Interaction (`behavioral_interaction`, 9)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `click_through_rate` | Click-Through Rate | 0 to 100 | percentage | neutral |  |  |
| `interaction_events` | Interaction Events |  | integer | neutral |  |  |
| `retention_time` | Retention / Dwell Time |  | seconds | higher_is_better |  |  |
| `gaze_on_road` | Gaze on Road Percentage | 0 to 100 | percentage | higher_is_better |  |  |
| `pedestrian_crossing_time` | Pedestrian Crossing Time |  | seconds | neutral |  |  |
| `crossing_initiation_time` | Crossing Initiation Time |  | seconds | neutral |  |  |
| `time_after_crossing` | Time After Crossing |  | seconds | neutral |  |  |
| `perceived_vehicle_influence` | Perceived Vehicle Influence on Decision | 1 to 5 | 5-point (1–5) | neutral |  |  |
| `intervention_responsiveness` | Intervention Responsiveness (latency) |  | seconds | neutral |  |  |

## Cognitive Load (`cognitive_load`, 22)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `TLX_SCORE` | NASA-TLX Composite Score | 0 to 100 | weighted composite | lower_is_better | NASA-TLX |  |
| `mental_demand` | Mental Demand | 0 to 20 | 21-point (0-20) | lower_is_better | NASA-TLX (Global or Subscales) |  |
| `paas_mental_effort` | Mental Effort (Paas scale) | 1 to 9 | 9-point (1-9) | lower_is_better | Paas Scale |  |
| `rsme` | Rating Scale Mental Effort (RSME) | 0 to 150 | 0-150 | lower_is_better | RSME |  |
| `physical_demand` | Physical Demand | 0 to 20 | 21-point (0–20) | lower_is_better | NASA-TLX |  |
| `temporal_demand` | Temporal Demand | 0 to 20 | 21-point (0–20) | lower_is_better | NASA-TLX |  |
| `performance` | Performance (TLX) | 0 to 20 | 21-point (0–20) | lower_is_better | NASA-TLX |  |
| `effort` | Effort | 0 to 20 | 21-point (0–20) | lower_is_better | NASA-TLX |  |
| `frustration` | Frustration | 0 to 20 | 21-point (0–20) | lower_is_better | NASA-TLX |  |
| `dali_total` | DALI Total Score | 0 to 5 | 6-point (0–5) | lower_is_better | DALI |  |
| `dali_effort_of_attention` | DALI — Effort of Attention | 0 to 5 | 6-point (0–5) | lower_is_better | DALI |  |
| `dali_visual_demand` | DALI — Visual Demand | 0 to 5 | 6-point (0–5) | lower_is_better | DALI |  |
| `dali_auditory_demand` | DALI — Auditory Demand | 0 to 5 | 6-point (0–5) | lower_is_better | DALI |  |
| `dali_temporal_demand` | DALI — Temporal Demand | 0 to 5 | 6-point (0–5) | lower_is_better | DALI |  |
| `dali_interference` | DALI — Interference | 0 to 5 | 6-point (0–5) | lower_is_better | DALI |  |
| `dali_situational_stress` | DALI — Situational Stress | 0 to 5 | 6-point (0–5) | lower_is_better | DALI |  |
| `sart_total` | sART Total Score |  | composite | higher_is_better | SART |  |
| `sart_understanding` | sART Understanding Subscale | 0 to 20 | 21-point (0–20) | higher_is_better | SART |  |
| `sart_demand` | sART Demand Subscale | 0 to 20 | 21-point (0–20) | lower_is_better | SART |  |
| `sart_supply` | sART Supply Subscale | 0 to 20 | 21-point (0–20) | higher_is_better | SART |  |
| `situational_awareness` | Situational Awareness (generic) |  | score | higher_is_better | SAGAT; SPAM |  |
| `understanding_rating` | Understanding / Comprehension (eHMI) | 1 to 5 | 5-point (1–5) | higher_is_better |  |  |

## Safety & Trust (`safety_trust`, 37)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `perceived_safety` | Perceived Safety | -3 to 3 | 7-point (-3–3) | higher_is_better |  |  |
| `perceived_risk` | Perceived Risk | 1 to 100 | VAS (1–100) | lower_is_better |  |  |
| `trust_rating` | Trust in Automation / eHMI | 1 to 5 | 5-point (1–5) | higher_is_better |  |  |
| `tia_reliability_competence` | TiA — Reliability/Competence | 1 to 5 | 5-point (1–5) | higher_is_better | Körber TiA |  |
| `tia_understanding_predictability` | TiA — Understanding/Predictability | 1 to 5 | 5-point (1–5) | higher_is_better | Körber TiA |  |
| `tia_familiarity` | TiA — Familiarity | 1 to 5 | 5-point (1–5) | higher_is_better | Körber TiA |  |
| `tia_intention_of_developers` | TiA — Intention of Developers | 1 to 5 | 5-point (1–5) | higher_is_better | Körber TiA |  |
| `tia_propensity_to_trust` | TiA — Propensity to Trust | 1 to 5 | 5-point (1–5) | higher_is_better | Körber TiA |  |
| `tia_trust_in_automation` | TiA — Trust in Automation | 1 to 5 | 5-point (1–5) | higher_is_better | Körber TiA |  |
| `jian_trust` | Jian Trust Scale | 1 to 7 | 7-point (1–7) | higher_is_better | Jian et al. Trust Scale |  |
| `godspeed_perceived_safety` | Godspeed — Perceived Safety | 1 to 5 | 5-point (1–5) | higher_is_better | Godspeed |  |
| `ttc` | Time to Collision |  | seconds | higher_is_better |  |  |
| `headway` | Headway / Following Distance |  | seconds | higher_is_better |  |  |
| `number_of_collisions` | Number of Collisions |  | count | lower_is_better |  |  |
| `perceived_safety_item_01` | Perceived Safety (item 1) | -3 to 3 | 7-point (-3-3) | higher_is_better |  |  |
| `perceived_safety_item_02` | Perceived Safety (item 2) | -3 to 3 | 7-point (-3-3) | higher_is_better |  |  |
| `perceived_safety_item_03` | Perceived Safety (item 3) | -3 to 3 | 7-point (-3-3) | higher_is_better |  |  |
| `perceived_safety_item_04` | Perceived Safety (item 4) | -3 to 3 | 7-point (-3-3) | higher_is_better |  |  |
| `tia_item_01` | TiA trust item 1 | 1 to 5 | 5-point (1-5) | higher_is_better |  |  |
| `tia_item_02` | TiA trust item 2 | 1 to 5 | 5-point (1-5) | higher_is_better |  |  |
| `tia_item_03` | TiA trust item 3 | 1 to 5 | 5-point (1-5) | higher_is_better |  |  |
| `tia_item_04` | TiA trust item 4 | 1 to 5 | 5-point (1-5) | higher_is_better |  |  |
| `tia_item_05` | TiA trust item 5 | 1 to 5 | 5-point (1-5) | higher_is_better |  |  |
| `tia_item_06` | TiA trust item 6 | 1 to 5 | 5-point (1-5) | higher_is_better |  |  |
| `perceived_conflict` | Perceived User–Vehicle Conflict | 1 to 5 | 5-point (1–5) | lower_is_better |  |  |
| `conflict_item_01` | Perceived Conflict (item 1) | 1 to 5 | 5-point (1–5) | lower_is_better |  | perceived_conflict |
| `conflict_item_02` | Perceived Conflict (item 2) | 1 to 5 | 5-point (1–5) | lower_is_better |  | perceived_conflict |
| `conflict_item_03` | Perceived Conflict (item 3) | 1 to 5 | 5-point (1–5) | lower_is_better |  | perceived_conflict |
| `conflict_item_04` | Perceived Conflict (item 4) | 1 to 5 | 5-point (1–5) | lower_is_better |  | perceived_conflict |
| `conflict_item_05` | Perceived Conflict (item 5) | 1 to 5 | 5-point (1–5) | lower_is_better |  | perceived_conflict |
| `sts_ad_trust` | STS-AD Situational Trust (composite) | 1 to 7 | 7-point (1–7) | higher_is_better | STS-AD (Holthausen, Wintersberger & Walker 2020) |  |
| `sts_ad_item_01` | STS-AD trust item 1 | 1 to 7 | 7-point (1–7) | higher_is_better |  | sts_ad_trust |
| `sts_ad_item_02` | STS-AD trust item 2 | 1 to 7 | 7-point (1–7) | higher_is_better |  | sts_ad_trust |
| `sts_ad_item_03` | STS-AD trust item 3 | 1 to 7 | 7-point (1–7) | higher_is_better |  | sts_ad_trust |
| `sts_ad_item_04` | STS-AD trust item 4 | 1 to 7 | 7-point (1–7) | higher_is_better |  | sts_ad_trust |
| `sts_ad_item_05` | STS-AD trust item 5 | 1 to 7 | 7-point (1–7) | higher_is_better |  | sts_ad_trust |
| `sts_ad_item_06` | STS-AD trust item 6 | 1 to 7 | 7-point (1–7) | higher_is_better |  | sts_ad_trust |

## UX & Satisfaction (`ux_satisfaction`, 90)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `aesthetics_rating` | Aesthetic Evaluation | 1 to 7 | 7-point (1–7) | higher_is_better |  |  |
| `acceptance_rating` | Acceptance (generic) | 1 to 7 | 7-point (1–7) | higher_is_better |  |  |
| `user_satisfaction` | User Satisfaction | 1 to 7 | 7-point | higher_is_better | UEQ; UMUX / UMUX-Lite; CSAT; NPS |  |
| `usability` | Usability (SUS Total) | 0 to 100 | SUS (0–100) | higher_is_better | SUS |  |
| `SUS1` | SUS Item 1 | 1 to 5 | 5-point | higher_is_better | SUS |  |
| `SUS2` | SUS Item 2 | 1 to 5 | 5-point | lower_is_better | SUS |  |
| `SUS3` | SUS Item 3 | 1 to 5 | 5-point | higher_is_better | SUS |  |
| `SUS4` | SUS Item 4 | 1 to 5 | 5-point | lower_is_better | SUS |  |
| `SUS5` | SUS Item 5 | 1 to 5 | 5-point | higher_is_better | SUS |  |
| `SUS6` | SUS Item 6 | 1 to 5 | 5-point | lower_is_better | SUS |  |
| `SUS7` | SUS Item 7 | 1 to 5 | 5-point | higher_is_better | SUS |  |
| `SUS8` | SUS Item 8 | 1 to 5 | 5-point | lower_is_better | SUS |  |
| `SUS9` | SUS Item 9 | 1 to 5 | 5-point | higher_is_better | SUS |  |
| `SUS10` | SUS Item 10 | 1 to 5 | 5-point | lower_is_better | SUS |  |
| `ueq_attractiveness` | UEQ — Attractiveness | -3 to 3 | 7-point (-3–3) | higher_is_better | UEQ; UEQ+ |  |
| `ueq_perspicuity` | UEQ — Perspicuity | -3 to 3 | 7-point (-3–3) | higher_is_better | UEQ; UEQ+ |  |
| `ueq_efficiency` | UEQ — Efficiency | -3 to 3 | 7-point (-3–3) | higher_is_better | UEQ; UEQ+ |  |
| `ueq_dependability` | UEQ — Dependability | -3 to 3 | 7-point (-3–3) | higher_is_better | UEQ; UEQ+ |  |
| `ueq_stimulation` | UEQ — Stimulation | -3 to 3 | 7-point (-3–3) | higher_is_better | UEQ; UEQ+ |  |
| `ueq_novelty` | UEQ — Novelty | -3 to 3 | 7-point (-3–3) | higher_is_better | UEQ; UEQ+ |  |
| `ueq_pragmatic` | UEQ — Pragmatic Quality (PQ) | -3 to 3 | 7-point (-3–3) | higher_is_better | UEQ-S; UEQ |  |
| `ueq_hedonic` | UEQ — Hedonic Quality (HQ) | -3 to 3 | 7-point (-3–3) | higher_is_better | UEQ-S; UEQ |  |
| `attrakdiff_pq` | AttrakDiff — Pragmatic Quality | -3 to 3 | 7-point (-3–3) | higher_is_better | AttrakDiff |  |
| `attrakdiff_hqs` | AttrakDiff — Hedonic Quality (Stimulation) | -3 to 3 | 7-point (-3–3) | higher_is_better | AttrakDiff |  |
| `attrakdiff_hqi` | AttrakDiff — Hedonic Quality (Identity) | -3 to 3 | 7-point (-3–3) | higher_is_better | AttrakDiff |  |
| `attrakdiff_att` | AttrakDiff — Attractiveness | -3 to 3 | 7-point (-3–3) | higher_is_better | AttrakDiff |  |
| `utaut_performance_expectancy` | UTAUT — Performance Expectancy | 1 to 7 | 7-point (1–7) | higher_is_better | UTAUT; UTAUT2 |  |
| `utaut_effort_expectancy` | UTAUT — Effort Expectancy | 1 to 7 | 7-point (1–7) | higher_is_better | UTAUT; UTAUT2 |  |
| `utaut_social_influence` | UTAUT — Social Influence | 1 to 7 | 7-point (1–7) | higher_is_better | UTAUT; UTAUT2 |  |
| `utaut_facilitating_conditions` | UTAUT — Facilitating Conditions | 1 to 7 | 7-point (1–7) | higher_is_better | UTAUT; UTAUT2 |  |
| `utaut_behavioral_intention` | UTAUT — Behavioral Intention | 1 to 7 | 7-point (1–7) | higher_is_better | UTAUT; UTAUT2; TAM |  |
| `utaut_hedonic_motivation` | UTAUT2 — Hedonic Motivation | 1 to 7 | 7-point (1–7) | higher_is_better | UTAUT2 |  |
| `tam_perceived_usefulness` | TAM — Perceived Usefulness | 1 to 7 | 7-point (1–7) | higher_is_better | TAM; TAM2; TAM3 |  |
| `tam_perceived_ease_of_use` | TAM — Perceived Ease of Use | 1 to 7 | 7-point (1–7) | higher_is_better | TAM; TAM2; TAM3 |  |
| `AOA_USEFULNESS` | van der Laan Acceptance — Usefulness Subscale | -2 to 2 | 5-point (-2 to +2) | higher_is_better | van der Laan Acceptance Scale |  |
| `AOA_SATISFYING` | van der Laan Acceptance — Satisfying Subscale | -2 to 2 | 5-point (-2 to +2) | higher_is_better | van der Laan Acceptance Scale |  |
| `AOA1` | van der Laan — Item 1 | -2 to 2 | 5-point (-2 to +2) | lower_is_better | van der Laan Acceptance Scale |  |
| `AOA2` | van der Laan — Item 2 | -2 to 2 | 5-point (-2 to +2) | lower_is_better | van der Laan Acceptance Scale |  |
| `AOA3` | van der Laan — Item 3 | -2 to 2 | 5-point (-2 to +2) | higher_is_better | van der Laan Acceptance Scale |  |
| `AOA4` | van der Laan — Item 4 | -2 to 2 | 5-point (-2 to +2) | lower_is_better | van der Laan Acceptance Scale |  |
| `AOA5` | van der Laan — Item 5 | -2 to 2 | 5-point (-2 to +2) | lower_is_better | van der Laan Acceptance Scale |  |
| `AOA6` | van der Laan — Item 6 | -2 to 2 | 5-point (-2 to +2) | higher_is_better | van der Laan Acceptance Scale |  |
| `AOA7` | van der Laan — Item 7 | -2 to 2 | 5-point (-2 to +2) | lower_is_better | van der Laan Acceptance Scale |  |
| `AOA8` | van der Laan — Item 8 | -2 to 2 | 5-point (-2 to +2) | higher_is_better | van der Laan Acceptance Scale |  |
| `AOA9` | van der Laan — Item 9 | -2 to 2 | 5-point (-2 to +2) | lower_is_better | van der Laan Acceptance Scale |  |
| `ipq_general_presence` | IPQ — General Presence | 0 to 6 | 7-point (0–6) | higher_is_better | IPQ |  |
| `ipq_spatial_presence` | IPQ — Spatial Presence | 0 to 6 | 7-point (0–6) | higher_is_better | IPQ |  |
| `ipq_involvement` | IPQ — Involvement | 0 to 6 | 7-point (0–6) | higher_is_better | IPQ |  |
| `ipq_experienced_realism` | IPQ — Experienced Realism | 0 to 6 | 7-point (0–6) | higher_is_better | IPQ |  |
| `godspeed_anthropomorphism` | Godspeed — Anthropomorphism | 1 to 5 | 5-point (1–5) | higher_is_better | Godspeed (Bartneck et al. 2009) |  |
| `godspeed_animacy` | Godspeed — Animacy | 1 to 5 | 5-point (1–5) | higher_is_better | Godspeed |  |
| `godspeed_likeability` | Godspeed — Likeability | 1 to 5 | 5-point (1–5) | higher_is_better | Godspeed |  |
| `godspeed_perceived_intelligence` | Godspeed — Perceived Intelligence | 1 to 5 | 5-point (1–5) | higher_is_better | Godspeed |  |
| `usability_problems` | Usability Problems |  | integer | lower_is_better |  |  |
| `task_feasibility` | Perceived Task Feasibility | 1 to 5 | 5-point (1–5) | higher_is_better |  |  |
| `task_feasibility_item_01` | Task Feasibility (item 1) | 1 to 5 | 5-point (1–5) | higher_is_better |  |  |
| `task_feasibility_item_02` | Task Feasibility (item 2) | 1 to 5 | 5-point (1–5) | higher_is_better |  |  |
| `umux_score` | UMUX (Usability Metric for User Experience) | 0 to 100 | 0–100 | higher_is_better | UMUX (Finstad 2010) |  |
| `umux_lite_score` | UMUX-Lite | 0 to 100 | 0–100 | higher_is_better | UMUX-Lite (Lewis, Utesch & Maher 2013) |  |
| `seq_rating` | Single Ease Question (SEQ) | 1 to 7 | 7-point (1–7) | higher_is_better | SEQ (Sauro & Dumas 2009) |  |
| `ues_focused_attention` | UES-SF – Focused Attention | 1 to 5 | 5-point (1–5) | higher_is_better | UES-SF (O'Brien, Cairns & Hall 2018) | engagement_level |
| `ues_perceived_usability` | UES-SF – Perceived Usability | 1 to 5 | 5-point (1–5) | higher_is_better | UES-SF (O'Brien, Cairns & Hall 2018) | engagement_level |
| `ues_aesthetic_appeal` | UES-SF – Aesthetic Appeal | 1 to 5 | 5-point (1–5) | higher_is_better | UES-SF (O'Brien, Cairns & Hall 2018) | engagement_level |
| `ues_reward` | UES-SF – Reward | 1 to 5 | 5-point (1–5) | higher_is_better | UES-SF (O'Brien, Cairns & Hall 2018) | engagement_level |
| `csi_score` | Creativity Support Index (CSI) | 0 to 100 | 0–100 | higher_is_better | CSI (Cherry & Latulipe 2014) |  |
| `pxi_ease_of_control` | PXI – Ease of Control | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_progress_feedback` | PXI – Progress Feedback | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_audiovisual_appeal` | PXI – Audiovisual Appeal | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_clarity_of_goals` | PXI – Goals and Rules | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_challenge` | PXI – Challenge | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_mastery` | PXI – Mastery | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_curiosity` | PXI – Curiosity | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_immersion` | PXI – Immersion | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_autonomy` | PXI – Autonomy | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_meaning` | PXI – Meaning | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `pxi_enjoyment` | PXI – Enjoyment | -3 to 3 | 7-point (−3–3) | higher_is_better | PXI (Vanden Abeele et al. 2020) |  |
| `geq_competence` | GEQ – Competence | 0 to 4 | 5-point (0–4) | higher_is_better | GEQ core module (IJsselsteijn, de Kort & Poels 2013) |  |
| `geq_immersion` | GEQ – Sensory and Imaginative Immersion | 0 to 4 | 5-point (0–4) | higher_is_better | GEQ core module (IJsselsteijn, de Kort & Poels 2013) |  |
| `geq_flow` | GEQ – Flow | 0 to 4 | 5-point (0–4) | higher_is_better | GEQ core module (IJsselsteijn, de Kort & Poels 2013) |  |
| `geq_challenge` | GEQ – Challenge | 0 to 4 | 5-point (0–4) | neutral | GEQ core module (IJsselsteijn, de Kort & Poels 2013) |  |
| `imi_interest_enjoyment` | IMI – Interest/Enjoyment | 1 to 7 | 7-point (1–7) | higher_is_better | IMI (Ryan 1982; McAuley, Duncan & Tammen 1989) |  |
| `imi_perceived_competence` | IMI – Perceived Competence | 1 to 7 | 7-point (1–7) | higher_is_better | IMI (Ryan 1982; McAuley, Duncan & Tammen 1989) |  |
| `imi_effort_importance` | IMI – Effort/Importance | 1 to 7 | 7-point (1–7) | neutral | IMI (Ryan 1982; McAuley, Duncan & Tammen 1989) |  |
| `imi_perceived_choice` | IMI – Perceived Choice | 1 to 7 | 7-point (1–7) | higher_is_better | IMI (Ryan 1982; McAuley, Duncan & Tammen 1989) |  |
| `imi_value_usefulness` | IMI – Value/Usefulness | 1 to 7 | 7-point (1–7) | higher_is_better | IMI (Ryan 1982; McAuley, Duncan & Tammen 1989) |  |
| `slater_usoh_steed_presence` | Presence (Slater–Usoh–Steed questionnaire) | 1 to 7 | 7-point (1–7) | higher_is_better | SUS presence questionnaire (Slater, Usoh & Steed 1994; Usoh et al. 2000) |  |
| `flow_fks_total` | Flow Short Scale (FKS) – Flow | 1 to 7 | 7-point (1–7) | higher_is_better | FKS (Rheinberg, Vollmeyer & Engeser 2003) |  |
| `flow_fks_fluency` | Flow Short Scale – Fluency of Performance | 1 to 7 | 7-point (1–7) | higher_is_better | FKS (Rheinberg, Vollmeyer & Engeser 2003) | flow_fks_total |
| `flow_fks_absorption` | Flow Short Scale – Absorption by Activity | 1 to 7 | 7-point (1–7) | higher_is_better | FKS (Rheinberg, Vollmeyer & Engeser 2003) | flow_fks_total |
| `embodiment_score` | Avatar Embodiment | 1 to 7 | 7-point (1–7) | higher_is_better | Avatar Embodiment Questionnaire (Peck & Gonzalez-Franco 2021) |  |

## Physiological & Biometric (`physiological_biometric`, 18)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `ssq_total` | SSQ Total Score | 0 to 235.62 | SSQ weighted | lower_is_better | SSQ (Kennedy et al. 1993) |  |
| `ssq_nausea` | SSQ — Nausea | 0 to 200.34 | SSQ weighted | lower_is_better | SSQ |  |
| `ssq_oculomotor` | SSQ — Oculomotor | 0 to 159.18 | SSQ weighted | lower_is_better | SSQ |  |
| `ssq_disorientation` | SSQ — Disorientation | 0 to 292.32 | SSQ weighted | lower_is_better | SSQ |  |
| `vrsq_score` | VRSQ Score | 0 to 100 | percentage (0–100) | lower_is_better | VRSQ (Kim et al. 2018) |  |
| `fms_score` | Fast Motion Sickness Score | 0 to 20 | VAS (0–20) | lower_is_better | FMS (Keshavarz & Hecht 2011) |  |
| `eda_gsr` | Electrodermal Activity (EDA/GSR) |  | microsiemens | neutral | Shimmer GSR; Empatica E4 |  |
| `hr_bpm` | Heart Rate |  | bpm | neutral | Polar H10; Empatica E4 |  |
| `hrv_rmssd` | Heart Rate Variability (RMSSD) |  | milliseconds | neutral | Polar H10; Empatica E4 |  |
| `emg` | Electromyography |  | microvolts | neutral |  |  |
| `pupil_dilation` | Pupil Dilation |  | millimeters | neutral | Tobii Pro; Pupil Labs |  |
| `fixation_duration` | Fixation Duration |  | milliseconds | neutral | Tobii Pro; EyeLink |  |
| `fixation_count` | Fixation Count |  | integer | neutral | Tobii Pro; EyeLink |  |
| `saccade_rate` | Saccade Rate |  | saccades/s | neutral | Tobii Pro; EyeLink |  |
| `dwell_time` | Dwell Time (AOI) |  | milliseconds | neutral | Tobii Pro; EyeLink |  |
| `scanpath_length` | Scanpath Length |  | pixels | neutral | Tobii Pro |  |
| `eye_tracking` | Eye-Tracking Metrics (generic) |  | milliseconds | neutral | Tobii Pro; EyeLink |  |
| `misc_rating` | Misery Scale (MISC) motion-sickness rating | 0 to 10 | 11-point (0–10) | lower_is_better | MISC (Bos, MacKinnon & Patterson 2005) |  |

## Affective & Emotional Response (`affective_emotion`, 13)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `valence` | Valence | 1 to 9 | 9-point (SAM) | higher_is_better | SAM (valence); PANAS |  |
| `arousal` | Arousal | 1 to 9 | 9-point (SAM) | neutral | SAM (arousal) |  |
| `dominance` | Dominance | 1 to 9 | 9-point (SAM) | higher_is_better | SAM (dominance) |  |
| `panas_positive` | PANAS — Positive Affect | 1 to 5 | 5-point (1–5) | higher_is_better | PANAS |  |
| `panas_negative` | PANAS — Negative Affect | 1 to 5 | 5-point (1–5) | lower_is_better | PANAS |  |
| `engagement_level` | Engagement | 1 to 5 | 5-point (1–5) | higher_is_better | UES; SAM (arousal) |  |
| `stress` | Stress | 1 to 5 | 5-point (1–5) | lower_is_better | PSS |  |
| `psychological_reactance` | Psychological Reactance | 1 to 5 | 5-point (1–5) | lower_is_better | Hong Psychological Reactance Scale |  |
| `sleepiness` | Sleepiness (KSS) | 1 to 9 | 9-point (1–9) | neutral | KSS (Karolinska Sleepiness Scale) |  |
| `geq_tension` | GEQ – Tension/Annoyance | 0 to 4 | 5-point (0–4) | lower_is_better | GEQ core module (IJsselsteijn, de Kort & Poels 2013) |  |
| `geq_negative_affect` | GEQ – Negative Affect | 0 to 4 | 5-point (0–4) | lower_is_better | GEQ core module (IJsselsteijn, de Kort & Poels 2013) |  |
| `geq_positive_affect` | GEQ – Positive Affect | 0 to 4 | 5-point (0–4) | higher_is_better | GEQ core module (IJsselsteijn, de Kort & Poels 2013) |  |
| `imi_pressure_tension` | IMI – Pressure/Tension | 1 to 7 | 7-point (1–7) | lower_is_better | IMI (Ryan 1982; McAuley, Duncan & Tammen 1989) |  |

## Social & Collaboration (`social_collaboration`, 2)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `social_presence` | Social Presence | 1 to 7 | 7-point (1–7) | higher_is_better | Networked Minds Social Presence |  |
| `collaboration_quality` | Collaboration Quality | 1 to 7 | 7-point (1–7) | higher_is_better |  |  |

## Accessibility & Inclusion (`accessibility_inclusion`, 1)

| id | label | range | unit | direction | instruments | item of |
|---|---|---|---|---|---|---|
| `accessibility_outcome` | Accessibility Outcome | 0 to 1 | success_failure | higher_is_better |  |  |
