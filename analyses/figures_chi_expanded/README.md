# Expanded corpus (14 datasets) — figure set

Same pipeline and figure script as [`../figures_chi/`](../figures_chi/), but run
on `data/raw/study_catalog_expanded.csv` = the original 10 catalog sources plus
**5 newly discovered datasets** (found via a web-research + adversarial-
verification workflow, then tested through the real pipeline):

| new source | type | canonical DVs it contributes |
|---|---|---|
| `opticarvis_chi25` (M-Colley) | github | `AOA1`, `AOA2` (raises both from k=1 to k=2) |
| `ehmi_japan_germany_hitl_mobo` (M-Colley) | github | `mental_demand`, `trust_rating`, `aesthetics_rating` (DE/JP) |
| `mimbcd_uta7_nasa_tlx` (MIMBCD-UI) | github | full NASA-TLX — *cross-domain (medical HCI)* |
| `partial_automation_workload_real_traffic` | 4TU | full NASA-TLX, on-road |
| `td2d_takeover_distracted_l2` | Zenodo | full NASA-TLX + `task_completion_time` |

Reproduce (PowerShell — this repo's documented shell):
```powershell
python scripts/run_catalog_meta_analysis.py --catalog data/raw/study_catalog_expanded.csv `
  --url-column dataset_url --source-id-column study_name --source-type-column source_type `
  --context-columns domain,task --output-dir data/processed/catalog_expanded `
  --disable-llm-deduction --cache-dir data/processed/.cache/sources
python analyses/multi_study_analysis.py `
  --input-dir data/processed/catalog_expanded/standardized --output-dir analyses/output_python_expanded
$env:CHI_ANALYSIS_DIR = "analyses/output_python_expanded"
$env:CHI_BATCH_DIR    = "data/processed/catalog_expanded"
$env:CHI_FIGURES_DIR  = "analyses/figures_chi_expanded"
python analyses/make_chi_figures.py
```

> **Stage 1 needs the network and is not reproducible from a clone alone.**
> `data/processed/` is gitignored, so a fresh clone has neither the
> standardized tables nor the source cache. Stages 2–3 reproduce exactly from
> `data/processed/catalog_expanded/standardized/` if you still have it. One
> catalog source (ACM DL) is served behind a Cloudflare challenge and always
> reports `access_restricted`.

## Headline numbers
| quantity | value | source file |
|---|---|---|
| Catalog sources | 15 | `run_summary.json` |
| Retrieved (1 access-restricted) | 14 | `run_summary.json` |
| Files standardized | 11,608 | `run_summary.json` |
| Columns schema-mapped | 69,882 / 135,234 (52%), 20,104 blocked | `run_summary.json` (`mapping_metrics`) |
| Canonical DVs surfaced | 41 | `dv_presence_matrix.csv` |
| Datasets with ≥1 canonical DV | 11 of 14 | `dv_presence_matrix.csv` |
| Meta-analyzable DVs (k ≥ 2) | 25 | `meta_analysis_summary.csv` |
| DVs pooled across ≥ 3 studies | 9 | `meta_analysis_summary.csv` |
| Mean pairwise DV overlap | Jaccard = 0.105 (all pairs) / 0.169 (DV-bearing pairs) | `dv_overlap_details.csv` |

## What the extra datasets buy (vs the 9-study corpus)
| quantity | 9-study | 14-study |
|---|---|---|
| Datasets (with ≥1 DV) | 9 (6) | **14 (11)** |
| Meta-analyzable DVs (k≥2) | 17 | **25** |
| DVs pooled at k≥3 | 3 | **9** |
| `mental_demand` k | 6 | **10** |
| `trust_rating` k | 3 | **4** |
| NASA-TLX battery (effort, frustration, performance, physical/temporal demand, TLX_SCORE) | k=1 each | **k=4 each (newly poolable)** |
| Mean pairwise Jaccard, all pairs | 0.070 | **0.105** |
| Mean pairwise Jaccard, DV-bearing pairs | 0.155 | **0.169** |

Adding five datasets turns the whole NASA-TLX battery into a random-effects
meta-analysis and pushes Mental Demand to k=10 — more standardized inputs →
more synthesizable evidence.

## How to read the overlap gain
The all-pairs Jaccard (0.070 → 0.105) averages over pairs involving the three
datasets that contributed **zero** canonical DVs, which can only score 0.0.
Restricted to DV-bearing pairs the corpus moves 0.155 → 0.169, and **that gain
is entirely due to three near-duplicate NASA-TLX-only datasets**:

| pair | Jaccard |
|---|---|
| `mimbcd_uta7_nasa_tlx` ↔ `partial_automation_workload_real_traffic` | 1.000 |
| `mimbcd_uta7_nasa_tlx` ↔ `td2d_takeover_distracted_l2` | 0.875 |
| `partial_automation_workload_real_traffic` ↔ `td2d_takeover_distracted_l2` | 0.875 |

Drop those three pairs and the DV-bearing mean falls to **0.125** — *below* the
9-study corpus. The honest reading: the new datasets make the TLX battery
poolable, but they do not make the corpus as a whole more mutually comparable.

## Caveats
- **`TLX_SCORE` is pooled on the 0–20 metric**, not the 0–100 raw-TLX
  convention, because this tool derives it as the unweighted mean of the six
  0–20 subscales. `schemas/standard_dv_mapping.yaml` v3.1 changed
  `canonical_range` to `[0, 20]` to match; `td2d_takeover_distracted_l2` ships
  a native 0–100 TLX and is rescaled down (see `irt_rescaling_warnings.csv`).
- **NASA-TLX `performance` polarity is not reconciled.** The subscale's anchor
  direction is inverted in the original instrument, and studies differ in
  which convention they used — `mimbcd_uta7_nasa_tlx` means 14.68 against
  6.4–9.0 elsewhere. The tool has no per-study anchor metadata, so it flags
  rather than silently flips: see `polarity_warning` in
  `meta_analysis_summary.csv` and `reverse_coding_warnings.csv`. Treat the
  `performance` pool as not-yet-harmonized.
- **Every pooled estimate is a pooled descriptive mean, not a contrast-based
  effect size.** There is no IV and no control condition, so I² here reflects
  instrument, population and protocol differences rather than treatment
  heterogeneity. `standardized_effects.csv` is likewise a descriptive
  study-vs-pool deviation.
- **k=2 rows are fixed-effect estimates** (`pooling_method =
  fixed_effects_k_lt_3`). Their intervals use Knapp-Hartung with t(k−1), which
  is why they are wide; a narrow interval at k=2 would have been an artifact of
  large per-study n rather than evidence of precision.
- **Egger's test and trim-and-fill are reported but not interpretable** at
  k=3–4. `eggers_test.csv` carries `underpowered_k_lt_10`.
- **`ehmi_japan_germany_hitl_mobo`** is a human-in-the-loop Bayesian-
  optimization study: its `User_ID` is closer to an optimization-iteration id
  than a person, so the auto participant-level pool (2,000 rows → 1,520) is
  likely still inflated — treat its weight as anti-conservative, like FACT-AV
  (n=10,354, no participant id, stays row-level).
- **`mimbcd_uta7_nasa_tlx`** is medical HCI (breast-cancer diagnosis workload),
  a deliberate cross-domain stress test. Its 90 rows are 45 clinicians × 2
  conditions, not 90 independent participants.
- **`colley_dissertation_pooled`** was found and verified but **excluded**
  because it re-pools participants from other M-Colley studies. Note that eight
  of the included datasets also come from that lab, so participant overlap
  across them has not been ruled out.
- Several verified datasets (eHMI trust surveys, MIMBCD SUS, takeover-time
  logs) downloaded cleanly but mapped **0 DVs** — an alias-coverage gap, not a
  data problem. See `data/processed/_dataset_candidates_raw.json`.
