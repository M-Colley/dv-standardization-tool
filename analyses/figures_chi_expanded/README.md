# Expanded corpus (14 datasets) — figure set

Same pipeline and figure script as [`../figures_chi/`](../figures_chi/), but run
on `data/raw/study_catalog_expanded.csv` = the original 10 catalog sources plus
**5 newly discovered, verified datasets** (found via a web-research + adversarial-
verification workflow, then tested through the real pipeline):

| new source | type | adds |
|---|---|---|
| `opticarvis_chi25` (M-Colley) | github | AOA usefulness items |
| `ehmi_japan_germany_hitl_mobo` (M-Colley) | github | mental_demand, trust, aesthetics (DE/JP) |
| `mimbcd_uta7_nasa_tlx` (MIMBCD-UI) | github | full NASA-TLX — *cross-domain (medical HCI)* |
| `partial_automation_workload_real_traffic` | 4TU | NASA-TLX on-road |
| `td2d_takeover_distracted_l2` | Zenodo | TLX subscales, HR, task time |

Reproduce:
```
python scripts/run_catalog_meta_analysis.py --catalog data/raw/study_catalog_expanded.csv \
  --url-column dataset_url --source-id-column study_name --source-type-column source_type \
  --context-columns domain,task --output-dir data/processed/catalog_expanded \
  --disable-llm-deduction --cache-dir data/processed/.cache/sources
python analyses/multi_study_analysis.py \
  --input-dir data/processed/catalog_expanded/standardized --output-dir analyses/output_python_expanded
CHI_ANALYSIS_DIR=analyses/output_python_expanded CHI_BATCH_DIR=data/processed/catalog_expanded \
  CHI_FIGURES_DIR=analyses/figures_chi_expanded python analyses/make_chi_figures.py
```

## What the extra datasets buy (vs the 9-study corpus)
| quantity | 9-study | 14-study |
|---|---|---|
| Datasets (with ≥1 DV) | 9 (6) | **14 (11)** |
| Meta-analyzable DVs (k≥2) | 17 | **25** |
| DVs pooled at k≥3 | 3 | **9** |
| `mental_demand` k | 6 | **10** |
| `trust_rating` k | 3 | **4** |
| NASA-TLX battery (effort, frustration, performance, physical/temporal demand, TLX_SCORE) | k=0–1 | **k=4 each (newly poolable)** |
| Mean pairwise Jaccard | 0.070 | **0.105** |

The headline: **adding five datasets turns the whole NASA-TLX battery into a
random-effects meta-analysis** and pushes Mental Demand to k=10 — a direct
demonstration of the tool's value (more standardized inputs → more synthesizable
evidence).

## Caveats specific to the new sources
- **`ehmi_japan_germany_hitl_mobo`** is a human-in-the-loop Bayesian-optimization
  study: its `User_ID` is closer to an optimization-iteration id than a person, so
  the auto participant-level pool (2,000 rows → 1,520) is likely still inflated —
  treat its weight as anti-conservative, like FACT-AV.
- **`mimbcd_uta7_nasa_tlx`** is medical HCI (breast-cancer diagnosis workload), not
  automotive — a deliberate cross-domain stress test of the TLX mapping.
- **`colley_dissertation`** was found and verified but **excluded** from this corpus
  because it re-pools participants from other M-Colley studies (double-counting risk).
- Several verified datasets (eHMI trust surveys, MIMBCD SUS, takeover-time logs)
  downloaded cleanly but mapped **0 DVs** — their instrument labels aren't in the
  deterministic schema yet (an alias-coverage gap, not a data problem). See
  `data/processed/_dataset_candidates_raw.json` for the full candidate list.
