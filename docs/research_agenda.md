# Research Agenda: From a Harmonization Tool to a Paper

This document proposes how OpenDV-HCI can become a full research contribution
(target venues: CHI, TOCHI, *Behavior Research Methods*, *Research Synthesis
Methods*, *Scientific Data*). It separates **evidence that already exists in
this repository** (with the scripts that reproduce it) from **proposed
studies**, ranked by expected scientific value per unit of effort.

---

## 1. Framing

**Working title.** *What Can We Pool? Measuring and Repairing the
Comparability Gap in Open HCI Data.*

**Core argument.** Open HCI data is rarely reusable for evidence synthesis.
Shared files are not the obstacle; the dependent variables (DVs) inside them
are named, scaled and structured inconsistently. Automated harmonization
(schemas, fuzzy matching, LLMs) closes part of that gap but introduces
*silent* failure modes that change the scientific answer. A credible pipeline
has to measure its own error, abstain when unsure, and expose how much each
harmonization decision moves the result.

**Research questions.**

- **RQ1 — Fragmentation.** How much DV vocabulary do open datasets from one
  research area actually share after standardization?
- **RQ2 — Automated mapping.** How accurate is LLM-assisted DV mapping on
  real repositories, and can cheap plausibility constraints make it safe?
- **RQ3 — Analytic robustness.** How sensitive are pooled estimates to
  harmonization choices (mapping provenance, rescaling, unit of analysis,
  estimator, interval method)?
- **RQ4 — Silent failures.** Which failure modes of automated evidence
  synthesis are invisible in ordinary output, and how can a tool expose them?

**Contributions.**

1. A canonical DV schema (160 DVs in thematic clusters, with measurement
   metadata and item → construct structure) and an open, tested pipeline.
2. An empirical audit of 11 open automotive-HCI datasets (RQ1).
3. A leave-alias-out benchmark of LLM DV mapping with a plausibility layer
   (RQ2), seeded by `analyses/evaluate_llm_mapping.py`.
4. A harmonization multiverse showing how much pooled estimates depend on
   analytic choices (RQ3), seeded by `analyses/multiverse_analysis.py`.
5. A taxonomy of silent failures with concrete instances from this
   pipeline (RQ4) and the guards that now catch them.

---

## 2. Evidence already produced

All numbers below are reproducible from this repository; see the script named
in each row. "E1" uses real dataset columns (leave-alias-out against curated
mappings); "E2" holds out schema aliases. Model: `google/gemma-4-E2B-it`
(local, RTX 5080). Proportions are given with Wilson 95% CIs in
`analyses/output_llm_evaluation/`.

### 2.1 Silent failure modes found in the pipeline (RQ4)

| Failure mode | Instance | Effect before the fix | Guard now in place |
|---|---|---|---|
| Wrong estimator implementation | `trim_and_fill` mirrored every study on one side of the mean | Hackshaw (1998) data: 19 imputed studies vs. 7 in `metafor`. In the committed 11-study run every evaluated DV was "adjusted" (e.g. `retention_time` 85.4 → 33.8) | Duval & Tweedie L0/R0 implementation validated against `metafor::trimfill` (`tests/test_meta_analysis_reference.py`) |
| Imputation of empty rows | PCA composite median-imputed rows with no shared DV observed | `ehmi_for_all_chi26` counted 924,223 rows (mostly telemetry); PC1 "explained variance" 0.50 → 0.24 on real observations | Rows with no observed shared DV are dropped before imputation |
| Semantically absurd LLM mappings | Participant **age** → `retention_time`; a survey's **random seed** → `trust_rating`; total survey duration → `sart_total`; participant IDs → Godspeed intelligence | Pooled `retention_time` (k = 3) in the committed results was built partly from participant ages | Plausibility layer (`schemas/llm_plausibility_rules.yaml`): name, data-type and range rules, all rejections logged |
| Tautological causal edges | Item columns (`perceived_safety_item_0x`) entered the LiNGAM DAG next to their own scale score | DAG edges "item → scale" reported as causal structure | Construct → item hierarchy derived from the schema (`subitem_of`, `<construct>_item_NN`) |
| Silent no-ops | Bootstrap edge stability and PC skeleton returned nothing whenever DVs only partly overlapped | No stability estimate for the reported DAG | Both now run on the exact matrix the DAG was fit on |
| Silent data loss | `psych_conflict_handling` stores Likert answers as text labels ("Rather agree") | The study was in the committed 11-study run but contributed **0** numeric DVs | Declarative `value_labels` in mapping YAMLs; all 432 rows of each item recoded, unmatched labels counted |
| Silent truncation of a scale | LimeSurvey exports negative AOA answers as `neg1`/`neg2`; numeric coercion turned them into missing values | eHMI-for-All and ROADS lost 30% of AOA answers (1,868/6,480 and 749/2,484 cells), all of them negative, so the AOA means were biased upward | `negN` codes recoded to −N in survey-like tables, counted in `-quality.json` |
| Wrong scale assumption | The derived AOA usefulness/satisfying formula hard-coded the 1–5 midpoint (3) | For studies coding items −2…+2 the composite was shifted and then mis-rescaled | Each study's coding midpoint is used, so 1–5 and −2…+2 give identical scores (regression test) |
| Anti-conservative intervals at small k | Most pooled DVs have k = 2–3 | Wald intervals far narrower than warranted | Hartung-Knapp-Sidik-Jonkman intervals reported next to Wald (validated against `metafor`, `test = "knha"`) |

### 2.2 LLM mapping is weak on real columns; plausibility checks are cheap (RQ2)

Pilot results (plausibility rules v1.0, before any tuning):

| Metric | E1 unchecked | E1 checked | E2 unchecked | E2 checked |
|---|---|---|---|---|
| Gold DV in the candidate shortlist | 42% (45/106) | — | 77% (230/300) | — |
| Accuracy on true DV columns | 18% | 16% | 55% | 55% |
| Precision of accepted DV mappings | 21% | 25% | 62% | 62% |
| Curated non-DV columns mapped to a DV | 41% (7/17) | 24% (4/17) | — | — |
| Blocklisted non-DVs mapped to a DV | 41% (18/44) | 5% (2/44) | — | — |
| True mappings lost to the check | — | 2/19 | — | 2/166 (1.2%); 1/166 (0.6%) with rules v1.1 |

With rules v1.1 every other number in the table is unchanged; the model's
answers are identical across runs (greedy decoding). The two E1 losses are
STS-AD items stored as text labels ("strongly agree"). The `value_labels`
recoding added afterwards makes them numeric, so the check no longer rejects
them in a recoded run.

Readings:

- **Retrieval, not generation, is the bottleneck.** For 58% of real columns
  the correct DV never reaches the model, because the fuzzy-string shortlist
  misses it. Ranking DVs by their *definitions* instead (§2.4) fixes most of
  this.
- **A small local model is not safe as an unsupervised mapper.** On real
  columns only about one in four of its accepted DV mappings is correct.
  LLM-deduced rows must be quarantined (the `*_llm_excluded` outputs) or
  human-reviewed.
- **The plausibility layer removes the most damaging errors** (ages, seeds,
  IDs, timestamps, free text) at about 1% cost on genuine DV names. Its two
  E2 false rejections led to one rule refinement (v1.1, documented in the
  YAML); `tia_id` remains a known trade-off of the identifier rule.

### 2.4 Definition-aware retrieval more than doubles LLM mapping quality (RQ2)

`scripts/dv_retrieval.py` ranks DVs by character n-gram TF-IDF similarity to
their definitions: the mean of a name similarity (id, label, aliases,
instruments) and a full-definition similarity (adding notes, cluster and
unit). It uses no model download, only scikit-learn.

Pilot on the same held-out items (10 studies, rules v1.1, Gemma-4-E2B):

| Metric | fuzzy shortlist | definition shortlist |
|---|---|---|
| E1 recall@8 (gold DV among candidates) | 42% | 83% |
| E1 accuracy on true DV columns (checked) | 16% | 39% |
| E1 precision of accepted DV mappings (checked) | 25% | 63% |
| E2 accuracy (checked) | 55% | 67% |
| E2 precision (checked) | 62% | 75% |
| E1 curated non-DVs mapped to a DV (checked) | 24% | 29% |

Definition retrieval is now the pipeline default (`--llm-shortlist fuzzy`
restores the old behaviour). Fusing the two rankings (reciprocal-rank fusion)
is worse than definition retrieval alone. Caveat: the 50/50 blend weight was
chosen on these items, so P1 must confirm it on a held-out split. The final
11-study numbers are in `analyses/output_llm_evaluation/` (definition) and
`analyses/output_llm_evaluation/shortlist_fuzzy/` (ablation).

### 2.3 Pooled estimates depend on harmonization choices (RQ3)

The multiverse crosses five binary choices (32 specifications). In a pilot on
10 studies:

- `understanding_rating` ranges from 3.76 to 7.56 depending only on whether
  scales are harmonized;
- `perceived_safety` ranges from 0.69 to 2.52;
- 95% interval widths differ by up to two orders of magnitude between Wald
  and HKSJ at k = 2.

Final numbers for the 11-study run are in `analyses/output_multiverse/`.

---

## 3. Proposed studies (ranked by value / effort)

### P1. OpenDV-Bench: a public benchmark for DV harmonization *(high value, low effort)*

Turn the E1/E2 harness into a released benchmark: gold (column → DV | none)
labels from curated mappings across all catalog datasets, plus a human-labeled
set of currently unmapped columns (two raters, Cohen's κ). Report
retrieval@k, accuracy, abstention and precision at fixed coverage for
string matching, embeddings, small and large LLMs, and LLM + plausibility.
**Deliverable:** benchmark + leaderboard; this alone is a strong
*Scientific Data* / *Behavior Research Methods* paper.

### P2. Retrieval-augmented, definition-aware mapping *(high value — lexical version done)*

The lexical definition retriever (§2.4) already lifts recall@8 on real columns
from 42% to 83% and precision from 25% to 63%. Next steps:

- **Dense retrieval.** `sentence-transformers` is already a dependency; test
  dense embeddings of the same definitions, alone and fused with TF-IDF.
- **Value-aware features.** Use the observed range, number of distinct levels,
  integer-ness, and correlation with already-mapped items of the same
  instrument.
- **Context.** Add repository context (README, paper abstract) to the query.

Hypothesis: recall@10 > 95% and precision > 80% at 80% coverage. Evaluate on
P1's held-out split.

### P3. Conformal abstention: mappings with a precision guarantee *(high value, medium effort)*

Treat LLM mapping as selective prediction. Calibrate a confidence score
(self-consistency over sampled answers, agreement between two models, or the
answer's token log-probability) on P1's calibration split, then use split
conformal prediction to choose the abstention threshold so that accepted
mappings reach a user-chosen precision (e.g. ≥ 90%). The pipeline would
then accept LLM mappings only with a stated error guarantee, which turns the
current warning ("LLM rows may be wrong") into a quantified contract.

### P4. Psychometrics-in-the-loop instrument recognition *(medium–high value, medium effort)*

Many DVs are items of standard instruments (NASA-TLX, SUS, TiA, UEQ-S,
STS-AD, SART). Detect instruments from *structure* rather than names: the
number of items, response scale, inter-item correlations (Cronbach's α,
already computed in `scripts/item_composition.py`) and one-factor fit
(`semopy` is installed with lingam). Map whole instruments at once,
including reverse-keyed items. This also enables
**measurement-invariance tests** across studies (configural / metric /
scalar), which ask whether "trust" means the same thing in five eHMI
studies before anyone pools it.

### P5. The harmonization multiverse as a reporting standard *(high value, low effort — prototype exists)*

Extend `analyses/multiverse_analysis.py` with:

- participant-level vs. observation-level variance (cluster-robust
  variance estimation, three-level models);
- linear vs. equipercentile / POMP rescaling;
- inclusion of composites vs. items.

Propose a "harmonization robustness statement" for papers that pool open
data: the share of specifications that support the headline claim. Pair it
with a simulation study quantifying the bias of linear Likert rescaling
(5 → 7 points) under realistic response distributions.

### P6. Studies as environments: invariant causal prediction *(medium value, medium effort)*

Pooled LiNGAM across heterogeneous studies is hard to defend: within-study
z-scoring removes level differences but not design differences. Treat
studies as *environments* and use invariant causal prediction (Peters,
Bühlmann & Meinshausen, 2016) or invariant risk minimization to keep only
DV–DV relations that are stable across studies. Report which relations
survive (e.g. perceived safety → trust), and compare with the bootstrap
stability that is now computed.

### P7. Human-in-the-loop curation study *(high HCI value, medium effort)*

A controlled study in which HCI researchers curate mappings for an unfamiliar
dataset in the Streamlit UI under three conditions:

1. column names only;
2. plus LLM suggestions;
3. plus LLM suggestions with plausibility evidence (observed range,
   rejection reason, instrument guess).

Measures: accuracy against gold, time, calibration of trust in the
suggestions, and over-acceptance of wrong suggestions (automation bias).
This is the CHI-shaped paper of the agenda.

### P8. Vocabulary saturation: how many datasets until the schema converges? *(medium value, low effort)*

Fit a Heaps'-law curve to the number of *new* aliases per added dataset
(from `meta_view.csv` / `unknown_alias_summary.json` as datasets are added in
catalog order). Saturation would mean a community schema is feasible for a
subfield; non-saturation quantifies the long tail and motivates P2/P3.

### P9. FAIR packaging of harmonization decisions *(medium value, low effort)*

Emit each run as an RO-Crate (or MLCommons Croissant) package. Record the
mapping decisions as W3C PROV:

- explicit / LLM / rejected mapping, with the rule that fired;
- schema version and model id.

Publish a per-dataset *harmonization card* listing DVs, scales,
transformations and quarantined mappings, so a reader can audit a pooled
estimate down to the source column.

### P10. Living, sequential evidence synthesis *(medium value, higher effort)*

Make the meta-analysis continuous: every new dataset (a pull request to the
catalog) re-runs standardization and updates pooled estimates with
cumulative meta-analysis and trial-sequential monitoring boundaries. This
shows when a question is answered and when more studies are futile.

---

## 4. Threats to validity (to address in the paper)

- **Domain concentration.** 10 of the 11 datasets come from automotive /
  eHMI research, several from the same lab. Extend the catalog to other HCI
  subfields before generalizing RQ1.
- **Gold-label circularity.** E1 gold labels come from mappings written by
  the tool's authors. P1's independent human labels address this.
- **Observation-level n.** Pooled estimates use row-level n, which is
  anti-conservative for repeated measures. P5 reports the participant-level
  alternative; the multiverse already includes it.
- **Model and hardware dependence.** The LLM results are for one small
  local model (Gemma-4-E2B; E4B does not fit a 16 GB GPU under the default
  memory check). P1 should compare model sizes and hosted models.
- **Rule tuning on the evaluation set.** The plausibility rules were refined
  once after the v1.0 evaluation. Report v1.0 and v1.1 (both kept in
  `analyses/output_llm_evaluation/`), and freeze the rules before P1's test
  split.

---

## 5. Reproduction

```bash
# Deterministic run (gold labels for the evaluation)
python -m scripts.run_batch_standardization --manifest sources_manifest_no4tu.yaml \
  --output-dir data/processed/det_run --cache-dir data/processed/cache --disable-llm-deduction

# LLM mapping benchmark (E1 + E2)
python analyses/evaluate_llm_mapping.py --manifest sources_manifest_no4tu.yaml \
  --batch-dir data/processed/det_run --cache-dir data/processed/cache

# Full LLM-assisted run, analysis, multiverse and figures
python -m scripts.run_batch_standardization --manifest sources_manifest_no4tu.yaml \
  --output-dir data/processed/catalog_meta_analysis --cache-dir data/processed/cache --snapshot-manifest
python analyses/multi_study_analysis.py --input-dir data/processed/catalog_meta_analysis/standardized \
  --output-dir analyses/output_python_full --exclude-llm-deduced
python analyses/multiverse_analysis.py --input-dir data/processed/catalog_meta_analysis/standardized
python analyses/make_chi_figures.py
```
