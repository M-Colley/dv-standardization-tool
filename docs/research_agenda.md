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

Pilot results (10 studies, fuzzy shortlist, plausibility rules v1.0, before
any tuning; superseded by the final table in §2.4):

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
- **A small local model is not safe as an unsupervised mapper.** With the
  fuzzy shortlist only about one in four accepted DV mappings is correct on
  real columns. Even with definition retrieval it is 59% (§2.4). LLM-deduced
  rows must be quarantined (the `*_llm_excluded` outputs) or human-reviewed.
- **The plausibility layer removes the most damaging errors** (ages, seeds,
  IDs, timestamps, free text) at about 1% cost on genuine DV names. Its two
  E2 false rejections led to one rule refinement (v1.1, documented in the
  YAML); `tia_id` remains a known trade-off of the identifier rule.

### 2.4 Definition-aware retrieval more than doubles LLM mapping quality (RQ2)

`scripts/dv_retrieval.py` ranks DVs by character n-gram TF-IDF similarity to
their definitions: the mean of a name similarity (id, label, aliases,
instruments) and a full-definition similarity (adding notes, cluster and
unit). It needs no model download, only scikit-learn.

Final evaluation (11 studies, rules v1.1, Gemma-4-E2B, check on; E1:
106 real DV columns, 69 non-DV columns; E2: 300 schema aliases):

| Metric | fuzzy shortlist | definition shortlist (default) |
|---|---|---|
| E1 recall@8 (gold DV among candidates) | 42% (44/106) | 83% (88/106) |
| E1 accuracy on true DV columns | 18% | 49% |
| E1 precision of accepted DV mappings | 23% (19/83) | 59% (52/88) |
| E1 curated non-DVs mapped to a DV, unchecked → checked | 30% → 17% | 74% → 35% |
| E1 blocklisted non-DVs mapped to a DV, unchecked → checked | 41% → 4% | 54% → 7% |
| E2 accuracy | 55% | 67% |
| E2 precision | 62% | 75% |
| True mappings lost to the check (E1; E2) | 0/19; 1/166 | 0/52; 0/200 |

Rank fusion of the two retrievers is worse than definition retrieval alone
(E1 recall@8 55%).

**Dense retrieval (P2, 2026-10-18).** Sentence embeddings of the same
definitions, same leave-alias-out items, CPU, no tuning
(`analyses/evaluate_shortlist_retrieval.py --dense-model ...`):

| E1 (106 real columns) | recall@1 | recall@3 | recall@8 [95% CI] |
|---|---|---|---|
| TF-IDF definitions | 56% | 59% | 83% [75, 89] |
| bge-small-en-v1.5 | 72% | 94% | 97% [92, 99] |
| multilingual-e5-small | 75% | 85% | 95% [89, 98] |
| TF-IDF + bge (rank fusion) | 58% | 86% | 97% [92, 99] |

On E2 (300 schema aliases) recall@8 rises from 84% to 88% (bge), 91%
(fusion) and 92% (e5). The dense model alone can miss exact lexical matches
("MentalDemand" ranks `sart_demand` first), so the pipeline option
`--llm-shortlist dense` uses the fusion. The TF-IDF blend weight was tuned on
the E1 items and the dense models were not, so the comparison if anything
favours TF-IDF.

**End to end, higher recall did not help (2026-10-19).** Same LLM
(Gemma-4-E2B), plausibility check on, same 101 real DV columns (E1; the
items changed slightly with the UEQ-S remapping) and 300 schema aliases (E2):

| | fuzzy | definition (default) | dense (fused) |
|---|---|---|---|
| E1 gold DV among candidates | 39 | 83 | 97 |
| E1 correct mappings | 17 | 52 | 41 |
| E1 precision of accepted DV mappings | 22% | 62% | 47% |
| E1 true mappings lost to the check | 0 | 0 | 3 |
| E2 correct mappings | 169 | 203 | 196 |

Paired on the same items, definition vs. dense differs by 25 vs. 14 E1
columns that only one of them gets right (exact McNemar p = 0.11; E2 34 vs.
27, p = 0.44). The mechanism is visible: on the 83 E1 columns whose gold DV
is in *both* shortlists the model picks it 52 times from the TF-IDF list but
33 times from the fused list, while it picks 11 of the 14 gold DVs that only
dense retrieval surfaces. Dense candidates are semantically closer to each
other, and a 2B model confuses near-synonyms. Recall@k is therefore not a
sufficient objective for the shortlist (cf. P3): candidate *diversity* and
ordering matter. TF-IDF stays the default; `--llm-shortlist dense` remains
available. Next: append only the dense-only candidates to the TF-IDF list,
or use dense retrieval to decide when to abstain.

**Pre-specified follow-up (written 2026-10-24, before running it).** The
*mixed* shortlist keeps the TF-IDF top 6 in order and gives the last 2 of 8
slots to the best dense candidates not among them
(`scripts.dv_retrieval.mix_rankings`, `N_DENSE_EXTRA = 2`).
Hypothesis: it keeps most of the recall gain while the model chooses as
well as from the TF-IDF list. Primary outcome: E1 correct mappings with
the check on, mixed vs. definition, paired exact McNemar test on the same
items (two-sided, α = 0.05). Secondary: E1 precision of accepted mappings,
true mappings lost to the check, E2 correct mappings. Same model, items,
seed and harness as the 2026-10-19 runs. A variant with 4 dense slots
(`mixed4`) is exploratory. The default changes only if mixed is not worse
than definition on E1 correct mappings and precision and loses no true
mapping.

**Result (2026-10-24).** The mixed list keeps nearly all of the recall
(gold DV in the top 8 for 96 of 101 real columns, vs. 83 for TF-IDF and 98
for dense), but the model maps fewer columns correctly:

| E1 (101 real DV columns), check on | definition | mixed (primary) | mixed4 (exploratory) | dense |
|---|---|---|---|---|
| Correct mappings | 52 | 46 | 36 | 41 |
| Paired exact McNemar vs. definition | | p = 0.38 | p = 0.020 | p = 0.11 |
| Precision of accepted mappings | 62% | 52% | 41% | 47% |
| True mappings lost to the check | 0 | 0 | 2 | 3 |
| E2 correct (of 300) | 203 | 206 | 210 | 196 |

The primary hypothesis is not supported, so TF-IDF stays the default. The
loss is not noise: of the 19 E1 columns that definition maps correctly and
mixed does not, 18 went to one of the two added dense candidates, almost
always the scale-level DV instead of the item (`SUS3`…`SUS10` mapped to
`usability`). Dense retrieval adds the parent construct, and the small model
prefers it over the item. The shortlist should therefore respect the
construct hierarchy (never offer a scale together with its own items, or
decide scale vs. item in a separate step) rather than chase recall. This
also ties the retrieval result to the construct/instrument/item conflation
in the schema.

A GPU-only rerun of the definition baseline (the 2026-10-19 run had part of
the model offloaded to the CPU) gave the same LLM answer on all 470 items:
generation is deterministic (greedy decoding) and the comparisons above are
on equal terms.

**Pre-specified follow-up (written 2026-10-25, before running it).** The
*hierarchy filter* (`scripts.dv_retrieval.apply_hierarchy`) walks a ranking
and skips a candidate when its scale or one of its own items is already
listed, so the higher-ranked of the two stays and lower candidates fill the
slot. Scale/item pairs come from the instrument registry, `subitem_of` and
the `<construct>_item_NN` convention (`construct_hierarchy`).
Hypothesis: with the filter, the dense fusion's recall gain turns into
correct mappings. Primary outcome: E1 correct mappings with the check on,
`dense_h` (fusion + filter, pipeline option `--llm-shortlist hierarchical`)
vs. definition, paired exact McNemar (two-sided, α = 0.05). Secondary: E1
precision, true mappings lost to the check, E2 correct mappings.
Exploratory: `definition_h` (filter on the TF-IDF list) and `mixed_h`. Same
model, items, seed and harness as before. The default becomes
`hierarchical` only if `dense_h` has more E1 correct mappings than
definition, precision at least as high, and loses no true mapping; the
p-value is reported either way.

**Result (2026-10-25).**

| E1 (101 real DV columns), check on | definition | dense_h (primary) | mixed_h (expl.) | definition_h (expl.) |
|---|---|---|---|---|
| Gold DV among candidates | 83 | 96 | 95 | 82 |
| Correct mappings | 52 | 61 | 65 | 55 |
| Paired exact McNemar vs. definition | | p = 0.14 (19 vs. 10 discordant) | p = 0.004 (16 vs. 3) | p = 0.38 |
| Precision of accepted mappings | 62% | 73% | 76% | 62% |
| True mappings lost to the check | 0 | 4 | 4 | 4 |
| E2 correct (of 300) | 203 | 201 | 205 | 195 (p = 0.021) |

The filter reverses the dense-retrieval result: with scale/item conflicts
removed, more recall now means more correct mappings (dense 41 → dense_h
61; mixed 46 → mixed_h 65), and precision rises to 73–76%.

The pre-specified decision rule is **not met**: dense_h has more correct
mappings (not significantly, p = 0.14) and higher precision, but "loses" 4
true mappings to the check. All filtered variants lose the same 4:
eHMI-for-All's `PerSafe01`–`04`, which the check screens out *before* the
LLM as non-numeric in every condition. Without the filter the LLM picked
the scale (`perceived_safety`, wrong) for them; with it, the item (right),
so the unchecked answer became correct while the checked result stayed
"unmapped". The criterion therefore counted columns the deployed pipeline
never maps under any shortlist; the rule was nonetheless pre-specified, so
the default stays `definition`. `mixed_h` is the best variant, but it is
exploratory and one of several comparisons on the same items (p = 0.004
survives a Bonferroni correction for the six comparisons run so far); it
needs a confirmatory test on new items (P1's held-out split). Applying the
filter to the TF-IDF list alone gains little on E1 and loses 8 schema
aliases on E2, where the scale is often the right answer.

**Data bug found on the way.** The screened columns are non-numeric because
eHMI-for-All exports negative answers as `1neg`/`2neg`/`3neg` (suffix), a
LimeSurvey variant that `recode_negative_answer_codes` (prefix `negN` only)
did not recognize. The four perceived-safety items therefore stayed text
columns and were left out of every analysis of that study (14–24% of their
answers carry a suffix code). Fixed on 2026-10-25: the items now enter with
all 720 answers, and each pooled perceived-safety item gains a study
(k 4 → 5, or 5 → 6 for item 1); other pooled estimates are unchanged.

**Post hoc, after the fix (2026-10-25).** Rerunning the same harness on the
corrected data (labelled post hoc: the pre-specified test above stands):

| E1 (101 real DV columns), check on | definition | dense_h | mixed_h |
|---|---|---|---|
| Correct mappings | 52 | 65 | 69 |
| Paired exact McNemar vs. definition | | p = 0.035 (23 vs. 10) | p < 0.001 (20 vs. 3) |
| Precision of accepted mappings | 59% | 75% | 77% |
| True mappings lost to the check | 0 | 0 | 0 |

With the `PerSafe` columns now reaching the LLM, the "lost" mappings
disappear: TF-IDF maps them to the scale (wrong, so its precision falls
from 62% to 59%) and the filtered lists to the item (right). On the
corrected data dense_h would meet the pre-specified rule. Because this was
not the pre-specified test, `definition` remains the default; a
confirmatory evaluation on new datasets (P1) should test `hierarchical`
(dense_h) as its primary hypothesis.

**Trade-off.** Because a plausible-looking candidate is always available, the
model maps non-DV columns more often with the definition shortlist (74% of
curated non-DVs, unchecked). The plausibility check halves that rate. In the
deployed pipeline, where the older LLM-exclusion list also applies, 3 of 23
curated non-DV columns end up mapped: Scrolling-in-the-Deep's binary context
variables. No true DV column is blocked.

**Caveat.** The 50/50 blend weight was chosen on the 10-study pilot items, so
P1 must confirm it on a held-out split.

### 2.3 Pooled estimates depend on harmonization choices (RQ3)

The multiverse crosses five binary choices (32 specifications). Results for
the 11-study run (`analyses/output_multiverse/`):

- **Robust:** NASA-TLX `mental_demand` (k = 7). All 32 estimates lie in
  7.59–8.02 and every interval covers the default estimate (run of
  2026-10-19; 7.37–7.58 in the run of 2026-10-17, before the LLM mapped
  `tlx1_sq001` of the longitudinal study to it).
- **Fragile:**
  - `perceived_safety` (k = 7) moves from −0.24 to 4.27 depending only on
    scale harmonization, and cannot be pooled at all without LLM-deduced rows;
  - `understanding_rating` ranges from 3.33 to 7.56;
  - `performance` ranges from 8.78 to 15.01.
- **Interval width:** 95% interval widths differ by up to two orders of
  magnitude between Wald and HKSJ at small k (e.g. `performance`: 0.8 to 103).

The multiverse turns "we pooled k studies" into "this estimate survives N of
32 defensible pipelines" — a reporting standard worth proposing (P5).

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

### P2. Retrieval-augmented, definition-aware mapping *(high value — lexical and dense versions done)*

The lexical definition retriever (§2.4) lifts recall@8 on real columns from
42% to 83% and precision from 23% to 59%; dense embeddings fused with it
reach 97% recall@8 (§2.4). Next steps:

- **Shortlist design beyond recall.** Dense retrieval reaches 97% recall@8
  but lowers end-to-end accuracy (52 → 41 of 101) because its candidates are
  near-synonyms (§2.4). Test diversity-aware shortlists (TF-IDF list plus
  dense-only additions) and larger models; German labels need the
  multilingual model.
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

The instrument exists: the Review Mappings tab and `opendv review`
(`scripts/mapping_review.py`) show exactly condition 3's evidence (observed
range vs. the suggested DV's range, rejection reason, retrieval candidates)
and log every decision with reviewer and timestamp, so agreement between
reviewers and over-acceptance can be computed from
`schemas/mapping_review_log.csv`. Conditions 1 and 2 need only hiding
columns of the queue.

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
- **Measurement non-invariance.** Pooling means assumes a scale measures
  the same construct on the same metric in every study (scalar invariance).
  Tested where item data exist (`analyses/measurement_invariance.py`,
  lavaan multi-group CFA with participant-clustered MLR; 2026-10-24):

  | Scale | Studies | Level reached | Why |
  |---|---|---|---|
  | SUS (10 items) | eHMI-for-All, ROADS | none | one factor misfits (robust RMSEA 0.12); items 4 and 10 load weakly, in line with SUS's usability/learnability split |
  | NASA-TLX (6 subscales) | ROADS, Running into Traffic | none | misfit (RMSEA 0.14); *performance* loads +0.25 in ROADS but −0.54 in Running into Traffic |
  | Van der Laan usefulness | eHMI-for-All, ROADS | metric | scalar fails (ΔCFI −0.014, ΔRMSEA +0.026): means are not comparable; item 9 barely loads (±0.14) |
  | Van der Laan satisfying | eHMI-for-All, ROADS | scalar | means comparable |

  Only Van der Laan *satisfying* supports pooling means; `mental_demand`
  (pooled over 7 studies) is untested in 5 of them and fails in the 2 with
  full TLX items. The negative *performance* loading suggests Running into
  Traffic's "success" item is keyed opposite to the TLX performance scale
  (high = success rather than failure), like the reverse-poled UEQ-S items
  of the same study; check the questionnaire and add it to `reverse_items`.
  `meta_analysis_summary.csv` now carries a `measurement_invariance` column.
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
