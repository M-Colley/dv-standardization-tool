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
   (RQ2), seeded by `opendv/analysis/evaluate_llm_mapping.py`.
4. A harmonization multiverse showing how much pooled estimates depend on
   analytic choices (RQ3), seeded by `opendv/analysis/multiverse_analysis.py`.
5. A taxonomy of silent failures with concrete instances from this
   pipeline (RQ4) and the guards that now catch them.
6. A harmonization ladder and a conclusion-level evaluation of LLM
   harmonization: which cross-study relations each step (identical names,
   shared schema, curated mappings, LLM) makes estimable, and which
   conclusions survive when the LLM stands in for curation (§2.5,
   `opendv/analysis/construct_relations.py`).

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
| Off-range scales read as 0–20 | Harmonization took the first listed alternative scale that held the data, and 0–20 came first | 1–5 Van der Laan answers (canonical −2…+2) became −1.7 with SD 0.1; 12 study means in six of the original studies were squeezed the same way (e.g. Running into Traffic's perceived safety −1.93 instead of +0.85 on −3…+3) | The narrowest common scale that holds every value is taken (regression test) |
| Reversed item keying | ROADS and eHMI-for-All store Van der Laan items with +2 at each item's first word; the registry's −2…+2 coding puts −2 there | Their usefulness and satisfying scores had the wrong sign (−0.5 and −0.9 instead of +0.5 and +0.9) and correlated −0.72 to −0.83 with SUS; found when a third study (+0.42) joined | `reverse_items` declared in both studies' mapping files; correlation with SUS is now a check worth running for every bipolar scale |
| Mixed units | One added study records crossing initiation times in ms, pooled with another study's seconds | A pooled estimate across units | `unit_conversions` in mapping files |
| Fit errors reported as non-invariance | lavaan refuses participant clusters when a study has one observation per participant; the error was reported as level "none" | Every scale shared with an added study showed "none", like a real misfit | Such scales are fitted without clusters (noted per row); an unfittable model is `not_fitted` |
| Anti-conservative intervals at small k | Most pooled DVs have k = 2–3 | Wald intervals far narrower than warranted | Hartung-Knapp-Sidik-Jonkman intervals reported next to Wald (validated against `metafor`, `test = "knha"`) |
| Columns combined by the sign of their correlation | When several columns mapped to one DV, the batch averaged them and first flipped any that correlated negatively with an anchor column | AV Acceptance VR's three curated perceived-safety items were flipped to match LLM-mapped anxiety items; 28 column groups in the 23 studies would have been flipped | Never flip on the evidence of the data; leave such groups apart; never combine LLM columns with deterministic ones (2026-11-04 review) |
| An optimizer's copy of an answer | eHMI Optimization logs each rating twice, the optimizer's mental demand inverted (11 − answer) | Participant means of the two cancelled out; pooled `mental_demand` (k = 13) shifted | Per-file `drop_columns` table option |
| A score on the wrong scale | The registry scored raw NASA-TLX on its items' 0–20 coding; `TLX_SCORE` is 0–100 | Pooled `TLX_SCORE` (LLM-excluded) 5.95 instead of 34.7 | Subscales put on 0–100 before averaging; declared `scale_conversions` for narrower recording scales (WoZ's 0–10 TLX) |
| A baseline as the study's value | FOV Snap-Turn's eight FMS time points mapped to one DV could not be averaged | The minute-0 rating (mean 0.18) stood for a 20-minute VR session | Session average mapped; uncomposed duplicates averaged |
| The benchmark showed the answer | The prompt lists three aliases per candidate DV, including the held-out one | 86 of 265 E1 and 216 of 300 E2 items had it in view (E2 accuracy 0.82 vs. 0.74 without it) | Aliases equal to the queried column are hidden (no effect in the pipeline) |
| LLM data in LLM-excluded outputs | Composite items were read as real scale items; scores computed from LLM-mapped items had no provenance | Pooled `perceived_safety_item_04` (LLM-excluded) rested on 5 studies instead of 3, two of them LLM composites with wrong item positions | Composite items not read as scale items; scores inherit their items' provenance |
| Fixed-effect meta-regression | The slope test ignored residual heterogeneity | p < .05 for 60% of null slopes (τ = 0.5, k = 6); 36 of 50 DVs "significant" | Random-effects meta-regression with the Knapp-Hartung test |
| Participant ids from the wrong column | Aggregation used the first ID-like column | eHMI for All lost every survey DV at the participant level | Declared participant columns for all studies (23 then, 39 now) |

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

`opendv/dv_retrieval.py` ranks DVs by character n-gram TF-IDF similarity to
their definitions: the mean of a name similarity (id, label, aliases,
instruments) and a full-definition similarity (adding notes, cluster and
unit). It needs no model download, only scikit-learn.

**Rerun after the 2026-11-04 review.** Until then the prompt listed each
candidate DV's first three schema aliases including the held-out one, for
86 of 265 E1 and 216 of 300 E2 items, and E1 skipped the unit conversions
the pipeline applies before the LLM step. The rerun on the 23 development
studies (definition shortlist, rules v1.3, the default sample of 40 items
per source and label kind) gives:

| Metric | before the fix | rerun |
|---|---|---|
| E1 accuracy on true DV columns | 45% (120/265) | 41% (105/255) |
| E1 precision of accepted DV mappings | 37% (120/323) | 31% (105/340) |
| E1 curated non-DVs mapped to a DV (checked) | 52% | 50% |
| E1 abstentions on DV columns (checked) | 21% | 10% |
| E2 accuracy | 68% (203/300) | 60% (181/300) |
| E2 precision | 74% | 65% |

Fewer abstentions and more wrong answers on DV columns: with milliseconds
converted to seconds, wrong suggestions for crossing times now pass the
range check, as they do in the pipeline. The tables below are the earlier
runs and overstate accuracy on columns whose name is a schema alias.

Final evaluation (11 studies, rules v1.2, Gemma-4-E2B, check on; E1:
106 real DV columns, 69 non-DV columns; E2: 300 schema aliases):

| Metric | fuzzy shortlist | definition shortlist (default) |
|---|---|---|
| E1 recall@8 (gold DV among candidates) | 42% (44/106) | 83% (88/106) |
| E1 accuracy on true DV columns | 18% | 49% |
| E1 precision of accepted DV mappings | 23% (19/83) | 60% (52/87) |
| E1 curated non-DVs mapped to a DV, unchecked → checked | 30% → 17% | 74% → 30% |
| E1 blocklisted non-DVs mapped to a DV, unchecked → checked | 41% → 4% | 54% → 7% |
| E2 accuracy | 55% | 67% |
| E2 precision | 62% | 75% |
| True mappings lost to the check (E1; E2) | 0/19; 1/166 | 0/52; 0/200 |

Rank fusion of the two retrievers is worse than definition retrieval alone
(E1 recall@8 55%).

**Dense retrieval (P2, 2026-10-18).** Sentence embeddings of the same
definitions, same leave-alias-out items, CPU, no tuning
(`opendv/analysis/evaluate_shortlist_retrieval.py --dense-model ...`):

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
(`opendv.dv_retrieval.mix_rankings`, `N_DENSE_EXTRA = 2`).
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
*hierarchy filter* (`opendv.dv_retrieval.apply_hierarchy`) walks a ranking
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
| Precision of accepted mappings | 60% | 76% | 78% |
| True mappings lost to the check | 0 | 0 | 0 |

With the `PerSafe` columns now reaching the LLM, the "lost" mappings
disappear: TF-IDF maps them to the scale (wrong, so its precision falls
from 62% to 59%) and the filtered lists to the item (right). On the
corrected data dense_h would meet the pre-specified rule. Because this was
not the pre-specified test, `definition` remains the default; a
confirmatory evaluation on new datasets (P1) should test `hierarchical`
(dense_h) as its primary hypothesis.

**Confirmatory test (2026-10-26, pre-registered in
`docs/preregistration_confirmatory_shortlist.md`).** On 168 DV columns of
seven new, non-automotive studies with frozen gold labels, `dense_h` mapped
53 correctly and `definition` 55 (8 vs. 10 discordant, p = 0.81); `mixed_h`
56 (p = 1.0). The post hoc advantage does not replicate, and `definition`
stays the default. All shortlists do much worse on new studies than on the
development set: 33% of DV columns correct (51% before), 35% precision (59%),
gold among the candidates for 68% (82%). The losses come from outside the
shortlist: text answer codes screened out before the LLM, long
question-text headers, and wrong choices although the right DV was offered
(44 of 78 errors). Details: `docs/confirmatory_shortlist_results.md`.

**Post hoc, after fixing two of those gaps (2026-11-03).** This is not part of the
pre-registered test, and header matching was designed on these studies'
headers. Answer codes and (since 2026-11-04) `True`/`False` columns are now
recoded before the screen, so 29 of the 31 screened DV columns reach the
LLM. `definition` then maps 64 columns correctly (8 more Hoffman trust items
and the VR study's `Success`, none lost, p = 0.004), while `dense_h` and
`mixed_h` reach 54 and 57, so the shortlist conclusion holds. Header
matching against instrument item wording assigns 45 question-text columns
(Hi-Audio, VR; English and German), all correct, with no non-DV column
matched. Header matching first and the LLM for the rest gives 106 correct
with `definition` (precision 60% [53, 67], up from 36%). New studies are
needed to confirm both gains. Still open: the LLM's wrong choices among
correct candidates, and the name rule that screens the VR study's `Time`.

**Reproducibility of LLM results (2026-11-04).** Two ways for a rerun to
change without any input changing were found and closed. (1) A failed
generation call (e.g. CUDA out of memory on a shared GPU) was logged at
DEBUG and counted as an abstention; in one development batch it silently
removed all 32 FACT-AV LLM mappings. Failures are now recorded, retried once
after an out-of-memory error, reported in `run_summary.json`, and make the
benchmark harness exit with an error. (2) The prompt's manuscript context was
fetched live from the DOI landing page, which served a different PDF excerpt
for Hi-Audio in one run and changed 41 `mixed_h` answers. Fetched text is now
cached per source and shared by the batch and every shortlist, and each
source's context fingerprint is recorded. Neither problem affected the
committed results: no committed run shows the fast unanswered calls a
failure leaves, and the three pre-registered shortlist runs logged the same
PDF context. A full development rerun with both fixes reproduced every
benchmark item of the three shortlists.

**Twelve added studies (2026-11-04, not pre-registered).** Twelve open-data
studies joined the development set (`docs/open_data_candidates.md`). Their
curated mappings were committed before any LLM run on them, and none of the
current methods was tuned on them, so their first benchmark run is reported
apart from the original 11 (`evaluate_llm_mapping.py --subset`):

| E1, check on | definition | dense_h | mixed_h |
|---|---|---|---|
| Correct, of 164 DV columns | 68 (41%) | 70 (43%) | 70 (43%) |
| Paired exact McNemar vs. definition | | p = 0.85 (15 vs. 13) | p = 0.81 (10 vs. 8) |
| Gold among the candidates | 77% | 78% | 73% |
| Precision of accepted mappings | 29% | 30% | 29% |
| Curated non-DVs mapped to a DV | 54% | 54% | 56% |
| Original 11 studies: correct, of 101 | 52 | 65 | 69 |

Rerun after the 2026-11-04 review (held-out alias no longer shown, unit and
scale conversions applied, rules 1.3; the review fixes changed the DV
columns of four studies, so there are 155 and 100):

| E1, check on | definition | dense_h | mixed_h |
|---|---|---|---|
| 12 added studies: correct, of 155 | 57 (37%) | 63 (41%) | 62 (40%) |
| Paired exact McNemar vs. definition | | p = 0.34 (17 vs. 11) | p = 0.27 (9 vs. 4) |
| Precision of accepted mappings | 23% | 25% | 24% |
| Original 11 studies: correct, of 100 | 48 | 63 (p = 0.011) | 60 (p = 0.023) |
| All 23: correct, of 255 | 105 (41%) | 126 (49%, p = 0.009) | 122 (48%, p = 0.008) |
| All 23: precision | 31% | 37% | 36% |
| E2 accuracy, of 300 | 60% | 56% | 62% |

The fix cost `definition` more than the filtered dense shortlists, which now
lead over all 23 studies. The lead comes from the original 11 studies, on
which they were designed; on the 12 added studies, as in the pre-registered
confirmatory test, it is not significant. `definition` therefore stays the
default.

In the first run the original 11 studies' items were identical to the
previous run, and, as in the confirmatory test, the filtered dense shortlists
did not beat `definition` on studies they were not tuned on. Three patterns
accounted for 80 of `definition`'s 96 errors:

- **Item numbers misread (31).** Zero-based SUS items (`sus0`–`sus9`) are
  mapped one item too low, and the numbered NASA-TLX columns `TLX 2`–`TLX 6`
  all become mental demand.
- **An abbreviation the shortlist misses (27).** `CIT_*` (crossing initiation
  time) does not bring `crossing_initiation_time` into the candidates. The LLM
  picks trust items instead, which the range check rejects (the values are
  milliseconds).
- **A scale for its parts (22).** SART items become the SART total,
  perceived-safety items the scale, NASA-TLX subscales the TLX score.

Precision falls mainly because columns of questionnaires the schema does not
cover (ATI, MSSQ, privacy concern), demographics and condition aggregates are
mapped to some DV. Any fix for these patterns would be designed on these
studies, so confirming it needs another held-out set.

**Trade-off.** Because a plausible-looking candidate is always available, the
model maps non-DV columns more often with the definition shortlist (74% of
curated non-DVs, unchecked). The plausibility check more than halves that
rate (to 30%). In the deployed pipeline, where the older LLM-exclusion list
also applies, 2 of 23 curated non-DV columns end up mapped:
Scrolling-in-the-Deep's context variables `Current_Activity` and
`Multitasking`. A third, the 0/1 indicator `At_Home`, has been rejected
since rules v1.2, because a 0/1 column is never accepted for a rating-scale
DV. No true DV column is blocked.

**Caveat.** The 50/50 blend weight was chosen on the 10-study pilot items, so
P1 must confirm it on a held-out split.

### 2.3 Pooled estimates depend on harmonization choices (RQ3)

The multiverse crosses five binary choices (32 specifications). Results for
the 39 development studies (2026-11-06, `analyses/output_multiverse/`; the
23-study run after the 2026-11-04 review fixes in parentheses):

- **Scale harmonization now decides the NASA-TLX subscales.** With 23
  studies `mental_demand` was robust: all 32 estimates lay in 7.35–8.13 and
  every interval covered the default. With 39 studies (k = 14–21) the
  estimates range from 7.44 to 21.12 and 47% of the intervals miss the
  default. The harmonized specifications stay together (7.44–8.41); the raw
  ones (13.0–21.1) pool six added studies' 0–100 ratings with 0–20 ones.
  Effort, frustration, temporal and physical demand and performance behave
  the same (harmonized ranges within 1, raw ones 13–23). A choice that did
  not matter for 23 studies decides the estimate once studies with another
  response format join.
- **Robust:** the NASA-TLX total `TLX_SCORE` (k = 11–14), 32.97–38.69 on its
  0–100 scale (27.4–34.7), every interval covering the default; SUS
  `usability` (k = 5–6), 74.2–75.5.
- **Fragile:**
  - `perceived_safety` (k = 5–12) moves from 0.85 to 3.70 (0.93 to 4.04),
    entirely with scale harmonization (raw 2.98–3.70, harmonized 0.85–1.13);
    25% of the intervals miss the default (28%);
  - `understanding_rating` ranges from 3.51 to 7.64 (3.71–7.56), again by
    scale harmonization alone.
- **Fragility from LLM rows:** with LLM-deduced columns included,
  `task_completion_time` spans 1.2 to 900 (to 1,331), because the LLM maps
  columns in milliseconds (AV Prevalence's decision times, myocontrol's test
  trials), which the curated mappings leave out. Without LLM rows it now
  spans 1.24 to 52.66 (0.70–0.90) by the τ² estimator alone: an NL2TAP
  scenario takes 209 s, a trial of the other three studies (VR dragging,
  touch selection, memory reaction time) 0.6–2 s, and REML weights the four
  studies nearly equally. Times of different tasks are
  not one quantity. `pedestrian_crossing_time` is still pooled only with
  LLM-mapped columns (6.2–13.9 over 7–8 studies). The plausibility rules
  check names, types and ranges, not whether a DV fits the study's domain:
  the LLM reads SenseCobot's EEG beta/alpha ratios as mental demand, and
  that study then carries 17.5% of the weight in the all-mappings pool.
- **Fragility that was a data error:** `performance` ranged from 8.73 to
  15.01, the widest range of any DV, until Running into Traffic's
  reverse-keyed "success" item was fixed (2026-10-26); with 23 studies it
  lay in 6.31–7.30 (k = 5–7), with 39 in 7.17–8.11 when harmonized
  (k = 12–15). Van der Laan usefulness was negative in every specification
  (−0.63) until the item keying of ROADS and eHMI-for-All was fixed; it now
  lies in 0.73–1.57 (0.81–1.85). A wide or implausible multiverse range can
  flag a polarity error, not only a consequential harmonization choice.
- **Interval width:** 95% interval widths differ by up to an order of
  magnitude between Wald and HKSJ at small k (e.g. `perceived_safety`: 1.4
  to 8.9; `mental_demand` 2.7 to 22.5). More studies narrow the widest:
  `TLX_SCORE`'s widths, 12 to 143 over 3–5 studies, are 16 to 36 over 11–14.

The multiverse turns "we pooled k studies" into "this estimate survives N of
32 defensible pipelines" — a reporting standard worth proposing (P5).

### 2.5 Harmonization reveals relations no single study shows (RQ1, RQ3)

Means need scalar invariance, which no shared scale reaches (§4);
correlations do not change under rescaling. `opendv/analysis/construct_relations.py`
therefore pools within-study correlations between harmonized constructs
(Fisher z, REML, HKSJ, prediction intervals; between persons on participant
means and within persons as repeated-measures correlations) and traces every
column to the step that mapped it. Details and figures:
[`docs/cross_study_insights.md`](cross_study_insights.md). Exploratory; the
finding rule was fixed after a first look at the first 23 studies. Numbers
for the 39 development studies (2026-11-05); the 23-study values of
2026-11-04 in parentheses.

- **Harmonization ladder.** Pairs of constructs measured together in at
  least three studies (between persons): 14 (0) when studies are pooled by
  identical column names, 21 (0) with the shared schema and header matching,
  35 (29) with the curated study mappings, 83 (54) with the LLM fallback;
  within persons 12 / 21 / 24 / 52 (0 / 0 / 23 / 41). All pairs the two
  cheap steps reach are NASA-TLX subscale pairs, which four added studies
  name by the subscales' own words. Pairs of *different* constructs in three
  or more studies need the curated mappings: 0 / 0 / 12 / 60 between and
  0 / 0 / 2 / 30 within persons.
- **Workload and usability (new).** Between persons, participants who rated
  the task more mentally demanding rated the system less usable (SUS) in all
  four studies that measured both: r = −0.42, HKSJ [−0.51, −0.31], BH
  q = .004, I² = 0%, prediction interval [−0.66, −0.09]. Frustration,
  effort, temporal demand and the TLX total do the same in three studies.
  With 23 studies only two measured both.
- **Workload and perceived safety (unchanged).** Within persons, occasions
  with higher mental demand are rated less safe in all four studies that
  measured both repeatedly (pooled r = −0.44, HKSJ [−0.72, −0.05], BH
  q = .04; I² = 94%); no added study measured both. Between persons the
  relation is uncertain (r = −0.27 [−0.62, 0.16], five studies): a
  state-level coupling, not a trait-level one.
- **NASA-TLX structure.** The subscales covary within persons in all eight
  studies that measured them repeatedly (four before), and now between
  persons over 10–12 studies (12 of 15 pairs; none with 23 studies).
  *Performance* is the weakest link (between persons r = 0.13–0.24),
  consistent with the TLX's one-factor misfit (§4).
- **The LLM's contribution.** It reaches 35 studies (29 without it). Its
  extra cross-construct findings need review: perceived safety × TiA trust
  (four studies, r = 0.63) rests on correct mappings in two studies, a
  partly wrong one in a third and a wrong one (relaxation as trust) in the
  fourth, yet looks homogeneous; trust × UEQ-S hedonic quality (new) rests
  partly on an immersion score read as trust (ARTV ornaments), and the LLM
  also reads minimum heart rate as trust (music tempo). On the 16 added
  studies it gives the curated DV for 16% of the curated columns (41% on
  the first 23).
- **Can the LLM replace curation? No.** With the LLM's leave-alias-out
  answers in place of the curated mappings, 67 of 311 curated DV columns
  keep their DV. Within persons the replay recovers 13 of the 16 curated
  findings, but the shared schema alone recovers 14, because the added
  TLX studies name their subscales plainly; between persons it recovers
  one of 19, and none of the usability–workload relations. Where both pool
  a pair the correlations agree closely (median |Δr| 0.06–0.07), so the
  cost is coverage and misassignment, not distorted estimates. This is a
  conclusion-level evaluation of LLM harmonization; column accuracy alone
  does not show which findings are lost. With 23 studies the replay
  recovered none of the 16 within-person findings (one with `dense_h`).

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
already computed in `opendv/item_composition.py`) and one-factor fit
(`semopy` is installed with lingam). Map whole instruments at once,
including reverse-keyed items. This also enables
**measurement-invariance tests** across studies (configural / metric /
scalar), which ask whether "trust" means the same thing in five eHMI
studies before anyone pools it.

### P5. The harmonization multiverse as a reporting standard *(high value, low effort — prototype exists)*

Extend `opendv/analysis/multiverse_analysis.py` with:

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
(`opendv/mapping_review.py`) show exactly condition 3's evidence (observed
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

- **Domain concentration.** 10 of the original 11 datasets come from
  automotive / eHMI research, several from the same lab. The 12 studies
  added on 2026-11-04 bring 7 from other HCI fields (authentication, VR
  interaction, touch input, games, adaptive interfaces), and the 16 added
  on 2026-11-05 another 8 (software tools, human-robot interaction, audio
  interfaces, augmented reality, media, human factors), so 23 of the 39
  are automotive (15 of 23 before). Extend the catalog further before
  generalizing RQ1.
- **Gold-label circularity.** E1 gold labels come from mappings written by
  the tool's authors. P1's independent human labels address this.
- **Measurement non-invariance.** Pooling means assumes a scale measures
  the same construct on the same metric in every study (scalar invariance).
  Tested where item data exist (`opendv/analysis/measurement_invariance.py`,
  lavaan multi-group CFA with robust MLR; 39 studies, 2026-11-06; the
  23-study result of 2026-11-04 in the last column):

  | Scale | Studies | Level reached | Why | 23 studies |
  |---|---|---|---|---|
  | SUS (10 items) | eHMI-for-All, ROADS, FIDO2, PairSonic, NL2TAP | none | one factor misfits (robust RMSEA 0.13, CFI 0.89); with four studies items 4 and 10 loaded weakly everywhere, in line with SUS's usability/learnability split, and FIDO2's loadings were low throughout (ceiling: SUS 90) | none (RMSEA 0.13) |
  | NASA-TLX (6 subscales) | ten studies (five added) | not fitted | NL2TAP's physical demand is 0 for every participant (a desk task), and lavaan refuses a group whose item has no variance | none (RMSEA 0.17); *performance* loaded −0.04 to +0.59 |
  | Van der Laan usefulness | eHMI-for-All, ROADS, FIDO2, AR pedestrian online | none | RMSEA 0.082 against the 0.08 limit (CFI 0.99); the AR survey adds 1,174 answers | configural; item 9 barely loaded (−0.14 to +0.26) |
  | Van der Laan satisfying | eHMI-for-All, ROADS, FIDO2, AR pedestrian online | none | misfit (RMSEA 0.33, CFI 0.93) | metric (scalar failed: ΔCFI −0.022) |
  | UEQ-S pragmatic | Running into Traffic, three automotive studies of 2026-11-04 | none | misfit (RMSEA 0.13) | unchanged |
  | UEQ-S hedonic | the same four | none | misfit (RMSEA 0.28); item 7 (inventive) loads 0.44–0.58 in three of the four studies | unchanged |
  | Perceived safety (4 items, scored since 2026-11-04) | BVI auditory HAV, eHMI for All | none | misfit (RMSEA 0.21, CFI 0.97), participant-clustered | unchanged |

  With the original two studies, Van der Laan *satisfying* reached scalar
  and *usefulness* metric invariance; adding FIDO2 removed both, and the AR
  survey removes the rest. No shared scale reaches even configural
  invariance, so none supports pooling means. Whether NASA-TLX would fit
  without NL2TAP's physical-demand item (or without NL2TAP) is untested;
  the module fits all studies that share a scale or none. All 39 studies
  declare their participants in `schemas/study_designs.yaml`; a scale is
  still fitted without the cluster correction when one of its studies has
  one observation per participant (FIDO2, PairSonic, NL2TAP, WoZ
  Adaptation, music tempo, AV Prevalence, eHMI State/Intent), because
  lavaan refuses singleton clusters.
  Correlations need less than means (metric invariance at most), which is
  why §2.5 pools relations instead.
  The negative *performance* loading of the 2-study run revealed that Running
  into Traffic's "success" item is keyed opposite to the TLX performance
  scale (high = success rather than failure; mean 17.7 of 21, negative
  correlations with the other five subscales), like the reverse-poled UEQ-S
  items of the same study. It is reverse-scored since 2026-10-26
  (`schemas/running_into_traffic_mapping.yaml`).
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
python -m opendv.run_batch_standardization --manifest sources_manifest_no4tu.yaml \
  --output-dir data/processed/det_run --cache-dir data/processed/cache --disable-llm-deduction

# LLM mapping benchmark (E1 + E2); --max-per-source 1000 asks about every
# curated column, which the LLM replay of construct_relations needs
python -m opendv.analysis.evaluate_llm_mapping --manifest sources_manifest_no4tu.yaml \
  --batch-dir data/processed/det_run --cache-dir data/processed/cache --max-per-source 1000 \
  --subset added_2026_11_04=analyses/development_sources/sources.yaml

# Full LLM-assisted run, analysis, multiverse and figures
python -m opendv.run_batch_standardization --manifest sources_manifest_no4tu.yaml \
  --output-dir data/processed/catalog_meta_analysis --cache-dir data/processed/cache --snapshot-manifest
python -m opendv.analysis.multi_study_analysis --input-dir data/processed/catalog_meta_analysis/standardized \
  --output-dir analyses/output_python_full --exclude-llm-deduced
python -m opendv.analysis.multiverse_analysis --input-dir data/processed/catalog_meta_analysis/standardized
python -m opendv.analysis.make_chi_figures
python -m opendv.analysis.construct_relations --input-dir data/processed/catalog_meta_analysis/standardized \
  --output-dir analyses/output_relations \
  --llm-replay-items analyses/output_llm_evaluation/llm_mapping_evaluation_items.csv
```
