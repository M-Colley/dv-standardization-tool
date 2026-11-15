# Schema stability of the LLM mappings: design

Written on 2026-11-15, before any LLM answer under a changed schema was generated. Everything below (perturbations, draws, metrics, comparisons) is fixed by this file, `analyses/schema_stability/` and `analyses/output_llm_evaluation/schema_stability/design.json`, committed before the runs. The only thing run before this note was committed is a dry run that builds every prompt without asking the LLM, to count how many prompts each perturbation changes and so size the runs (see "Compute").

Code: `opendv/analysis/schema_stability.py` (a thin layer over `opendv.analysis.evaluate_llm_mapping`), tests in `tests/test_schema_stability.py`. Results go to `analyses/output_llm_evaluation/schema_stability/` and `docs/research_agenda.md` (section 2.6).

## Why

Schema 3.3 added one DV, `touch_offset`. In the rerun of 2026-11-08 that alone changed 19 LLM column mappings in five studies whose curated mappings did not change (changelog, 2026-11-08): Music tempo's maximum heart rates and two SenseCobot EEG statistics (`Cz_alpha Max`, `Pz_alpha Max`) became maximum lateral acceleration, FOV Snap-Turn's answers changed, and Curved Dragging's `avg_abs_dx` (degrees) became `touch_offset`. The model (`google/gemma-4-E2B-it`) decodes greedily, so the change came from the schema text.

The primary analyses exclude LLM-mapped columns, but the tool's value rests on how reliable the LLM mappings are. A mapping that changes when an unrelated DV is added is a threat to validity, and its size is unknown.

## How the schema reaches the LLM

Three parts of a column's prompt come from the DV schema; knowing which ones a change touches explains where an answer can move.

1. **Whether the LLM is asked, and how many candidates it gets:** the fuzzy alias shortlist over the resolved mapping (schema aliases plus the study's curated mapping) gives the top score (asked only if ≥ 65) and the number of candidates (at most 8).
2. **Which candidates, in which order:** `definition` ranks DVs by character n-gram TF-IDF similarity to their definitions (id, label, instruments, aliases, notes, cluster description, unit). Adding or removing a DV changes the vocabulary and the inverse document frequencies, so every score moves a little and close ranks can swap. `dense_h` fuses that ranking with sentence embeddings of the definitions and drops a scale when its own item ranks higher (and vice versa).
3. **What the prompt says about each candidate:** one line per candidate with its label, cluster, category, direction, up to three aliases, two instruments and the first 120 characters of its notes, in shortlist order.

The order of the DV list matters only through exact score ties (stable sorting), and the definitions of DVs outside the shortlist matter only through the TF-IDF statistics.

## Items (fixed)

The E1 items of the committed benchmark, unchanged: leave-alias-out, `--max-per-source 1000`, seed 20261017, from the two runs without the LLM of 2026-11-14 (`det23l`: 23 studies, 735 items; `det16l`: 16 studies, 815 items), with their source contexts (download cache of 2026-11-04), plausibility rules 1.3 and the model snapshots of `docs/reproducibility.md`.

- **Analysis set:** the 681 DV columns (gold label: their curated DV) and the 787 curated non-DV columns (gold: no DV), 1,468 items in 39 studies.
- The 82 blocklisted columns are asked as in the benchmark but not analysed; the pipeline never asks the LLM about them.
- E2 (schema alias recovery) is not run: it has no columns, and its items are the schema's own aliases.

## Perturbations

**Unrelated DVs.** A perturbation may remove or reword only the 62 of the 163 schema DVs that are neither the gold label of an item nor named anywhere in the curated mapping of one of the 39 studies (`design.json`, `unrelated_dvs`). So no perturbation removes or rewords a column's gold DV.

**Added DVs** come from a pool of 20 new DVs written in the schema's own format (`analyses/schema_stability/added_dv_pool.yaml`): established measures from HCI, driving and psychophysiology, split over the clusters in proportion to the schema (7 UX, 5 safety and trust, 3 cognitive load, 2 physiological, 1 each performance, interaction and affect). None is another instrument for a construct the schema has (no usability, trust, workload, sickness or presence scale), none is measured in any of the 39 studies (keyword check of every column and curated name), and no id or alias collides with one in the schema. An added DV is inserted after the last DV of its cluster, as a maintainer would add it. Every answer that moves to an added DV is therefore wrong.

**Rewordings** (`analyses/schema_stability/reworded_definitions.yaml`): one meaning-preserving rewording of the label of each unrelated DV, and of its notes where it has notes (17 of 62). Instrument names, abbreviations, citations, numbers and ranges are kept; ids, aliases, instruments and measurement fields never change. Examples: "DALI — Visual Demand" → "Visual Demand (DALI)", "Pupil Dilation" → "Dilation of the Pupil".

| Set | What changes | k | Draws | Purpose |
|---|---|---|---|---|
| `add_k1`, `add_k5` | k DVs of the pool added | 1, 5 | 5 each | the 3.3 event, and five times as much |
| `remove_k1`, `remove_k5` | k unrelated DVs removed | 1, 5 | 5 each | |
| `reword_k1`, `reword_k5` | label (and notes) of k unrelated DVs reworded | 1, 5 | 5 each | edits that keep the meaning |
| `schema_order` | the DV list permuted | – | 2 | negative control: retrieval depends on order only through exact ties |
| `candidate_order` | the shortlist listed in a random order in the prompt (same set) | – | 3 | the LLM's sensitivity to order alone |
| `identity` | the reference rerun without the response cache | – | 1 | noise floor; checks the cache (below) |
| `real_3_2` | `touch_offset` removed: schema 3.2 exactly (only its version and date lines differ) | 1 | 1 | the observed event, reversed |
| `sample_seed` | the reference with another sampling seed (self-consistency only) | – | 2 | the vote's own sampling noise |

The six `add`/`remove`/`reword` sets together (30 perturbations) are the **schema edits**, the primary endpoint's set.

**Draws.** Every set draws from its own generator, `numpy.random.default_rng([20261115, family, k, draw])`, without replacement within a draw; the schema permutations use `[20261115, 99, draw]`, the prompt shuffles a per-column generator seeded by the draw and the column name. Adding draws later never changes the existing ones. The drawn ids are listed in `design.json`; for example `add_k1` adds `respiration_rate`, `n_back_accuracy`, `willingness_to_pay`, `boredom`, `flow_experience`, and `remove_k1` removes `dwell_time`, `ueq_dependability`, `tam_perceived_ease_of_use`, `rsme`, `stress`.

**Validity.** A (column, perturbation) pair is left out of the flip rate if the perturbation removed the DV the column's reference answer named (a *forced* change: the answer must change, so it says nothing about stability; counted and reported) or the column's gold DV (only `real_3_2`, for the two `touch_offset` columns), or if a generation failed in either run.

## Conditions

| Condition | Model | Shortlist | Prompt order | Decoding |
|---|---|---|---|---|
| `e2b_definition` (reference) | gemma-4-E2B-it | `definition` (the pipeline's default) | rank | greedy |
| `e2b_dense_h` | gemma-4-E2B-it | `dense_h` | rank | greedy |
| `e4b_definition` | gemma-4-E4B-it | `definition` | rank | greedy |
| `e2b_definition_sc5` | gemma-4-E2B-it | `definition` | rank | 5 samples, majority vote |
| `e2b_definition_alpha` (exploratory) | gemma-4-E2B-it | `definition` | alphabetical | greedy |

- **Self-consistency:** 5 answers sampled in one call with the model's own sampling settings (temperature 1.0, top-k 64, top-p 0.95, its `generation_config.json`), the seed set before every call (20261115, plus 1 or 2 in `sample_seed`), so an identical prompt draws identical samples. The majority answer wins, "no candidate in the output" counts as a vote, and a tie goes to the candidate listed first (no answer last).
- **Alphabetical order** is exploratory, added because it removes order changes by construction: a perturbation can change such a prompt only by changing the set of candidates or their reference lines. It does not run `candidate_order`.
- **E4B:** 8.0 B parameters (16 GB in bf16) do not fit the 16 GB GPU, but 2.8 B of them are the per-layer embedding table, which is only looked up. The run loads the model on the CPU, moves everything else to the GPU (10.4 GB) and does the lookups on the CPU (`install_cpu_ple_pipeline`); a lookup is a gather and its scale a power of two, so the numbers are those of the whole model on the GPU. Checked on E2B, which fits either way: the logits of a benchmark prompt are bit-identical (largest difference 0.0) and four test answers are the same. E4B then needs 10.4 GB of the GPU and about 0.3–1 s per prompt. The pinned revision (`ee0ef60`) has only its tokenizer, chat template and config in the local cache; the weights come from the cached snapshot `fee6332` (`model.safetensors` SHA-256 `cfbd3d2f…3503`, the object id of an interrupted download in the same cache), whose config is byte-identical. Its tokenizer and chat template are the pinned revision's, which are byte-identical to the E2B snapshot's.
- **Order of runs:** reference, `dense_h`, self-consistency, alphabetical, E4B. The GPU is shared; a condition that cannot be run is reported as not run, and the comparisons that ran are the Holm family.

## Metrics

The **answer** of a column is its checked mapping, the DV the pipeline would accept after the plausibility check, or "no DV" (not asked, no candidate in the output, or rejected). The unchecked LLM answer is reported as a secondary outcome.

- **Flip rate of a column:** the share of its valid perturbations of a set whose answer differs from the condition's reference answer ("no DV" is an answer). **Flip rate of a set:** the mean over columns (columns with at least one valid perturbation).
- **Primary endpoint:** the flip rate over the 30 schema-edit perturbations, checked answers, reference condition.
- **Accuracy change** on DV columns: the mean over columns of (share of perturbations answering the gold DV) − (reference answer is the gold DV). Forced pairs count here, since they are what such a schema change would do; `real_3_2` leaves out the two `touch_offset` columns. For curated non-DV columns: the change in the share mapped to a DV.
- **Direction of flips:** among valid pairs whose reference answer is "no DV", the share answered with a DV (*no DV → DV*); among those with a DV, the shares answered with no DV and with another DV.
- **Mechanism:** the share of pairs whose prompt changed (SHA-256 of the prompt text), whose shortlist changed as a set, and the flip rate given a changed prompt and given a change of order only; flips to an added or reworded DV (*attraction*) and from one; flips with an unchanged prompt (zero if the cache assumption holds).
- **Uncertainty:** 95% percentile bootstrap intervals that resample studies (39 clusters; 10,000 resamples, seed 20261115), as the columns of a study share a context and a mapping file.

## Comparisons

Each condition against the reference, on the primary endpoint, paired by column (columns with a valid perturbation in both): the difference of mean flip rates with its cluster-bootstrap interval, and a two-sided p from a sign-flip permutation test on study totals (100,000 random sign vectors, seed 20261115; under the null the two conditions are exchangeable within a study). Holm over the comparisons that ran (at most four). Unchecked answers and the other sets are reported with intervals, without tests.

Accuracy of the reference answers is reported per condition (Wilson intervals), since a condition that is stable but less accurate is no improvement.

## Response cache

A perturbation changes few prompts, so answers are cached per (models, exact prompt text, decoding settings), and a perturbation generates only the prompts it changed. This is exact if a repeated prompt gets the same answer. The `identity` perturbation reruns the whole reference without the cache: any flip there is nondeterminism of the GPU, reported as the noise floor, and the flip rates of the other sets then omit that noise for the prompts they did not change.

## Compute

The draws were set from the cost, not from any answer. In the first draws of the dry run (`definition`, prompts only), adding one DV changed 107–186 of the 1,500 prompts the LLM is asked (7–12%) and adding five 222–387 (15–26%). At 0.5–1 s per prompt (E2B), a greedy E2B condition then needs about 14,000 generations, 3–4 GPU hours: the reference and its `identity` rerun 1,500 each, `candidate_order` about 4,400, the 30 schema edits about 6,600. Self-consistency and E4B take about twice as long. The complete dry-run counts of both shortlists are reported with the results.

## Deviations

Any departure from this note is listed with the results.
