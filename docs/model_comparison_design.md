# E4B against E2B on the full LLM mapping benchmark: design

Written on 2026-11-16, before any answer of `google/gemma-4-E4B-it` on these inputs, with the `dense_h` shortlist or on E2 was generated. Everything below (items, conditions, outcomes, tests) is fixed by this file. Deviations are listed with the results in `docs/research_agenda.md` (§2.4).

## Why

Every committed benchmark and pipeline run used `gemma-4-E2B-it` (2 B effective parameters), the model that fits the 16 GB GPU. The schema-stability test (`docs/schema_stability_design.md`, results in §2.6) also ran `gemma-4-E4B-it` on 1,468 E1 items with the `definition` shortlist. There, E4B answered 310 of 681 DV columns with their curated DV and E2B 176 (paired +19.7 points [9.4, 30.2]), and E4B's answers moved less under schema edits. That test was about stability; accuracy was a side result on one shortlist, without E2, precision or non-DV columns. This test measures the model's effect on the benchmark's own outcomes, with both shortlists the agenda reports (`definition`, the pipeline default, and `dense_h`, the pre-specified primary of 2026-10-25).

**What is already known.** The `definition` comparison is not an independent replication. 680 of its 724 DV columns and 788 of its 1,136 curated non-DV columns were items of the schema-stability test (the pre-audit runs `det23l`/`det16l`; one DV column there is a curated non-DV column now). E4B's accuracy on them is the result quoted above. Its answers on the non-DV columns are in that branch's committed outputs; this note was written without looking at them. New evidence comes from `dense_h`, from E2, from the 392 items that the LLM-mapping audit and the curated fixes after it added (44 DV and 348 curated non-DV columns in 21 studies), and from the outcomes the schema-stability test did not report.

## Items (fixed)

The full benchmark as committed on `main` (`analyses/output_llm_evaluation/`, `postfix/shortlist_dense_h/`, `added_2026_11_05/`): leave-alias-out, `--max-per-source 1000`, seed 20261017, plausibility rules 1.3, DV schema 3.3, source contexts from the download caches of 2026-11-04.

- **E1, first 23 studies:** the run without the LLM `det23b` (audit39b, 2026-11-15): 301 DV columns, 540 curated non-DV columns, 69 blocklisted columns.
- **E1, added 16 studies:** `det16b`: 423 DV, 596 curated non-DV, 13 blocklisted columns.
- **Analysis set E1:** both halves pooled, 39 studies: 724 DV columns (gold label: their curated DV) and 1,136 curated non-DV columns (gold: no DV). The 82 blocklisted columns are asked, as in the benchmark, but not analysed; the pipeline never asks the LLM about them.
- **E2:** the 300 held-out schema aliases (seed 20261017). E2 needs no data, so it runs once per model and shortlist, with the first half, as committed (`--e2-samples 0` for the added 16).

## Conditions

| | `definition` | `dense_h` |
|---|---|---|
| `gemma-4-E2B-it` (snapshot `3e22461`, whole model on the GPU) | E2B rerun | E2B rerun (new for the added 16) |
| `gemma-4-E4B-it` (weights `fee6332`, tokenizer and chat template `ee0ef60`, per-layer embeddings on the CPU) | new | new |

Everything else is the committed benchmark's: candidates in rank order, greedy decoding, the same prompt. The shortlists, and so the prompts, do not depend on the model.

- **E4B on a 16 GB GPU.** E4B has 8.0 B parameters (16 GB in bf16); 2.8 B of them are the per-layer embedding table, which is only looked up. `install_cpu_ple_pipeline` (from the schema-stability test) keeps that table on the CPU and the rest on the GPU (10.4 GB). On E2B the logits of a benchmark prompt were bit-identical to the normal load. The pinned E4B revision (`ee0ef60`, `docs/reproducibility.md`) has no weights in the local cache. The weights come from the cached snapshot `fee6332`, whose config is byte-identical. The tokenizer and chat template are the pinned revision's, byte-identical to E2B's. The same placement and files gave the schema-stability E4B runs.
- **E2B is rerun** under the same code as E4B and written to new folders, so the pair differs only in the model. The committed E2B outputs are not overwritten: they serve as a reproduction check (below).

## Outcomes

Each for the **checked** answer (the DV the pipeline would accept after the plausibility check; primary) and the **unchecked** LLM answer (secondary):

1. **Accuracy** (E1): the share of DV columns answered with their gold DV.
2. **Non-DV mapped** (E1): the share of curated non-DV columns answered with a DV (lower is better).
3. **Precision** (E1): among DV and curated non-DV columns answered with a DV, the share answered with their gold DV (the benchmark's `precision_of_accepted_dv_mappings`).
4. **E2 accuracy**: the share of the 300 held-out aliases answered with their DV (secondary).

**Hypothesis (H1):** E4B's checked E1 accuracy is higher than E2B's, with each shortlist. The tests are two-sided and the other outcomes have no directional hypothesis. The headline is checked accuracy with `definition`, the pipeline's default.

## Analysis

- **Pairing:** by item (E1: half, study, dataset, column; E2: alias). The four conditions must have the same items. Any difference in the item lists stops the analysis.
- **Estimates:** each model's rate with a Wilson 95% interval, and the difference E4B − E2B in percentage points. For accuracy the discordant items are counted: E4B right and E2B wrong, and the reverse.
- **Intervals:** 95% percentile bootstrap of the difference, resampling the 39 studies with replacement (10,000 resamples, seed 20261116); both models' answers of a drawn study enter together. For E2 the clusters are the aliases' DVs (123), since aliases of one DV share its definition.
- **Test:** a two-sided paired permutation test at the cluster level. In each of 100,000 permutations (seed 20261116) the two models' answers are swapped in each study with probability 1/2. The statistic is the difference of the outcome. p = (1 + number of |permuted| ≥ |observed|) / (1 + 100,000). For accuracy and non-DV mapped this is the sign-flip test on study totals of §2.6; for precision, a ratio, the same exchangeability argument holds.
- **Multiplicity:** Holm over the 12 E1 tests (3 outcomes × checked/unchecked × 2 shortlists), α = 0.05. E2's 4 tests (checked/unchecked × 2 shortlists) are a separate Holm family.
- **Reported without tests:** each half separately; the 392 items new since the schema-stability test separately; abstentions and wrong DVs on DV columns; how often the gold DV is in the shortlist (the same for both models) and accuracy given that it is; seconds per prompt.

## Checks

- **Generation errors:** none allowed. A run with errors is rerun when the GPU is free; it is not reported.
- **E2B reproduction:** the E2B rerun should give the committed E2B answers (`suggestion`, `final_checked`) for every item of `definition` (both halves) and `dense_h` (first 23). Differences are counted and reported. The comparison uses the rerun either way, since it shares E4B's code.
- **Determinism across runs:** an E1 item whose prompt (SHA-256) equals that of an item in the schema-stability reference runs (`base` of `e2b_definition`, `e2b_dense_h`, `e4b_definition`) should get the same answer from the same model. Differences are reported.

## Runs and outputs

One run at a time on the shared GPU: the four E2B runs first, then E4B with `definition`, then with `dense_h`. At 0.3–1 s per prompt and about 2,240 prompts per model and shortlist (1,942 E1, 300 E2), each E4B shortlist takes about 20–40 minutes. Every run goes to `analyses/output_llm_evaluation/model_comparison/<model>_<shortlist>/<half>/` (items and summary as `evaluate_llm_mapping` writes them). The analysis (`python -m opendv.analysis.model_comparison`, run by `analyses/model_comparison/run.sh`) writes `rates.csv`, `comparisons.csv`, `reproduction.csv` and `summary.json` to `analyses/output_llm_evaluation/model_comparison/`.

## What this does not decide

The pipeline's default model stays as it is. `select_local_model_candidates` already prefers E4B where its memory estimate fits (about 20 GB of GPU memory), and only an explicit option places the per-layer embeddings on the CPU. The result feeds the "Model and hardware dependence" threat (§4): it says how far the committed E2B numbers understate a larger model on the same benchmark.
