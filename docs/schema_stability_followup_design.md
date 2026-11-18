# Schema stability follow-up: no zero-score padding and order-robust answering (design)

Written on 2026-11-16, before any answer of the conditions below was generated. It builds on the schema-stability experiment ([`schema_stability_design.md`](schema_stability_design.md), results in `docs/research_agenda.md` §2.6). Items, perturbations, draws, the flip-rate definition, the bootstrap and the sign-flip test are taken from there unchanged; this note fixes only what is new. Code: `opendv/analysis/schema_stability.py` (new conditions and the `freeze-mappings` and `followup` subcommands), the options in `opendv/dv_retrieval.py`, `opendv/llm_utils.py` and `opendv/analysis/evaluate_llm_mapping.py`. Outputs: `analyses/output_llm_evaluation/schema_stability_followup/`; the committed `schema_stability/` outputs are read, never written.

## Why

The schema-stability experiment's exploratory analyses (added after its results were seen) suggested two changes to how the LLM is asked. Neither was pre-specified there.

1. **Zero-score padding.** The `definition` shortlist takes as many candidates as the fuzzy alias shortlist (up to 8), even when fewer DVs share any character n-gram with the column. 39 of the 1,433 asked reference shortlists are padded with DVs whose definition score is 0; their order is then the order of the DV list (stable sort). Permuting the DV list flipped 20.5% of these columns and 0.2% of those without a tie; schema edits flipped 3.5% and 2.9%. A zero-score DV is a candidate only because the list has room for it.
2. **Candidate order.** With the shortlist listed in a random order the model answers with the first-listed candidate in 26% of prompts (about 11% if position did not matter), and a random order changes 33.5% of the answers, ten times the effect of a schema edit (2.8%). Three quarters of the flips to a DV under schema edits (839 of 1,119) go to a DV already on the column's reference shortlist, so flips look like re-choices among the same candidates, which is what an order change produces. An answer that holds under several orders should be less sensitive to the small prompt changes a schema edit makes, and the orders' disagreement is a natural reason to abstain.

These hypotheses come from the same 1,468 items they are tested on. The answers of the new procedures have not been seen, but the columns are not new data, so the results show whether the changes do what the exploratory analyses suggest on these items, not that they generalize to new studies.

## Conditions

All use `google/gemma-4-E2B-it` (the snapshot of `docs/reproducibility.md`), greedy decoding, the prompt template of the reference, and the plausibility rules 1.3 on the final answer.

| Condition | Shortlist | Prompts per column | Answer |
|---|---|---|---|
| `e2b_definition` (reference) | `definition` | rank order | the answer (committed answers, not rerun) |
| `e2b_definition_nopad` | `definition` without zero-score DVs | rank order | the answer |
| `e2b_definition_vote3` | `definition` | rank order and two seeded shuffles | majority of the three |
| `e2b_definition_vote3u` | `definition` | the same three | the common answer if all three agree, else no DV |

- **No zero-score padding (`nopad`).** A DV leaves the ranking if its definition score (the TF-IDF blend of name and full-definition similarity, leave-alias-out as in the benchmark) is exactly 0, i.e. it shares no character 3–5-gram with the column name. The shortlist is then the first *n* of the remaining ranking (*n* is the fuzzy shortlist's length, as before), so it can be shorter than *n*; if nothing remains, the column is not asked (no DV). The gate (fuzzy top score ≥ 65) is unchanged. On the unchanged schema this changes exactly the 39 padded shortlists of the analysed columns (9 of them have no candidate left and are not asked).
- **Three orders (`vote3`, `vote3u`).** The column is asked three times: with the shortlist in rank order (the reference's prompt) and listed in two random orders, generated as the reference's `candidate_order` perturbation does (a per-column generator seeded by SHA-256 of the seed and the column name) with seeds 1 and 2. On the unchanged schema all three prompts are therefore ones the reference already asked (its base run and `candidate_order` draws 1 and 2); those answers were generated before this note but were never aggregated or examined as a vote. Each prompt is answered greedily and its answer extracted as before ("no candidate in the output" is an answer, *no answer*).
    - **Majority (`vote3`):** the answer given under at least two of the three orders; if all three differ, the one ranked highest by retrieval among them, *no answer* last (`llm_utils.majority_vote`, the rule of the self-consistency condition).
    - **Unanimous (`vote3u`):** the answer if all three orders give the same one (possibly *no answer*); otherwise no DV (abstention).
    - With one candidate the three prompts are identical. The vote's final answer goes through the plausibility check like any answer.

## Perturbations, items and inputs

- **Perturbations:** every one in the committed `analyses/output_llm_evaluation/schema_stability/design.json` (the 30 schema edits, the DV list permuted, schema 3.2, the identity rerun, the shortlist in random order, the sampling-seed control), with their draws. Which sets each condition runs:
    - `nopad` runs every set the reference ran: identity, `candidate_order` draws 1–3, `schema_order`, the 30 edits, `real_3_2`.
    - `vote3` and `vote3u` run the 30 edits, `schema_order` and `real_3_2`. `candidate_order` does not apply (the vote already varies the order). `identity` is not run for them: each of their prompts is a greedy prompt like the reference's, and `nopad`'s identity rerun checks determinism on the day. The `sample_seed` draws 1 and 2 run as **order seeds**: the two shuffles use seeds 3 and 4 (draw 1) or 5 and 6 (draw 2) instead of 1 and 2, measuring how much the vote depends on which orders it uses (as `sample_seed` did for self-consistency).
- **Items:** the same 1,468 analysed E1 items (681 DV columns, 787 curated non-DV columns, 39 studies; the 82 blocklisted columns are asked but not analysed), from `det23l` and `det16l`, the same manifests and download caches.
- **Mapping files.** The benchmark reads each study's curated mapping file (`schemas/*_mapping.yaml`) for the fuzzy gate, the number of candidates and the plausibility check. The reference ran with the files of commit `ffabda7` (unchanged up to `e2df354`); since then the LLM-mapping audit of 2026-11-15 changed 24 of them on `main`. The follow-up reads copies of the `ffabda7` files (`python -m opendv.analysis.schema_stability freeze-mappings`, which writes manifests pointing at them), so every prompt that a condition does not change by design is the reference's prompt.
- **Integrity check** (before any analysis; a failure is reported and investigated, not analysed around): wherever `nopad`'s shortlist equals the reference's, its prompt hash and its checked answer must equal the committed reference's, for every item and perturbation both ran; for `vote3`, the rank-order answer must equal the reference's answer for every item and perturbation both ran.

## Response cache

Answers are cached per (models, exact prompt text, decoding settings) as before. The follow-up starts from a read-only copy of the reference's response cache (its SHA-256 is recorded in the outputs), so a prompt the reference asked is not asked again. The vote caches each of its three prompts separately under the same greedy key, so `vote3` and `vote3u` share every generation, and the reference's cache covers all three of their prompts on the unchanged schema and the rank-order prompt under every perturbation. `nopad`'s identity rerun (no cache) checks that the GPU still gives the cached answers (the reference's identity rerun changed 0 of 1,468).

## Outcomes and comparisons

The answer is the checked answer (the DV accepted after the plausibility check, or no DV), as in the original design; unchecked answers are reported without tests.

**Primary outcomes**, each condition against the committed reference, paired by column:

1. **DV accuracy:** the share of the 681 DV columns whose answer under the unchanged schema is their gold DV.
2. **Curated non-DV columns mapped:** the share of the 787 curated non-DV columns answered with any DV under the unchanged schema (lower is better).
3. **Precision of accepted DV mappings:** among the analysed columns (DV and curated non-DV; blocklisted ones never reach the LLM in the pipeline) answered with a DV under the unchanged schema, the share answered with their gold DV, as in the benchmark summary.
4. **Schema-edit flip rate:** the original primary endpoint (mean over columns of the share of valid schema edits whose answer differs from the condition's own answer under the unchanged schema).

**Tests.** Differences with 95% percentile intervals from the paired cluster bootstrap over the 39 studies (10,000 resamples, seed 20261115). Two-sided p-values from the sign-flip permutation test on study totals of the per-column differences (100,000 sign vectors, seed 20261115) for outcomes 1, 2 and 4. Precision is a ratio whose denominator differs between conditions, so it has no per-column difference; its bootstrap recomputes both precisions on each resample of studies, and its test swaps the two conditions' answers within randomly chosen whole studies (100,000 swaps, same seed) and recomputes the difference. For a mean this is the sign-flip test. Holm's correction over the 12 p-values (three conditions × four outcomes).

**Expected directions** (stated in advance; the tests are two-sided):

- `nopad` changes at most a few dozen shortlists, so the four whole-benchmark outcomes should hardly move, and a non-significant result there says little about the padded columns. Its effect is reported on the padded columns themselves (descriptive), and the flip rate under the permuted DV list should fall, since those flips sat in padded shortlists.
- `vote3` should flip less under schema edits if flips are order-like re-choices. Its accuracy may move either way: in rank order the first candidate is the answer in 36% of prompts, and the retrieval rank is informative.
- `vote3u` should accept fewer DV mappings with higher precision, map fewer non-DV columns, and get fewer DV columns right (abstention trades recall for precision). Its flip rate can move either way, since an abstention becomes an answer when a perturbation makes the orders agree.

**Reading the results.** A condition counts as an improvement on an outcome if its Holm-adjusted p < 0.05 in the better direction. A condition that improves one outcome and worsens another significantly is reported as a trade-off, not an improvement.

**Secondary (intervals, no tests):** the flip rate and mechanism of every perturbation set, for checked and unchecked answers (the original `analyze` tables); `nopad` on the padded columns (their answers in both conditions, and how many lose a wrong or a correct DV); the vote's agreement (the share of asked columns whose three orders agree, and precision of accepted answers when all three agree and when two do); the order-seed flip rate of `vote3` and `vote3u`.

## Compute

From the reference's runs (about 1.2–1.6 s per generation on the shared RTX 5080): `nopad` needs its identity rerun (about 1,500 generations) plus the few padded prompts per perturbation; `vote3` needs two new prompts for each prompt a schema edit changes (about 13,000 for the 30 edits), the order seeds about 4,300 (seeds 4–6 are new; seed 3 is the reference's `candidate_order` draw 3), and the permuted lists and schema 3.2 about 800. About 20,000 generations, 7–9 GPU hours in all; `vote3u` generates nothing. A dry run (prompts only) counts the new prompts before the runs. Order of runs: `nopad`, `vote3`, `vote3u`. If the GPU is needed elsewhere, the order seeds run last and are reported as not run.

## Deviations

Any departure from this note is listed with the results.
