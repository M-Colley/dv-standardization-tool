#!/usr/bin/env python3
"""Compare two LLMs on the full LLM mapping benchmark (docs/model_comparison_design.md).

Reads the eight benchmark runs that ``analyses/model_comparison/run.sh`` writes
with ``evaluate_llm_mapping`` (two models x two shortlists x two halves), pairs
the models' answers by item and writes ``rates.csv``, ``comparisons.csv``,
``reproduction.csv``, ``determinism.csv`` and ``summary.json``.

Outcomes, each for the checked and the unchecked answer: E1 accuracy on DV
columns, E1 curated non-DV columns mapped to a DV, E1 precision of accepted DV
mappings, and E2 accuracy. Differences are candidate minus reference, with 95%
percentile bootstrap intervals that resample clusters (studies; for E2 the
aliases' DVs) and a two-sided permutation test that swaps the two models'
answers within random clusters. Holm over the 12 E1 tests, and separately over
the 4 E2 tests.

Example::

    python -m opendv.analysis.model_comparison
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("model_comparison")

MODELS = {"e2b": "google/gemma-4-E2B-it", "e4b": "google/gemma-4-E4B-it"}
REFERENCE, CANDIDATE = "e2b", "e4b"
SHORTLISTS = ("definition", "dense_h")
HALVES = ("first_23", "added_16")
ANSWERS = ("checked", "unchecked")
E1_OUTCOMES = ("accuracy", "non_dv_mapped", "precision")
DESCRIPTIVE_OUTCOMES = ("abstained_on_dv", "wrong_dv_on_dv")
ANALYSIS_KINDS = ("dv", "curated_non_dv")  # blocklisted columns never reach the LLM in the pipeline
E1_KEY = ["half", "source_id", "dataset_id", "alias"]
E2_KEY = ["alias", "gold"]
SEED = 20261116
N_BOOT = 10_000
N_PERM = 100_000
OUTPUT_DIR = Path("analyses/output_llm_evaluation/model_comparison")
# E2B outputs committed before this test, (shortlist, half) -> folder: the reproduction check.
COMMITTED_E2B = {
    ("definition", "first_23"): Path("analyses/output_llm_evaluation"),
    ("dense_h", "first_23"): Path("analyses/output_llm_evaluation/postfix/shortlist_dense_h"),
    ("definition", "added_16"): Path("analyses/output_llm_evaluation/added_2026_11_05"),
}
SCHEMA_STABILITY_DIR = Path("analyses/output_llm_evaluation/schema_stability")
# (model, shortlist) -> schema-stability condition whose `base` run asked the same model.
SCHEMA_STABILITY_RUNS = {
    ("e2b", "definition"): "e2b_definition",
    ("e2b", "dense_h"): "e2b_dense_h",
    ("e4b", "definition"): "e4b_definition",
}


# ── Reading the runs ─────────────────────────────────────────────────────────


def cell_dir(output_dir: Path, model: str, shortlist: str, half: str) -> Path:
    """Folder of one benchmark run: ``<output_dir>/<model>_<shortlist>/<half>``."""
    return Path(output_dir) / f"{model}_{shortlist}" / half


def _read_items(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])


def load_runs(output_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    """Every run's items (with ``model``, ``shortlist``, ``half``) and summaries.

    Stops if a run is missing, had generation errors or used another model.
    """
    frames, summaries, missing = [], {}, []
    for model in MODELS:
        for shortlist in SHORTLISTS:
            for half in HALVES:
                folder = cell_dir(output_dir, model, shortlist, half)
                items_path = folder / "llm_mapping_evaluation_items.csv"
                summary_path = folder / "llm_mapping_evaluation_summary.json"
                if not items_path.is_file() or not summary_path.is_file():
                    missing.append(str(folder))
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("llm_generation_errors"):
                    raise ValueError(f"{folder}: {summary['llm_generation_errors']} generation errors; rerun it")
                if summary.get("models") != [MODELS[model]]:
                    raise ValueError(f"{folder}: models {summary.get('models')}, expected [{MODELS[model]!r}]")
                if summary.get("shortlist") != shortlist:
                    raise ValueError(f"{folder}: shortlist {summary.get('shortlist')!r}, expected {shortlist!r}")
                summaries[(model, shortlist, half)] = summary
                frames.append(_read_items(items_path).assign(model=model, shortlist=shortlist, half=half))
    if missing:
        raise FileNotFoundError("benchmark runs missing: " + ", ".join(missing))
    return pd.concat(frames, ignore_index=True), summaries


def _item_set(frame: pd.DataFrame, key: list[str]) -> set[tuple]:
    columns = [*key, "label_kind", "gold"] if "gold" not in key else [*key, "label_kind"]
    return set(frame[columns].astype(str).itertuples(index=False, name=None))


def check_same_items(items: pd.DataFrame) -> dict[str, int]:
    """The four conditions must ask about the same items, with the same labels (ValueError otherwise)."""
    counts = {}
    for experiment, key in (("E1", E1_KEY), ("E2", E2_KEY)):
        exp = items[items["experiment"] == experiment]
        sets = {
            (model, shortlist): _item_set(group, key)
            for (model, shortlist), group in exp.groupby(["model", "shortlist"], sort=True)
        }
        reference = next(iter(sets.values()))
        for condition, item_set in sets.items():
            if item_set != reference:
                raise ValueError(
                    f"{experiment} items of {condition} differ from the others "
                    f"({len(item_set - reference)} extra, {len(reference - item_set)} missing)"
                )
        for condition, group in exp.groupby(["model", "shortlist"], sort=True):
            if group.duplicated(key).any():
                raise ValueError(f"{experiment} items of {condition} repeat a key {key}")
        counts[experiment] = len(reference)
    return counts


# ── Outcomes ─────────────────────────────────────────────────────────────────


def outcome_items(frame: pd.DataFrame, outcome: str, answer: str, dv_ids: set[str]) -> pd.DataFrame:
    """One row per item that counts for ``outcome``: its cluster, numerator and denominator.

    The rate of a set of items is sum(num) / sum(den). Accuracy, non-DV mapped,
    abstained and wrong DV have den = 1 per item; precision counts an item in its
    denominator when the answer is a DV.
    """
    final = frame[f"final_{answer}"]
    is_dv_answer = final.isin(dv_ids)
    hit = final.notna() & (final == frame["gold"])
    kind = frame["label_kind"]
    if outcome == "e2_accuracy":
        mask = (frame["experiment"] == "E2") & (kind == "dv")
        num, den, cluster = hit, pd.Series(1, index=frame.index), frame["gold"]
    else:
        e1 = frame["experiment"] == "E1"
        cluster = frame["source_id"]
        den = pd.Series(1, index=frame.index)
        if outcome == "accuracy":
            mask, num = e1 & (kind == "dv"), hit
        elif outcome == "non_dv_mapped":
            mask, num = e1 & (kind == "curated_non_dv"), is_dv_answer
        elif outcome == "precision":
            mask, num, den = e1 & kind.isin(ANALYSIS_KINDS), is_dv_answer & hit, is_dv_answer
        elif outcome == "abstained_on_dv":
            mask, num = e1 & (kind == "dv"), final.isna()
        elif outcome == "wrong_dv_on_dv":
            mask, num = e1 & (kind == "dv"), final.notna() & ~hit
        else:
            raise ValueError(f"unknown outcome {outcome!r}")
    key = E2_KEY if outcome == "e2_accuracy" else E1_KEY
    out = frame.loc[mask, key].copy()
    out["cluster"] = cluster[mask].astype(str).to_numpy()
    out["num"] = num[mask].astype(float).to_numpy()
    out["den"] = den[mask].astype(float).to_numpy()
    return out.reset_index(drop=True)


def cluster_sums(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Per cluster, both models' summed numerators and denominators (num_ref, den_ref, num_cand, den_cand)."""
    ref = reference.groupby("cluster")[["num", "den"]].sum()
    cand = candidate.groupby("cluster")[["num", "den"]].sum()
    joined = ref.join(cand, how="outer", lsuffix="_ref", rsuffix="_cand").fillna(0.0)
    return joined.sort_index()


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def paired_difference(sums: pd.DataFrame) -> float:
    """Candidate rate minus reference rate over all clusters."""
    totals = sums.sum()
    return float(_ratio(np.array(totals["num_cand"]), np.array(totals["den_cand"]))
                 - _ratio(np.array(totals["num_ref"]), np.array(totals["den_ref"])))


def paired_bootstrap(sums: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, int]:
    """95% percentile interval of the difference, resampling clusters; both models' answers move together.

    Returns (low, high, number of resamples where a rate is undefined), the latter left out.
    """
    arrays = {c: sums[c].to_numpy(dtype=float) for c in ("num_ref", "den_ref", "num_cand", "den_cand")}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    drawn = {c: a[picks].sum(axis=1) for c, a in arrays.items()}
    diff = _ratio(drawn["num_cand"], drawn["den_cand"]) - _ratio(drawn["num_ref"], drawn["den_ref"])
    undefined = int(np.isnan(diff).sum())
    if undefined == n_boot:
        return (np.nan, np.nan, undefined)
    return (float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5)), undefined)


def paired_permutation_p(
    sums: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED, chunk: int = 20_000
) -> float:
    """Two-sided p: the two models' answers are swapped in each cluster with probability 1/2.

    For a rate with the same denominators for both models (accuracy, non-DV
    mapped) this is the sign-flip test on cluster totals; for precision, a ratio,
    the same exchangeability under the null holds.
    """
    num_ref, den_ref = sums["num_ref"].to_numpy(dtype=float), sums["den_ref"].to_numpy(dtype=float)
    num_cand, den_cand = sums["num_cand"].to_numpy(dtype=float), sums["den_cand"].to_numpy(dtype=float)
    observed = paired_difference(sums)
    if np.isnan(observed):
        return float("nan")
    if observed == 0:
        return 1.0
    d_num, d_den = num_cand - num_ref, den_cand - den_ref
    totals = (num_ref.sum(), den_ref.sum(), num_cand.sum(), den_cand.sum())
    rng = np.random.default_rng(seed)
    exceed, done = 0, 0
    while done < n_perm:
        size = min(chunk, n_perm - done)
        swap = rng.integers(0, 2, size=(size, len(sums))).astype(float)
        moved_num, moved_den = swap @ d_num, swap @ d_den
        cand = _ratio(totals[2] - moved_num, totals[3] - moved_den)
        diff = cand - _ratio(totals[0] + moved_num, totals[1] + moved_den)
        exceed += int(np.sum(np.abs(np.nan_to_num(diff, nan=0.0)) >= abs(observed) - 1e-12))
        done += size
    return float((1 + exceed) / (1 + n_perm))


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p values (same order as given)."""
    from statsmodels.stats.multitest import multipletests

    return multipletests(p_values, method="holm")[1].tolist()


def wilson_row(num: float, den: float) -> dict[str, Any]:
    from opendv.analysis.evaluate_llm_mapping import wilson

    result = wilson(int(round(num)), int(round(den)))
    return {"k": result["k"], "n": result["n"], "rate": result["rate"],
            "ci_low": result["ci95"][0], "ci_high": result["ci95"][1]}


# ── Analysis ─────────────────────────────────────────────────────────────────


def compare_models(
    items: pd.DataFrame, dv_ids: set[str], n_boot: int = N_BOOT, n_perm: int = N_PERM, seed: int = SEED
) -> pd.DataFrame:
    """The pre-specified comparisons, candidate against reference, with Holm within each family."""
    rows = []
    for shortlist in SHORTLISTS:
        frames = {m: items[(items["model"] == m) & (items["shortlist"] == shortlist)] for m in (REFERENCE, CANDIDATE)}
        for answer in ANSWERS:
            for outcome in (*E1_OUTCOMES, "e2_accuracy"):
                ref, cand = (outcome_items(frames[m], outcome, answer, dv_ids) for m in (REFERENCE, CANDIDATE))
                sums = cluster_sums(ref, cand)
                low, high, undefined = paired_bootstrap(sums, n_boot, seed)
                row = {
                    "family": "E2" if outcome == "e2_accuracy" else "E1",
                    "shortlist": shortlist, "answer": answer, "outcome": outcome,
                    "reference": MODELS[REFERENCE], "candidate": MODELS[CANDIDATE],
                    "n_items": int(len(ref)), "n_clusters": int(len(sums)),
                    "k_reference": int(ref["num"].sum()), "n_reference": int(ref["den"].sum()),
                    "k_candidate": int(cand["num"].sum()), "n_candidate": int(cand["den"].sum()),
                    "rate_reference": float(_ratio(np.array(ref["num"].sum()), np.array(ref["den"].sum()))),
                    "rate_candidate": float(_ratio(np.array(cand["num"].sum()), np.array(cand["den"].sum()))),
                    "difference": paired_difference(sums),
                    "ci_low": low, "ci_high": high, "bootstrap_undefined": undefined,
                    "p": paired_permutation_p(sums, n_perm, seed),
                }
                if outcome != "precision":
                    key = E2_KEY if outcome == "e2_accuracy" else E1_KEY
                    paired = ref.merge(cand, on=key, suffixes=("_ref", "_cand"), validate="one_to_one")
                    row["n_candidate_only"] = int(((paired["num_cand"] > 0) & (paired["num_ref"] == 0)).sum())
                    row["n_reference_only"] = int(((paired["num_ref"] > 0) & (paired["num_cand"] == 0)).sum())
                rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, group in table.groupby("family"):
        table.loc[group.index, "p_holm"] = holm(group["p"].tolist())
    return table


def schema_stability_items(stability_dir: Path = SCHEMA_STABILITY_DIR) -> set[tuple]:
    """E1 item keys of the schema-stability test (its reference run's ``base``)."""
    path = stability_dir / "answers_e2b_definition.csv.gz"
    answers = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""],
                          usecols=["perturbation", *E1_KEY])
    base = answers[answers["perturbation"] == "base"]
    return set(base[E1_KEY].astype(str).itertuples(index=False, name=None))


def rates(items: pd.DataFrame, dv_ids: set[str], earlier_items: set[tuple] | None = None) -> pd.DataFrame:
    """Each condition's rates with Wilson intervals: pooled, per half, on new items, given the gold in the shortlist."""
    rows = []
    for (model, shortlist), frame in items.groupby(["model", "shortlist"], sort=True):
        keys = frame[E1_KEY].astype(str).apply(tuple, axis=1)
        scopes = {"pooled": pd.Series(True, index=frame.index)}
        scopes |= {half: frame["half"] == half for half in HALVES}
        if earlier_items is not None:
            scopes["new_since_schema_stability"] = (frame["experiment"] == "E1") & ~keys.isin(earlier_items)
        scopes["gold_in_shortlist"] = frame["gold_in_shortlist"].astype(str).str.lower() == "true"
        for answer in ANSWERS:
            for outcome in (*E1_OUTCOMES, *DESCRIPTIVE_OUTCOMES, "e2_accuracy"):
                for scope, mask in scopes.items():
                    if outcome == "e2_accuracy" and scope not in ("pooled", "gold_in_shortlist"):
                        continue
                    if scope == "gold_in_shortlist" and outcome not in ("accuracy", "e2_accuracy"):
                        continue
                    part = outcome_items(frame[mask], outcome, answer, dv_ids)
                    rows.append({"model": MODELS[model], "shortlist": shortlist, "answer": answer,
                                 "outcome": outcome, "scope": scope,
                                 **wilson_row(part["num"].sum(), part["den"].sum())})
        asked = frame[(frame["experiment"] == "E1") & frame["prompt_sha256"].notna()]
        rows.append({"model": MODELS[model], "shortlist": shortlist, "answer": "", "outcome": "seconds_per_prompt",
                     "scope": "pooled", "k": None, "n": int(len(asked)),
                     "rate": float(asked["seconds"].mean()) if len(asked) else None,
                     "ci_low": None, "ci_high": None})
    return pd.DataFrame(rows)


def reproduction(items: pd.DataFrame, committed: dict[tuple[str, str], Path] = COMMITTED_E2B) -> pd.DataFrame:
    """The E2B rerun against the committed E2B outputs, per committed run and experiment."""
    rows = []
    for (shortlist, half), folder in committed.items():
        old = _read_items(folder / "llm_mapping_evaluation_items.csv")
        new = items[(items["model"] == REFERENCE) & (items["shortlist"] == shortlist) & (items["half"] == half)]
        for experiment, key in (("E1", ["source_id", "dataset_id", "alias"]), ("E2", E2_KEY)):
            a, b = old[old["experiment"] == experiment], new[new["experiment"] == experiment]
            if a.empty and b.empty:
                continue
            merged = a.merge(b, on=key, how="outer", suffixes=("_committed", "_rerun"), indicator=True)
            both = merged[merged["_merge"] == "both"]
            row = {"shortlist": shortlist, "half": half, "experiment": experiment, "committed": str(folder),
                   "n_committed": int(len(a)), "n_rerun": int(len(b)),
                   "only_committed": int((merged["_merge"] == "left_only").sum()),
                   "only_rerun": int((merged["_merge"] == "right_only").sum())}
            for column in ("suggestion", "final_unchecked", "final_checked"):
                x, y = both[f"{column}_committed"], both[f"{column}_rerun"]
                row[f"{column}_differ"] = int((~((x == y) | (x.isna() & y.isna()))).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def determinism(items: pd.DataFrame, stability_dir: Path = SCHEMA_STABILITY_DIR) -> pd.DataFrame:
    """Answers to prompts that the schema-stability reference runs asked too (same model, same prompt)."""
    rows = []
    for (model, shortlist), condition in SCHEMA_STABILITY_RUNS.items():
        path = stability_dir / f"answers_{condition}.csv.gz"
        if not path.is_file():
            continue
        earlier = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""],
                              usecols=["perturbation", "prompt_sha256", "suggestion"])
        earlier = earlier[(earlier["perturbation"] == "base") & earlier["prompt_sha256"].notna()]
        answers = earlier.groupby("prompt_sha256")["suggestion"].agg(lambda s: set(s.fillna("<none>")))
        frame = items[(items["model"] == model) & (items["shortlist"] == shortlist) & (items["experiment"] == "E1")]
        asked = frame[frame["prompt_sha256"].notna()]
        matched = asked[asked["prompt_sha256"].isin(answers.index)]
        same = [
            suggestion in answers[sha] and len(answers[sha]) == 1
            for sha, suggestion in zip(matched["prompt_sha256"], matched["suggestion"].fillna("<none>"), strict=True)
        ]
        rows.append({"model": MODELS[model], "shortlist": shortlist, "schema_stability_condition": condition,
                     "n_asked": int(len(asked)), "n_same_prompt": int(len(matched)),
                     "n_same_answer": int(sum(same)), "n_different_answer": int(len(same) - sum(same))})
    return pd.DataFrame(rows)


def run_cells(summaries: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"model": MODELS[model], "shortlist": shortlist, "half": half,
         "runtime_seconds": summary.get("runtime_seconds"),
         "cuda_peak_memory_gb": summary.get("cuda_peak_memory_gb"),
         "model_placement": summary.get("model_placement"),
         "generated_at": summary.get("generated_at"),
         "llm_generation_errors": summary.get("llm_generation_errors")}
        for (model, shortlist, half), summary in sorted(summaries.items())
    ]


def analyze(
    output_dir: Path = OUTPUT_DIR,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    committed: dict[tuple[str, str], Path] = COMMITTED_E2B,
    stability_dir: Path = SCHEMA_STABILITY_DIR,
    dv_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Read the runs, check them, compare the models and write the outputs to ``output_dir``."""
    if dv_ids is None:
        from opendv.analysis.schema_stability import base_dv_ids

        dv_ids = base_dv_ids()
    items, summaries = load_runs(output_dir)
    counts = check_same_items(items)
    earlier = (
        schema_stability_items(stability_dir) if (stability_dir / "answers_e2b_definition.csv.gz").is_file() else None
    )
    comparisons = compare_models(items, dv_ids, n_boot, n_perm)
    rate_table = rates(items, dv_ids, earlier)
    repro = reproduction(items, committed)
    determ = determinism(items, stability_dir)
    comparisons.to_csv(output_dir / "comparisons.csv", index=False)
    rate_table.to_csv(output_dir / "rates.csv", index=False)
    repro.to_csv(output_dir / "reproduction.csv", index=False)
    determ.to_csv(output_dir / "determinism.csv", index=False)
    summary = {
        "design": "docs/model_comparison_design.md",
        "reference": MODELS[REFERENCE], "candidate": MODELS[CANDIDATE],
        "n_items": counts, "n_boot": n_boot, "n_perm": n_perm, "seed": SEED,
        "runs": run_cells(summaries),
        "comparisons": _records(comparisons),
        "reproduction": _records(repro),
        "determinism": _records(determ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary


def shortlists_by_model(
    items: pd.DataFrame, dv_ids: set[str], n_boot: int = N_BOOT, n_perm: int = N_PERM, seed: int = SEED
) -> pd.DataFrame:
    """Exploratory, added after the runs: ``dense_h`` minus ``definition`` within each model, and between models.

    Within a model the two shortlists' answers are paired by item, with the
    pre-specified interval and permutation test (shortlists swapped within
    studies; unadjusted). The difference between the two models' shortlist
    differences gets a cluster-bootstrap interval only: no exchangeability
    argument gives a test of it.
    """
    rows = []
    for answer in ANSWERS:
        for outcome in (*E1_OUTCOMES, "e2_accuracy"):
            frames = {
                (m, s): outcome_items(items[(items["model"] == m) & (items["shortlist"] == s)], outcome, answer, dv_ids)
                for m in MODELS for s in SHORTLISTS
            }
            sums = {m: cluster_sums(frames[(m, "definition")], frames[(m, "dense_h")]) for m in MODELS}
            for model in MODELS:
                low, high, _ = paired_bootstrap(sums[model], n_boot, seed)
                rows.append({"answer": answer, "outcome": outcome, "contrast": f"dense_h - definition, {model}",
                             "difference": paired_difference(sums[model]), "ci_low": low, "ci_high": high,
                             "p_unadjusted": paired_permutation_p(sums[model], n_perm, seed)})
            joined = sums[CANDIDATE].join(sums[REFERENCE], how="outer", lsuffix="_c", rsuffix="_r").fillna(0.0)
            arrays = {name: joined[name].to_numpy(dtype=float) for name in joined}
            picks = np.random.default_rng(seed).integers(0, len(joined), size=(n_boot, len(joined)))
            boot = _shortlist_gain(arrays, "c", picks) - _shortlist_gain(arrays, "r", picks)
            rows.append({"answer": answer, "outcome": outcome,
                         "contrast": f"(dense_h - definition) of {CANDIDATE} minus that of {REFERENCE}",
                         "difference": float(_shortlist_gain(arrays, "c") - _shortlist_gain(arrays, "r")),
                         "ci_low": float(np.nanpercentile(boot, 2.5)), "ci_high": float(np.nanpercentile(boot, 97.5)),
                         "p_unadjusted": None})
    return pd.DataFrame(rows)


def _shortlist_gain(arrays: dict[str, np.ndarray], suffix: str, index: Any = slice(None)) -> np.ndarray:
    """One model's dense_h rate minus its definition rate, over all clusters or each bootstrap draw (``index``)."""
    def total(name: str) -> np.ndarray:
        return arrays[f"{name}_{suffix}"][index].sum(axis=-1)

    return _ratio(total("num_cand"), total("den_cand")) - _ratio(total("num_ref"), total("den_ref"))


def explore(output_dir: Path = OUTPUT_DIR, n_boot: int = N_BOOT, n_perm: int = N_PERM,
            dv_ids: set[str] | None = None) -> pd.DataFrame:
    """Exploratory analyses of the finished runs (not in the design note): ``exploratory_shortlists.csv``."""
    if dv_ids is None:
        from opendv.analysis.schema_stability import base_dv_ids

        dv_ids = base_dv_ids()
    items, _ = load_runs(output_dir)
    check_same_items(items)
    table = shortlists_by_model(items, dv_ids, n_boot, n_perm)
    table.to_csv(output_dir / "exploratory_shortlists.csv", index=False)
    return table


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as dicts with None for missing values (``NaN`` is not JSON)."""
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--explore", action="store_true",
                        help="Run the exploratory analyses added after the runs instead (exploratory_shortlists.csv).")
    args = parser.parse_args(argv)
    if args.explore:
        table = explore(args.output_dir, args.n_boot, args.n_perm)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return
    summary = analyze(args.output_dir, args.n_boot, args.n_perm)
    table = pd.DataFrame(summary["comparisons"])
    columns = ["family", "shortlist", "answer", "outcome", "rate_reference", "rate_candidate", "difference",
               "ci_low", "ci_high", "p", "p_holm"]
    print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
