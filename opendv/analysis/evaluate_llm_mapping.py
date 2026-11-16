#!/usr/bin/env python3
"""Evaluate local-LLM DV mapping against curated mappings (leave-alias-out).

The batch runner only asks the LLM about columns no explicit mapping covers,
so its accuracy is never visible in a normal run. This script measures it by
hiding mappings whose answer is known and replaying the pipeline's LLM step
(same candidate shortlist, prompt, model and repository context):

* **E1 — in-dataset columns.** Columns a schema or curated repository
  mapping maps to a canonical DV are positives; columns a curated mapping
  locks as ``meta_*`` (demographics, conditions, context) or the never-map
  blocklist names are negatives. Each column's own alias is removed from the
  mapping before the shortlist is built, and its real values feed the
  plausibility check.
* **E2 — schema alias recovery.** Aliases of DVs with several aliases are
  held out one at a time; the remaining aliases are the only evidence. No
  data is available, so only the name rules apply — this measures how often
  the plausibility check rejects genuine DV names.

The plausibility check (``scripts/llm_plausibility.py``) is applied offline to
the same LLM answers, so "checked" and "unchecked" are compared on identical
model outputs. Proportions carry Wilson 95% CIs (``scipy.stats.binomtest``).

Example::

    python analyses/evaluate_llm_mapping.py \\
        --manifest sources_manifest_no4tu.yaml \\
        --batch-dir data/processed/bench/new2_w8/batch \\
        --cache-dir data/processed/bench/cache \\
        --output-dir analyses/output_llm_evaluation

``gemma-4-E4B-it`` on a 16 GB GPU, its per-layer embeddings in CPU memory
(``llm_utils.configure_cpu_ple``; ``analyses/model_comparison/run.sh``)::

    python -m opendv.analysis.evaluate_llm_mapping ... --models google/gemma-4-E4B-it \\
        --cpu-ple $HUB/models--google--gemma-4-E4B-it/snapshots/<weights revision> \\
                  $HUB/models--google--gemma-4-E4B-it/snapshots/<tokenizer revision>

A model named in ``--models`` that would not be used (its memory estimate does
not fit, or ``DV_LLM_MODELS`` replaces it) stops the run instead of leaving
every item unanswered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest


from opendv import llm_utils  # noqa: E402
from opendv.batch_profiles import classify_dataset_type, llm_eligible_dataset_types  # noqa: E402
from opendv.dv_retrieval import (  # noqa: E402
    DEFAULT_DV_SCHEMA_PATH,
    N_DENSE_EXTRA,
    apply_hierarchy,
    construct_hierarchy,
    get_dense_retriever,
    get_retriever,
    mix_rankings,
    reciprocal_rank_fusion,
)
from opendv.value_labels import (  # noqa: E402
    apply_missing_values,
    apply_reverse_keying,
    apply_row_exclusions,
    apply_scale_conversions,
    apply_unit_conversions,
    apply_value_labels,
    recode_negative_answer_codes,
    recode_survey_answer_codes,
)
from opendv.llm_coordinator import _select_llm_candidate_shortlist  # noqa: E402
from opendv.llm_plausibility import (  # noqa: E402
    check_llm_suggestion,
    data_rejection_reason,
    load_rules,
    name_rejection_reason,
    normalize_name,
    profile_columns,
)
from opendv.data_loaders import table_options_for  # noqa: E402
from opendv.run_batch_standardization import (  # noqa: E402
    NEVER_MAP_NORMALIZED_COLUMNS,
    SchemaFamilies,
    _coerce_manifest_text_list,
    _find_repository_mapping_candidates,
    _load_any_table,
    _MappingResolver,
    _resolve_repository_mapping_path,
    discover_source_files,
    load_manifest,
)
from opendv._paths import SCHEMAS_DIR

logger = logging.getLogger("evaluate_llm_mapping")
MIN_ATTEMPT_SCORE = 65.0  # same gate as llm_coordinator._augment_mapping_with_llm_deductions


def wilson(k: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": [None, None]}
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return {"k": int(k), "n": int(n), "rate": k / n, "ci95": [float(ci.low), float(ci.high)]}


def _without_alias(mapping: dict[str, str], alias: str) -> dict[str, str]:
    """Drop every key that normalizes to the held-out alias."""
    target = normalize_name(alias)
    return {key: value for key, value in mapping.items() if normalize_name(key) != target}


# Shortlists built from the definition (TF-IDF) and dense rankings; "_h" variants
# apply the scale/item hierarchy filter (scripts.dv_retrieval.apply_hierarchy).
HARNESS_SHORTLISTS = ("definition", "dense", "mixed", "mixed4", "definition_h", "dense_h", "mixed_h")
CANDIDATE_ORDERS = ("rank", "shuffle", "alphabetical")


@dataclass(frozen=True)
class AskOptions:
    """How the LLM is asked; the defaults are the committed benchmark's.

    ``schema_path`` reads every DV definition (shortlist retrieval, the scale/item
    filter, the prompt's candidate reference) from another DV schema file.
    ``candidate_order`` lists the shortlist in the prompt by retrieval rank, in a
    random order (seeded by ``order_seed`` and the column name) or alphabetically.
    ``num_samples > 1`` asks for that many sampled answers and takes their majority
    (``temperature``, ``sample_seed``).
    """

    schema_path: str | None = None
    candidate_order: str = "rank"
    order_seed: int = 0
    num_samples: int = 1
    temperature: float = 1.0
    sample_seed: int = 0

    @classmethod
    def from_args(cls, args: Any) -> AskOptions:
        return cls(
            schema_path=getattr(args, "schema", None),
            candidate_order=getattr(args, "candidate_order", "rank"),
            order_seed=getattr(args, "order_seed", 0),
            num_samples=getattr(args, "samples", 1),
            temperature=getattr(args, "temperature", 1.0),
            sample_seed=getattr(args, "sample_seed", 0),
        )


def order_candidates(candidates: list[str], alias: str, order: str, seed: int = 0) -> list[str]:
    """The shortlist in the order the prompt lists it (see :class:`AskOptions`)."""
    if order == "rank":
        return list(candidates)
    if order == "alphabetical":
        return sorted(candidates, key=lambda dv: (dv.lower(), dv))
    if order == "shuffle":
        digest = hashlib.sha256(f"{seed}\x1f{alias}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return [candidates[i] for i in rng.permutation(len(candidates))]
    raise ValueError(f"candidate order must be one of {CANDIDATE_ORDERS}, got {order!r}")


def response_cache_key(models: list[str], prompt: str, options: AskOptions) -> str:
    """Key of one LLM answer: the models, the exact prompt and the decoding settings."""
    decoding = {"num_samples": options.num_samples}
    if options.num_samples > 1:
        decoding |= {"temperature": options.temperature, "sample_seed": options.sample_seed}
    payload = json.dumps({"models": list(models), "prompt": prompt, "decoding": decoding}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ask_llm(
    alias: str,
    mapping: dict[str, str],
    context: str,
    preferred_models: list[str] | None,
    shortlist: str = "definition",
    options: AskOptions | None = None,
    response_cache: dict[str, str | None] | None = None,
) -> tuple[list[str], float, str | None]:
    """Shortlist, top fuzzy score and the LLM's answer for one column.

    ``response_cache`` maps :func:`response_cache_key` to an earlier answer, so a
    prompt that a rerun repeats exactly is not generated again (decoding is greedy,
    or sampled with a fixed seed). Failed generations are not cached.
    """
    options = options or AskOptions()
    schema_path = options.schema_path or str(DEFAULT_DV_SCHEMA_PATH)
    candidates, top_score = _select_llm_candidate_shortlist(mapping, alias)
    if shortlist in HARNESS_SHORTLISTS and candidates:
        # Leave-alias-out: the held-out alias is removed from every definition.
        n = len(candidates)
        depth = max(3 * n, 24)
        lexical = [dv for dv, _ in get_retriever(schema_path).rank(alias, k=depth, exclude_alias=alias)]
        if shortlist not in ("definition", "definition_h"):
            dense = [
                dv for dv, _ in get_dense_retriever(schema_path=schema_path).rank(alias, k=depth, exclude_alias=alias)
            ]
        related = construct_hierarchy(schema_path)
        if shortlist == "dense_h":
            # Pre-specified primary (2026-10-25): the fusion with the scale/item filter.
            candidates = apply_hierarchy(reciprocal_rank_fusion(lexical, dense), n, related)
        elif shortlist == "definition_h":
            candidates = apply_hierarchy(lexical, n, related)
        elif shortlist == "mixed_h":
            mixed = mix_rankings(lexical, dense, n, N_DENSE_EXTRA)
            candidates = apply_hierarchy([*mixed, *lexical, *dense], n, related)
        elif shortlist == "dense":
            # Same fusion as scripts.dv_retrieval.dense_shortlist.
            candidates = reciprocal_rank_fusion(lexical, dense)[:n]
        elif shortlist in ("mixed", "mixed4"):
            # mixed: the pipeline option (N_DENSE_EXTRA dense-only candidates);
            # mixed4: exploratory variant with four.
            candidates = mix_rankings(lexical, dense, n, N_DENSE_EXTRA if shortlist == "mixed" else 4)
        else:
            candidates = lexical[:n]
    candidates = order_candidates(candidates, alias, options.candidate_order, options.order_seed)
    if not candidates or top_score < MIN_ATTEMPT_SCORE:
        return candidates, top_score, None
    key = None
    if response_cache is not None:
        prompt = llm_utils.build_mapping_prompt(alias, candidates, context, options.schema_path)
        key = response_cache_key(llm_utils.select_local_model_candidates(preferred_models), prompt, options)
        if key in response_cache:
            return candidates, top_score, response_cache[key]
    before = len(llm_utils.generation_errors())
    suggestion = llm_utils.deduce_standard_name_with_local_llm(
        raw_column_name=alias,
        canonical_candidates=candidates,
        preferred_models=preferred_models,
        repository_context=context,
        schema_path=options.schema_path,
        num_samples=options.num_samples,
        temperature=options.temperature,
        sample_seed=options.sample_seed,
    )
    if key is not None and len(llm_utils.generation_errors()) == before:
        response_cache[key] = suggestion
    return candidates, top_score, suggestion


def prompt_fingerprint(alias: str, candidates: list[str], context: str, options: AskOptions) -> str:
    """SHA-256 of the prompt an asked column gets (equal fingerprints, equal prompts)."""
    prompt = llm_utils.build_mapping_prompt(alias, candidates, context, options.schema_path)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _source_context(source: dict[str, Any], base_dir: Path, cache_dir: Path | None = None) -> str:
    extra = _coerce_manifest_text_list(source.get("llm_context"))
    extra.extend(_coerce_manifest_text_list(source.get("publication_context")))
    return llm_utils.collect_repository_context(
        base_dir,
        explicit_dois=_coerce_manifest_text_list(source.get("publication_doi"))
        + _coerce_manifest_text_list(source.get("doi")),
        explicit_pdf_urls=_coerce_manifest_text_list(source.get("publication_pdf_url"))
        + _coerce_manifest_text_list(source.get("pdf_url")),
        extra_context=extra,
        cache_dir=cache_dir,
    )


def _ask_llm_logged(*args: Any, **kwargs: Any) -> tuple[list[str], float, str | None, str | None]:
    """``_ask_llm`` plus the generation error of this call, if one failed."""
    before = len(llm_utils.generation_errors())
    shortlist, top_score, suggestion = _ask_llm(*args, **kwargs)
    errors = llm_utils.generation_errors()[before:]
    return shortlist, top_score, suggestion, errors[-1]["error"] if errors else None


def gold_items(meta_view: pd.DataFrame, dv_ids: set[str], max_per_source: int, seed: int) -> pd.DataFrame:
    """One row per (source, alias) with its gold label (canonical DV or None)."""
    mv = meta_view.copy()
    if "composite_items" in mv:
        # Each item of a composite is a column with its own row; the
        # composite's row repeats its first item's alias.
        mv = mv[mv["composite_items"].isna()]
    mv["original_alias"] = mv["original_alias"].astype(str)
    positives = mv[(mv["mapping_method"] == "mapping") & mv["canonical_dv"].isin(dv_ids)].assign(
        label_kind="dv", gold=lambda d: d["canonical_dv"]
    )
    curated_neg = mv[
        (mv["mapping_method"] == "mapping") & mv["canonical_dv"].astype(str).str.startswith("meta_")
    ].assign(label_kind="curated_non_dv", gold=None)
    blocked = mv[mv["mapping_method"] == "blocked"].assign(label_kind="blocklisted_non_dv", gold=None)
    items = pd.concat([positives, curated_neg, blocked], ignore_index=True)
    items = items.drop_duplicates(["source_id", "original_alias"])
    sampled = [
        group.sample(n=min(len(group), max_per_source), random_state=seed)
        for _, group in items.groupby(["source_id", "label_kind"], sort=True)
    ]
    return pd.concat(sampled, ignore_index=True)[
        ["source_id", "dataset_id", "original_alias", "label_kind", "gold"]
    ].sort_values(["source_id", "label_kind", "original_alias"]).reset_index(drop=True)


def run_e1(
    args,
    families: SchemaFamilies,
    dv_ids: set[str],
    items: pd.DataFrame | None = None,
    response_cache: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """E1 answers, one row per item. ``items`` fixes the item list (default: drawn from
    the batch's ``meta_view.csv`` with ``dv_ids``); ``response_cache`` is passed to ``_ask_llm``."""
    sources = {s["source_id"]: s for s in load_manifest(Path(args.manifest))}
    if items is None:
        meta_view = pd.read_csv(Path(args.batch_dir) / "meta_view.csv", low_memory=False)
        items = gold_items(meta_view, dv_ids, args.max_per_source, args.seed)
    options = AskOptions.from_args(args)
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory() as tmp:
        for source_id, group in items.groupby("source_id", sort=True):
            source = sources[source_id]
            base_dir, _, _ = discover_source_files(source, Path(tmp), cache_root=Path(args.cache_dir))
            # The same cache as the batch runner, so every shortlist and rerun
            # prompts with the same fetched manuscript text.
            context = _source_context(source, base_dir, Path(args.cache_dir) / "llm_context")
            context_sha256 = llm_utils.context_fingerprint(context)
            candidates_paths = _find_repository_mapping_candidates(base_dir)
            resolver = _MappingResolver(families, bool(source.get("override_standard_aliases", False)))
            requested = str(source["mapping_path"]).strip() if source.get("mapping_path") else None
            frames: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}
            for item in group.itertuples(index=False):
                relative = Path(item.dataset_id)
                if item.dataset_id not in frames:
                    raw = _load_any_table(
                        base_dir / item.dataset_id, table_options_for(source.get("table_options"), item.dataset_id)
                    )
                    raw_columns = [str(c) for c in raw.columns]
                    dataset_type = classify_dataset_type(relative, raw_columns)
                    mapping_path = _resolve_repository_mapping_path(
                        base_dir, candidates_paths, requested_mapping_path=requested, dataset_path=base_dir / relative
                    )
                    resolved = resolver.resolve(dataset_type, mapping_path, item.dataset_id)
                    # Same row filter and response recoding as the pipeline,
                    # before profiling.
                    raw, _ = apply_row_exclusions(raw, resolved.exclude_rows)
                    raw, _ = apply_missing_values(raw, resolved.missing_values)
                    raw, _ = apply_value_labels(raw, resolved.value_labels)
                    if dataset_type in llm_eligible_dataset_types():
                        raw, _ = recode_negative_answer_codes(raw)
                        raw, _ = recode_survey_answer_codes(raw)
                    raw, _ = apply_reverse_keying(raw, resolved.reverse_items)
                    # The pipeline converts units and scales before the LLM
                    # step, so the plausibility check sees seconds, not ms.
                    raw, _ = apply_unit_conversions(raw, resolved.unit_conversions)
                    raw, _ = apply_scale_conversions(raw, resolved.scale_conversions)
                    frames[item.dataset_id] = (raw, resolved.mapping)
                df, mapping = frames[item.dataset_id]
                columns = [str(c) for c in df.columns]
                alias = item.original_alias
                profile = profile_columns(df, [alias]).get(alias)
                t0 = time.perf_counter()
                shortlist, top_score, suggestion, llm_error = _ask_llm_logged(
                    alias, _without_alias(mapping, alias), context, args.models, args.shortlist,
                    options=options, response_cache=response_cache,
                )
                asked = bool(shortlist) and top_score >= MIN_ATTEMPT_SCORE
                pre = name_rejection_reason(alias, columns) or (data_rejection_reason(profile) if profile else None)
                post = (
                    check_llm_suggestion(alias, suggestion, profile, families.canonical_ranges,
                                         rating_scales=families.rating_scales)
                    if suggestion and not pre else None
                )
                rows.append({
                    "experiment": "E1",
                    "source_id": source_id,
                    "dataset_id": item.dataset_id,
                    "alias": alias,
                    "label_kind": item.label_kind,
                    "gold": item.gold,
                    "gold_in_shortlist": bool(item.gold) and item.gold in shortlist,
                    "shortlist_top_score": round(top_score, 1),
                    "shortlist": "|".join(shortlist),
                    "prompt_sha256": prompt_fingerprint(alias, shortlist, context, options) if asked else None,
                    "suggestion": suggestion,
                    "pre_llm_rejection": pre,
                    "post_llm_rejection": post["reason"] if post else None,
                    "final_unchecked": suggestion,
                    "final_checked": None if (pre or post) else suggestion,
                    "observed_min": profile.minimum if profile else None,
                    "observed_max": profile.maximum if profile else None,
                    "seconds": round(time.perf_counter() - t0, 3),
                    "llm_error": llm_error,
                    "context_sha256": context_sha256,
                })
            logger.info("E1 %s: %d items", source_id, len(group))
    return pd.DataFrame(rows)


def run_e2(args, families: SchemaFamilies) -> pd.DataFrame:
    import yaml

    schema = yaml.safe_load(Path(getattr(args, "schema", None) or DEFAULT_DV_SCHEMA_PATH).read_text(encoding="utf-8"))
    pairs = [
        (str(alias), str(entry["id"]))
        for entry in schema.get("dvs", [])
        if isinstance(entry, dict) and len(entry.get("aliases") or []) >= 2
        for alias in entry["aliases"]
        if isinstance(alias, str) and normalize_name(alias) != normalize_name(entry["id"])
    ]
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(pairs), size=min(args.e2_samples, len(pairs)), replace=False)
    base_mapping = families.standard_mapping_for("questionnaire")
    rows = []
    for index in sorted(picks):
        alias, gold = pairs[index]
        t0 = time.perf_counter()
        shortlist, top_score, suggestion, llm_error = _ask_llm_logged(
            alias, _without_alias(base_mapping, alias), "", args.models, args.shortlist,
            options=AskOptions.from_args(args),
        )
        pre = name_rejection_reason(alias)
        rows.append({
            "experiment": "E2",
            "alias": alias,
            "label_kind": "dv",
            "gold": gold,
            "gold_in_shortlist": gold in shortlist,
            "shortlist_top_score": round(top_score, 1),
            "suggestion": suggestion,
            "pre_llm_rejection": pre,
            "post_llm_rejection": None,
            "final_unchecked": suggestion,
            "final_checked": None if pre else suggestion,
            "seconds": round(time.perf_counter() - t0, 3),
            "llm_error": llm_error,
        })
    return pd.DataFrame(rows)


def summarize(items: pd.DataFrame, dv_ids: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for experiment, exp_items in items.groupby("experiment", sort=True):
        pos = exp_items[exp_items["label_kind"] == "dv"]
        neg = exp_items[exp_items["label_kind"] != "dv"]
        result: dict[str, Any] = {
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "shortlist_contains_gold": wilson(int(pos["gold_in_shortlist"].sum()), len(pos)),
        }
        # Blocklisted columns never reach the LLM in the pipeline; they are
        # asked here only to measure the model's own false-mapping tendency,
        # so precision is computed over what the pipeline could actually emit.
        reachable = exp_items[exp_items["label_kind"] != "blocklisted_non_dv"]
        for condition in ("unchecked", "checked"):
            final = f"final_{condition}"
            correct = int((pos[final] == pos["gold"]).sum())
            accepted_dv = reachable[reachable[final].isin(dv_ids)]
            correct_accepted = int((accepted_dv[final] == accepted_dv["gold"]).sum())
            result[condition] = {
                "accuracy_on_dv_columns": wilson(correct, len(pos)),
                "wrong_dv_on_dv_columns": wilson(
                    int(((pos[final] != pos["gold"]) & pos[final].notna()).sum()), len(pos)
                ),
                "abstained_on_dv_columns": wilson(int(pos[final].isna().sum()), len(pos)),
                "precision_of_accepted_dv_mappings": wilson(correct_accepted, len(accepted_dv)),
                **{
                    f"{kind}_mapped_to_a_dv": wilson(int(group[final].isin(dv_ids).sum()), len(group))
                    for kind, group in neg.groupby("label_kind", sort=True)
                },
            }
        flagged = pos[(pos["final_unchecked"] == pos["gold"]) & pos["final_checked"].isna()]
        result["true_mappings_lost_to_check"] = wilson(len(flagged), int((pos["final_unchecked"] == pos["gold"]).sum()))
        result["rejections_by_reason"] = (
            exp_items["pre_llm_rejection"].fillna(exp_items["post_llm_rejection"]).dropna().value_counts().to_dict()
        )
        out[experiment] = result
    return out


def check_requested_models(requested: list[str] | None) -> list[str]:
    """The models the benchmark will ask, or ``ValueError`` if one it was told to use is not among them.

    ``select_local_model_candidates`` drops a model whose memory estimate does not
    fit, and ``DV_LLM_MODELS`` replaces ``--models``; either would leave the
    benchmark answering with another model, or with none (every item an
    abstention), without an error.
    """
    selected = llm_utils.select_local_model_candidates(requested)
    missing = [model for model in requested or [] if model not in selected]
    if missing:
        override = os.getenv("DV_LLM_MODELS", "").strip()
        reason = (
            f"DV_LLM_MODELS={override!r} replaces --models" if override
            else "its memory estimate exceeds the available memory; for a Gemma 'E' model, --cpu-ple keeps "
                 "its per-layer embeddings on the CPU"
        )
        raise ValueError(f"requested model(s) {', '.join(missing)} would not be used: {reason}")
    if not selected:
        raise ValueError("no LLM fits this machine's memory (select_local_model_candidates returned none)")
    return selected


def model_placement(models: list[str]) -> dict[str, Any]:
    """How each model was loaded: the snapshot folders of a CPU per-layer-embedding placement."""
    out: dict[str, Any] = {}
    for model in models:
        placement = llm_utils.cpu_ple_placement(model)
        out[model] = (
            {"per_layer_embeddings": "cpu", "weights_snapshot": placement.weights_dir.name,
             "tokenizer_snapshot": placement.tokenizer_dir.name, "device": placement.device}
            if placement else {"per_layer_embeddings": "default"}
        )
    return out


def cuda_peak_memory_gb() -> float | None:
    """Largest GPU memory this process allocated (None without CUDA in use)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def load_subsets(specs: list[str]) -> dict[str, set[str]]:
    """``NAME=SOURCES_YAML`` -> {name: source ids listed under the file's ``sources:``}."""
    import yaml

    subsets: dict[str, set[str]] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not name.strip() or not path.strip():
            raise ValueError(f"--subset expects NAME=SOURCES_YAML, got {spec!r}")
        data = yaml.safe_load(Path(path.strip()).read_text(encoding="utf-8")) or {}
        subsets[name.strip()] = {str(source["source_id"]) for source in data.get("sources") or []}
    return subsets


def summarize_subsets(items: pd.DataFrame, dv_ids: set[str], subsets: dict[str, set[str]]) -> dict[str, Any]:
    """E1 summarized separately for each subset of sources and for the remaining sources."""
    e1 = items[items["experiment"] == "E1"]
    out: dict[str, Any] = {}
    for name, source_ids in subsets.items():
        inside = e1["source_id"].isin(source_ids)
        out[name] = summarize(e1[inside], dv_ids).get("E1")
        out[f"not_{name}"] = summarize(e1[~inside], dv_ids).get("E1")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-dir", required=True,
                        help="Output dir of a run WITHOUT LLM deduction (its mappings are the gold labels).")
    parser.add_argument("--cache-dir", required=True, help="Source cache used by that run.")
    parser.add_argument("--output-dir", default="analyses/output_llm_evaluation")
    parser.add_argument("--max-per-source", type=int, default=40,
                        help="Max items per (source, label kind) in E1 (default: %(default)s).")
    parser.add_argument("--e2-samples", type=int, default=300, help="Aliases sampled for E2 (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: pipeline default).")
    parser.add_argument("--shortlist", choices=["fuzzy", *HARNESS_SHORTLISTS],
                        default="definition",
                        help="Candidate retrieval for the LLM (default: %(default)s).")
    parser.add_argument("--subset", action="append", default=[], metavar="NAME=SOURCES_YAML",
                        help="Also summarize E1 for the sources listed in this YAML and for the rest (repeatable).")
    parser.add_argument("--schema", default=None,
                        help="DV schema file read for every definition (default: schemas/standard_dv_mapping.yaml).")
    parser.add_argument("--candidate-order", choices=CANDIDATE_ORDERS, default="rank",
                        help="Order of the shortlist in the prompt (default: %(default)s).")
    parser.add_argument("--order-seed", type=int, default=0, help="Seed of --candidate-order shuffle.")
    parser.add_argument("--samples", type=int, default=1,
                        help="Sampled answers per column, majority vote (default: 1 = greedy).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for --samples > 1.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed set before each sampled generation.")
    parser.add_argument("--cpu-ple", nargs=2, default=None, metavar=("WEIGHTS_DIR", "TOKENIZER_DIR"),
                        help="Load the one --models model (a Gemma 'E' model) with its per-layer embeddings in CPU "
                             "memory, weights and tokenizer from these local snapshot folders "
                             "(llm_utils.configure_cpu_ple; gemma-4-E4B-it on a 16 GB GPU).")
    args = parser.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    if args.cpu_ple:
        if not args.models or len(args.models) != 1:
            parser.error("--cpu-ple needs exactly one model in --models")
        for folder in args.cpu_ple:
            if not Path(folder).is_dir():
                parser.error(f"--cpu-ple: {folder} is not a folder")
        try:
            llm_utils.configure_cpu_ple(args.models[0], *args.cpu_ple)
        except ValueError as exc:
            parser.error(f"--cpu-ple: {exc}")
    try:
        models = check_requested_models(args.models)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        subsets = load_subsets(args.subset)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"--subset: {exc}")

    run_summary = json.loads((Path(args.batch_dir) / "run_summary.json").read_text(encoding="utf-8"))
    if run_summary.get("llm_deductions_count"):
        parser.error("--batch-dir must come from a run without LLM deductions (they would leak into the gold labels)")
    families = SchemaFamilies.load(Path(args.schema) if args.schema else SCHEMAS_DIR / "standard_dv_mapping.yaml")
    dv_ids = {canonical for canonical, domain in families.canonical_domain_lookup.items() if domain == "dv"}

    started = time.perf_counter()
    items = pd.concat([run_e1(args, families, dv_ids), run_e2(args, families)], ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items.to_csv(output_dir / "llm_mapping_evaluation_items.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "model_placement": model_placement(models),
        "cuda_peak_memory_gb": cuda_peak_memory_gb(),
        "plausibility_rules_version": load_rules().version,
        "shortlist": args.shortlist,
        "ask_options": AskOptions.from_args(args).__dict__,
        "dv_schema_version": families.schema_version,
        "seed": args.seed,
        "max_per_source": args.max_per_source,
        "e2_samples": args.e2_samples,
        "blocklist_size": len(NEVER_MAP_NORMALIZED_COLUMNS),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "llm_generation_errors": int(items["llm_error"].notna().sum()),
        "context_sha256": (
            items.dropna(subset=["context_sha256"]).groupby("source_id")["context_sha256"].first().to_dict()
            if "context_sha256" in items else {}
        ),
        "results": summarize(items, dv_ids),
    }
    if subsets:
        summary["e1_by_subset"] = summarize_subsets(items, dv_ids, subsets)
    (output_dir / "llm_mapping_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["results"], indent=2))
    if summary["llm_generation_errors"]:
        # A failed call is not an abstention: these results must not be reported.
        sys.exit(
            f"{summary['llm_generation_errors']} LLM generation call(s) failed; their items count as "
            "abstentions in the summary. Rerun when the GPU is free (llm_error column)."
        )


if __name__ == "__main__":
    main()
