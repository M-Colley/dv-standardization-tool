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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import llm_utils  # noqa: E402
from scripts.batch_profiles import classify_dataset_type, llm_eligible_dataset_types  # noqa: E402
from scripts.dv_retrieval import (  # noqa: E402
    N_DENSE_EXTRA,
    get_dense_retriever,
    get_retriever,
    mix_rankings,
    reciprocal_rank_fusion,
)
from scripts.value_labels import (  # noqa: E402
    apply_reverse_keying,
    apply_value_labels,
    recode_negative_answer_codes,
)
from scripts.llm_coordinator import _select_llm_candidate_shortlist  # noqa: E402
from scripts.llm_plausibility import (  # noqa: E402
    check_llm_suggestion,
    data_rejection_reason,
    load_rules,
    name_rejection_reason,
    normalize_name,
    profile_columns,
)
from scripts.run_batch_standardization import (  # noqa: E402
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


def _ask_llm(
    alias: str,
    mapping: dict[str, str],
    context: str,
    preferred_models: list[str] | None,
    shortlist: str = "definition",
) -> tuple[list[str], float, str | None]:
    candidates, top_score = _select_llm_candidate_shortlist(mapping, alias)
    if shortlist in ("definition", "dense", "mixed", "mixed4") and candidates:
        # Leave-alias-out: the held-out alias is removed from every definition.
        n = len(candidates)
        depth = max(2 * n, 16)
        lexical = [dv for dv, _ in get_retriever().rank(alias, k=depth, exclude_alias=alias)]
        if shortlist != "definition":
            dense = [dv for dv, _ in get_dense_retriever().rank(alias, k=depth, exclude_alias=alias)]
        if shortlist == "dense":
            # Same fusion as scripts.dv_retrieval.dense_shortlist.
            candidates = reciprocal_rank_fusion(lexical, dense)[:n]
        elif shortlist in ("mixed", "mixed4"):
            # mixed: the pipeline option (N_DENSE_EXTRA dense-only candidates);
            # mixed4: exploratory variant with four.
            candidates = mix_rankings(lexical, dense, n, N_DENSE_EXTRA if shortlist == "mixed" else 4)
        else:
            candidates = lexical[:n]
    if not candidates or top_score < MIN_ATTEMPT_SCORE:
        return candidates, top_score, None
    suggestion = llm_utils.deduce_standard_name_with_local_llm(
        raw_column_name=alias,
        canonical_candidates=candidates,
        preferred_models=preferred_models,
        repository_context=context,
    )
    return candidates, top_score, suggestion


def _source_context(source: dict[str, Any], base_dir: Path) -> str:
    extra = _coerce_manifest_text_list(source.get("llm_context"))
    extra.extend(_coerce_manifest_text_list(source.get("publication_context")))
    return llm_utils.collect_repository_context(
        base_dir,
        explicit_dois=_coerce_manifest_text_list(source.get("publication_doi"))
        + _coerce_manifest_text_list(source.get("doi")),
        explicit_pdf_urls=_coerce_manifest_text_list(source.get("publication_pdf_url"))
        + _coerce_manifest_text_list(source.get("pdf_url")),
        extra_context=extra,
    )


def gold_items(meta_view: pd.DataFrame, dv_ids: set[str], max_per_source: int, seed: int) -> pd.DataFrame:
    """One row per (source, alias) with its gold label (canonical DV or None)."""
    mv = meta_view.copy()
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


def run_e1(args, families: SchemaFamilies, dv_ids: set[str]) -> pd.DataFrame:
    sources = {s["source_id"]: s for s in load_manifest(Path(args.manifest))}
    meta_view = pd.read_csv(Path(args.batch_dir) / "meta_view.csv", low_memory=False)
    items = gold_items(meta_view, dv_ids, args.max_per_source, args.seed)
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory() as tmp:
        for source_id, group in items.groupby("source_id", sort=True):
            source = sources[source_id]
            base_dir, _, _ = discover_source_files(source, Path(tmp), cache_root=Path(args.cache_dir))
            context = _source_context(source, base_dir)
            candidates_paths = _find_repository_mapping_candidates(base_dir)
            resolver = _MappingResolver(families, bool(source.get("override_standard_aliases", False)))
            requested = str(source["mapping_path"]).strip() if source.get("mapping_path") else None
            frames: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}
            for item in group.itertuples(index=False):
                relative = Path(item.dataset_id)
                if item.dataset_id not in frames:
                    raw = _load_any_table(base_dir / item.dataset_id)
                    raw_columns = [str(c) for c in raw.columns]
                    dataset_type = classify_dataset_type(relative, raw_columns)
                    mapping_path = _resolve_repository_mapping_path(
                        base_dir, candidates_paths, requested_mapping_path=requested, dataset_path=base_dir / relative
                    )
                    resolved = resolver.resolve(dataset_type, mapping_path, item.dataset_id)
                    # Same response recoding as the pipeline, before profiling.
                    raw, _ = apply_value_labels(raw, resolved.value_labels)
                    if dataset_type in llm_eligible_dataset_types():
                        raw, _ = recode_negative_answer_codes(raw)
                    raw, _ = apply_reverse_keying(raw, resolved.reverse_items)
                    frames[item.dataset_id] = (raw, resolved.mapping)
                df, mapping = frames[item.dataset_id]
                columns = [str(c) for c in df.columns]
                alias = item.original_alias
                profile = profile_columns(df, [alias]).get(alias)
                t0 = time.perf_counter()
                shortlist, top_score, suggestion = _ask_llm(
                    alias, _without_alias(mapping, alias), context, args.models, args.shortlist
                )
                pre = name_rejection_reason(alias, columns) or (data_rejection_reason(profile) if profile else None)
                post = (
                    check_llm_suggestion(alias, suggestion, profile, families.canonical_ranges)
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
                    "suggestion": suggestion,
                    "pre_llm_rejection": pre,
                    "post_llm_rejection": post["reason"] if post else None,
                    "final_unchecked": suggestion,
                    "final_checked": None if (pre or post) else suggestion,
                    "observed_min": profile.minimum if profile else None,
                    "observed_max": profile.maximum if profile else None,
                    "seconds": round(time.perf_counter() - t0, 3),
                })
            logger.info("E1 %s: %d items", source_id, len(group))
    return pd.DataFrame(rows)


def run_e2(args, families: SchemaFamilies) -> pd.DataFrame:
    import yaml

    schema = yaml.safe_load((REPO_ROOT / "schemas" / "standard_dv_mapping.yaml").read_text(encoding="utf-8"))
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
        shortlist, top_score, suggestion = _ask_llm(
            alias, _without_alias(base_mapping, alias), "", args.models, args.shortlist
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
    parser.add_argument("--shortlist", choices=["fuzzy", "definition", "dense", "mixed", "mixed4"],
                        default="definition",
                        help="Candidate retrieval for the LLM (default: %(default)s).")
    args = parser.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    run_summary = json.loads((Path(args.batch_dir) / "run_summary.json").read_text(encoding="utf-8"))
    if run_summary.get("llm_deductions_count"):
        parser.error("--batch-dir must come from a run without LLM deductions (they would leak into the gold labels)")
    families = SchemaFamilies.load(REPO_ROOT / "schemas" / "standard_dv_mapping.yaml")
    dv_ids = {canonical for canonical, domain in families.canonical_domain_lookup.items() if domain == "dv"}

    started = time.perf_counter()
    items = pd.concat([run_e1(args, families, dv_ids), run_e2(args, families)], ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items.to_csv(output_dir / "llm_mapping_evaluation_items.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": llm_utils.select_local_model_candidates(args.models),
        "plausibility_rules_version": load_rules().version,
        "shortlist": args.shortlist,
        "seed": args.seed,
        "max_per_source": args.max_per_source,
        "e2_samples": args.e2_samples,
        "blocklist_size": len(NEVER_MAP_NORMALIZED_COLUMNS),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "results": summarize(items, dv_ids),
    }
    (output_dir / "llm_mapping_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
